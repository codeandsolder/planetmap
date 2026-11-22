#[cfg(feature = "fearless_simd")]
use fearless_simd::Simd;
use na::RealField;

#[cfg(feature = "fearless_simd")]
use crate::cubemap::SampleIterSimd;
use crate::cubemap::{Coords, Edge, Face, SampleIter};

/// A node of a quadtree on a particular cubemap face
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct Chunk {
    /// Coordinates within the set of nodes at this depth in the quadtree
    pub(crate) coords: Coords,
    /// Depth in the quadtree
    pub(crate) depth: u8,
}

/// Error constructing a quadtree chunk outside the addressable grid.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChunkError {
    /// More than 31 subdivisions cannot be addressed with `u32` grid resolution.
    Depth(u8),
    /// Chunk coordinates must be smaller than `1 << depth` on both axes.
    Coordinates,
    /// A direction must be finite and nonzero.
    Direction,
    /// A cone half-angle must be finite and nonnegative.
    Angle,
}

impl std::fmt::Display for ChunkError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Depth(depth) => write!(f, "chunk depth {depth} exceeds the maximum of 31"),
            Self::Coordinates => write!(f, "chunk coordinates exceed the quadtree depth"),
            Self::Direction => write!(f, "chunk direction must be finite and nonzero"),
            Self::Angle => write!(f, "cone half-angle must be finite and nonnegative"),
        }
    }
}

impl std::error::Error for ChunkError {}

impl Chunk {
    /// Greatest addressable chunk depth (the grid has `2^31` chunks per face edge).
    pub const MAX_DEPTH: u8 = 31;

    /// Create a chunk after validating its depth and coordinates.
    ///
    /// # Errors
    ///
    /// Rejects depth greater than 31 and coordinates outside the grid at that depth.
    pub const fn try_new(coords: Coords, depth: u8) -> Result<Self, ChunkError> {
        if depth > Self::MAX_DEPTH {
            return Err(ChunkError::Depth(depth));
        }
        let resolution = 1u32 << depth;
        if coords.x >= resolution || coords.y >= resolution {
            return Err(ChunkError::Coordinates);
        }
        Ok(Self { coords, depth })
    }

    /// Coordinates identifying the chunk within its face at the current depth.
    #[must_use]
    pub const fn coords(&self) -> Coords {
        self.coords
    }

    /// Depth of the chunk within the face quadtree.
    #[must_use]
    pub const fn depth(&self) -> u8 {
        self.depth
    }

    /// The top-level chunk corresponding to a particular cubemap face
    #[must_use]
    pub const fn root(face: Face) -> Self {
        Self {
            coords: Coords { x: 0, y: 0, face },
            depth: 0,
        }
    }

    /// Compute the chunk at `depth` that intersects the direction `dir`.
    ///
    /// # Errors
    ///
    /// Rejects depths outside the addressable quadtree and non-finite or zero directions.
    pub fn from_vector(depth: u8, dir: &na::Vector3<f64>) -> Result<Self, ChunkError> {
        if depth > Self::MAX_DEPTH {
            return Err(ChunkError::Depth(depth));
        }
        if !dir.iter().all(|x| x.is_finite()) || dir.iter().all(|&x| x == 0.0) {
            return Err(ChunkError::Direction);
        }
        Self::try_new(Coords::from_vector(1u32 << depth, dir), depth)
    }

    /// Select all `Chunk`s of a certain depth intersecting a cone opening towards `direction` with
    /// half-angle `theta`.
    ///
    /// # Errors
    ///
    /// Rejects depths above 31, non-finite or zero directions, and non-finite
    /// or negative half-angles.
    pub fn neighborhood(
        depth: u8,
        direction: na::Vector3<f64>,
        theta: f64,
    ) -> Result<impl Iterator<Item = Self>, ChunkError> {
        if depth > Self::MAX_DEPTH {
            return Err(ChunkError::Depth(depth));
        }
        if !direction.iter().all(|x| x.is_finite()) || direction.iter().all(|&x| x == 0.0) {
            return Err(ChunkError::Direction);
        }
        if !theta.is_finite() || theta < 0.0 {
            return Err(ChunkError::Angle);
        }
        Ok(Coords::neighborhood(1u32 << depth, direction, theta)
            .map(move |coords| Self { coords, depth }))
    }

    /// The smallest chunk that contains this chunk
    #[must_use]
    pub fn parent(&self) -> Option<Self> {
        let depth = self.depth.checked_sub(1)?;
        Some(Self {
            coords: Coords {
                x: self.coords.x / 2,
                y: self.coords.y / 2,
                face: self.coords.face,
            },
            depth,
        })
    }

    /// Iterator over the path from this chunk to the root, inclusive
    #[must_use]
    pub const fn path(self) -> Path {
        Path { chunk: Some(self) }
    }

    /// The four direct children, or `None` at the maximum representable depth.
    #[must_use]
    pub const fn children(&self) -> Option<[Self; 4]> {
        if self.depth == Self::MAX_DEPTH {
            return None;
        }
        let depth = self.depth + 1;
        let (x, y) = (self.coords.x * 2, self.coords.y * 2);
        let face = self.coords.face;
        Some([
            Self {
                coords: Coords { x, y, face },
                depth,
            },
            Self {
                coords: Coords { x, y: y + 1, face },
                depth,
            },
            Self {
                coords: Coords { x: x + 1, y, face },
                depth,
            },
            Self {
                coords: Coords {
                    x: x + 1,
                    y: y + 1,
                    face,
                },
                depth,
            },
        ])
    }

    /// Chunks that share an edge with this chunk
    #[must_use]
    pub fn neighbors(&self) -> [Self; 4] {
        let x = self.coords.neighbors(self.resolution());
        let depth = self.depth;
        [
            Self {
                coords: x[Edge::Nx],
                depth,
            },
            Self {
                coords: x[Edge::Ny],
                depth,
            },
            Self {
                coords: x[Edge::Px],
                depth,
            },
            Self {
                coords: x[Edge::Py],
                depth,
            },
        ]
    }

    /// Length of one of this chunk's edges before mapping onto the sphere
    #[must_use]
    pub fn edge_length<N: RealField>(&self) -> N {
        Coords::edge_length(self.resolution())
    }

    /// Location of the center of this chunk on the surface of the sphere
    ///
    /// Shorthand for `direction(&[0.5; 2].into())`.
    #[must_use]
    pub fn origin<N: RealField + Copy>(&self) -> na::Unit<na::Vector3<N>> {
        self.coords.center(self.resolution())
    }

    /// Returns a grid of resolution^2 directions contained by the chunk, in scan-line order
    #[must_use]
    pub fn samples<N: RealField + Copy>(&self, resolution: u32) -> SampleIter<N> {
        self.coords.samples(self.resolution(), resolution)
    }

    /// Returns a grid of resolution^2 directions contained by the chunk, in scan-line order
    ///
    /// Because this returns data in batches of `S::VF32_WIDTH`, a few excess values will be
    /// computed at the end for any `resolution` whose square is not a multiple of the batch size.
    #[cfg(feature = "fearless_simd")]
    #[must_use]
    pub fn samples_ps<S: Simd>(&self, simd: S, resolution: u32) -> SampleIterSimd<S> {
        self.coords.samples_ps(simd, self.resolution(), resolution)
    }

    /// Compute the direction identified by a [0..1]^2 vector on this chunk
    #[must_use]
    pub fn direction<N: RealField + Copy + Into<f64>>(
        &self,
        coords: &na::Point2<N>,
    ) -> na::Unit<na::Vector3<N>> {
        self.coords.direction(self.resolution(), coords)
    }

    /// Number of chunks at this depth along a cubemap edge, i.e. 2^depth
    #[must_use]
    pub fn resolution(&self) -> u32 {
        2u32.pow(u32::from(self.depth))
    }
}

pub struct Path {
    chunk: Option<Chunk>,
}

impl Iterator for Path {
    type Item = Chunk;
    fn next(&mut self) -> Option<Chunk> {
        let chunk = self.chunk?;
        self.chunk = chunk.parent();
        Some(chunk)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.len(), Some(self.len()))
    }
}

impl ExactSizeIterator for Path {
    fn len(&self) -> usize {
        self.chunk.map_or(0, |x| x.depth as usize + 1)
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use approx::*;

    #[test]
    fn depth_and_coordinates_are_validated_at_construction() {
        let face = Face::Px;
        let max = (1_u32 << Chunk::MAX_DEPTH) - 1;
        let edge = Coords {
            x: max,
            y: max,
            face,
        };
        let deepest = Chunk::try_new(edge, Chunk::MAX_DEPTH).unwrap();
        assert_eq!(deepest.resolution(), 1_u32 << Chunk::MAX_DEPTH);
        assert!(deepest.children().is_none());
        assert_eq!(deepest.path().len(), usize::from(Chunk::MAX_DEPTH) + 1);
        for neighbor in deepest.neighbors() {
            assert_eq!(neighbor.depth(), Chunk::MAX_DEPTH);
            assert!(neighbor.coords().x < neighbor.resolution());
            assert!(neighbor.coords().y < neighbor.resolution());
        }
        assert_eq!(
            Chunk::try_new(edge, Chunk::MAX_DEPTH + 1),
            Err(ChunkError::Depth(32))
        );
        assert_eq!(
            Chunk::from_vector(32, &na::Vector3::x()),
            Err(ChunkError::Depth(32))
        );
        assert!(matches!(
            Chunk::neighborhood(32, na::Vector3::x(), 0.1),
            Err(ChunkError::Depth(32))
        ));
        assert_eq!(
            Chunk::try_new(Coords { x: 1, y: 0, face }, 0),
            Err(ChunkError::Coordinates)
        );
        assert_eq!(
            Chunk::try_new(Coords { x: 0, y: 1, face }, 0),
            Err(ChunkError::Coordinates)
        );
        let parent = Chunk::try_new(
            Coords {
                x: max / 2,
                y: max / 2,
                face,
            },
            30,
        )
        .unwrap();
        assert!(parent.children().unwrap().contains(&deepest));
    }

    #[test]
    fn direction_and_cone_inputs_are_checked() {
        let zero = na::Vector3::<f64>::zeros();
        let nan = na::Vector3::new(f64::NAN, 1.0, 0.0);
        let infinite = na::Vector3::new(f64::INFINITY, 1.0, 0.0);
        for invalid in [zero, nan, infinite] {
            assert_eq!(Chunk::from_vector(0, &invalid), Err(ChunkError::Direction));
            assert!(matches!(
                Chunk::neighborhood(0, invalid, 0.1),
                Err(ChunkError::Direction)
            ));
        }
        assert!(matches!(
            Chunk::neighborhood(0, na::Vector3::x(), -0.1),
            Err(ChunkError::Angle)
        ));
        assert!(matches!(
            Chunk::neighborhood(0, na::Vector3::x(), f64::INFINITY),
            Err(ChunkError::Angle)
        ));
        assert!(matches!(
            Chunk::neighborhood(0, na::Vector3::x(), f64::NAN),
            Err(ChunkError::Angle)
        ));
    }

    #[test]
    fn neighbors() {
        use self::Face::*;
        // Every face chunk's neighbors are the face's neighbors
        for face in Face::iter() {
            for (&(neighbor_face, _, _), &neighbor_chunk) in
                face.neighbors().iter().zip(&Chunk::root(face).neighbors())
            {
                assert_eq!(Chunk::root(neighbor_face), neighbor_chunk);
            }
        }

        let chunk = Chunk::root(Face::Pz).children().unwrap()[0];
        assert_eq!(
            chunk.neighbors()[0],
            Chunk {
                coords: Coords {
                    x: 0,
                    y: 1,
                    face: Nx
                },
                depth: 1,
            }
        );
        assert_eq!(
            chunk.neighbors()[1],
            Chunk {
                coords: Coords {
                    x: 0,
                    y: 1,
                    face: Ny
                },
                depth: 1,
            }
        );
        assert_eq!(
            chunk.neighbors()[2],
            Chunk {
                coords: Coords {
                    x: 1,
                    y: 0,
                    face: Pz
                },
                depth: 1,
            }
        );
        assert_eq!(
            chunk.neighbors()[3],
            Chunk {
                coords: Coords {
                    x: 0,
                    y: 1,
                    face: Pz
                },
                depth: 1,
            }
        );

        let chunk = Chunk {
            depth: 1,
            coords: Coords {
                x: 1,
                y: 0,
                face: Px,
            },
        };
        assert_eq!(chunk.neighbors()[0b01].neighbors()[0b10], chunk);

        // Every chunk is its neighbor's neighbor
        // Depth 2 to ensure we get all the interesting cases
        for grandparent in Face::iter().map(Chunk::root) {
            for parent in grandparent.children().unwrap().iter() {
                for chunk in parent.children().unwrap().iter() {
                    for &neighbor in chunk.neighbors().iter() {
                        assert!(neighbor.neighbors().iter().any(|x| x == chunk));
                    }
                }
            }
        }
    }

    #[test]
    fn parents() {
        for grandparent in Face::iter().map(Chunk::root) {
            for &parent in grandparent.children().unwrap().iter() {
                assert_eq!(parent.parent(), Some(grandparent));
                for chunk in parent.children().unwrap().iter() {
                    assert_eq!(chunk.parent(), Some(parent));
                }
            }
        }
    }

    #[test]
    fn sample_count() {
        let chunk = Chunk::root(Face::Pz);
        for i in 0..4 {
            assert_eq!(chunk.samples::<f32>(i).count(), (i * i) as usize);
        }
    }

    #[test]
    fn sample_sanity() {
        assert_eq!(
            Chunk::root(Face::Pz).samples::<f32>(1).next().unwrap(),
            na::Vector3::z_axis()
        );

        let chunk = Chunk::root(Face::Pz).children().unwrap()[1];
        assert!(chunk.samples::<f32>(2).any(|x| x == na::Vector3::z_axis()));

        // Every face's
        for face in Face::iter() {
            // immediate children
            for chunk in &Chunk::root(face).children().unwrap() {
                // each have one sample at exactly the center of the face
                assert_eq!(
                    chunk
                        .samples::<f32>(2)
                        .filter(|&x| x == face.basis() * na::Vector3::z_axis())
                        .count(),
                    1
                );
                // and another at a corner
                let corner = 1.0 / 3.0f32.sqrt();
                assert_eq!(
                    chunk
                        .samples::<f32>(2)
                        .filter(|&x| abs_diff_eq!(x.x.abs(), corner)
                            && abs_diff_eq!(x.y.abs(), corner)
                            && abs_diff_eq!(x.z.abs(), corner))
                        .count(),
                    1
                );
            }
        }
    }

    #[test]
    fn sample_lod_boundaries() {
        let chunk = Chunk {
            depth: 10,
            coords: Coords {
                x: 12,
                y: 34,
                face: Face::Pz,
            },
        };
        let children = chunk.children().unwrap();
        let neighbor = chunk.neighbors()[0];
        assert_eq!(
            children[0]
                .samples::<f32>(5)
                .map(|child_vert| neighbor
                    .samples(5)
                    .filter(|&neighbor_vert| neighbor_vert == child_vert)
                    .count())
                .sum::<usize>(),
            3
        );
    }

    #[test]
    fn depth_31_chunk_origin_is_distinct_and_addressable() {
        let left = Chunk::try_new(
            Coords {
                x: (1 << 30) - 1,
                y: 1 << 30,
                face: Face::Pz,
            },
            31,
        )
        .unwrap();
        let right = left.neighbors()[Edge::Px];
        assert_ne!(left.origin::<f64>(), right.origin::<f64>());
        for chunk in [left, right] {
            assert_eq!(Chunk::from_vector(31, &chunk.origin()).unwrap(), chunk);
            assert_eq!(
                chunk.origin::<f64>(),
                chunk.direction(&na::Point2::new(0.5, 0.5))
            );
        }
    }

    #[test]
    fn origin_sanity() {
        assert_eq!(Chunk::root(Face::Pz).origin::<f32>(), na::Vector3::z_axis());

        let chunk = Chunk {
            coords: Coords {
                x: 12,
                y: 47,
                face: Face::Px,
            },
            depth: 12,
        };
        assert_eq!(
            chunk,
            Chunk::from_vector(chunk.depth, chunk.origin().as_ref()).unwrap()
        );

        assert_abs_diff_eq!(chunk.origin(), chunk.direction(&na::Point2::new(0.5, 0.5)));
    }

    #[test]
    #[cfg(feature = "fearless_simd")]
    fn simd_samples() {
        use fearless_simd::{Fallback, SimdBase};

        let chunk = Chunk::root(Face::Pz);

        let mut samples = chunk.samples(5);
        for coords in chunk.samples_ps(Fallback::new(), 5) {
            let [x, y, z] = coords.map(<[f32; <Fallback as Simd>::f32s::LEN]>::from);
            for i in 0..<Fallback as Simd>::f32s::LEN {
                let reference = if let Some(v) = samples.next() {
                    v
                } else {
                    break;
                };
                assert_relative_eq!(x[i], reference.x, epsilon = 1e-4);
                assert_relative_eq!(y[i], reference.y, epsilon = 1e-4);
                assert_relative_eq!(z[i], reference.z, epsilon = 1e-4);
            }
        }
    }

    #[test]
    fn path() {
        let leaf = Chunk {
            coords: Coords {
                x: 1,
                y: 1,
                face: Face::Py,
            },
            depth: 1,
        };
        assert_eq!(
            leaf.path().collect::<Vec<_>>(),
            [leaf, leaf.parent().unwrap()]
        );
    }
}
