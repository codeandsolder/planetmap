#[cfg(feature = "fearless_simd")]
use fearless_simd::Simd;
use na::RealField;

#[cfg(feature = "fearless_simd")]
use crate::cubemap::SampleIterSimd;
use crate::cubemap::{Coords, Face, SampleIter};

/// A node of a quadtree on a particular cubemap face
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct Chunk {
    coords: Coords,
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
    pub const fn try_new(depth: u8, face: Face, x: u32, y: u32) -> Result<Self, ChunkError> {
        if depth > Self::MAX_DEPTH {
            return Err(ChunkError::Depth(depth));
        }
        match Coords::new(1u32 << depth, face, x, y) {
            Some(coords) => Ok(Self { coords }),
            None => Err(ChunkError::Coordinates),
        }
    }

    /// Coordinates identifying the chunk within its face at the current depth.
    #[must_use]
    pub const fn coords(&self) -> Coords {
        self.coords
    }

    /// Depth of the chunk within the face quadtree.
    #[must_use]
    #[expect(
        clippy::cast_possible_truncation,
        reason = "Validated quadtree resolution is a u32 power of two, so its log2 is at most 31"
    )]
    pub const fn depth(&self) -> u8 {
        self.resolution().ilog2() as u8
    }

    /// The top-level chunk corresponding to a particular cubemap face
    #[must_use]
    pub const fn root(face: Face) -> Self {
        Self {
            coords: Coords::root(face),
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
        let coords = Coords::from_vector(1u32 << depth, dir).ok_or(ChunkError::Direction)?;
        Ok(Self { coords })
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
        Ok(Coords::neighborhood(1u32 << depth, direction, theta).map(|coords| Self { coords }))
    }

    /// The smallest chunk that contains this chunk
    #[must_use]
    pub fn parent(&self) -> Option<Self> {
        let coords = Coords::new(
            self.resolution() / 2,
            self.coords.face(),
            self.coords.x() / 2,
            self.coords.y() / 2,
        )?;
        Some(Self { coords })
    }

    /// Iterator over the path from this chunk to the root, inclusive
    #[must_use]
    pub const fn path(self) -> Path {
        Path { chunk: Some(self) }
    }

    /// The four direct children, or `None` at the maximum representable depth.
    #[must_use]
    pub fn children(&self) -> Option<[Self; 4]> {
        let resolution = self.resolution().checked_mul(2)?;
        let (x, y) = (self.coords.x() * 2, self.coords.y() * 2);
        let face = self.coords.face();
        Some([
            Self {
                coords: Coords::new(resolution, face, x, y)?,
            },
            Self {
                coords: Coords::new(resolution, face, x, y + 1)?,
            },
            Self {
                coords: Coords::new(resolution, face, x + 1, y)?,
            },
            Self {
                coords: Coords::new(resolution, face, x + 1, y + 1)?,
            },
        ])
    }

    /// Chunks that share an edge with this chunk
    #[must_use]
    pub fn neighbors(&self) -> [Self; 4] {
        self.coords.neighbors().map(|coords| Self { coords })
    }

    /// Length of one of this chunk's edges before mapping onto the sphere
    #[must_use]
    pub fn edge_length<N: RealField>(&self) -> N {
        self.coords.edge_length()
    }

    /// Location of the center of this chunk on the surface of the sphere
    ///
    /// Shorthand for `direction(&[0.5; 2].into())`.
    #[must_use]
    pub fn origin<N: RealField + Copy>(&self) -> na::Unit<na::Vector3<N>> {
        self.coords.center()
    }

    /// Returns a grid of resolution^2 directions contained by the chunk, in scan-line order
    #[must_use]
    pub fn samples<N: RealField + Copy>(&self, resolution: u32) -> SampleIter<N> {
        self.coords.samples(resolution)
    }

    /// Returns a grid of resolution^2 directions contained by the chunk, in scan-line order
    ///
    /// Because this returns data in batches of `S::VF32_WIDTH`, a few excess values will be
    /// computed at the end for any `resolution` whose square is not a multiple of the batch size.
    #[cfg(feature = "fearless_simd")]
    #[must_use]
    pub fn samples_ps<S: Simd>(&self, simd: S, resolution: u32) -> SampleIterSimd<S> {
        self.coords.samples_ps(simd, resolution)
    }

    /// Compute the direction identified by a [0..1]^2 vector on this chunk
    #[must_use]
    pub fn direction<N: RealField + Copy + Into<f64>>(
        &self,
        coords: &na::Point2<N>,
    ) -> na::Unit<na::Vector3<N>> {
        self.coords.direction(coords)
    }

    /// Number of chunks at this depth along a cubemap edge, i.e. 2^depth
    #[must_use]
    pub const fn resolution(&self) -> u32 {
        self.coords.resolution()
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
        self.chunk.map_or(0, |x| usize::from(x.depth()) + 1)
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
        let deepest = Chunk::try_new(Chunk::MAX_DEPTH, face, max, max).unwrap();
        assert_eq!(deepest.resolution(), 1_u32 << Chunk::MAX_DEPTH);
        assert!(deepest.children().is_none());
        assert_eq!(deepest.path().len(), usize::from(Chunk::MAX_DEPTH) + 1);
        for neighbor in deepest.neighbors() {
            assert_eq!(neighbor.depth(), Chunk::MAX_DEPTH);
            assert!(neighbor.coords().x() < neighbor.resolution());
            assert!(neighbor.coords().y() < neighbor.resolution());
        }
        assert_eq!(
            Chunk::try_new(Chunk::MAX_DEPTH + 1, face, max, max),
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
        assert_eq!(Chunk::try_new(0, face, 1, 0), Err(ChunkError::Coordinates));
        assert_eq!(Chunk::try_new(0, face, 0, 1), Err(ChunkError::Coordinates));
        let parent = Chunk::try_new(30, face, max / 2, max / 2).unwrap();
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
        assert_eq!(chunk.neighbors()[0], Chunk::try_new(1, Nx, 0, 1).unwrap());
        assert_eq!(chunk.neighbors()[1], Chunk::try_new(1, Ny, 0, 1).unwrap());
        assert_eq!(chunk.neighbors()[2], Chunk::try_new(1, Pz, 1, 0).unwrap());
        assert_eq!(chunk.neighbors()[3], Chunk::try_new(1, Pz, 0, 1).unwrap());

        let chunk = Chunk::try_new(1, Px, 1, 0).unwrap();
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
        let chunk = Chunk::try_new(10, Face::Pz, 12, 34).unwrap();
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
        let left = Chunk::try_new(31, Face::Pz, (1 << 30) - 1, 1 << 30).unwrap();
        let right = left.neighbors()[crate::cubemap::Edge::Px];
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

        let chunk = Chunk::try_new(12, Face::Px, 12, 47).unwrap();
        assert_eq!(
            chunk,
            Chunk::from_vector(chunk.depth(), chunk.origin().as_ref()).unwrap()
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
        let leaf = Chunk::try_new(1, Face::Py, 1, 1).unwrap();
        assert_eq!(
            leaf.path().collect::<Vec<_>>(),
            [leaf, leaf.parent().unwrap()]
        );
    }
}
