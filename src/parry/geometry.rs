use crate::cubemap::Coords;
use parry3d_f64::math::{Matrix, Vec2, Vector};
use parry3d_f64::shape::Triangle;

/// Direction on a face at `[0..1]^2` coordinates within a chunk
pub(super) fn corner(coords: &Coords, face_resolution: u32, uv: [f64; 2]) -> Vector {
    let dir = coords
        .direction(face_resolution, &na::Point2::from(uv))
        .into_inner();
    Vector::new(dir.x, dir.y, dir.z)
}

/// Quad defined by a chunk pre-displacement
///
/// Generally neither flat nor square.
#[derive(Copy, Clone, Debug)]
pub(super) struct Patch {
    // (0, 0) a--b
    //        |\ |
    //        | \|
    //        c--d (1,1)
    pub(super) a: Vector,
    pub(super) b: Vector,
    pub(super) c: Vector,
    pub(super) d: Vector,
}

impl Patch {
    pub(super) fn new(coords: &Coords, face_resolution: u32) -> Self {
        Self {
            a: corner(coords, face_resolution, [0.0, 0.0]),
            b: corner(coords, face_resolution, [1.0, 0.0]),
            c: corner(coords, face_resolution, [0.0, 1.0]),
            d: corner(coords, face_resolution, [1.0, 1.0]),
        }
    }

    /// Map a point from patch space to a direction in sphere space
    pub(super) fn get(&self, p: &Vec2) -> Vector {
        // Extend the triangle into a parallelogram, then bilinearly interpolate. This guarantees a
        // numerically exact result at each vertex, because in that case every vertex's contribution
        // is multiplied by 0 or 1 exactly and then summed. This precision ensures that there won't
        // be cracks between patches.
        let (b, c) = if p.x > p.y {
            (self.b, self.a + self.d - self.b)
        } else {
            (self.a + self.d - self.c, self.c)
        };
        let y0 = self.a * (1.0 - p.x) + b * p.x;
        let y1 = c * (1.0 - p.x) + self.d * p.x;
        y0 * (1.0 - p.y) + y1 * p.y
    }

    /// Map a direction in sphere space to a point in patch space
    #[inline(always)]
    pub(super) fn project(&self, dir: &Vector) -> Vec2 {
        // Project onto each triangle, then select the in-bounds result
        fn project(p: Vector, x: Vector, y: Vector, dir: Vector) -> Vec2 {
            // t * dir = p + u * x + v * y
            // -p = x * u + y * v - t * dir
            //    = [x y dir] [u v -t]^T
            // [u v -t]^T = [x y dir]^-1 . -p
            let m = Matrix::from_cols(x, y, dir);
            let Some(inverse) = m.try_inverse() else {
                // A tangent direction has no unique ray/patch intersection.
                return Vec2::ZERO;
            };
            -(inverse * p).truncate()
        }

        let left = project(self.a, self.d - self.c, self.c - self.a, *dir);
        let result = if left.x <= left.y {
            left
        } else {
            project(self.a, self.b - self.a, self.d - self.b, *dir)
        };
        result.clamp(Vec2::ZERO, Vec2::ONE)
    }

    pub(super) fn quad_at_direction(&self, side: u32, direction: &Vector) -> Quad {
        let uv = self.project(direction);
        let (x, y) = crate::cubemap::discretize(side, na::Point2::new(uv.x, uv.y));
        Quad::new(self, side, na::Vector2::new(x, y))
    }

    pub(super) fn quads(&self, chunk_resolution: u32) -> impl Iterator<Item = Quad> + '_ {
        let quad_resolution = chunk_resolution - 1;
        (0..quad_resolution).flat_map(move |y| {
            (0..quad_resolution)
                .map(move |x| Quad::new(self, quad_resolution, na::Vector2::new(x, y)))
        })
    }

    #[cfg(test)]
    pub(super) fn triangles<'a>(
        &'a self,
        radius: f64,
        chunk_resolution: u32,
        samples: &'a [f32],
    ) -> impl Iterator<Item = (u32, Triangle)> + 'a {
        self.quads(chunk_resolution).flat_map(move |quad| {
            let index = quad.index(chunk_resolution);
            quad.displace(radius, chunk_resolution, samples)
                .triangles()
                .into_iter()
                .zip(0u32..)
                .map(move |(tri, i)| ((index << 1) | i, tri))
        })
    }
}

/// Identifies a pair of triangles within a patch
pub(super) struct Quad {
    /// Row-major order
    pub(super) corners: [Vector; 4],
    pub(super) position: na::Vector2<u32>,
}

impl Quad {
    pub(super) fn new(patch: &Patch, resolution: u32, position: na::Vector2<u32>) -> Self {
        let offsets = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let position_f = Vec2::new(f64::from(position.x), f64::from(position.y));
        Self {
            corners: offsets
                .map(|x| patch.get(&((position_f + Vec2::from(x)) / f64::from(resolution)))),
            position,
        }
    }

    pub(super) fn index(&self, chunk_resolution: u32) -> u32 {
        self.position.y * (chunk_resolution - 1) + self.position.x
    }

    pub(super) fn displace(
        &self,
        radius: f64,
        chunk_resolution: u32,
        chunk_samples: &[f32],
    ) -> DisplacedQuad {
        let offsets = [[0, 0], [1, 0], [0, 1], [1, 1]];
        let mut result = self.corners;
        for (v, offset) in result.iter_mut().zip(offsets) {
            let sample = self.position + na::Vector2::from(offset);
            let displacement = chunk_samples[(sample.y * chunk_resolution + sample.x) as usize];
            // We deliberately don't normalize `v` in `v * radius` because we're displacing a
            // subdivided patch, not the surface of the sphere directly.
            *v = *v * radius + v.normalize() * f64::from(displacement);
        }
        DisplacedQuad { corners: result }
    }
}

pub(super) struct DisplacedQuad {
    /// Row-major order
    pub(super) corners: [Vector; 4],
}

impl DisplacedQuad {
    pub(super) fn triangles(&self) -> [Triangle; 2] {
        let [p0, p1, p2, p3] = self.corners;
        [Triangle::new(p0, p1, p3), Triangle::new(p3, p2, p0)]
    }
}
