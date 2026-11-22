use super::{Planet, outer_displaced_radius};
use crate::cubemap::Coords;
use parry3d_f64::bounding_volume::Aabb;
use parry3d_f64::math::{Pose, Vector};

/// An integer grid rectangle with exclusive upper bounds. Splitting an
/// odd-sized region partitions every cell exactly once, including when
/// only one axis can still be subdivided.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub(super) struct GridRect {
    pub(super) x0: u32,
    pub(super) x1: u32,
    pub(super) y0: u32,
    pub(super) y1: u32,
}

impl GridRect {
    pub(super) const fn new(x0: u32, x1: u32, y0: u32, y1: u32) -> Self {
        Self { x0, x1, y0, y1 }
    }

    pub(super) const fn is_leaf(self) -> bool {
        self.x1 - self.x0 == 1 && self.y1 - self.y0 == 1
    }

    pub(super) fn children(self) -> impl Iterator<Item = Self> {
        let mid_x = u32::midpoint(self.x0, self.x1);
        let mid_y = u32::midpoint(self.y0, self.y1);
        [
            Self::new(self.x0, mid_x, self.y0, mid_y),
            Self::new(self.x0, mid_x, mid_y, self.y1),
            Self::new(mid_x, self.x1, self.y0, mid_y),
            Self::new(mid_x, self.x1, mid_y, self.y1),
        ]
        .into_iter()
        .filter(move |r| !self.is_leaf() && r.x0 < r.x1 && r.y0 < r.y1)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct FaceRegion {
    resolution: u32,
    pub(super) face: crate::cubemap::Face,
    pub(super) cells: GridRect,
}

impl FaceRegion {
    pub(super) const fn new(
        resolution: u32,
        face: crate::cubemap::Face,
        x0: u32,
        x1: u32,
        y0: u32,
        y1: u32,
    ) -> Self {
        Self {
            resolution,
            face,
            cells: GridRect::new(x0, x1, y0, y1),
        }
    }

    pub(super) const fn from_coords(coords: Coords) -> Self {
        Self::new(
            coords.resolution(),
            coords.face(),
            coords.x(),
            coords.x() + 1,
            coords.y(),
            coords.y() + 1,
        )
    }

    pub(super) fn roots(resolution: u32) -> impl Iterator<Item = Self> {
        crate::cubemap::Face::iter()
            .map(move |face| Self::new(resolution, face, 0, resolution, 0, resolution))
    }

    pub(super) const fn leaf(self) -> Option<Coords> {
        if self.cells.is_leaf() {
            Coords::new(self.resolution, self.face, self.cells.x0, self.cells.y0)
        } else {
            None
        }
    }

    pub(super) fn children(self) -> impl Iterator<Item = Self> {
        self.cells
            .children()
            .map(move |cells| Self { cells, ..self })
    }
}

impl PartialOrd for FaceRegion {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FaceRegion {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        (self.resolution, self.face as u8, self.cells).cmp(&(
            other.resolution,
            other.face as u8,
            other.cells,
        ))
    }
}

/// A cubemap region with a conservative lower bound for a single
/// best-first query. Each heap uses its own units (distance squared, ray
/// time, or plane support); the ordering and partition are identical.
#[derive(Clone, Copy, Debug)]
pub(super) struct RankedRegion {
    pub(super) region: FaceRegion,
    pub(super) lower_bound: f64,
}

/// The immutable query geometry shared by all cubemap hierarchy nodes.
pub(super) struct ProjectionQuery {
    pub(super) resolution: u32,
    pub(super) query_axis: Vector,
    pub(super) query_radius: f64,
    pub(super) planet_radius: f64,
    pub(super) min_height: f64,
    pub(super) max_height: f64,
}

/// A conservative unit-direction cone, represented by its maximum chord.
/// Unlike cosine alone, this remains well-conditioned as the aperture
/// approaches zero (e.g. millimetre tiles on a planetary-scale sphere).
#[derive(Clone, Copy, Debug)]
pub(super) struct DirectionCone {
    pub(super) axis: Vector,
    pub(super) chord: f64,
}

impl DirectionCone {
    // Allow for normalization, warp, and vector arithmetic rounding at the
    // unit-sphere scale, rather than inflating every angle by sqrt(epsilon).
    const ROUNDOFF: f64 = 64.0 * f64::EPSILON;

    pub(super) fn from_region(region: FaceRegion) -> Self {
        let FaceRegion {
            resolution,
            face,
            cells: GridRect { x0, x1, y0, y1 },
        } = region;
        let scale = f64::from(resolution);
        let face_direction = |x: f64, y: f64| {
            let d = face.direction(&na::Point2::new(
                x.mul_add(2.0, -scale) / scale,
                y.mul_add(2.0, -scale) / scale,
            ));
            Vector::new(d.x, d.y, d.z)
        };
        let axis = face_direction(
            f64::from(x0).midpoint(f64::from(x1)),
            f64::from(y0).midpoint(f64::from(y1)),
        );
        // The warped projective rectangle is in the positive cone of its
        // corners. Normalizing a convex combination cannot decrease its
        // positive axis dot below the minimum corner dot.
        let mut chord = 0.0_f64;
        for x in [x0, x1] {
            for y in [y0, y1] {
                chord = chord.max(axis.distance(face_direction(f64::from(x), f64::from(y))));
            }
        }
        Self {
            axis,
            chord: (chord + Self::ROUNDOFF).min(2.0),
        }
    }

    fn cosine(self) -> f64 {
        (-0.5 * self.chord).mul_add(self.chord, 1.0)
    }

    fn half_cosine(self) -> f64 {
        (-0.25 * self.chord)
            .mul_add(self.chord, 1.0)
            .max(0.0)
            .sqrt()
    }

    /// Conservative radial lower bound, including chord sag and negative
    /// displacement. A nonpositive bound means angular clipping is unsafe.
    pub(super) fn minimum_radius(self, radius: f64, min_height: f64) -> f64 {
        let cos = (self.cosine() - Self::ROUNDOFF).max(0.0);
        let height = if min_height >= 0.0 {
            min_height * cos
        } else {
            min_height
        };
        radius.mul_add(cos, height)
    }

    /// Smallest unit-direction chord separating a query direction from the
    /// cone. Use the half-angle subtraction identity, without subtracting
    /// nearly equal dot products or taking acos near one.
    fn separation_chord(self, direction: Vector) -> f64 {
        let sin_half = 0.5 * self.axis.distance(direction);
        let cos_half = 0.5 * (self.axis + direction).length();
        let difference = sin_half.mul_add(self.half_cosine(), -cos_half * 0.5 * self.chord);
        difference.mul_add(2.0, -Self::ROUNDOFF).max(0.0)
    }

    /// Support of the cone with any radius in the supplied interval.
    pub(super) fn radial_interval(
        self,
        direction: Vector,
        min_radius: f64,
        max_radius: f64,
    ) -> (f64, f64) {
        if min_radius <= 0.0 {
            return (-max_radius, max_radius);
        }
        let cosine = self.cosine();
        let transverse = self.axis.cross(direction).length() * self.chord * self.half_cosine();
        let center = self.axis.dot(direction).clamp(-1.0, 1.0);
        let low = if (self.axis + direction).length() <= self.chord {
            -1.0
        } else {
            center.mul_add(cosine, -transverse) - Self::ROUNDOFF
        }
        .max(-1.0);
        let high = if self.axis.distance(direction) <= self.chord {
            1.0
        } else {
            center.mul_add(cosine, transverse) + Self::ROUNDOFF
        }
        .min(1.0);
        (
            low * if low >= 0.0 { min_radius } else { max_radius },
            high * if high >= 0.0 { max_radius } else { min_radius },
        )
    }

    fn rotated(self, pose: &Pose) -> Self {
        Self {
            axis: pose.rotation * self.axis,
            ..self
        }
    }

    pub(super) fn aabb(self, min_radius: f64, max_radius: f64, pad: Vector) -> Aabb {
        let (lx, ux) = self.radial_interval(Vector::X, min_radius, max_radius);
        let (ly, uy) = self.radial_interval(Vector::Y, min_radius, max_radius);
        let (lz, uz) = self.radial_interval(Vector::Z, min_radius, max_radius);
        Aabb::new(Vector::new(lx, ly, lz) - pad, Vector::new(ux, uy, uz) + pad)
    }
}

impl RankedRegion {
    pub(super) fn projection(region: FaceRegion, query: &ProjectionQuery) -> Self {
        let cone = DirectionCone::from_region(region);
        let min_radius = cone.minimum_radius(query.planet_radius, query.min_height);
        let max_radius =
            outer_displaced_radius(query.planet_radius, query.min_height, query.max_height);
        let lower_distance2 = if min_radius <= 0.0 {
            // A large negative allowed height can invert a coarse patch's
            // local radial direction. Do not prune unless all possible
            // triangle points have a positive radius in the cone.
            0.0
        } else {
            let chord = cone.separation_chord(query.query_axis);
            let chord_squared = chord * chord;
            let cos_distance = (-0.5 * chord).mul_add(chord, 1.0);
            // Squared distance = radial delta squared + product of the
            // radii times chord squared. Avoid cancellation at small angles.
            let radius = (query.query_radius * cos_distance).clamp(min_radius, max_radius);
            let delta = query.query_radius - radius;
            let bound2 = delta.mul_add(delta, query.query_radius * radius * chord_squared);
            let rounding =
                128.0 * f64::EPSILON * (query.query_radius + max_radius + query.planet_radius);
            (bound2.max(0.0).sqrt() - rounding).max(0.0).powi(2)
        };
        Self {
            region,
            lower_bound: lower_distance2,
        }
    }
}

impl PartialEq for RankedRegion {
    fn eq(&self, other: &Self) -> bool {
        self.lower_bound.total_cmp(&other.lower_bound).is_eq() && self.region == other.region
    }
}
impl Eq for RankedRegion {}
impl PartialOrd for RankedRegion {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for RankedRegion {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .lower_bound
            .total_cmp(&self.lower_bound)
            .then_with(|| self.region.cmp(&other.region))
    }
}

/// One cubemap rectangle and its conservative world-space AABB.
/// The bounds enclose displaced triangle interiors, including seams and
/// deeply negative terrain; they are the same cone bounds as the ray BVH.
#[derive(Clone, Copy, Debug)]
pub(super) struct PatchBounds {
    pub(super) region: FaceRegion,
    pub(super) aabb: Aabb,
    // Shape1 frame; the cone contains all displaced triangle interiors.
    pub(super) origin: Vector,
    pub(super) cone: DirectionCone,
    pub(super) radial_min: f64,
    pub(super) radial_max: f64,
    pub(super) sphere_center: Vector,
    pub(super) sphere_radius: f64,
    pub(super) padding: f64,
}

impl PatchBounds {
    pub(super) fn new(planet: &Planet, pose: &Pose, region: FaceRegion) -> Self {
        let cone = DirectionCone::from_region(region);
        let min_height = f64::from(planet.min_height);
        let max_height = f64::from(planet.max_height);
        let radial_min = cone.minimum_radius(planet.radius, min_height);
        let radial_max = outer_displaced_radius(planet.radius, min_height, max_height);
        let pad = Vector::splat(
            512.0
                * f64::EPSILON
                * (radial_max + planet.radius + pose.translation.abs().max_element()),
        );
        // Convexity of the squared distance to a point on the cone axis
        // makes the extremal radial value occur at an endpoint.
        let world_cone = cone.rotated(pose);
        let (sphere_center, sphere_radius) = if radial_min <= 0.0 {
            (pose.translation, radial_max + pad.x)
        } else {
            let center_radius = radial_min.midpoint(radial_max);
            let radius_at = |radial: f64| {
                let delta = radial - center_radius;
                delta
                    .mul_add(delta, radial * center_radius * cone.chord.powi(2))
                    .sqrt()
            };
            (
                pose.translation + world_cone.axis * center_radius,
                radius_at(radial_min).max(radius_at(radial_max)) + pad.x,
            )
        };
        let aabb = cone.aabb(radial_min, radial_max, pad).transform_by(pose);
        Self {
            region,
            aabb,
            origin: pose.translation,
            cone: world_cone,
            radial_min,
            radial_max,
            sphere_center,
            sphere_radius,
            padding: pad.x,
        }
    }

    /// Signed support interval enclosing every displaced triangle point
    /// along a unit world direction. Inverted cones use the outer sphere.
    pub(super) fn support_interval(&self, direction: Vector) -> (f64, f64) {
        let shift = self.origin.dot(direction);
        let (low, high) = self
            .cone
            .radial_interval(direction, self.radial_min, self.radial_max);
        (shift + low - self.padding, shift + high + self.padding)
    }

    pub(super) const fn is_leaf(&self) -> bool {
        self.region.cells.is_leaf()
    }

    /// Generate ready-to-use child bounds, dropping empty rectangles
    /// before evaluating costly angular cone geometry.
    pub(super) fn children<'a>(
        &self,
        planet: &'a Planet,
        pose: &'a Pose,
    ) -> impl Iterator<Item = Self> + 'a {
        self.region
            .children()
            .map(move |region| Self::new(planet, pose, region))
    }

    pub(super) fn extent_squared(&self) -> f64 {
        (self.aabb.maxs - self.aabb.mins).length_squared()
    }
}
