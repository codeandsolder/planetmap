use super::bounds::{DirectionCone, FaceRegion, RankedRegion};
use super::geometry::Patch;
use super::{Planet, cube_vector, outer_displaced_radius};
use crate::cubemap::Coords;
use parry3d_f64::bounding_volume::Aabb;
use parry3d_f64::math::{Real, Vector};
use parry3d_f64::query::{Ray, RayCast, RayIntersection};
use parry3d_f64::shape::{FeatureId, Triangle};
use std::collections::BinaryHeap;

/// Robust fallback for rays intersecting close to a triangle's edge.
///
/// Parry's fast triangle test rejects barycentric components strictly below
/// zero; rounding near shared edges can exclude *both* adjacent triangles.
/// The tolerance accounts for the world-space cancellation in `origin - a`
/// relative to the local triangle edge length rather than using a fixed
/// world-space epsilon (planets can be millions of units from their origin).
pub(super) fn cast_triangle_ray_near_edge(
    tri: &Triangle,
    ray: &Ray,
    max_toi: f64,
) -> Option<RayIntersection> {
    let ab = tri.b - tri.a;
    let ac = tri.c - tri.a;
    let normal = ab.cross(ac);
    let denominator = normal.dot(ray.dir);
    if denominator == 0.0 || !denominator.is_finite() {
        return None;
    }
    let time = normal.dot(tri.a - ray.origin) / denominator;
    if !time.is_finite() || !(0.0..=max_toi).contains(&time) {
        return None;
    }
    let hit = ray.origin + ray.dir * time;
    let ap = hit - tri.a;
    let d00 = ab.dot(ab);
    let d01 = ab.dot(ac);
    let d11 = ac.dot(ac);
    let d20 = ap.dot(ab);
    let d21 = ap.dot(ac);
    let det = d00.mul_add(d11, -(d01 * d01));
    if det <= 0.0 || !det.is_finite() {
        return None;
    }
    let v = d11.mul_add(d20, -(d01 * d21)) / det;
    let w = d00.mul_add(d21, -(d01 * d20)) / det;
    let u = 1.0 - (v + w);
    let edge_scale = ab.length().min(ac.length());
    if edge_scale == 0.0 {
        return None;
    }
    let world_scale = ray
        .origin
        .abs()
        .max_element()
        .max(tri.a.abs().max_element())
        .max(tri.b.abs().max_element())
        .max(tri.c.abs().max_element());
    let tolerance = 64.0 * f64::EPSILON * (1.0 + world_scale / edge_scale);
    if ![u, v, w]
        .iter()
        .all(|&component| component >= -tolerance && component <= 1.0 + tolerance)
    {
        return None;
    }
    let outward = normal.normalize();
    let (normal, face) = if denominator < 0.0 {
        (outward, 0)
    } else {
        (-outward, 1)
    };
    Some(RayIntersection::new(time, normal, FeatureId::Face(face)))
}

impl Planet {
    /// Exact ray-triangle intersections inside one chunk, using displaced
    /// quad AABBs only as a conservative early rejection.
    fn cast_ray_in_chunk(
        &self,
        coords: Coords,
        ray: &Ray,
        max_toi: f64,
        solid: bool,
    ) -> Option<(u32, RayIntersection)> {
        let (_, data) = self.cached_chunk(&coords);
        let patch = Patch::new(&coords, self.face_resolution);
        let padding = Vector::splat(
            256.0 * f64::EPSILON * (self.max_radius() + ray.origin.abs().max_element()),
        );
        let mut closest: Option<(u32, RayIntersection)> = None;
        for quad in patch.quads(self.chunk_resolution) {
            let displaced = quad.displace(self.radius, self.chunk_resolution, &data.samples);
            let [a, b, c, d] = displaced.corners;
            let lower = a.min(b).min(c).min(d) - padding;
            let upper = a.max(b).max(c).max(d) + padding;
            let limit = closest.as_ref().map_or(max_toi, |x| x.1.time_of_impact);
            if Aabb::new(lower, upper)
                .cast_local_ray(ray, limit, true)
                .is_none()
            {
                continue;
            }
            for (tri_idx, triangle) in displaced.triangles().into_iter().enumerate() {
                let Some(hit) = triangle
                    .cast_local_ray_and_get_normal(ray, limit, solid)
                    .or_else(|| cast_triangle_ray_near_edge(&triangle, ray, limit))
                else {
                    continue;
                };
                if closest
                    .as_ref()
                    .is_none_or(|best| hit.time_of_impact < best.1.time_of_impact)
                {
                    let index = (quad.index(self.chunk_resolution) << 1) | u32::from(tri_idx == 1);
                    closest = Some((index, hit));
                }
            }
        }
        closest
    }
}

impl Planet {
    /// Fast path for a ray segment that provably remains within one
    /// perspective quad. The four edge half-spaces define a convex cone, so
    /// if both segment endpoints are strictly inside, *every* earlier hit
    /// must be one of the quad's two triangles. Near seams, origin changes,
    /// or deep inward displacements, the global hierarchy remains necessary.
    pub(super) fn cast_ray_inside_one_quad(
        &self,
        ray: &Ray,
        max_toi: f64,
        solid: bool,
    ) -> Option<RayIntersection> {
        if !max_toi.is_finite() || max_toi < 0.0 {
            return None;
        }
        // When moving away from a planet we generally miss its surface;
        // avoid constructing any local geometry in that common case.
        if ray.origin.dot(ray.dir) >= 0.0 && ray.origin.length() > self.max_radius() {
            return None;
        }
        if self.radius / 3.0_f64.sqrt() + f64::from(self.min_height) <= 0.0 {
            return None;
        }
        let end = ray.origin + ray.dir * max_toi;
        if !end.is_finite() {
            return None;
        }
        let coords = Coords::from_vector(self.face_resolution, &cube_vector(ray.origin));
        let patch = Patch::new(&coords, self.face_resolution);
        let quad = patch.quad_at_direction(self.chunk_resolution - 1, &ray.origin);
        let [a, b, c, d] = quad.corners;
        let margin = 512.0
            * f64::EPSILON
            * (self.max_radius() + ray.origin.abs().max_element() + end.abs().max_element());
        // Inward normals of the four great-circle side planes.
        for (v1, v2) in [(c, a), (a, b), (b, d), (d, c)] {
            let normal = v1.cross(v2);
            if normal.dot(ray.origin) <= margin * normal.length()
                || normal.dot(end) <= margin * normal.length()
            {
                return None;
            }
        }
        let (_, data) = self.cached_chunk(&coords);
        let mut closest: Option<(u32, RayIntersection)> = None;
        for (i, triangle) in quad
            .displace(self.radius, self.chunk_resolution, &data.samples)
            .triangles()
            .into_iter()
            .enumerate()
        {
            let Some(hit) = triangle
                .cast_local_ray_and_get_normal(ray, max_toi, solid)
                .or_else(|| cast_triangle_ray_near_edge(&triangle, ray, max_toi))
            else {
                continue;
            };
            if closest
                .as_ref()
                .is_none_or(|x| hit.time_of_impact < x.1.time_of_impact)
            {
                closest = Some((
                    (quad.index(self.chunk_resolution) << 1) | u32::from(i == 1),
                    hit,
                ));
            }
        }
        let (triangle, mut hit) = closest?;
        let (slot, _) = self.cached_chunk(&coords);
        hit.subshape = self.subshape_id(slot, triangle);
        Some(hit)
    }
}

impl RayCast for Planet {
    fn cast_local_ray_and_get_normal(
        &self,
        ray: &Ray,
        max_toi: Real,
        solid: bool,
    ) -> Option<RayIntersection> {
        if max_toi.is_nan()
            || max_toi < 0.0
            || !ray.origin.is_finite()
            || !ray.dir.is_finite()
            || ray.dir == Vector::ZERO
        {
            return None;
        }
        if let Some(hit) = self.cast_ray_inside_one_quad(ray, max_toi, solid) {
            return Some(hit);
        }
        let mut candidates: BinaryHeap<_> = FaceRegion::roots(self.face_resolution)
            .filter_map(|region| RankedRegion::ray(self, region, ray, max_toi))
            .collect();
        // The LRU slot can be reassigned while visiting adjacent patches.
        // Retain the winning tile and triangle until all possible earlier
        // intersections have been excluded.
        let mut closest: Option<(f64, Coords, u32, RayIntersection)> = None;
        while let Some(candidate) = candidates.pop() {
            let limit = closest.as_ref().map_or(max_toi, |x| x.0);
            if candidate.lower_bound > limit {
                break;
            }
            if let Some(coords) = candidate.region.leaf() {
                if let Some((idx, hit)) = self.cast_ray_in_chunk(coords, ray, limit, solid)
                    && closest
                        .as_ref()
                        .is_none_or(|best| hit.time_of_impact < best.0)
                {
                    closest = Some((hit.time_of_impact, coords, idx, hit));
                }
                continue;
            }
            candidates.extend(
                candidate
                    .region
                    .children()
                    .filter_map(|child| RankedRegion::ray(self, child, ray, limit)),
            );
        }
        let (_, coords, tri_idx, mut hit) = closest?;
        let (slot, _) = self.cached_chunk(&coords);
        hit.subshape = self.subshape_id(slot, tri_idx);
        Some(hit)
    }
}

impl RankedRegion {
    fn ray(planet: &Planet, region: FaceRegion, ray: &Ray, max_toi: f64) -> Option<Self> {
        let cone = DirectionCone::from_region(region, planet.face_resolution);
        let radius = planet.radius;
        let min_height = f64::from(planet.min_height);
        let max_height = f64::from(planet.max_height);
        let radial_min = cone.minimum_radius(radius, min_height);
        let radial_max = outer_displaced_radius(radius, min_height, max_height);
        let pad = Vector::splat(
            256.0 * f64::EPSILON * (radial_max + ray.origin.abs().max_element() + radius),
        );
        let aabb = cone.aabb(radial_min, radial_max, pad);
        let lower_time = aabb.cast_local_ray(ray, max_toi, true)?;
        Some(Self {
            region,
            lower_bound: lower_time,
        })
    }
}
