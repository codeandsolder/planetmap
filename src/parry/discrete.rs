use super::bounds::{DirectionCone, FaceRegion, PatchBounds, ProjectionQuery, RankedRegion};
use super::broadphase::MAX_DIRECT_CAP_CELLS;
use super::cast::{compute_nonlinear_toi, compute_toi};
use super::geometry::Patch;
use super::{Planet, PlanetDispatcher, cube_vector};
use crate::cubemap::Coords;
use parry3d_f64::bounding_volume::{Aabb, BoundingSphere, BoundingVolume};
use parry3d_f64::math::{Pose, Real, Vector};
use parry3d_f64::query::{
    ClosestPoints, Contact, DefaultQueryDispatcher, NonlinearRigidMotion, QueryDispatcher,
    ShapeCastHit, ShapeCastOptions, ShapeDistance, ShapeIntersection, Unsupported,
};
use parry3d_f64::shape::{HalfSpace, Shape, SubShapeId, Triangle};
use std::collections::BinaryHeap;
use std::sync::Arc;

impl Planet {
    /// Seed proximity from the perspective quad closest to the other shape
    /// center, evaluating just two real triangles. Any valid candidate is a
    /// global *upper* distance bound, never an assumption about which patch
    /// wins; all closer faces are still visited by the proven hierarchy.
    fn visit_proximity_seed_quad(
        &self,
        center: Vector,
        visitor: &mut impl FnMut(Coords, u32, &Triangle) -> bool,
    ) -> bool {
        let axis = if center.length_squared() > 0.0 {
            center
        } else {
            Vector::X
        };
        let resolution = self.face_resolution;
        let coords = Coords::from_vector(resolution, &cube_vector(axis));
        let patch = Patch::new(&coords, resolution);
        let quad = patch.quad_at_direction(self.chunk_resolution - 1, &axis);
        let (_, data) = self.cached_chunk(&coords);
        for (index, triangle) in quad
            .displace(self.radius, self.chunk_resolution, &data.samples)
            .triangles()
            .into_iter()
            .enumerate()
        {
            let triangle_id = (quad.index(self.chunk_resolution) << 1) | u32::from(index == 1);
            if !visitor(coords, triangle_id, &triangle) {
                return false;
            }
        }
        true
    }

    /// A locally complete fast path for positive-radius terrain: if a
    /// triangle can beat the current minimum distance from the other shape,
    /// it must lie within a Euclidean ball of (shape radius + best distance)
    /// around that shape's bounding-sphere center. The angular cone of this
    /// ball covers every possible winner, including across face seams.
    /// Returns None when the cap cannot be certified; the global hierarchy
    /// then remains the correctness-preserving fallback.
    fn visit_proximity_cap(
        &self,
        bounds: &BoundingSphere,
        other_aabb: &Aabb,
        max_distance: &mut Real,
        visitor: &mut impl FnMut(Coords, u32, &Triangle, &mut Real) -> bool,
    ) -> Option<bool> {
        let distance = bounds.center().length();
        let cap_radius = bounds.radius() + *max_distance;
        let inscribed = self.radius / 3.0_f64.sqrt() + f64::from(self.min_height);
        if inscribed <= 64.0 * f64::EPSILON * self.radius
            || !distance.is_finite()
            || distance <= cap_radius
            || !cap_radius.is_finite()
        {
            return None;
        }
        let theta = (cap_radius / distance).asin();
        if theta * f64::from(self.face_resolution) > MAX_DIRECT_CAP_CELLS {
            return None;
        }
        let axis = cube_vector(bounds.center() / distance);
        let face_resolution = self.face_resolution;
        for coords in Coords::neighborhood(face_resolution, axis, theta) {
            let rounding = 512.0
                * f64::EPSILON
                * (self.max_radius() + bounds.center().abs().max_element() + cap_radius);
            let limit = (*max_distance + rounding).max(0.0);
            let query_aabb = other_aabb.loosened(limit);
            let query_bounds = BoundingSphere::new(bounds.center(), bounds.radius() + limit);
            let mut continue_search = true;
            self.map_elements_in_chunk(
                coords,
                &query_bounds,
                &query_aabb,
                &mut |&at, _, index, triangle| {
                    continue_search = visitor(at, index, triangle, max_distance);
                    continue_search
                },
            );
            if !continue_search {
                return Some(false);
            }
        }
        Some(true)
    }

    /// Visit a region's single terrain chunk after distance pruning. A
    /// finite bound allows exact AABB/sphere filtering; only an unsupported
    /// or non-finite initial result forces transient full-chunk inspection.
    fn visit_proximity_leaf(
        &self,
        coords: Coords,
        bounds: &BoundingSphere,
        other_aabb: &Aabb,
        roundoff: f64,
        max_distance: &mut Real,
        visitor: &mut impl FnMut(Coords, u32, &Triangle, &mut Real) -> bool,
    ) -> bool {
        if max_distance.is_infinite() {
            let patch = Patch::new(&coords, self.face_resolution);
            let (_, data) = self.cached_chunk(&coords);
            for quad in patch.quads(self.chunk_resolution) {
                for (index, triangle) in quad
                    .displace(self.radius, self.chunk_resolution, &data.samples)
                    .triangles()
                    .into_iter()
                    .enumerate()
                {
                    let triangle_id =
                        (quad.index(self.chunk_resolution) << 1) | u32::from(index == 1);
                    if !visitor(coords, triangle_id, &triangle, max_distance) {
                        return false;
                    }
                }
            }
            true
        } else {
            let limit = (*max_distance + roundoff).max(0.0);
            let search_aabb = other_aabb.loosened(limit);
            let search_sphere = BoundingSphere::new(bounds.center(), bounds.radius() + limit);
            let mut continue_search = true;
            self.map_elements_in_chunk(
                coords,
                &search_sphere,
                &search_aabb,
                &mut |&at, _, index, triangle| {
                    continue_search = visitor(at, index, triangle, max_distance);
                    continue_search
                },
            );
            continue_search
        }
    }

    /// Explore the complete surface for a shape-to-shape minimum-distance
    /// query. Each region is bounded by its distance to the center of the
    /// other shape's containing sphere, minus that sphere's radius. This is
    /// conservative for *every* actual triangle, including face seams and
    /// inward-flipped heights, and applies to nonconvex other shapes too.
    /// `max_distance` is tightened by the callback as better pairs are found.
    fn visit_proximity_triangles(
        &self,
        pos12: &Pose,
        other: &dyn Shape,
        max_distance: &mut Real,
        mut visitor: impl FnMut(Coords, u32, &Triangle, &mut Real) -> bool,
    ) {
        let bounds = other.compute_bounding_sphere(pos12);
        let center = bounds.center();
        let center_radius = center.length();
        let query = ProjectionQuery {
            resolution: self.face_resolution,
            query_axis: if center_radius > 0.0 {
                center / center_radius
            } else {
                Vector::X
            },
            query_radius: center_radius,
            planet_radius: self.radius,
            min_height: f64::from(self.min_height),
            max_height: f64::from(self.max_height),
        };
        // The two seed triangles establish a finite, geometrically valid
        // upper bound without enumerating every triangle in a chunk.
        if !self.visit_proximity_seed_quad(center, &mut |at, index, triangle| {
            visitor(at, index, triangle, max_distance)
        }) {
            return;
        }
        let other_aabb = other.compute_aabb(pos12);
        if self
            .visit_proximity_cap(&bounds, &other_aabb, max_distance, &mut visitor)
            .is_some()
        {
            return;
        }
        let roundoff = 512.0
            * f64::EPSILON
            * (self.max_radius() + center.abs().max_element() + bounds.radius());
        let mut regions = BinaryHeap::new();
        regions.extend(
            FaceRegion::roots(query.resolution)
                .map(|region| RankedRegion::projection(region, &query)),
        );
        while let Some(region) = regions.pop() {
            let threshold = bounds.radius() + *max_distance + roundoff;
            if region.lower_bound > threshold * threshold {
                break;
            }
            if let Some(coords) = region.region.leaf() {
                if !self.visit_proximity_leaf(
                    coords,
                    &bounds,
                    &other_aabb,
                    roundoff,
                    max_distance,
                    &mut visitor,
                ) {
                    return;
                }
                continue;
            }
            regions.extend(
                region
                    .region
                    .children()
                    .map(|child| RankedRegion::projection(child, &query))
                    .filter(|candidate| candidate.lower_bound <= threshold * threshold),
            );
        }
    }
}

struct PlaneSupportWinner {
    coords: Coords,
    triangle: u32,
    point: Vector,
    dot: f64,
}

impl Planet {
    fn update_plane_support(
        winner: &mut Option<PlaneSupportWinner>,
        normal: Vector,
        coords: Coords,
        triangle: u32,
        shape: &Triangle,
    ) {
        for point in [shape.a, shape.b, shape.c] {
            let dot = point.dot(normal);
            if winner.as_ref().is_none_or(|best| dot < best.dot) {
                *winner = Some(PlaneSupportWinner {
                    coords,
                    triangle,
                    point,
                    dot,
                });
            }
        }
    }

    /// Find the supporting point of the entire triangulated terrain in the
    /// negative plane-normal direction. Unlike finite shapes, `HalfSpace`
    /// has a near-infinite Parry bounding sphere, so the general proximity
    /// hierarchy cannot prune it. A global directional extremum is exact
    /// and requires only a single supporting triangle witness.
    fn supporting_terrain_point(&self, normal: Vector) -> Option<PlaneSupportWinner> {
        let mut winner: Option<PlaneSupportWinner> = None;
        let _ = self.visit_proximity_seed_quad(
            -normal * self.radius,
            &mut |coords, triangle, shape| {
                Self::update_plane_support(&mut winner, normal, coords, triangle, shape);
                true
            },
        );
        let mut regions = BinaryHeap::new();
        let resolution = self.face_resolution;
        regions.extend(
            FaceRegion::roots(resolution).map(|region| RankedRegion::plane(self, region, normal)),
        );
        while let Some(region) = regions.pop() {
            if winner
                .as_ref()
                .is_some_and(|best| region.lower_bound > best.dot)
            {
                break;
            }
            if let Some(coords) = region.region.leaf() {
                let (_, data) = self.cached_chunk(&coords);
                let patch = Patch::new(&coords, resolution);
                for quad in patch.quads(self.chunk_resolution) {
                    for (index, triangle) in quad
                        .displace(self.radius, self.chunk_resolution, &data.samples)
                        .triangles()
                        .into_iter()
                        .enumerate()
                    {
                        let triangle_id =
                            (quad.index(self.chunk_resolution) << 1) | u32::from(index == 1);
                        Self::update_plane_support(
                            &mut winner,
                            normal,
                            coords,
                            triangle_id,
                            &triangle,
                        );
                    }
                }
                continue;
            }
            regions.extend(
                region
                    .region
                    .children()
                    .map(|child| RankedRegion::plane(self, child, normal))
                    .filter(|candidate| {
                        winner
                            .as_ref()
                            .is_none_or(|best| candidate.lower_bound <= best.dot)
                    }),
            );
        }
        // Every validated planet has at least one populated triangle, and
        // the seed evaluates both triangles of a valid quad.
        winner
    }
}

struct PlanetPlaneResult {
    winner: PlaneSupportWinner,
    signed: f64,
    normal: Vector,
}
impl PlanetPlaneResult {
    pub(super) fn new(planet: &Planet, pose: &Pose, plane: &HalfSpace) -> Option<Self> {
        let normal = pose.rotation * plane.normal;
        let winner = planet.supporting_terrain_point(normal)?;
        let signed = winner.dot - normal.dot(pose.translation);
        Some(Self {
            winner,
            signed,
            normal,
        })
    }
}

fn planet_halfspace_distance(
    pose: &Pose,
    planet: &Planet,
    plane: &HalfSpace,
) -> Result<ShapeDistance, Unsupported> {
    let result = PlanetPlaneResult::new(planet, pose, plane).ok_or(Unsupported)?;
    Ok(ShapeDistance::new(result.signed.max(0.0)).with_subshapes(
        planet.stable_triangle_id(result.winner.coords, result.winner.triangle),
        0,
    ))
}

fn planet_halfspace_closest_points(
    pose: &Pose,
    planet: &Planet,
    plane: &HalfSpace,
    max_dist: Real,
) -> Result<ClosestPoints, Unsupported> {
    let result = PlanetPlaneResult::new(planet, pose, plane).ok_or(Unsupported)?;
    if result.signed <= 0.0 {
        Ok(ClosestPoints::Intersecting)
    } else if result.signed <= max_dist {
        let surface = result.winner.point;
        let on_plane = surface - result.normal * result.signed;
        Ok(ClosestPoints::WithinMargin(
            surface,
            pose.inverse_transform_point(on_plane),
        ))
    } else {
        Ok(ClosestPoints::Disjoint)
    }
}

fn planet_halfspace_contact(
    pose: &Pose,
    planet: &Planet,
    plane: &HalfSpace,
    prediction: Real,
) -> Result<Option<Contact>, Unsupported> {
    let result = PlanetPlaneResult::new(planet, pose, plane).ok_or(Unsupported)?;
    if result.signed > prediction {
        return Ok(None);
    }
    let surface = result.winner.point;
    let on_plane = surface - result.normal * result.signed;
    Ok(Some(
        Contact::new(
            surface,
            pose.inverse_transform_point(on_plane),
            -result.normal,
            pose.rotation.inverse() * result.normal,
            result.signed,
        )
        .with_subshapes(
            planet.stable_triangle_id(result.winner.coords, result.winner.triangle),
            0,
        ),
    ))
}

#[derive(Clone, Copy, Debug)]
pub(super) struct PlanetPairRegion {
    pub(super) first: PatchBounds,
    pub(super) second: PatchBounds,
    pub(super) lower: f64,
}

impl PlanetPairRegion {
    pub(super) fn new(first: PatchBounds, second: PatchBounds) -> Self {
        let minkowski = Aabb::new(
            first.aabb.mins - second.aabb.maxs,
            first.aabb.maxs - second.aabb.mins,
        );
        let center_delta = second.sphere_center - first.sphere_center;
        let center_distance = center_delta.length();
        let sphere_bound = (center_distance - first.sphere_radius - second.sphere_radius).max(0.0);
        let between_centers = second.origin - first.origin;
        let axis = if center_distance > 0.0 {
            center_delta / center_distance
        } else if between_centers.length_squared() > 0.0 {
            between_centers.normalize()
        } else {
            Vector::X
        };
        let (_, first_max) = first.support_interval(axis);
        let (second_min, _) = second.support_interval(axis);
        let separation = (second_min - first_max).max(0.0);
        let global_delta = second.origin - first.origin;
        let global_separation = if global_delta.length_squared() > 0.0 {
            let global_axis = global_delta.normalize();
            let (_, global_first_max) = first.support_interval(global_axis);
            let (global_second_min, _) = second.support_interval(global_axis);
            (global_second_min - global_first_max).max(0.0)
        } else {
            0.0
        };
        Self {
            first,
            second,
            lower: minkowski
                .distance_to_origin()
                .max(sphere_bound)
                .max(separation)
                .max(global_separation),
        }
    }
}
impl PartialEq for PlanetPairRegion {
    fn eq(&self, other: &Self) -> bool {
        self.lower.total_cmp(&other.lower).is_eq()
            && self.first.region == other.first.region
            && self.second.region == other.second.region
    }
}
impl Eq for PlanetPairRegion {}
impl PartialOrd for PlanetPairRegion {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for PlanetPairRegion {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .lower
            .total_cmp(&self.lower)
            .then_with(|| self.first.region.cmp(&other.first.region))
            .then_with(|| self.second.region.cmp(&other.second.region))
    }
}

/// Every candidate pair with distance <= limit is visited, regardless of
/// cubemap face, patch boundary, relative rotation or negative displacement.
/// AABBs are in shape1's frame; triangle witness frames are untouched.
/// The callback can reduce limit for nearest queries, or terminate early.
fn visit_planet_pair_triangles(
    first: &Planet,
    second: &Planet,
    pos12: &Pose,
    limit: &mut Real,
    mut visitor: impl FnMut(Coords, u32, Coords, u32, &Triangle, &Triangle, &mut Real) -> bool,
) {
    let pos21 = pos12.inverse();
    let mut seeds_second = Vec::with_capacity(2);
    second.visit_proximity_seed_quad(pos21.translation, &mut |at, id, triangle| {
        seeds_second.push((at, id, *triangle));
        true
    });
    if !first.visit_proximity_seed_quad(pos12.translation, &mut |at1, id1, tri1| {
        for &(at2, id2, tri2) in &seeds_second {
            if !visitor(at1, id1, at2, id2, tri1, &tri2, limit) {
                return false;
            }
        }
        true
    }) {
        return;
    }

    let roundoff = 1024.0
        * f64::EPSILON
        * (first.max_radius() + second.max_radius() + pos12.translation.abs().max_element());
    let mut queue = BinaryHeap::new();
    let res1 = first.face_resolution;
    let res2 = second.face_resolution;
    for region1 in FaceRegion::roots(res1) {
        let a = PatchBounds::new(first, &Pose::IDENTITY, region1);
        for region2 in FaceRegion::roots(res2) {
            let b = PatchBounds::new(second, pos12, region2);
            let candidate = PlanetPairRegion::new(a, b);
            if candidate.lower <= *limit + roundoff {
                queue.push(candidate);
            }
        }
    }
    while let Some(candidate) = queue.pop() {
        if candidate.lower > *limit + roundoff {
            break;
        }
        if let (Some(coords1), Some(coords2)) = (
            candidate.first.region.leaf(),
            candidate.second.region.leaf(),
        ) {
            let enlarged = candidate.second.aabb.loosened(*limit + roundoff);
            let bounds = enlarged.bounding_sphere();
            let mut continue_query = true;
            first.map_elements_in_chunk(coords1, &bounds, &enlarged, &mut |&at1, _, id1, tri1| {
                let tri_box = tri1.compute_aabb(&pos21).loosened(*limit + roundoff);
                let tri_bounds = tri_box.bounding_sphere();
                second.map_elements_in_chunk(
                    coords2,
                    &tri_bounds,
                    &tri_box,
                    &mut |&at2, _, id2, tri2| {
                        continue_query = visitor(at1, id1, at2, id2, tri1, tri2, limit);
                        continue_query
                    },
                );
                continue_query
            });
            if !continue_query {
                return;
            }
            continue;
        }
        let split_first = !candidate.first.is_leaf()
            && (candidate.second.is_leaf()
                || candidate.first.extent_squared() >= candidate.second.extent_squared());
        if split_first {
            for a in candidate.first.children(first, &Pose::IDENTITY) {
                let child = PlanetPairRegion::new(a, candidate.second);
                if child.lower <= *limit + roundoff {
                    queue.push(child);
                }
            }
        } else {
            for b in candidate.second.children(second, pos12) {
                let child = PlanetPairRegion::new(candidate.first, b);
                if child.lower <= *limit + roundoff {
                    queue.push(child);
                }
            }
        }
    }
}

/// Two independent Planets each own their LRU, but a Planet compared
/// with itself shares a single slot table. Resolve both winning triangles
/// while holding one cache lock so the second lookup cannot evict the
/// first. With capacity one and different winning chunks there is no
/// representable pair of simultaneously live Parry subshape IDs.
fn stable_planet_pair_ids(
    first: &Planet,
    at1: Coords,
    tri1: u32,
    second: &Planet,
    at2: Coords,
    tri2: u32,
) -> Result<(SubShapeId, SubShapeId), Unsupported> {
    if std::ptr::eq(first, second) {
        if at1 != at2 && first.lock_cache().max_size < 2 {
            return Err(Unsupported);
        }
        let (_, data1) = first.cached_chunk(&at1);
        let data2 = if at1 == at2 {
            Arc::clone(&data1)
        } else {
            first.cached_chunk(&at2).1
        };
        let mut cache = first.lock_cache();
        let (slot1, _) = cache.insert(data1);
        let (slot2, _) = cache.insert(data2);
        drop(cache);
        Ok((
            first.subshape_id(slot1, tri1),
            first.subshape_id(slot2, tri2),
        ))
    } else {
        Ok((
            first.stable_triangle_id(at1, tri1),
            second.stable_triangle_id(at2, tri2),
        ))
    }
}

fn planet_planet_distance(
    pos12: &Pose,
    first: &Planet,
    second: &Planet,
) -> Result<ShapeDistance, Unsupported> {
    let dispatcher = DefaultQueryDispatcher;
    let mut bound = f64::INFINITY;
    let mut best: Option<(Coords, u32, Coords, u32, ShapeDistance)> = None;
    let mut unsupported = false;
    visit_planet_pair_triangles(
        first,
        second,
        pos12,
        &mut bound,
        |at1, id1, at2, id2, tri1, tri2, limit| {
            dispatcher.distance(pos12, tri1, tri2).map_or_else(
                |_| {
                    unsupported = true;
                    false
                },
                |distance| {
                    if best
                        .as_ref()
                        .is_none_or(|(_, _, _, _, prev)| distance.distance < prev.distance)
                    {
                        *limit = distance.distance;
                        best = Some((at1, id1, at2, id2, distance));
                    }
                    *limit > 0.0
                },
            )
        },
    );
    if unsupported {
        return Err(Unsupported);
    }
    let Some((at1, id1, at2, id2, mut result)) = best else {
        return Err(Unsupported);
    };
    (result.subshape1, result.subshape2) =
        stable_planet_pair_ids(first, at1, id1, second, at2, id2)?;
    Ok(result)
}

fn planet_planet_closest_points(
    pos12: &Pose,
    first: &Planet,
    second: &Planet,
    max_dist: Real,
) -> Result<ClosestPoints, Unsupported> {
    if max_dist.is_nan() || max_dist < 0.0 {
        return Err(Unsupported);
    }
    let dispatcher = DefaultQueryDispatcher;
    let mut bound = max_dist;
    let mut best = ClosestPoints::Disjoint;
    let mut unsupported = false;
    visit_planet_pair_triangles(
        first,
        second,
        pos12,
        &mut bound,
        |_, _, _, _, tri1, tri2, limit| match dispatcher.closest_points(pos12, tri1, tri2, *limit) {
            Ok(ClosestPoints::Intersecting) => {
                best = ClosestPoints::Intersecting;
                false
            }
            Ok(ClosestPoints::WithinMargin(p1, p2)) => {
                let dist = p1.distance(*pos12 * p2);
                if matches!(best, ClosestPoints::Disjoint) || dist < *limit {
                    best = ClosestPoints::WithinMargin(p1, p2);
                    *limit = dist;
                }
                true
            }
            Ok(ClosestPoints::Disjoint) => true,
            Err(_) => {
                unsupported = true;
                false
            }
        },
    );
    if unsupported {
        Err(Unsupported)
    } else {
        Ok(best)
    }
}

fn planet_planet_contact(
    pos12: &Pose,
    first: &Planet,
    second: &Planet,
    prediction: Real,
) -> Result<Option<Contact>, Unsupported> {
    let dispatcher = DefaultQueryDispatcher;
    let mut bound = prediction.max(0.0);
    let mut best: Option<(Coords, u32, Coords, u32, Contact)> = None;
    let mut unsupported = false;
    visit_planet_pair_triangles(
        first,
        second,
        pos12,
        &mut bound,
        |at1, id1, at2, id2, tri1, tri2, limit| {
            // A more deeply penetrating pair cannot have positive
            // geometric separation. Once we know a positive best contact,
            // its signed distance is also a valid upper limit on the
            // spatial distance of any potential replacement. Tighten only
            // nonnegative bounds; negative/deep contacts must still search
            // all intersecting triangle pairs.
            match dispatcher.contact(pos12, tri1, tri2, prediction.min(*limit)) {
                Ok(Some(contact)) => {
                    if best
                        .as_ref()
                        .is_none_or(|(_, _, _, _, prev)| contact.dist < prev.dist)
                    {
                        *limit = (*limit).min(contact.dist.max(0.0));
                        best = Some((at1, id1, at2, id2, contact));
                    }
                    true
                }
                Ok(None) => true,
                Err(_) => {
                    unsupported = true;
                    false
                }
            }
        },
    );
    if unsupported {
        return Err(Unsupported);
    }
    let Some((at1, id1, at2, id2, mut contact)) = best else {
        return Ok(None);
    };
    (contact.subshape1, contact.subshape2) =
        stable_planet_pair_ids(first, at1, id1, second, at2, id2)?;
    Ok(Some(contact))
}

fn planet_planet_intersection(
    pos12: &Pose,
    first: &Planet,
    second: &Planet,
) -> Result<ShapeIntersection, Unsupported> {
    let dispatcher = DefaultQueryDispatcher;
    let mut bound = 0.0;
    let mut best = None;
    let mut unsupported = false;
    visit_planet_pair_triangles(
        first,
        second,
        pos12,
        &mut bound,
        |at1, id1, at2, id2, tri1, tri2, _| match dispatcher.intersection_test(pos12, tri1, tri2) {
            Ok(result) if result.intersecting => {
                best = Some((at1, id1, at2, id2));
                false
            }
            Ok(_) => true,
            Err(_) => {
                unsupported = true;
                false
            }
        },
    );
    if unsupported {
        return Err(Unsupported);
    }
    let Some((at1, id1, at2, id2)) = best else {
        return Ok(ShapeIntersection::new(false));
    };
    let (id1, id2) = stable_planet_pair_ids(first, at1, id1, second, at2, id2)?;
    Ok(ShapeIntersection::new(true).with_subshapes(id1, id2))
}

fn planet_shape_distance(
    pos12: &Pose,
    planet: &Planet,
    other: &dyn Shape,
) -> Result<ShapeDistance, Unsupported> {
    if let Some(plane) = other.as_shape::<HalfSpace>() {
        return planet_halfspace_distance(pos12, planet, plane);
    }
    let dispatcher = DefaultQueryDispatcher;
    let mut max_distance = f64::INFINITY;
    let mut best: Option<(Coords, u32, ShapeDistance)> = None;
    let mut unsupported = false;
    planet.visit_proximity_triangles(
        pos12,
        other,
        &mut max_distance,
        |coords, index, triangle, limit| {
            dispatcher.distance(pos12, triangle, other).map_or_else(
                |_| {
                    unsupported = true;
                    false
                },
                |result| {
                    if best.is_none_or(|(_, _, previous)| result.distance < previous.distance) {
                        *limit = result.distance;
                        best = Some((coords, index, result));
                    }
                    *limit > 0.0
                },
            )
        },
    );
    if unsupported {
        return Err(Unsupported);
    }
    let Some((coords, index, mut result)) = best else {
        return Err(Unsupported);
    };
    result.subshape1 = planet.stable_triangle_id(coords, index);
    Ok(result)
}

fn planet_shape_closest_points(
    pos12: &Pose,
    planet: &Planet,
    other: &dyn Shape,
    max_dist: Real,
) -> Result<ClosestPoints, Unsupported> {
    if max_dist.is_nan() || max_dist < 0.0 {
        return Err(Unsupported);
    }
    if let Some(plane) = other.as_shape::<HalfSpace>() {
        return planet_halfspace_closest_points(pos12, planet, plane, max_dist);
    }
    let dispatcher = DefaultQueryDispatcher;
    let mut limit = max_dist;
    let mut best = ClosestPoints::Disjoint;
    let mut unsupported = false;
    planet.visit_proximity_triangles(pos12, other, &mut limit, |_, _, triangle, limit| {
        match dispatcher.closest_points(pos12, triangle, other, *limit) {
            Ok(ClosestPoints::Intersecting) => {
                best = ClosestPoints::Intersecting;
                false
            }
            Ok(ClosestPoints::WithinMargin(p1, p2)) => {
                let distance = p1.distance(*pos12 * p2);
                if matches!(best, ClosestPoints::Disjoint) || distance < *limit {
                    *limit = distance;
                    best = ClosestPoints::WithinMargin(p1, p2);
                }
                true
            }
            Ok(ClosestPoints::Disjoint) => true,
            Err(_) => {
                unsupported = true;
                false
            }
        }
    });
    if unsupported {
        Err(Unsupported)
    } else {
        Ok(best)
    }
}

pub(super) fn planet_shape_contact(
    pos12: &Pose,
    planet: &Planet,
    other: &dyn Shape,
    prediction: Real,
) -> Result<Option<Contact>, Unsupported> {
    if let Some(plane) = other.as_shape::<HalfSpace>() {
        return planet_halfspace_contact(pos12, planet, plane, prediction);
    }
    let dispatcher = DefaultQueryDispatcher;
    let envelope = other.compute_aabb(pos12).loosened(prediction.max(0.0));
    let bound = envelope.bounding_sphere();
    let mut best: Option<(Coords, u32, Contact)> = None;
    let mut unsupported = false;
    planet.map_elements_in_local_sphere(&bound, &envelope, |&coords, _, index, triangle| {
        match dispatcher.contact(pos12, triangle, other, prediction) {
            Ok(Some(contact)) => {
                if best.is_none_or(|(_, _, previous)| contact.dist < previous.dist) {
                    best = Some((coords, index, contact));
                }
                true
            }
            Ok(None) => true,
            Err(_) => {
                unsupported = true;
                false
            }
        }
    });
    if unsupported {
        return Err(Unsupported);
    }
    let Some((coords, index, mut contact)) = best else {
        return Ok(None);
    };
    contact.subshape1 = planet.stable_triangle_id(coords, index);
    Ok(Some(contact))
}

/// Normalize shape order once for all discrete queries. The specialized
/// pair path preserves both terrain geometries; only the single-planet
/// path needs a result transformation when the input order is reversed.
fn dispatch_discrete<T>(
    pose: &Pose,
    first: &dyn Shape,
    second: &dyn Shape,
    single: impl FnOnce(&Pose, &Planet, &dyn Shape) -> Result<T, Unsupported>,
    pair: impl FnOnce(&Pose, &Planet, &Planet) -> Result<T, Unsupported>,
    swap: impl FnOnce(T) -> T,
) -> Result<T, Unsupported> {
    match (first.as_shape::<Planet>(), second.as_shape::<Planet>()) {
        (Some(a), Some(b)) => pair(pose, a, b),
        (Some(a), None) => single(pose, a, second),
        (None, Some(b)) => single(&pose.inverse(), b, first).map(swap),
        (None, None) => Err(Unsupported),
    }
}

impl QueryDispatcher for PlanetDispatcher {
    fn intersection_test(
        &self,
        pos12: &Pose,
        g1: &dyn Shape,
        g2: &dyn Shape,
    ) -> Result<ShapeIntersection, Unsupported> {
        dispatch_discrete(
            pos12,
            g1,
            g2,
            intersects,
            planet_planet_intersection,
            ShapeIntersection::swapped,
        )
    }

    fn distance(
        &self,
        pos12: &Pose,
        g1: &dyn Shape,
        g2: &dyn Shape,
    ) -> Result<ShapeDistance, Unsupported> {
        dispatch_discrete(
            pos12,
            g1,
            g2,
            planet_shape_distance,
            planet_planet_distance,
            ShapeDistance::swapped,
        )
    }

    fn contact(
        &self,
        pos12: &Pose,
        g1: &dyn Shape,
        g2: &dyn Shape,
        prediction: Real,
    ) -> Result<Option<Contact>, Unsupported> {
        dispatch_discrete(
            pos12,
            g1,
            g2,
            |pose, planet, other| planet_shape_contact(pose, planet, other, prediction),
            |pose, first, second| planet_planet_contact(pose, first, second, prediction),
            |contact| contact.map(Contact::flipped),
        )
    }

    fn closest_points(
        &self,
        pos12: &Pose,
        g1: &dyn Shape,
        g2: &dyn Shape,
        max_dist: Real,
    ) -> Result<ClosestPoints, Unsupported> {
        dispatch_discrete(
            pos12,
            g1,
            g2,
            |pose, planet, other| planet_shape_closest_points(pose, planet, other, max_dist),
            |pose, first, second| planet_planet_closest_points(pose, first, second, max_dist),
            |points| points.flipped(),
        )
    }

    fn cast_shapes(
        &self,
        pos12: &Pose,
        vel12: Vector,
        g1: &dyn Shape,
        g2: &dyn Shape,
        options: ShapeCastOptions,
    ) -> Result<Option<ShapeCastHit>, Unsupported> {
        // The single-planet CCD adapter delegates to Parry's
        // Triangle/other-shape primitive; Planet/Planet is not one of
        // those primitives. Report Unsupported instead of a false miss.
        if g1.as_shape::<Planet>().is_some() && g2.as_shape::<Planet>().is_some() {
            return Err(Unsupported);
        }
        if let Some(p1) = g1.as_shape::<Planet>() {
            return compute_toi(pos12, vel12, p1, g2, options);
        }
        if let Some(p2) = g2.as_shape::<Planet>() {
            return compute_toi(
                &pos12.inverse(),
                pos12.rotation.inverse() * -vel12,
                p2,
                g1,
                options,
            )
            .map(|hit| hit.map(ShapeCastHit::swapped));
        }
        Err(Unsupported)
    }

    fn cast_shapes_nonlinear(
        &self,
        motion1: &NonlinearRigidMotion,
        g1: &dyn Shape,
        motion2: &NonlinearRigidMotion,
        g2: &dyn Shape,
        start_time: Real,
        end_time: Real,
        stop_at_penetration: bool,
    ) -> Result<Option<ShapeCastHit>, Unsupported> {
        // The single-planet CCD adapter delegates to Parry's
        // Triangle/other-shape primitive; Planet/Planet is not one of
        // those primitives. Report Unsupported instead of a false miss.
        if g1.as_shape::<Planet>().is_some() && g2.as_shape::<Planet>().is_some() {
            return Err(Unsupported);
        }
        if let Some(p1) = g1.as_shape::<Planet>() {
            return compute_nonlinear_toi(
                motion1,
                p1,
                motion2,
                g2,
                start_time,
                end_time,
                stop_at_penetration,
                false,
            );
        }
        if let Some(p2) = g2.as_shape::<Planet>() {
            return compute_nonlinear_toi(
                motion2,
                p2,
                motion1,
                g1,
                start_time,
                end_time,
                stop_at_penetration,
                true,
            );
        }
        Err(Unsupported)
    }
}

/// Reports whether `other` intersects `planet`, along with the sub-shapes of both that intersect.
fn intersects(
    pos12: &Pose,
    planet: &Planet,
    other: &dyn Shape,
) -> Result<ShapeIntersection, Unsupported> {
    if let Some(plane) = other.as_shape::<HalfSpace>() {
        let result = PlanetPlaneResult::new(planet, pos12, plane).ok_or(Unsupported)?;
        return Ok(if result.signed <= 0.0 {
            ShapeIntersection::new(true).with_subshapes(
                planet.stable_triangle_id(result.winner.coords, result.winner.triangle),
                0,
            )
        } else {
            ShapeIntersection::new(false)
        });
    }
    let dispatcher = DefaultQueryDispatcher;
    let bounds = other.compute_bounding_sphere(pos12);
    let aabb = other.compute_aabb(pos12);
    let mut hit = Ok(ShapeIntersection::new(false));
    planet.map_elements_in_local_sphere(&bounds, &aabb, |&coords, _, index, triangle| {
        match dispatcher.intersection_test(pos12, triangle, other) {
            Ok(mut result) if result.intersecting => {
                result.subshape1 = planet.stable_triangle_id(coords, index);
                hit = Ok(result);
                false
            }
            Ok(_) => true,
            Err(error) => {
                hit = Err(error);
                false
            }
        }
    });
    hit
}

impl RankedRegion {
    fn plane(planet: &Planet, region: FaceRegion, normal: Vector) -> Self {
        let cone = DirectionCone::from_region(region, planet.face_resolution);
        let min_height = f64::from(planet.min_height);
        let min_radius = cone.minimum_radius(planet.radius, min_height);
        let max_radius = planet.max_radius();
        let (minimum, _) = cone.radial_interval(normal, min_radius, max_radius);
        let slack = 512.0 * f64::EPSILON * (max_radius + planet.radius);
        Self {
            region,
            lower_bound: minimum - slack,
        }
    }
}
