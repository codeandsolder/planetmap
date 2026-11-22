use super::bounds::{FaceRegion, ProjectionQuery, RankedRegion};
use super::geometry::{DisplacedQuad, Patch};
use super::tile_cache::MAX_CACHED_PROJECTION_VERTICES;
use super::{Planet, cube_vector};
use crate::cubemap::Coords;
use parry3d_f64::math::Vector;
use parry3d_f64::query::{PointProjection, PointQuery, Ray, RayCast};
use parry3d_f64::shape::FeatureId;
use std::collections::BinaryHeap;

/// Retain the winning tile rather than a cache slot that may be recycled.
struct ProjectionWinner {
    distance2: f64,
    coords: Coords,
    triangle: u32,
    projection: PointProjection,
    feature: FeatureId,
}

impl Planet {
    /// Evaluate the actual cached chunk geometry without holding the cache
    /// mutex during triangle projection. Its displaced vertices are lazy.
    fn update_projection_from_patch(
        &self,
        coords: Coords,
        pt: Vector,
        query_radius: f64,
        best: &mut Option<ProjectionWinner>,
    ) {
        let (_, data) = self.cached_chunk(&coords);
        // Actual displaced quad vertices bound both triangles by their
        // AABB. Skip Parry's expensive projection unless that AABB can
        // beat the nearest triangle found so far.
        let rounding = 128.0 * f64::EPSILON * (query_radius + self.max_radius() + self.radius);
        let padding = Vector::splat(rounding);
        let evaluate_quad =
            |quad_index: u32, corners: [Vector; 4], best: &mut Option<ProjectionWinner>| {
                let [a, b, c, d] = corners;
                let lower = a.min(b).min(c).min(d) - padding;
                let upper = a.max(b).max(c).max(d) + padding;
                if best.as_ref().is_some_and(|candidate| {
                    pt.clamp(lower, upper).distance_squared(pt) > candidate.distance2
                }) {
                    return;
                }
                for (i, tri) in (DisplacedQuad { corners })
                    .triangles()
                    .into_iter()
                    .enumerate()
                {
                    let (projection, feature) = tri.project_local_point_and_get_feature(pt);
                    let dist2 = projection.point.distance_squared(pt);
                    if best
                        .as_ref()
                        .is_none_or(|candidate| dist2 < candidate.distance2)
                    {
                        *best = Some(ProjectionWinner {
                            distance2: dist2,
                            coords,
                            triangle: (quad_index << 1) | u32::from(i == 1),
                            projection,
                            feature,
                        });
                    }
                }
            };
        if data.samples.len() <= MAX_CACHED_PROJECTION_VERTICES {
            // Reuse the full displaced grid on subsequent queries. The
            // OnceLock is initialized only for projection, not raycasts.
            let vertices = data.projection_vertices(self);
            let width = self.chunk_resolution as usize;
            let quad_width = self.chunk_resolution - 1;
            for y in 0..quad_width {
                for x in 0..quad_width {
                    let index = (y * self.chunk_resolution + x) as usize;
                    let corners = [
                        vertices[index],
                        vertices[index + 1],
                        vertices[index + width],
                        vertices[index + width + 1],
                    ];
                    evaluate_quad(y * quad_width + x, corners, best);
                }
            }
        } else {
            // Keep the API general for very large sample grids without
            // multiplying their persistent memory footprint by six.
            let patch = Patch::new(&coords, self.face_resolution);
            for quad in patch.quads(self.chunk_resolution) {
                let corners = quad
                    .displace(self.radius, self.chunk_resolution, &data.samples)
                    .corners;
                evaluate_quad(quad.index(self.chunk_resolution), corners, best);
            }
        }
    }
}

impl Planet {
    /// Best-first search for points that cannot be fully covered by a small
    /// near-surface angular cap. Search all six faces without preallocation.
    fn search_projection_regions(
        &self,
        pt: Vector,
        query: &ProjectionQuery,
        nearest: &mut Option<ProjectionWinner>,
    ) {
        let mut regions = BinaryHeap::new();
        regions.extend(
            FaceRegion::roots(query.resolution)
                .map(|region| RankedRegion::projection(region, query)),
        );
        while let Some(region) = regions.pop() {
            let best_distance2 = nearest.as_ref().map_or(f64::INFINITY, |x| x.distance2);
            if region.lower_bound > best_distance2 {
                break;
            }
            if let Some(coords) = region.region.leaf() {
                self.update_projection_from_patch(coords, pt, query.query_radius, nearest);
                continue;
            }

            regions.extend(
                region
                    .region
                    .children()
                    .map(|child| RankedRegion::projection(child, query))
                    .filter(|candidate| candidate.lower_bound <= best_distance2),
            );
        }
    }
}

impl Planet {
    /// Classify the radial solid under the *actual triangulated* terrain.
    /// Vertex radii alone are not valid interior bounds: triangles are
    /// chords and may be much closer to the origin than their vertices.
    fn contains_terrain_point(&self, pt: Vector) -> bool {
        if !pt.is_finite() {
            return false;
        }
        let radius = pt.length();
        let max_radius = self.max_radius();
        // Exact mesh vertices may round a few ULPs outside the nominal
        // maximum vertex radius, especially at three-face cube corners.
        let error = 256.0 * f64::EPSILON * max_radius;
        if !radius.is_finite() || radius > max_radius + error {
            return false;
        }
        // Every warped cubemap vertex lies in one of the six face cones
        // with axis projection at least 1/sqrt(3). This is also true of
        // convexly interpolated triangle interiors. For negative heights,
        // the worst possible projected height is the entire negative
        // displacement; for positive heights it is scaled by 1/sqrt(3).
        let min_height = f64::from(self.min_height);
        let inscribed = self.radius / 3.0_f64.sqrt()
            + if min_height >= 0.0 {
                min_height / 3.0_f64.sqrt()
            } else {
                min_height
            };
        if radius + error < inscribed {
            return true;
        }
        // For a radial heightfield with positive vertex/triangle radii,
        // each ray through the origin crosses its surface exactly once.
        // Query from pt outwards: a crossing means the point lies beneath
        // the surface, even at cube-face seams or near the actual mesh.
        // Reuse the same globally exact raycast as other collision queries.
        let axis = if radius > 0.0 { pt / radius } else { Vector::X };
        // Starting exactly on a vertex shared by two or three cube faces
        // can lose a zero-TOI hit to floating-point cancellation. Shift
        // inward by a few hundred ULPs and compare *where* the surface was
        // hit, not merely whether an intersection exists. This also keeps
        // points just outside the terrain classified outside.
        let ray = Ray {
            origin: pt - axis * error,
            dir: axis,
        };
        self.cast_local_ray_and_get_normal(&ray, 2.0_f64.mul_add(error, max_radius - radius), true)
            .is_some_and(|hit| {
                hit.time_of_impact >= (16.0 * f64::EPSILON).mul_add(-max_radius, error)
            })
    }
}

impl Planet {
    pub(super) fn project_surface_point_and_get_feature(
        &self,
        pt: Vector,
    ) -> (PointProjection, FeatureId) {
        // Search the actual nearest triangle, not merely the tile intersected
        // by the ray from the planet center through the query. Tall cliffs,
        // patch boundaries, and points near the center can all have their
        // nearest surface on a different tile or even a different cube face.
        let face_resolution = self.face_resolution;
        let query_radius = pt.length();
        if !query_radius.is_finite() {
            return (PointProjection::new(false, pt), FeatureId::Unknown);
        }
        let query = ProjectionQuery {
            resolution: face_resolution,
            query_radius,
            query_axis: if query_radius > 0.0 {
                pt / query_radius
            } else {
                Vector::X
            },
            planet_radius: self.radius,
            min_height: f64::from(self.min_height),
            max_height: f64::from(self.max_height),
        };
        // The tuple retains the winning grid location rather than a cache
        // slot that may be evicted while visiting other patches.
        let mut nearest: Option<ProjectionWinner> = None;
        // For ordinary near-surface queries, project onto the addressed
        // patch first, then restrict contenders to the angular cap swept by
        // a Euclidean ball of that distance around the query. The cap covers
        // EVERY possible closer surface point, regardless of local cliffs.
        // Only use this path when all surface radii are positive (otherwise
        // patch interpolation can flip the source ray) and the cone fits in
        // a handful of cells. More demanding queries use the hierarchy below.
        let mut cap_searched = false;
        if query_radius > 0.0 {
            let source_direction = cube_vector(query.query_axis);
            let seed = Coords::from_vector(face_resolution, &source_direction);
            self.update_projection_from_patch(seed, pt, query_radius, &mut nearest);
            let distance = nearest
                .as_ref()
                .map_or(f64::INFINITY, |x| x.distance2.sqrt());
            if distance < query_radius
                && self.radius / 3.0_f64.sqrt() + query.min_height
                    > self.radius * 64.0 * f64::EPSILON
            {
                let theta = (distance / query_radius).asin();
                if theta * f64::from(face_resolution) <= 2.0 {
                    for coords in Coords::neighborhood(face_resolution, source_direction, theta) {
                        if coords != seed {
                            self.update_projection_from_patch(
                                coords,
                                pt,
                                query_radius,
                                &mut nearest,
                            );
                        }
                    }
                    cap_searched = true;
                }
            }
        }

        if !cap_searched {
            self.search_projection_regions(pt, &query, &mut nearest);
        }
        let Some(ProjectionWinner {
            coords,
            triangle,
            mut projection,
            feature,
            ..
        }) = nearest
        else {
            return (PointProjection::new(false, pt), FeatureId::Unknown);
        };
        // Classifying containment can sample a different chunk and evict
        // the nearest projection triangle from a small LRU. Assign its live
        // subshape only after all containment raycasts have completed.
        let (slot, _) = self.cached_chunk(&coords);
        projection.subshape = self.subshape_id(slot, triangle);
        (projection, feature)
    }
}

impl PointQuery for Planet {
    fn project_local_point(&self, pt: Vector, solid: bool) -> PointProjection {
        let inside = self.contains_terrain_point(pt);
        if solid && inside {
            return PointProjection::new(true, pt);
        }
        let (mut projection, _) = self.project_surface_point_and_get_feature(pt);
        projection.is_inside = inside;
        projection
    }

    fn contains_local_point(&self, pt: Vector) -> bool {
        self.contains_terrain_point(pt)
    }

    fn project_local_point_and_get_feature(&self, pt: Vector) -> (PointProjection, FeatureId) {
        // Containment can touch other tiles: perform it before the nearest
        // surface search, so returned subshape IDs remain live under eviction.
        let inside = self.contains_terrain_point(pt);
        let (mut projection, feature) = self.project_surface_point_and_get_feature(pt);
        projection.is_inside = inside;
        (projection, feature)
    }
}
