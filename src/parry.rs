//! Collision detection for radial heightmaps
//!
//! Implement [`Terrain`] for your heightmap, then create colliders using it with a [`Planet`]. A
//! query dispatcher that handles both planets and standard shapes can be constructed with
//! `PlanetDispatcher.chain(DefaultQueryDispatcher)`.

use std::{
    collections::BinaryHeap,
    sync::{
        atomic::{AtomicU8, Ordering as AtomicOrdering},
        Arc, Mutex, OnceLock,
    },
};

use glam::UVec2;
use hashbrown::hash_map;
use hashbrown::HashMap;
use lru_slab::LruSlab;
use parry3d_f64::{
    bounding_volume::{Aabb, BoundingSphere, BoundingVolume},
    mass_properties::MassProperties,
    math::{Matrix, Pose, Real, Vec2, Vector},
    query::{
        details::NormalConstraints, ClosestPoints, Contact, ContactManifold,
        ContactManifoldsWorkspace, DefaultQueryDispatcher, NonlinearRigidMotion,
        PersistentQueryDispatcher, PointProjection, PointQuery, QueryDispatcher, Ray, RayCast,
        RayIntersection, ShapeCastHit, ShapeCastOptions, ShapeDistance, ShapeIntersection,
        TypedWorkspaceData, Unsupported, WorkspaceData,
    },
    shape::{CompositeShape, FeatureId, Shape, ShapeType, SubShapeId, Triangle, TypedShape},
    utils::PoseOpt,
};

use crate::cubemap::Coords;

/// Convert a `parry` vector to the nalgebra type consumed by the `cubemap` module
#[expect(
    clippy::cast_possible_truncation,
    reason = "Parry uses f64 world-space vectors, but terrain sampling is intentionally f32"
)]
const fn na3f32(v: Vector) -> na::Vector3<f32> {
    na::Vector3::new(v.x as f32, v.y as f32, v.z as f32)
}

/// Height data source for `Planet`
pub trait Terrain: Send + Sync + 'static {
    /// Generate a `resolution * resolution` grid of heights wrt. sea level
    fn sample(&self, coords: &Coords, out: &mut [f32]);
    /// Number of blocks of samples (chunks) along the edge of a cubemap face
    fn face_resolution(&self) -> u32;
    /// Number of samples along the edge of a single chunk. `sample` will be supplied buffers sized
    /// based for this. Must be at least 2.
    fn chunk_resolution(&self) -> u32;
    /// The maximum value that will ever be written by `sample`
    fn max_height(&self) -> f32;
    /// The minimum value that will ever be written by `sample`
    fn min_height(&self) -> f32;
}

/// Invalid radial terrain or collision-cache configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlanetConfigError {
    /// A cubemap face has no chunks.
    FaceResolution,
    /// At least two samples per chunk edge are needed to form triangles.
    ChunkResolution,
    /// The sample grid cannot fit within Rust allocation limits.
    SampleGridTooLarge,
    /// Collision-cache capacity must be positive.
    CacheSize,
    /// The planet radius or terrain-adjusted radius is not finite and positive.
    Radius,
    /// Height extrema must be finite and ordered.
    HeightRange,
    /// Parry's 32-bit subshape identifiers cannot represent the proposed cache.
    SubshapeIdOverflow,
}

impl std::fmt::Display for PlanetConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FaceResolution => write!(f, "face resolution must be positive"),
            Self::ChunkResolution => write!(f, "chunk resolution must be at least two"),
            Self::SampleGridTooLarge => write!(f, "terrain sample grid exceeds allocation limits"),
            Self::CacheSize => write!(f, "collision cache size must be positive"),
            Self::Radius => write!(
                f,
                "planet radius must remain finite and positive with terrain heights"
            ),
            Self::HeightRange => write!(f, "height bounds must be finite and ordered"),
            Self::SubshapeIdOverflow => write!(
                f,
                "cache and triangle count exceed 32-bit subshape identifier space"
            ),
        }
    }
}

impl std::error::Error for PlanetConfigError {}

fn validate_terrain_geometry(
    face_resolution: u32,
    chunk_resolution: u32,
) -> Result<(), PlanetConfigError> {
    if face_resolution == 0 {
        return Err(PlanetConfigError::FaceResolution);
    }
    if chunk_resolution < 2 {
        return Err(PlanetConfigError::ChunkResolution);
    }
    let sample_count = u64::from(chunk_resolution).pow(2);
    let max_elements = (isize::MAX as u64) / size_of::<f32>() as u64;
    if sample_count > max_elements {
        return Err(PlanetConfigError::SampleGridTooLarge);
    }
    Ok(())
}

/// Upper bound on the distance of any displaced vertex (and therefore
/// any point in a triangle) from the planet center. A sufficiently negative
/// displacement can cross the origin and reverse its direction, so
/// `radius + max_height` alone is not always an upper bound.
fn outer_displaced_radius(radius: f64, min_height: f64, max_height: f64) -> f64 {
    (radius + max_height).max(-min_height)
}

/// Trivial `Terrain` impl
#[derive(Debug, Copy, Clone)]
pub struct FlatTerrain {
    face_resolution: u32,
    chunk_resolution: u32,
}

impl FlatTerrain {
    /// Construct a flat terrain with a valid cubemap and triangle grid.
    ///
    /// # Errors
    ///
    /// Rejects zero face resolution or chunk resolutions that cannot form an
    /// addressable sample grid with at least two vertices per edge.
    pub fn new(face_resolution: u32, chunk_resolution: u32) -> Result<Self, PlanetConfigError> {
        validate_terrain_geometry(face_resolution, chunk_resolution)?;
        Ok(Self {
            face_resolution,
            chunk_resolution,
        })
    }
}

impl Terrain for FlatTerrain {
    fn face_resolution(&self) -> u32 {
        self.face_resolution
    }

    fn chunk_resolution(&self) -> u32 {
        self.chunk_resolution
    }

    fn max_height(&self) -> f32 {
        0.0
    }
    fn min_height(&self) -> f32 {
        0.0
    }

    fn sample(&self, _: &Coords, out: &mut [f32]) {
        for x in out {
            *x = 0.0;
        }
    }
}

/// A fixed-resolution partially-resident radial heightmap
///
/// Generates height data on-demand via `Terrain`, preserving it in a fixed-size LRU cache.
pub struct Planet {
    terrain: Arc<dyn Terrain>,
    radius: f64,
    chunk_resolution: u32,
    // Future work: could preallocate an arena for height samples
    cache: Mutex<Cache>,
}

impl Planet {
    /// Construct a new collision shape for a radial heightmap defined by `terrain`
    ///
    /// `terrain` - source of height samples
    /// `cache_size` - maximum number of chunks of height data to keep in memory
    /// `radius` - distance from origin of points with height 0
    ///
    /// # Errors
    ///
    /// Rejects invalid source geometry, non-finite/negative radial extents,
    /// zero cache capacity, and cache geometries that would alias Parry's
    /// 32-bit triangle subshape identifiers.
    pub fn new(
        terrain: Arc<dyn Terrain>,
        cache_size: u32,
        radius: f64,
    ) -> Result<Self, PlanetConfigError> {
        let chunk_resolution = terrain.chunk_resolution();
        validate_terrain_geometry(terrain.face_resolution(), chunk_resolution)?;
        if cache_size == 0 {
            return Err(PlanetConfigError::CacheSize);
        }
        let (min, max) = (terrain.min_height(), terrain.max_height());
        if !min.is_finite() || !max.is_finite() || min > max {
            return Err(PlanetConfigError::HeightRange);
        }
        if !radius.is_finite()
            || radius <= 0.0
            || !((radius + f64::from(min)).is_finite())
            || radius + f64::from(min) <= 0.0
            || !(radius + f64::from(max)).is_finite()
        {
            return Err(PlanetConfigError::Radius);
        }
        // Every cached slot owns two triangles per quad. The maximum
        // identifier, including the last slot, must fit in SubShapeId=u32.
        let quads_per_edge = u64::from(chunk_resolution - 1);
        let triangles_per_chunk = 2 * quads_per_edge * quads_per_edge;
        let total = u64::from(cache_size) * triangles_per_chunk;
        if total > u64::from(u32::MAX) + 1 {
            return Err(PlanetConfigError::SubshapeIdOverflow);
        }
        Ok(Self {
            chunk_resolution,
            terrain,
            radius,
            cache: Mutex::new(Cache::new(cache_size)),
        })
    }

    fn max_radius(&self) -> f64 {
        outer_displaced_radius(
            self.radius,
            f64::from(self.terrain.min_height()),
            f64::from(self.terrain.max_height()),
        )
    }

    fn sample(&self, coords: &Coords) -> Box<[f32]> {
        let mut samples =
            vec![0.0; self.chunk_resolution as usize * self.chunk_resolution as usize]
                .into_boxed_slice();
        self.terrain.sample(coords, &mut samples[..]);
        samples
    }

    /// Number of triangles in a single chunk, i.e. the stride between chunk slots in
    /// [`Self::subshape_id`].
    const fn triangles_per_chunk(&self) -> u32 {
        2 * (self.chunk_resolution - 1).pow(2)
    }

    /// Identifier of a triangle's part of the shape.
    ///
    /// Unique among the triangles currently cached, though a triangle's ID changes if its chunk
    /// is evicted and reloaded into a different slot.
    const fn subshape_id(&self, slot: u32, triangle: u32) -> SubShapeId {
        slot * self.triangles_per_chunk() + triangle
    }

    /// Evaluate a candidate tile without holding the cache lock during
    /// callbacks; return false when the visitor requested an early exit.
    fn map_elements_in_chunk(
        &self,
        chunk_coords: Coords,
        bounds: &BoundingSphere,
        aabb: &Aabb,
        visitor: &mut impl FnMut(&Coords, u32, u32, &Triangle) -> bool,
    ) -> bool {
        let dir = bounds.center();
        let distance = dir.length();
        // Never call an application-provided callback with our cache lock
        // held. An Arc keeps the geometry alive even if a nested query
        // evicts this slot before the callback completes.
        let (slot, data) = self
            .cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(self, &chunk_coords);
        // Both triangle interiors and displaced vertices lie in this
        // patch's angular cone; unlike the sample vertex radii, its
        // projected radial minimum is conservative for triangle chords.
        let (_, min_cos) = ProjectionRegion::cone(
            chunk_coords.face,
            chunk_coords.x,
            chunk_coords.x + 1,
            chunk_coords.y,
            chunk_coords.y + 1,
            self.terrain.face_resolution(),
        );
        let min_height = f64::from(data.min);
        let min_height_projection = if min_height >= 0.0 {
            min_height * min_cos
        } else {
            min_height
        };
        let minimum_radius = self.radius.mul_add(min_cos, min_height_projection);
        if outer_displaced_radius(self.radius, f64::from(data.min), f64::from(data.max))
            + bounds.radius()
            < distance
            || minimum_radius - bounds.radius() > distance
        {
            return true;
        }
        let patch = Patch::new(&chunk_coords, self.terrain.face_resolution());
        let mut emit = |quad: Quad| {
            for (i, triangle) in quad
                .displace(self.radius, self.chunk_resolution, &data.samples)
                .triangles()
                .into_iter()
                .enumerate()
            {
                let index = (quad.index(self.chunk_resolution) << 1) | u32::from(i == 1);
                if !visitor(&chunk_coords, slot, index, &triangle) {
                    return false;
                }
            }
            true
        };
        let side = self.chunk_resolution - 1;
        if data.samples.len() <= MAX_CACHED_PROJECTION_VERTICES
            && data.quad_bounds.get().is_none()
            && data
                .broadphase_queries
                .fetch_add(1, AtomicOrdering::Relaxed)
                < QUAD_INDEX_BUILD_AFTER_QUERIES
        {
            // Most one-off collision queries touch only a handful of
            // quads. Use conservative affine patch bounds and never build
            // the entire vertex grid or index for these cold chunks.
            let sparse = SparseQuadSearch {
                patch: &patch,
                samples: &data.samples,
                query: aabb,
                radius: self.radius,
                height_abs: f64::from(data.min).abs().max(f64::from(data.max).abs()),
                resolution: self.chunk_resolution,
            };
            if !sparse.visit(
                QuadCellRegion {
                    x0: 0,
                    x1: side,
                    y0: 0,
                    y1: side,
                },
                &mut emit,
            ) {
                return false;
            }
        } else if data.samples.len() <= MAX_CACHED_PROJECTION_VERTICES {
            // Frequently reused chunks switch to exact cached quad AABBs
            // and row/block summaries, preserving the warm fast path.
            // A cache of exact quad AABBs rejects non-overlapping quads
            // without reconstructing their displaced triangle geometry.
            let boxes = data.quad_bounds(self);
            let blocks_per_row = side.div_ceil(QUAD_BLOCK_WIDTH);
            for y in 0..side {
                if !boxes.rows[y as usize].intersects(aabb) {
                    continue;
                }
                for block in 0..blocks_per_row {
                    if !boxes.blocks[(y * blocks_per_row + block) as usize].intersects(aabb) {
                        continue;
                    }
                    let x_start = block * QUAD_BLOCK_WIDTH;
                    for x in x_start..(x_start + QUAD_BLOCK_WIDTH).min(side) {
                        let i = (y * side + x) as usize;
                        if boxes.quads[i].intersects(aabb)
                            && !emit(Quad::new(&patch, side, UVec2::new(x, y)))
                        {
                            return false;
                        }
                    }
                }
            }
        } else {
            // Extremely large chunks use bounded transient memory,
            // with exactly the same world-space quad AABB semantics.
            for quad in patch.quads(self.chunk_resolution) {
                let displaced = quad.displace(self.radius, self.chunk_resolution, &data.samples);
                let [top_left, top_right, bottom_left, bottom_right] = displaced.corners;
                let lower = top_left.min(top_right).min(bottom_left).min(bottom_right);
                let upper = top_left.max(top_right).max(bottom_left).max(bottom_right);
                if Aabb::new(lower, upper).intersects(aabb) && !emit(quad) {
                    return false;
                }
            }
        }
        true
    }

    /// Applies `f` to potentially overlapping displaced terrain triangles.
    /// Returns early when `f` returns false. The callback executes without
    /// the cache mutex and may reenter this planet. Cache slot identifiers
    /// are ephemeral and can be recycled by nested calls.
    ///
    /// Positive-radius radial terrain supports fast angular neighborhood
    /// discovery. Inward-displaced triangles and query spheres centered at
    /// the origin instead use conservative best-first global region bounds.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "The angular neighborhood uses f32 with a conservative 1e-6-radian pad"
    )]
    pub fn map_elements_in_local_sphere(
        &self,
        bounds: &BoundingSphere,
        aabb: &Aabb,
        mut f: impl FnMut(&Coords, u32, u32, &Triangle) -> bool,
    ) {
        let dir = bounds.center();
        let distance = dir.length();
        let min_height = f64::from(self.terrain.min_height());
        let max_height = f64::from(self.terrain.max_height());
        let face_resolution = self.terrain.face_resolution();
        // A full cube face has minimum axis projection 1/sqrt(3).
        // This bound covers triangle *interiors*, not only vertices.
        let inscribed = self.radius / 3.0_f64.sqrt()
            + if min_height >= 0.0 {
                min_height / 3.0_f64.sqrt()
            } else {
                min_height
            };
        let direction32 = na3f32(dir);
        if inscribed > 256.0 * f64::EPSILON * self.radius
            && distance.is_finite()
            && distance > 0.0
            && direction32.iter().all(|v| v.is_finite())
            && direction32.norm_squared() > 0.0
        {
            // A query bounding sphere at distance d subtends asin(r/d);
            // Coords::neighborhood adds 1e-6 radians for f32 discretization.
            let theta = if distance > bounds.radius() {
                (bounds.radius() / distance).asin()
            } else {
                std::f64::consts::PI
            };
            for coords in Coords::neighborhood(face_resolution, direction32, theta as f32) {
                if !self.map_elements_in_chunk(coords, bounds, aabb, &mut f) {
                    return;
                }
            }
            return;
        }

        // A deeply inward-displaced triangle can point toward an *opposite*
        // cube face; querying only the sphere's angular neighborhood is
        // unsound. Use the proof-backed global projection region lower
        // distance bound to visit every potentially intersecting tile.
        let query = ProjectionQuery {
            resolution: face_resolution,
            query_radius: if distance.is_finite() { distance } else { 0.0 },
            query_axis: if distance.is_finite() && distance > 0.0 {
                dir / distance
            } else {
                Vector::X
            },
            planet_radius: self.radius,
            min_height,
            max_height,
        };
        let threshold2 = bounds.radius().powi(2);
        let mut regions = BinaryHeap::new();
        for face in crate::cubemap::Face::iter() {
            regions.push(ProjectionRegion::new(
                face,
                0,
                face_resolution,
                0,
                face_resolution,
                &query,
            ));
        }
        while let Some(region) = regions.pop() {
            if region.lower_distance2 > threshold2 {
                break;
            }
            if region.x1 - region.x0 == 1 && region.y1 - region.y0 == 1 {
                let coords = Coords {
                    face: region.face,
                    x: region.x0,
                    y: region.y0,
                };
                if !self.map_elements_in_chunk(coords, bounds, aabb, &mut f) {
                    return;
                }
                continue;
            }
            let mid_x = region.x0 + (region.x1 - region.x0) / 2;
            let mid_y = region.y0 + (region.y1 - region.y0) / 2;
            for (x0, x1) in [(region.x0, mid_x), (mid_x, region.x1)] {
                if x0 == x1 {
                    continue;
                }
                for (y0, y1) in [(region.y0, mid_y), (mid_y, region.y1)] {
                    if y0 == y1 {
                        continue;
                    }
                    let candidate = ProjectionRegion::new(region.face, x0, x1, y0, y1, &query);
                    if candidate.lower_distance2 <= threshold2 {
                        regions.push(candidate);
                    }
                }
            }
        }
    }
}

impl Clone for Planet {
    fn clone(&self) -> Self {
        Self {
            terrain: self.terrain.clone(),
            cache: Mutex::new(
                self.cache
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .clone(),
            ),
            ..*self
        }
    }
}

/// Robust fallback for rays intersecting close to a triangle's edge.
///
/// Parry's fast triangle test rejects barycentric components strictly below
/// zero; rounding near shared edges can exclude *both* adjacent triangles.
/// The tolerance accounts for the world-space cancellation in `origin - a`
/// relative to the local triangle edge length rather than using a fixed
/// world-space epsilon (planets can be millions of units from their origin).
fn cast_triangle_ray_near_edge(tri: &Triangle, ray: &Ray, max_toi: f64) -> Option<RayIntersection> {
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

/// A ray-overlapping cubemap rectangle ordered by its earliest possible TOI.
/// AABB bounds enclose every displaced triangle interior, not just vertices.
#[derive(Clone, Copy, Debug)]
struct RayRegion {
    region: ProjectionRegion,
    lower_time: f64,
}

impl RayRegion {
    #[expect(
        clippy::too_many_arguments,
        reason = "Conservative bound depends on terrain extrema and grid rectangle"
    )]
    fn new(
        face: crate::cubemap::Face,
        x0: u32,
        x1: u32,
        y0: u32,
        y1: u32,
        resolution: u32,
        radius: f64,
        min_height: f64,
        max_height: f64,
        ray: &Ray,
        max_toi: f64,
    ) -> Option<Self> {
        let (axis, min_cos) = ProjectionRegion::cone(face, x0, x1, y0, y1, resolution);
        let region = ProjectionRegion {
            face,
            x0,
            x1,
            y0,
            y1,
            lower_distance2: 0.0,
        };
        let min_height_projection = if min_height >= 0.0 {
            min_height * min_cos
        } else {
            min_height
        };
        let radial_min = radius.mul_add(min_cos, min_height_projection);
        let radial_max = outer_displaced_radius(radius, min_height, max_height);
        let pad = Vector::splat(
            256.0 * f64::EPSILON * (radial_max + ray.origin.abs().max_element() + radius),
        );
        let aabb = if radial_min <= 0.0 {
            // An unusually deep inward displacement can invalidate angular
            // bounds. A full containing cube remains correct and eventually
            // subdivides to narrower, positive-radius regions.
            Aabb::from_half_extents(Vector::ZERO, Vector::splat(radial_max) + pad)
        } else {
            let sin_cone = min_cos.mul_add(-min_cos, 1.0).max(0.0).sqrt();
            let component_bounds = |component: f64| {
                let perpendicular = component.mul_add(-component, 1.0).max(0.0).sqrt() * sin_cone;
                let min_component = if -component >= min_cos {
                    -1.0
                } else {
                    component.mul_add(min_cos, -perpendicular)
                };
                let max_component = if component >= min_cos {
                    1.0
                } else {
                    component.mul_add(min_cos, perpendicular)
                };
                let lower = min_component
                    * if min_component >= 0.0 {
                        radial_min
                    } else {
                        radial_max
                    };
                let upper = max_component
                    * if max_component >= 0.0 {
                        radial_max
                    } else {
                        radial_min
                    };
                (lower, upper)
            };
            let (lx, ux) = component_bounds(axis.x);
            let (ly, uy) = component_bounds(axis.y);
            let (lz, uz) = component_bounds(axis.z);
            Aabb::new(Vector::new(lx, ly, lz) - pad, Vector::new(ux, uy, uz) + pad)
        };
        let lower_time = aabb.cast_local_ray(ray, max_toi, true)?;
        Some(Self { region, lower_time })
    }
}

impl PartialEq for RayRegion {
    fn eq(&self, other: &Self) -> bool {
        self.lower_time.total_cmp(&other.lower_time).is_eq() && self.region == other.region
    }
}
impl Eq for RayRegion {}
impl PartialOrd for RayRegion {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for RayRegion {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .lower_time
            .total_cmp(&self.lower_time)
            .then_with(|| self.region.cmp(&other.region))
    }
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
        let (_, data) = self
            .cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(self, &coords);
        let patch = Patch::new(&coords, self.terrain.face_resolution());
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
    fn cast_ray_inside_one_quad(
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
        if self.radius / 3.0_f64.sqrt() + f64::from(self.terrain.min_height()) <= 0.0 {
            return None;
        }
        let end = ray.origin + ray.dir * max_toi;
        if !end.is_finite() {
            return None;
        }
        let coords = Coords::from_vector(self.terrain.face_resolution(), &na3f32(ray.origin));
        let patch = Patch::new(&coords, self.terrain.face_resolution());
        let quad_resolution = self.chunk_resolution - 1;
        let uv = patch.project(&ray.origin) * f64::from(quad_resolution);
        #[expect(
            clippy::cast_possible_truncation,
            reason = "The clamped normalized patch coordinate is within the u32 quad range"
        )]
        #[expect(
            clippy::cast_sign_loss,
            reason = "Patch coordinates are clamped nonnegative"
        )]
        let address = UVec2::new(
            (uv.x.floor() as u32).min(quad_resolution - 1),
            (uv.y.floor() as u32).min(quad_resolution - 1),
        );
        let quad = Quad::new(&patch, quad_resolution, address);
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
        let (_, data) = self
            .cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(self, &coords);
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
        let (slot, _) = self
            .cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(self, &coords);
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
        let resolution = self.terrain.face_resolution();
        let min_height = f64::from(self.terrain.min_height());
        let max_height = f64::from(self.terrain.max_height());
        let mut candidates = BinaryHeap::new();
        for face in crate::cubemap::Face::iter() {
            if let Some(region) = RayRegion::new(
                face,
                0,
                resolution,
                0,
                resolution,
                resolution,
                self.radius,
                min_height,
                max_height,
                ray,
                max_toi,
            ) {
                candidates.push(region);
            }
        }
        // The LRU slot can be reassigned while visiting adjacent patches.
        // Retain the winning tile and triangle until all possible earlier
        // intersections have been excluded.
        let mut closest: Option<(f64, Coords, u32, RayIntersection)> = None;
        while let Some(candidate) = candidates.pop() {
            let limit = closest.as_ref().map_or(max_toi, |x| x.0);
            if candidate.lower_time > limit {
                break;
            }
            let ProjectionRegion {
                face,
                x0,
                x1,
                y0,
                y1,
                ..
            } = candidate.region;
            if x1 - x0 == 1 && y1 - y0 == 1 {
                let coords = Coords { face, x: x0, y: y0 };
                if let Some((idx, hit)) = self.cast_ray_in_chunk(coords, ray, limit, solid) {
                    if closest
                        .as_ref()
                        .is_none_or(|best| hit.time_of_impact < best.0)
                    {
                        closest = Some((hit.time_of_impact, coords, idx, hit));
                    }
                }
                continue;
            }
            let mid_x = x0 + (x1 - x0) / 2;
            let mid_y = y0 + (y1 - y0) / 2;
            for (a, b) in [(x0, mid_x), (mid_x, x1)] {
                if a == b {
                    continue;
                }
                for (c, d) in [(y0, mid_y), (mid_y, y1)] {
                    if c == d {
                        continue;
                    }
                    if let Some(region) = RayRegion::new(
                        face,
                        a,
                        b,
                        c,
                        d,
                        resolution,
                        self.radius,
                        min_height,
                        max_height,
                        ray,
                        limit,
                    ) {
                        candidates.push(region);
                    }
                }
            }
        }
        let (_, coords, tri_idx, mut hit) = closest?;
        let (slot, _) = self
            .cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(self, &coords);
        hit.subshape = self.subshape_id(slot, tri_idx);
        Some(hit)
    }
}

/// A rectangular group of terrain patches in the cubemap, ordered by a
/// conservative lower bound on the distance to any triangle it contains.
///
/// The patch's undisplaced vertices are convex combinations of unit corner
/// directions. A positive radial displacement preserves this cone, and
/// triangles interpolate between such displaced vertices. The conservative
/// radius lower bound must therefore enclose triangle interiors, not merely
/// the displaced vertices; use their projection along the cone axis.
#[derive(Clone, Copy, Debug)]
struct ProjectionRegion {
    face: crate::cubemap::Face,
    x0: u32,
    x1: u32,
    y0: u32,
    y1: u32,
    lower_distance2: f64,
}

/// At most 1.5 MiB of f64 vertices per tile is retained for warm queries.
/// Larger grids use the same projection algorithm with transient geometry.
const MAX_CACHED_PROJECTION_VERTICES: usize = 65_536;

/// Retain the winning tile rather than a cache slot that may be recycled.
struct ProjectionWinner {
    distance2: f64,
    coords: Coords,
    triangle: u32,
    projection: PointProjection,
    feature: FeatureId,
}

/// The immutable query geometry shared by all cubemap hierarchy nodes.
struct ProjectionQuery {
    resolution: u32,
    query_axis: Vector,
    query_radius: f64,
    planet_radius: f64,
    min_height: f64,
    max_height: f64,
}

impl ProjectionRegion {
    /// Conservative cone of directions containing all displaced triangle
    /// interiors of the cubemap rectangle, independent of terrain heights.
    fn cone(
        face: crate::cubemap::Face,
        x0: u32,
        x1: u32,
        y0: u32,
        y1: u32,
        resolution: u32,
    ) -> (Vector, f64) {
        let scale = 2.0 / f64::from(resolution);
        let face_direction = |x: f64, y: f64| {
            let d = face.direction(&na::Point2::new(
                x.mul_add(scale, -1.0),
                y.mul_add(scale, -1.0),
            ));
            Vector::new(d.x, d.y, d.z)
        };
        let center = face_direction(
            f64::from(x0).midpoint(f64::from(x1)),
            f64::from(y0).midpoint(f64::from(y1)),
        );
        // All unit directions in this warped cubemap rectangle lie inside
        // the positive cone spanned by its four corners. For our centered
        // axis, the minimum dot product over that cone is bounded by the
        // minimum over its corners (the normalization can only increase a
        // positive center dot product of a convex combination).
        let mut min_cos = 1.0_f64;
        for x in [x0, x1] {
            for y in [y0, y1] {
                min_cos = min_cos.min(center.dot(face_direction(f64::from(x), f64::from(y))));
            }
        }
        // Account for rounding of warp, the face rotation, and corner dot
        // products when using the bound to prune an entire subtree.
        min_cos = 64.0f64.mul_add(-f64::EPSILON, min_cos).clamp(0.0, 1.0);
        (center, min_cos)
    }

    fn new(
        face: crate::cubemap::Face,
        x0: u32,
        x1: u32,
        y0: u32,
        y1: u32,
        query: &ProjectionQuery,
    ) -> Self {
        let (center, min_cos) = Self::cone(face, x0, x1, y0, y1, query.resolution);
        // A displaced vertex is R*v + h*normalize(v), where v is a convex
        // combination of the face's unit corner directions. The dot product
        // with the cone axis is at least R*min_cos plus h*min_cos for h>=0
        // (or h itself for negative h). Every point inside a triangle is
        // itself a convex combination of its displaced vertices, so this
        // remains a valid radial *lower* bound for triangle interiors.
        // Bounding only vertex lengths would be unsound: a raised flat
        // triangle's interior can lie much closer to the planet center.
        let min_height_projection = if query.min_height >= 0.0 {
            query.min_height * min_cos
        } else {
            query.min_height
        };
        let min_radius = query.planet_radius.mul_add(min_cos, min_height_projection);
        let max_radius =
            outer_displaced_radius(query.planet_radius, query.min_height, query.max_height);
        let lower_distance2 = if min_radius <= 0.0 {
            // A large negative allowed height can invert a coarse patch's
            // local radial direction. Do not prune unless all possible
            // triangle points have a positive radius in the cone.
            0.0
        } else {
            let cos_center = center.dot(query.query_axis).clamp(-1.0, 1.0);
            let cos_distance = if cos_center >= min_cos {
                1.0
            } else {
                let sin_center = cos_center.mul_add(-cos_center, 1.0).max(0.0).sqrt();
                let sin_cone = min_cos.mul_add(-min_cos, 1.0).max(0.0).sqrt();
                cos_center.mul_add(min_cos, sin_center * sin_cone).min(1.0)
            };
            // Closest possible radius at this angular distance, clamped to
            // the global height envelope and chord sag of this region.
            let radius = (query.query_radius * cos_distance).clamp(min_radius, max_radius);
            let delta = query.query_radius - radius;
            let bound2 = delta.mul_add(
                delta,
                2.0 * query.query_radius * radius * (1.0 - cos_distance),
            );
            let rounding =
                128.0 * f64::EPSILON * (query.query_radius + max_radius + query.planet_radius);
            (bound2.max(0.0).sqrt() - rounding).max(0.0).powi(2)
        };
        Self {
            face,
            x0,
            x1,
            y0,
            y1,
            lower_distance2,
        }
    }
}

impl PartialEq for ProjectionRegion {
    fn eq(&self, other: &Self) -> bool {
        self.lower_distance2
            .total_cmp(&other.lower_distance2)
            .is_eq()
            && (self.face as u8, self.x0, self.x1, self.y0, self.y1)
                == (other.face as u8, other.x0, other.x1, other.y0, other.y1)
    }
}
impl Eq for ProjectionRegion {}
impl PartialOrd for ProjectionRegion {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for ProjectionRegion {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .lower_distance2
            .total_cmp(&self.lower_distance2)
            .then_with(|| {
                (self.face as u8, self.x0, self.x1, self.y0, self.y1).cmp(&(
                    other.face as u8,
                    other.x0,
                    other.x1,
                    other.y0,
                    other.y1,
                ))
            })
    }
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
        let (_, data) = self
            .cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(self, &coords);
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
            let patch = Patch::new(&coords, self.terrain.face_resolution());
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
        for face in crate::cubemap::Face::iter() {
            regions.push(ProjectionRegion::new(
                face,
                0,
                query.resolution,
                0,
                query.resolution,
                query,
            ));
        }
        while let Some(region) = regions.pop() {
            let best_distance2 = nearest.as_ref().map_or(f64::INFINITY, |x| x.distance2);
            if region.lower_distance2 > best_distance2 {
                break;
            }
            if region.x1 - region.x0 == 1 && region.y1 - region.y0 == 1 {
                let coords = Coords {
                    face: region.face,
                    x: region.x0,
                    y: region.y0,
                };
                self.update_projection_from_patch(coords, pt, query.query_radius, nearest);
                continue;
            }

            let mid_x = region.x0 + (region.x1 - region.x0) / 2;
            let mid_y = region.y0 + (region.y1 - region.y0) / 2;
            // Do not create empty rectangles when one axis is already one tile wide.
            for (x0, x1) in [(region.x0, mid_x), (mid_x, region.x1)] {
                if x0 == x1 {
                    continue;
                }
                for (y0, y1) in [(region.y0, mid_y), (mid_y, region.y1)] {
                    if y0 == y1 {
                        continue;
                    }
                    let candidate = ProjectionRegion::new(region.face, x0, x1, y0, y1, query);
                    if candidate.lower_distance2 <= best_distance2 {
                        regions.push(candidate);
                    }
                }
            }
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
        let min_height = f64::from(self.terrain.min_height());
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
    fn project_surface_point_and_get_feature(&self, pt: Vector) -> (PointProjection, FeatureId) {
        // Search the actual nearest triangle, not merely the tile intersected
        // by the ray from the planet center through the query. Tall cliffs,
        // patch boundaries, and points near the center can all have their
        // nearest surface on a different tile or even a different cube face.
        let face_resolution = self.terrain.face_resolution();
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
            min_height: f64::from(self.terrain.min_height()),
            max_height: f64::from(self.terrain.max_height()),
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
            let source_direction = na3f32(query.query_axis);
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
                // Leave space for cubemap angular and f32 conversion slack;
                // Coords::neighborhood also pads its cone by 1e-6 radians.
                if (theta + 2.0e-6) * f64::from(face_resolution) <= 2.0 {
                    #[expect(
                        clippy::cast_possible_truncation,
                        reason = "The validated near-surface cone is much smaller than pi radians"
                    )]
                    let theta = (theta + 2.0e-6) as f32;
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
        let (slot, _) = self
            .cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(self, &coords);
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

impl Shape for Planet {
    fn compute_local_aabb(&self) -> Aabb {
        Aabb::from_half_extents(Vector::ZERO, Vector::splat(self.max_radius()))
    }

    fn compute_local_bounding_sphere(&self) -> BoundingSphere {
        BoundingSphere::new(Vector::ZERO, self.max_radius())
    }

    fn mass_properties(&self, density: Real) -> MassProperties {
        parry3d_f64::shape::Ball {
            radius: self.radius,
        }
        .mass_properties(density)
    }

    fn shape_type(&self) -> ShapeType {
        ShapeType::Custom
    }

    fn as_typed_shape(&self) -> TypedShape<'_> {
        TypedShape::Custom(self)
    }

    fn ccd_thickness(&self) -> Real {
        0.0
    }

    fn ccd_angular_thickness(&self) -> Real {
        0.0
    }

    fn clone_dyn(&self) -> Box<dyn Shape> {
        Box::new(self.clone())
    }

    fn scale_dyn(&self, _scale: Vector, _num_subdivisions: u32) -> Option<Box<dyn Shape>> {
        // Non-uniform scale not supported
        None
    }
}

#[derive(Clone)]
struct Cache {
    slots: LruSlab<Arc<ChunkData>>,
    index: HashMap<Coords, u32>,
    max_size: u32,
}

impl Cache {
    pub fn new(capacity: u32) -> Self {
        Self {
            // LruSlab grows on demand. Do not preallocate potentially
            // enormous capacity merely because it fits the identifier space.
            slots: LruSlab::new(),
            index: HashMap::new(),
            max_size: capacity,
        }
    }

    pub fn get(&mut self, planet: &Planet, coords: &Coords) -> (u32, Arc<ChunkData>) {
        let (slot, old) = match self.index.entry(*coords) {
            hash_map::Entry::Occupied(e) => (*e.get(), None),
            hash_map::Entry::Vacant(e) => {
                let old = if self.slots.len() == self.max_size {
                    self.slots.lru().map(|lru| self.slots.remove(lru).coords)
                } else {
                    None
                };
                let slot = self
                    .slots
                    .insert(Arc::new(ChunkData::new(*coords, planet.sample(coords))));
                e.insert(slot);
                (slot, old)
            }
        };
        if let Some(old) = old {
            self.index.remove(&old);
        }
        (slot, Arc::clone(self.slots.get_mut(slot)))
    }
}

/// Row-major displaced-quad boxes with compact row and block summaries.
/// Query cost is proportional to nearby rows/blocks rather than the number
/// of quads in the whole chunk; every summary is a union of exact leaf AABBs.
#[derive(Debug)]
struct QuadBoundsIndex {
    quads: Box<[Aabb]>,
    rows: Box<[Aabb]>,
    blocks: Box<[Aabb]>,
}

const QUAD_BLOCK_WIDTH: u32 = 8;
/// Index on fourth collision query for a still-resident chunk; a single
/// collision query should not build every vertex and bounding box.
const QUAD_INDEX_BUILD_AFTER_QUERIES: u8 = 3;

/// A rectangle of quad cells, with exclusive upper bounds.
#[derive(Clone, Copy, Debug)]
struct QuadCellRegion {
    x0: u32,
    x1: u32,
    y0: u32,
    y1: u32,
}

/// Sparse, proof-backed search for quads in a cold terrain chunk. It uses
/// only four to six evaluations of the *undisplaced* piecewise-affine patch
/// for each visited region, and displaces geometry only at candidate leaves.
struct SparseQuadSearch<'a> {
    patch: &'a Patch,
    samples: &'a [f32],
    query: &'a Aabb,
    radius: f64,
    height_abs: f64,
    resolution: u32,
}

impl SparseQuadSearch<'_> {
    /// `Patch::get(u,v)` is affine on each side of `u=v`, and continuous
    /// along that diagonal. The extrema of its coordinates over a rectangle
    /// occur at the four corners or the diagonal's boundary intersections.
    /// A displaced vertex is R*v + h*normalize(v), whose height contribution
    /// lies component-wise within +/-max(|min_height|,|max_height|).
    /// Convex combinations preserve this AABB for complete triangle interiors.
    fn conservative_bounds(&self, region: QuadCellRegion) -> Aabb {
        let scale = 1.0 / f64::from(self.resolution - 1);
        let left = f64::from(region.x0) * scale;
        let right = f64::from(region.x1) * scale;
        let bottom = f64::from(region.y0) * scale;
        let top = f64::from(region.y1) * scale;
        let mut lower = Vector::splat(f64::INFINITY);
        let mut upper = Vector::splat(f64::NEG_INFINITY);
        for uv in [
            Vec2::new(left, bottom),
            Vec2::new(right, bottom),
            Vec2::new(left, top),
            Vec2::new(right, top),
        ] {
            let vertex = self.patch.get(&uv);
            lower = lower.min(vertex);
            upper = upper.max(vertex);
        }
        let diagonal_start = left.max(bottom);
        let diagonal_end = right.min(top);
        if diagonal_start <= diagonal_end {
            for coord in [diagonal_start, diagonal_end] {
                let vertex = self.patch.get(&Vec2::splat(coord));
                lower = lower.min(vertex);
                upper = upper.max(vertex);
            }
        }
        // Covers rounding in UV subdivision, affine interpolation, height
        // displacement and world-space AABB comparisons, including 6,371 km
        // planets and extremely small chunks.
        let roundoff = 512.0
            * f64::EPSILON
            * (self.radius
                + self.height_abs
                + self.query.mins.abs().max_element()
                + self.query.maxs.abs().max_element());
        let extension = Vector::splat(self.height_abs + roundoff);
        Aabb::new(
            lower * self.radius - extension,
            upper * self.radius + extension,
        )
    }

    /// Returns false only if the callback explicitly terminates traversal.
    fn visit(&self, region: QuadCellRegion, emit: &mut impl FnMut(Quad) -> bool) -> bool {
        if !self.conservative_bounds(region).intersects(self.query) {
            return true;
        }
        if region.x1 - region.x0 == 1 && region.y1 - region.y0 == 1 {
            let quad = Quad::new(
                self.patch,
                self.resolution - 1,
                UVec2::new(region.x0, region.y0),
            );
            let displaced = quad.displace(self.radius, self.resolution, self.samples);
            let [top_left, top_right, bottom_left, bottom_right] = displaced.corners;
            let lower = top_left.min(top_right).min(bottom_left).min(bottom_right);
            let upper = top_left.max(top_right).max(bottom_left).max(bottom_right);
            return !Aabb::new(lower, upper).intersects(self.query) || emit(quad);
        }
        let mid_x = region.x0 + (region.x1 - region.x0) / 2;
        let mid_y = region.y0 + (region.y1 - region.y0) / 2;
        for (y0, y1) in [(region.y0, mid_y), (mid_y, region.y1)] {
            if y0 == y1 {
                continue;
            }
            for (x0, x1) in [(region.x0, mid_x), (mid_x, region.x1)] {
                if x0 != x1 && !self.visit(QuadCellRegion { x0, x1, y0, y1 }, emit) {
                    return false;
                }
            }
        }
        true
    }
}

#[derive(Debug)]
struct ChunkData {
    samples: Box<[f32]>,
    coords: Coords,
    min: f32,
    max: f32,
    /// Lazily constructed displaced mesh vertices shared across point queries.
    projection_vertices: OnceLock<Box<[Vector]>>,
    /// Exact AABBs of displaced quads in row-major order; initialized only
    /// by collision broadphase queries and shared by the chunk LRU's Arc.
    quad_bounds: OnceLock<QuadBoundsIndex>,
    broadphase_queries: AtomicU8,
}

impl ChunkData {
    fn new(coords: Coords, samples: Box<[f32]>) -> Self {
        let mut iter = samples.iter().copied();
        let first = iter.next().unwrap_or(0.0);
        let mut min = first;
        let mut max = first;
        for sample in iter {
            if sample < min {
                min = sample;
            } else if sample > max {
                max = sample;
            }
        }
        Self {
            samples,
            coords,
            min,
            max,
            projection_vertices: OnceLock::new(),
            quad_bounds: OnceLock::new(),
            broadphase_queries: AtomicU8::new(0),
        }
    }

    /// Exact displaced-quad bounding boxes, shared across warm collision
    /// queries without rebuilding any triangle or projection geometry.
    fn quad_bounds(&self, planet: &Planet) -> &QuadBoundsIndex {
        self.quad_bounds.get_or_init(|| {
            let vertices = self.projection_vertices(planet);
            let width = planet.chunk_resolution as usize;
            let side = width - 1;
            let block_width = QUAD_BLOCK_WIDTH as usize;
            let blocks_per_row = side.div_ceil(block_width);
            let mut quads = Vec::with_capacity(side * side);
            let mut rows = Vec::with_capacity(side);
            let mut blocks = Vec::with_capacity(side * blocks_per_row);
            for y in 0..side {
                let row_start = quads.len();
                for x in 0..side {
                    let index = y * width + x;
                    let [top_left, top_right, bottom_left, bottom_right] = [
                        vertices[index],
                        vertices[index + 1],
                        vertices[index + width],
                        vertices[index + width + 1],
                    ];
                    let lower = top_left.min(top_right).min(bottom_left).min(bottom_right);
                    let upper = top_left.max(top_right).max(bottom_left).max(bottom_right);
                    quads.push(Aabb::new(lower, upper));
                }
                let row = &quads[row_start..];
                let mut row_box = row[0];
                for candidate in &row[1..] {
                    row_box = row_box.merged(candidate);
                }
                rows.push(row_box);
                for block in row.chunks(block_width) {
                    let mut block_box = block[0];
                    for candidate in &block[1..] {
                        block_box = block_box.merged(candidate);
                    }
                    blocks.push(block_box);
                }
            }
            QuadBoundsIndex {
                quads: quads.into_boxed_slice(),
                rows: rows.into_boxed_slice(),
                blocks: blocks.into_boxed_slice(),
            }
        })
    }

    /// Projected mesh positions, computed only if nearest-point queries need
    /// them. Raycasts and collision callbacks still use the original samples.
    fn projection_vertices(&self, planet: &Planet) -> &[Vector] {
        self.projection_vertices.get_or_init(|| {
            let patch = Patch::new(&self.coords, planet.terrain.face_resolution());
            let resolution = planet.chunk_resolution;
            let quads = f64::from(resolution - 1);
            let mut vertices = Vec::with_capacity(self.samples.len());
            for y in 0..resolution {
                for x in 0..resolution {
                    let p = Vec2::new(f64::from(x) / quads, f64::from(y) / quads);
                    let v = patch.get(&p);
                    let height = f64::from(self.samples[(y * resolution + x) as usize]);
                    vertices.push(v * planet.radius + v.normalize() * height);
                }
            }
            vertices.into_boxed_slice()
        })
    }
}

/// A [`PersistentQueryDispatcher`] that handles `Planet` shapes
pub struct PlanetDispatcher;

// TODO: Fill these in
impl QueryDispatcher for PlanetDispatcher {
    fn intersection_test(
        &self,
        pos12: &Pose,
        g1: &dyn Shape,
        g2: &dyn Shape,
    ) -> Result<ShapeIntersection, Unsupported> {
        if let Some(p1) = g1.as_shape::<Planet>() {
            return Ok(intersects(pos12, p1, g2));
        }
        if let Some(p2) = g2.as_shape::<Planet>() {
            return Ok(intersects(&pos12.inverse(), p2, g1).swapped());
        }
        Err(Unsupported)
    }

    fn distance(
        &self,
        _pos12: &Pose,
        _g1: &dyn Shape,
        _g2: &dyn Shape,
    ) -> Result<ShapeDistance, Unsupported> {
        Err(Unsupported)
    }

    fn contact(
        &self,
        _pos12: &Pose,
        _g1: &dyn Shape,
        _g2: &dyn Shape,
        _prediction: Real,
    ) -> Result<Option<Contact>, Unsupported> {
        Err(Unsupported)
    }

    fn closest_points(
        &self,
        _pos12: &Pose,
        _g1: &dyn Shape,
        _g2: &dyn Shape,
        _max_dist: Real,
    ) -> Result<ClosestPoints, Unsupported> {
        Err(Unsupported)
    }

    fn cast_shapes(
        &self,
        pos12: &Pose,
        vel12: Vector,
        g1: &dyn Shape,
        g2: &dyn Shape,
        options: ShapeCastOptions,
    ) -> Result<Option<ShapeCastHit>, Unsupported> {
        if let Some(p1) = g1.as_shape::<Planet>() {
            return Ok(compute_toi(pos12, vel12, p1, g2, options, false));
        }
        if let Some(p2) = g2.as_shape::<Planet>() {
            return Ok(compute_toi(&pos12.inverse(), -vel12, p2, g1, options, true));
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
        if let Some(p1) = g1.as_shape::<Planet>() {
            return Ok(compute_nonlinear_toi(
                motion1,
                p1,
                motion2,
                g2,
                start_time,
                end_time,
                stop_at_penetration,
                false,
            ));
        }
        if let Some(p2) = g2.as_shape::<Planet>() {
            return Ok(compute_nonlinear_toi(
                motion2,
                p2,
                motion1,
                g1,
                start_time,
                end_time,
                stop_at_penetration,
                true,
            ));
        }
        Err(Unsupported)
    }
}

/// Reports whether `other` intersects `planet`, along with the sub-shapes of both that intersect.
fn intersects(pos12: &Pose, planet: &Planet, other: &dyn Shape) -> ShapeIntersection {
    // TODO after https://github.com/dimforge/parry/issues/8
    let dispatcher = DefaultQueryDispatcher;
    let bounds = other.compute_bounding_sphere(pos12);
    let aabb = other.compute_aabb(pos12);
    let mut hit = ShapeIntersection::new(false);
    planet.map_elements_in_local_sphere(&bounds, &aabb, |_, slot, index, triangle| {
        match dispatcher.intersection_test(pos12, triangle, other) {
            Ok(mut result) if result.intersecting => {
                // The triangle is shape 1 and `other` is shape 2, so `result.subshape2` already
                // identifies the part of `other` that was hit, if it's a composite shape.
                result.subshape1 = planet.subshape_id(slot, index);
                hit = result;
                false
            }
            _ => true,
        }
    });
    hit
}

fn compute_toi(
    pos12: &Pose,
    vel12: Vector,
    planet: &Planet,
    other: &dyn Shape,
    options: ShapeCastOptions,
    flipped: bool,
) -> Option<ShapeCastHit> {
    // TODO after https://github.com/dimforge/parry/issues/8
    let dispatcher = DefaultQueryDispatcher;
    // TODO: Raycast vs. minkowski sum of chunk bounds and bounding sphere?
    let aabb = {
        let start = other.compute_aabb(pos12);
        let end = start.transform_by(&Pose::from_translation(options.max_time_of_impact * vel12));
        start.merged(&end)
    };
    let mut closest = None::<ShapeCastHit>;
    planet.map_elements_in_local_sphere(
        &aabb.bounding_sphere(),
        &aabb,
        |_, slot, index, triangle| {
            let impact = if flipped {
                dispatcher.cast_shapes(&pos12.inverse(), -vel12, other, triangle, options)
            } else {
                dispatcher.cast_shapes(pos12, vel12, triangle, other, options)
            };
            if let Ok(Some(mut impact)) = impact {
                let subshape = planet.subshape_id(slot, index);
                if flipped {
                    impact.subshape2 = subshape;
                } else {
                    impact.subshape1 = subshape;
                }
                closest = Some(match closest {
                    None => impact,
                    Some(x) if impact.time_of_impact < x.time_of_impact => impact,
                    Some(x) => x,
                });
            }
            true
        },
    );
    closest
}

#[expect(
    clippy::too_many_arguments,
    reason = "Parry dispatch requires separate motion, shape, time-bound and orientation arguments"
)]
fn compute_nonlinear_toi(
    motion_planet: &NonlinearRigidMotion,
    planet: &Planet,
    motion_other: &NonlinearRigidMotion,
    other: &dyn Shape,
    start_time: Real,
    end_time: Real,
    stop_at_penetration: bool,
    flipped: bool,
) -> Option<ShapeCastHit> {
    // TODO after https://github.com/dimforge/parry/issues/8
    let dispatcher = DefaultQueryDispatcher;
    // TODO: Select chunks/triangles more conservatively, as discussed in compute_toi
    let aabb = {
        let start_pos = motion_planet.position_at_time(start_time).inverse()
            * motion_other.position_at_time(start_time);
        let end_pos = motion_planet.position_at_time(end_time).inverse()
            * motion_other.position_at_time(end_time);
        let start = other.compute_aabb(&start_pos);
        let end = other.compute_aabb(&end_pos);
        start.merged(&end)
    };
    let mut closest = None::<ShapeCastHit>;
    planet.map_elements_in_local_sphere(
        &aabb.bounding_sphere(),
        &aabb,
        |_, slot, index, triangle| {
            let impact = if flipped {
                dispatcher.cast_shapes_nonlinear(
                    motion_other,
                    other,
                    motion_planet,
                    triangle,
                    start_time,
                    end_time,
                    stop_at_penetration,
                )
            } else {
                dispatcher.cast_shapes_nonlinear(
                    motion_planet,
                    triangle,
                    motion_other,
                    other,
                    start_time,
                    end_time,
                    stop_at_penetration,
                )
            };
            if let Ok(Some(mut impact)) = impact {
                let subshape = planet.subshape_id(slot, index);
                if flipped {
                    impact.subshape2 = subshape;
                } else {
                    impact.subshape1 = subshape;
                }
                closest = Some(match closest {
                    None => impact,
                    Some(x) if impact.time_of_impact < x.time_of_impact => impact,
                    Some(x) => x,
                });
            }
            true
        },
    );
    closest
}

impl<ManifoldData, ContactData> PersistentQueryDispatcher<ManifoldData, ContactData>
    for PlanetDispatcher
where
    ManifoldData: Default + Clone + Send + Sync,
    ContactData: Default + Copy + Send + Sync,
{
    fn contact_manifolds(
        &self,
        pos12: &Pose,
        g1: &dyn Shape,
        g2: &dyn Shape,
        prediction: Real,
        manifolds: &mut Vec<ContactManifold<ManifoldData, ContactData>>,
        workspace: &mut Option<ContactManifoldsWorkspace>,
    ) -> Result<(), Unsupported> {
        if let Some(p1) = g1.as_shape::<Planet>() {
            if let Some(composite) = g2.as_composite_shape() {
                compute_manifolds_vs_composite(
                    pos12,
                    &pos12.inverse(),
                    p1,
                    composite,
                    prediction,
                    manifolds,
                    workspace,
                    false,
                );
            } else {
                compute_manifolds(pos12, p1, g2, prediction, manifolds, workspace, false);
            }
            return Ok(());
        }
        if let Some(p2) = g2.as_shape::<Planet>() {
            if let Some(composite) = g1.as_composite_shape() {
                compute_manifolds_vs_composite(
                    &pos12.inverse(),
                    pos12,
                    p2,
                    composite,
                    prediction,
                    manifolds,
                    workspace,
                    true,
                );
            } else {
                compute_manifolds(
                    &pos12.inverse(),
                    p2,
                    g1,
                    prediction,
                    manifolds,
                    workspace,
                    true,
                );
            }
            return Ok(());
        }
        Err(Unsupported)
    }

    fn contact_manifold_convex_convex(
        &self,
        _pos12: &Pose,
        _g1: &dyn Shape,
        _g2: &dyn Shape,
        _normal_constraints1: Option<&dyn NormalConstraints>,
        _normal_constraints2: Option<&dyn NormalConstraints>,
        _prediction: Real,
        _manifold: &mut ContactManifold<ManifoldData, ContactData>,
    ) -> Result<(), Unsupported> {
        // Planets aren't guaranteed to be convex, so we have no cases to handle here
        Err(Unsupported)
    }
}

fn compute_manifolds<ManifoldData, ContactData>(
    pos12: &Pose,
    planet: &Planet,
    other: &dyn Shape,
    prediction: Real,
    manifolds: &mut Vec<ContactManifold<ManifoldData, ContactData>>,
    workspace: &mut Option<ContactManifoldsWorkspace>,
    flipped: bool,
) where
    ManifoldData: Default + Clone + Send + Sync,
    ContactData: Default + Copy + Send + Sync,
{
    let workspace =
        workspace.get_or_insert_with(|| ContactManifoldsWorkspace(Box::<Workspace>::default()));
    if !workspace.0.is::<Workspace>() {
        *workspace = ContactManifoldsWorkspace(Box::<Workspace>::default());
    }
    let Some(workspace) = workspace.0.downcast_mut::<Workspace>() else {
        return;
    };
    let dispatcher = DefaultQueryDispatcher; // TODO after https://github.com/dimforge/parry/issues/8

    workspace.phase ^= true;
    let phase = workspace.phase;

    let bounds = other.compute_bounding_sphere(pos12).loosened(prediction);
    let aabb = other.compute_aabb(pos12).loosened(prediction);
    let mut old_manifolds = std::mem::take(manifolds);
    planet.map_elements_in_local_sphere(&bounds, &aabb, |&coords, slot, index, triangle| {
        let tri_state = match workspace.state.entry((coords, index)) {
            hash_map::Entry::Occupied(e) => {
                let tri_state = e.into_mut();

                let manifold = old_manifolds[tri_state.manifold_index].take();
                tri_state.manifold_index = manifolds.len();
                tri_state.phase = phase;
                manifolds.push(manifold);

                tri_state
            }
            hash_map::Entry::Vacant(e) => {
                let tri_state = TriangleState {
                    manifold_index: manifolds.len(),
                    phase,
                };

                let subshape = planet.subshape_id(slot, index);
                // `other`'s subshape is 0 because only composite shapes have subshapes, and those
                // take the `compute_manifolds_vs_composite` path. TODO: Supporting other shapes
                // with subshapes (height fields, voxels) would mean dispatching through
                // `contact_manifolds` so that `other` can report its own subshape IDs.
                let (subshape1, subshape2) = if flipped {
                    (0, subshape)
                } else {
                    (subshape, 0)
                };
                manifolds.push(ContactManifold::with_data(
                    subshape1,
                    subshape2,
                    ManifoldData::default(),
                ));

                e.insert(tri_state)
            }
        };

        let manifold = &mut manifolds[tri_state.manifold_index];

        // TODO: Nonconvex, postprocess contact `fid`s once parry's feature ID story is worked out
        if flipped {
            let _ = dispatcher.contact_manifold_convex_convex(
                &pos12.inverse(),
                other,
                triangle,
                None,
                None,
                prediction,
                manifold,
            );
        } else {
            let _ = dispatcher.contact_manifold_convex_convex(
                pos12, triangle, other, None, None, prediction, manifold,
            );
        }
        true
    });

    workspace.state.retain(|_, x| x.phase == phase);
}

/// Narrow-phase collision detection state for `Planet`
#[derive(Default, Clone)]
pub struct Workspace {
    state: HashMap<(Coords, u32), TriangleState>,
    phase: bool,
}

impl WorkspaceData for Workspace {
    fn as_typed_workspace_data(&self) -> TypedWorkspaceData<'_> {
        TypedWorkspaceData::Custom
    }

    fn clone_dyn(&self) -> Box<dyn WorkspaceData> {
        Box::new(self.clone())
    }
}

#[derive(Clone)]
struct TriangleState {
    manifold_index: usize,
    phase: bool,
}

#[expect(
    clippy::too_many_arguments,
    reason = "Parry contact-manifold dispatch requires transforms, shapes, workspace, contacts and orientation"
)]
fn compute_manifolds_vs_composite<ManifoldData, ContactData>(
    pos12: &Pose,
    pos21: &Pose,
    planet: &Planet,
    other: &dyn CompositeShape,
    prediction: Real,
    manifolds: &mut Vec<ContactManifold<ManifoldData, ContactData>>,
    workspace: &mut Option<ContactManifoldsWorkspace>,
    flipped: bool,
) where
    ManifoldData: Default + Clone + Send + Sync,
    ContactData: Default + Copy + Send + Sync,
{
    let workspace = workspace
        .get_or_insert_with(|| ContactManifoldsWorkspace(Box::<WorkspaceVsComposite>::default()));
    if !workspace.0.is::<WorkspaceVsComposite>() {
        *workspace = ContactManifoldsWorkspace(Box::<WorkspaceVsComposite>::default());
    }
    let Some(workspace) = workspace.0.downcast_mut::<WorkspaceVsComposite>() else {
        return;
    };
    let dispatcher = DefaultQueryDispatcher; // TODO after https://github.com/dimforge/parry/issues/8

    workspace.phase ^= true;
    let phase = workspace.phase;

    let bvh = other.bvh();

    let bounds = bvh
        .root_aabb()
        .bounding_sphere()
        .transform_by(pos12)
        .loosened(prediction);
    let aabb = bvh.root_aabb().transform_by(pos12).loosened(prediction);
    let mut old_manifolds = std::mem::take(manifolds);
    planet.map_elements_in_local_sphere(&bounds, &aabb, |&coords, slot, index, triangle| {
        let tri_aabb = triangle.compute_aabb(pos21).loosened(prediction);

        for composite_subshape in bvh.intersect_aabb(&tri_aabb) {
            other.map_part_at(
                composite_subshape,
                &mut |composite_part_pos, composite_part_shape, normal_constraints| {
                    let key = CompositeKey {
                        chunk_coords: coords,
                        triangle: index,
                        composite_subshape,
                    };
                    // TODO: Dedup wrt. convex case
                    let tri_state = match workspace.state.entry(key) {
                        hash_map::Entry::Occupied(e) => {
                            let tri_state = e.into_mut();

                            let manifold = old_manifolds[tri_state.manifold_index].take();
                            tri_state.manifold_index = manifolds.len();
                            tri_state.phase = phase;
                            manifolds.push(manifold);

                            tri_state
                        }
                        hash_map::Entry::Vacant(e) => {
                            let mut manifold = ContactManifold::new();
                            let subshape = planet.subshape_id(slot, index);
                            if flipped {
                                manifold.subshape1 = composite_subshape;
                                manifold.subshape2 = subshape;
                                manifold.set_subshape_pos1(composite_part_pos.copied());
                            } else {
                                manifold.subshape1 = subshape;
                                manifold.subshape2 = composite_subshape;
                                manifold.set_subshape_pos2(composite_part_pos.copied());
                            }

                            let tri_state = TriangleState {
                                manifold_index: manifolds.len(),
                                phase,
                            };
                            manifolds.push(manifold);
                            e.insert(tri_state)
                        }
                    };

                    let manifold = &mut manifolds[tri_state.manifold_index];

                    if flipped {
                        let _ = dispatcher.contact_manifold_convex_convex(
                            &composite_part_pos.inv_mul(pos21),
                            composite_part_shape,
                            triangle,
                            normal_constraints,
                            None,
                            prediction,
                            manifold,
                        );
                    } else {
                        let _ = dispatcher.contact_manifold_convex_convex(
                            &composite_part_pos.prepend_to(pos12),
                            triangle,
                            composite_part_shape,
                            None,
                            normal_constraints,
                            prediction,
                            manifold,
                        );
                    }
                },
            );
        }

        true
    });

    workspace.state.retain(|_, x| x.phase == phase);
}

/// Narrow-phase collision detection state for `Planet`
#[derive(Default, Clone)]
pub struct WorkspaceVsComposite {
    state: HashMap<CompositeKey, TriangleState>,
    phase: bool,
}

#[derive(Eq, PartialEq, Hash, Copy, Clone)]
struct CompositeKey {
    chunk_coords: Coords,
    triangle: u32,
    composite_subshape: u32,
}

impl WorkspaceData for WorkspaceVsComposite {
    fn as_typed_workspace_data(&self) -> TypedWorkspaceData<'_> {
        TypedWorkspaceData::Custom
    }

    fn clone_dyn(&self) -> Box<dyn WorkspaceData> {
        Box::new(self.clone())
    }
}

/// Direction on a face at `[0..1]^2` coordinates within a chunk
fn corner(coords: &Coords, face_resolution: u32, uv: [f64; 2]) -> Vector {
    let dir = coords
        .direction(face_resolution, &na::Point2::from(uv))
        .into_inner();
    Vector::new(dir.x, dir.y, dir.z)
}

/// Quad defined by a chunk pre-displacement
///
/// Generally neither flat nor square.
#[derive(Copy, Clone, Debug)]
struct Patch {
    // (0, 0) a--b
    //        |\ |
    //        | \|
    //        c--d (1,1)
    a: Vector,
    b: Vector,
    c: Vector,
    d: Vector,
}

impl Patch {
    pub fn new(coords: &Coords, face_resolution: u32) -> Self {
        Self {
            a: corner(coords, face_resolution, [0.0, 0.0]),
            b: corner(coords, face_resolution, [1.0, 0.0]),
            c: corner(coords, face_resolution, [0.0, 1.0]),
            d: corner(coords, face_resolution, [1.0, 1.0]),
        }
    }

    /// Map a point from patch space to a direction in sphere space
    fn get(&self, p: &Vec2) -> Vector {
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
    fn project(&self, dir: &Vector) -> Vec2 {
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

    fn quads(&self, chunk_resolution: u32) -> impl Iterator<Item = Quad> + '_ {
        let quad_resolution = chunk_resolution - 1;
        (0..quad_resolution).flat_map(move |y| {
            (0..quad_resolution).map(move |x| Quad::new(self, quad_resolution, UVec2::new(x, y)))
        })
    }

    #[cfg(test)]
    fn triangles<'a>(
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
struct Quad {
    /// Row-major order
    corners: [Vector; 4],
    position: UVec2,
}

impl Quad {
    fn new(patch: &Patch, resolution: u32, position: UVec2) -> Self {
        let offsets = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let position_f = position.as_dvec2();
        Self {
            corners: offsets
                .map(|x| patch.get(&((position_f + Vec2::from(x)) / f64::from(resolution)))),
            position,
        }
    }

    const fn index(&self, chunk_resolution: u32) -> u32 {
        self.position.y * (chunk_resolution - 1) + self.position.x
    }

    fn displace(&self, radius: f64, chunk_resolution: u32, chunk_samples: &[f32]) -> DisplacedQuad {
        let offsets = [[0, 0], [1, 0], [0, 1], [1, 1]];
        let mut result = self.corners;
        for (v, offset) in result.iter_mut().zip(offsets) {
            let sample = self.position + UVec2::from(offset);
            let displacement = chunk_samples[(sample.y * chunk_resolution + sample.x) as usize];
            // We deliberately don't normalize `v` in `v * radius` because we're displacing a
            // subdivided patch, not the surface of the sphere directly.
            *v = *v * radius + v.normalize() * f64::from(displacement);
        }
        DisplacedQuad { corners: result }
    }
}

struct DisplacedQuad {
    /// Row-major order
    corners: [Vector; 4],
}

impl DisplacedQuad {
    fn triangles(&self) -> [Triangle; 2] {
        let [p0, p1, p2, p3] = self.corners;
        [Triangle::new(p0, p1, p3), Triangle::new(p3, p2, p0)]
    }
}

#[cfg(test)]
mod tests {
    use approx::{assert_abs_diff_eq, assert_relative_eq};
    use parry3d_f64::{
        math::Rotation,
        query::ShapeCastStatus,
        shape::{Ball, Cuboid, TriMesh},
    };

    use crate::cubemap::{Edge, Face};

    use super::*;

    #[derive(Clone)]
    struct TestTerrain {
        face: u32,
        chunk: u32,
        min: f32,
        max: f32,
    }

    impl Terrain for TestTerrain {
        fn sample(&self, _: &Coords, out: &mut [f32]) {
            out.fill(0.0);
        }
        fn face_resolution(&self) -> u32 {
            self.face
        }
        fn chunk_resolution(&self) -> u32 {
            self.chunk
        }
        fn min_height(&self) -> f32 {
            self.min
        }
        fn max_height(&self) -> f32 {
            self.max
        }
    }

    /// Terrain with deliberate patch-to-patch cliffs, so the globally
    /// closest triangle need not have the query point's cubemap address.
    struct ProjectionTerrain {
        face_resolution: u32,
        peak: Option<Coords>,
        patterned: bool,
        samples: std::sync::atomic::AtomicUsize,
    }

    impl Terrain for ProjectionTerrain {
        fn sample(&self, coords: &Coords, out: &mut [f32]) {
            self.samples
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let height = if self.patterned {
                let hash = (coords.face as u32)
                    .wrapping_mul(7)
                    .wrapping_add(coords.x.wrapping_mul(13))
                    .wrapping_add(coords.y.wrapping_mul(17));
                (hash % 6) as f32 - 1.0
            } else if self.peak == Some(*coords) {
                5.0
            } else {
                0.0
            };
            out.fill(height);
        }
        fn face_resolution(&self) -> u32 {
            self.face_resolution
        }
        fn chunk_resolution(&self) -> u32 {
            3
        }
        fn min_height(&self) -> f32 {
            if self.patterned {
                -1.0
            } else {
                0.0
            }
        }
        fn max_height(&self) -> f32 {
            5.0
        }
    }

    /// Exhaustive reference over *all* triangles, used only at tiny grid
    /// resolutions to test spatial pruning independently of the cache.
    fn brute_force_projection(planet: &Planet, pt: Vector) -> (f64, Vector, Coords) {
        let mut winner = None;
        for face in Face::iter() {
            for y in 0..planet.terrain.face_resolution() {
                for x in 0..planet.terrain.face_resolution() {
                    let coords = Coords { face, x, y };
                    let samples = planet.sample(&coords);
                    let patch = Patch::new(&coords, planet.terrain.face_resolution());
                    for (_, tri) in
                        patch.triangles(planet.radius, planet.chunk_resolution, &samples)
                    {
                        let projection = tri.project_local_point(pt, false);
                        let distance2 = projection.point.distance_squared(pt);
                        if winner
                            .as_ref()
                            .is_none_or(|(best, _, _): &(f64, Vector, Coords)| distance2 < *best)
                        {
                            winner = Some((distance2, projection.point, coords));
                        }
                    }
                }
            }
        }
        winner.expect("nonempty terrain mesh")
    }

    #[test]
    fn solid_classification_handles_extreme_positive_and_negative_heights() {
        struct ConstantHeight(f32);
        impl Terrain for ConstantHeight {
            fn sample(&self, _: &Coords, out: &mut [f32]) {
                out.fill(self.0);
            }
            fn face_resolution(&self) -> u32 {
                1
            }
            fn chunk_resolution(&self) -> u32 {
                2
            }
            fn min_height(&self) -> f32 {
                self.0
            }
            fn max_height(&self) -> f32 {
                self.0
            }
        }
        for height in [-9.0, -4.0, 0.0, 5.0] {
            let planet = Planet::new(Arc::new(ConstantHeight(height)), 1, 10.0).unwrap();
            let face = (10.0 + f64::from(height)) / 3.0_f64.sqrt();
            for (radius, inside) in [
                (0.0, true),
                (face * 0.9, true),
                (face * 1.1, false),
                (10.0 + f64::from(height) + 0.1, false),
            ] {
                let pt = Vector::X * radius;
                assert_eq!(
                    planet.contains_local_point(pt),
                    inside,
                    "height={height}, pt={pt:?}, actual face={face}"
                );
                let proj = planet.project_local_point(pt, false);
                assert_eq!(proj.is_inside, inside);
                assert_relative_eq!(
                    proj.point.distance(pt),
                    (radius - face).abs(),
                    epsilon = 1.0e-9
                );
                let solid = planet.project_local_point(pt, true);
                assert_eq!(solid.is_inside, inside);
                if inside {
                    assert_eq!(solid.point, pt);
                }
            }
        }
    }

    #[test]
    fn point_query_surface_points_count_as_inside() {
        let planet = Planet::new(Arc::new(FlatTerrain::new(1, 2).unwrap()), 1, 10.0).unwrap();
        let face_distance = 10.0 / 3.0_f64.sqrt();
        for pt in [
            Vector::X * face_distance,
            Vector::Y * face_distance,
            Vector::Z * face_distance,
            Vector::new(face_distance, face_distance, 0.0),
        ] {
            let solid = planet.project_local_point(pt, true);
            assert!(
                solid.is_inside,
                "on-surface point {pt:?} returned {solid:?}"
            );
            assert_eq!(solid.point, pt);
            let hollow = planet.project_local_point(pt, false);
            assert!(
                hollow.is_inside,
                "on-surface hollow query {pt:?}: {hollow:?}"
            );
            assert!(hollow.point.distance(pt) < 1.0e-10);
            assert_eq!(planet.distance_to_local_point(pt, true), 0.0);
        }
    }

    #[test]
    fn solid_classification_matches_exhaustive_radial_mesh() {
        for face_resolution in [1, 2, 3] {
            for patterned in [false, true] {
                let source = Arc::new(ProjectionTerrain {
                    face_resolution,
                    peak: None,
                    patterned,
                    samples: std::sync::atomic::AtomicUsize::new(0),
                });
                let planet = Planet::new(source, 1, 10.0).unwrap();
                let mut seed = 0x4141_0505_c3f0_0155_u64;
                let mut next = || {
                    seed = seed
                        .wrapping_mul(6364136223846793005)
                        .wrapping_add(1442695040888963407);
                    ((seed >> 32) as u32) as f64 / f64::from(u32::MAX)
                };
                for index in 0..144 {
                    let axis =
                        Vector::new(next() * 2.0 - 1.0, next() * 2.0 - 1.0, next() * 2.0 - 1.0)
                            .normalize();
                    let ray = Ray {
                        origin: Vector::ZERO,
                        dir: axis,
                    };
                    let boundary = exhaustive_ray_hit(&planet, &ray, 50.0)
                        .expect("positive radial terrain must have one intersection from origin")
                        .time_of_impact;
                    let near_point = if index % 3 == 0 {
                        boundary - 0.0001
                    } else if index % 3 == 1 {
                        boundary + 0.0001
                    } else {
                        boundary + 5.0
                    };
                    let point = axis * near_point;
                    let expected_inside = near_point < boundary;
                    assert_eq!(planet.contains_local_point(point), expected_inside,
                        "face_resolution={face_resolution}, patterned={patterned}, index={index}, radius={near_point}, boundary={boundary}, point={point:?}");
                    let projected = planet.project_local_point(point, true);
                    assert_eq!(projected.is_inside, expected_inside);
                    if expected_inside {
                        assert_eq!(projected.point, point);
                    }
                }
            }
        }
    }

    #[test]
    fn solid_classification_preserves_cube_edges_and_three_face_corners() {
        for (radius, delta) in [(10.0, 0.001), (6_371_000.0, 0.1)] {
            let planet = Planet::new(Arc::new(FlatTerrain::new(1, 2).unwrap()), 1, radius).unwrap();
            for face in Face::iter() {
                let coords = Coords { face, x: 0, y: 0 };
                let corners = Quad::new(&Patch::new(&coords, 1), 1, UVec2::ZERO)
                    .displace(radius, 2, &planet.sample(&coords))
                    .corners;
                let [a, b, c, d] = corners;
                for surface in [
                    a,
                    b,
                    c,
                    d,
                    (a + b) * 0.5,
                    (a + c) * 0.5,
                    (b + d) * 0.5,
                    (c + d) * 0.5,
                ] {
                    let outward = surface.normalize();
                    let inside = surface - outward * delta;
                    let outside = surface + outward * delta;
                    assert!(
                        planet.contains_local_point(inside),
                        "missing seam interior: face={face:?}, radius={radius}, point={inside:?}"
                    );
                    assert!(
                        !planet.contains_local_point(outside),
                        "false seam interior: face={face:?}, radius={radius}, point={outside:?}"
                    );
                    assert!(planet.contains_local_point(surface),
                        "on-surface cube edge/corner must be contained: face={face:?}, point={surface:?}");
                }
            }
        }
    }

    #[test]
    fn solid_classification_tracks_face_and_quad_triangle_offsets() {
        for (radius, displacement) in [(10.0, 1.0e-4), (6_371_000.0, 0.01)] {
            let planet = Planet::new(Arc::new(FlatTerrain::new(3, 5).unwrap()), 1, radius).unwrap();
            for face in Face::iter() {
                for (x, y) in [(0, 0), (1, 1), (2, 2), (0, 2), (2, 0)] {
                    let coords = Coords { face, x, y };
                    let patch = Patch::new(&coords, 3);
                    let samples = planet.sample(&coords);
                    for position in [UVec2::new(0, 0), UVec2::new(3, 3)] {
                        let quad = Quad::new(&patch, 4, position);
                        for triangle in quad.displace(radius, 5, &samples).triangles() {
                            let point = triangle.a * 0.25 + triangle.b * 0.35 + triangle.c * 0.40;
                            let outward = (triangle.b - triangle.a)
                                .cross(triangle.c - triangle.a)
                                .normalize();
                            assert!(point.dot(outward) > 0.0);
                            for (sign, inside) in [(-1.0, true), (1.0, false)] {
                                let query = point + outward * (sign * displacement);
                                let solid = planet.project_local_point(query, true);
                                let hollow = planet.project_local_point(query, false);
                                assert_eq!(solid.is_inside, inside,
                                    "radius={radius}, coords={coords:?}, query={query:?}, point={point:?}");
                                assert_eq!(
                                    hollow.is_inside, inside,
                                    "radius={radius}, coords={coords:?}, query={query:?}"
                                );
                                assert_eq!(planet.contains_local_point(query), inside);
                                assert!(
                                    (hollow.point.distance(query) - displacement).abs()
                                        < displacement * 0.05
                                );
                                if inside {
                                    assert_eq!(solid.point, query);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn contained_point_feature_keeps_winning_chunk_live_after_ray_classification() {
        // The closest surface is in a deep adjacent depression, but the
        // radial containment ray intersects the original higher patch.
        // With a one-slot LRU, the containment query can evict the winning
        // projection chunk and invalidate its previously allocated subshape.
        struct Depression(Coords);
        impl Terrain for Depression {
            fn sample(&self, coords: &Coords, out: &mut [f32]) {
                out.fill(if *coords == self.0 { -8.0 } else { 0.0 });
            }
            fn face_resolution(&self) -> u32 {
                4
            }
            fn chunk_resolution(&self) -> u32 {
                3
            }
            fn min_height(&self) -> f32 {
                -8.0
            }
            fn max_height(&self) -> f32 {
                0.0
            }
        }
        let depressed = Coords {
            face: Face::Pz,
            x: 2,
            y: 1,
        };
        let low = Coords {
            face: Face::Pz,
            x: 1,
            y: 1,
        };
        let planet = Planet::new(Arc::new(Depression(depressed)), 1, 10.0).unwrap();
        let direction = low.direction(4, &na::Point2::new(0.999, 0.5)).into_inner();
        let query = Vector::new(direction.x, direction.y, direction.z) * 3.0;
        assert_eq!(Coords::from_vector(4, &na3f32(query)), low);
        let (_, expected_surface, winner) = brute_force_projection(&planet, query);
        assert_eq!(winner, depressed);
        let (projection, _) = planet.project_local_point_and_get_feature(query);
        assert_relative_eq!(projection.point, expected_surface, epsilon = 1.0e-9);
        let cache = planet.cache.lock().unwrap();
        let slot = projection.subshape / planet.triangles_per_chunk();
        let data = cache
            .slots
            .iter()
            .find(|(index, _)| *index == slot)
            .expect("projected subshape must refer to a live cache slot")
            .1;
        assert_eq!(
            data.coords, winner,
            "subshape ID must remain attached to the winning chunk"
        );
    }

    #[test]
    fn point_query_solid_classifies_triangle_interiors_not_vertex_radius() {
        // Six cubemap faces with one quad each form an inscribed cube. Its
        // vertices have radius 10, but the +X face lies at 10/sqrt(3).
        // Vertex radius is *not* the planet's inscribed solid radius.
        let planet = Planet::new(Arc::new(FlatTerrain::new(1, 2).unwrap()), 1, 10.0).unwrap();
        let face = 10.0 / 3.0_f64.sqrt();
        for (coordinate, inside) in [
            (8.0, false),
            (0.0, true),
            (face - 0.1, true),
            (face + 0.1, false),
            (15.0, false),
        ] {
            let point = Vector::X * coordinate;
            let solid = planet.project_local_point(point, true);
            let hollow = planet.project_local_point(point, false);
            let (with_feature, _) = planet.project_local_point_and_get_feature(point);
            assert_eq!(solid.is_inside, inside, "solid {point:?}: {solid:?}");
            assert_eq!(hollow.is_inside, inside, "hollow {point:?}: {hollow:?}");
            assert_eq!(with_feature.is_inside, inside, "feature {point:?}");
            assert_eq!(
                planet.contains_local_point(point),
                inside,
                "contains {point:?}"
            );
            if coordinate > 0.0 {
                assert_relative_eq!(hollow.point.x, face, epsilon = 1.0e-10);
            } else {
                assert_relative_eq!(hollow.point.length(), face, epsilon = 1.0e-10);
            }
            if inside {
                assert_eq!(
                    solid.point, point,
                    "solid projection should retain an interior point"
                );
                assert!(planet.distance_to_local_point(point, false) < 0.0);
                assert_eq!(planet.distance_to_local_point(point, true), 0.0);
            } else {
                assert_relative_eq!(solid.point.x, face, epsilon = 1.0e-10);
                assert!(planet.distance_to_local_point(point, false) > 0.0);
            }
        }
    }

    #[test]
    fn projection_region_bound_includes_interiors_of_raised_triangles() {
        // A face with all four corner vertices displaced to radius 20 has
        // triangle points at radius 20/sqrt(3). A vertex-only radial lower
        // bound (10/sqrt(3)+10) would incorrectly prune these interiors.
        let query = ProjectionQuery {
            resolution: 1,
            query_radius: 0.0,
            query_axis: Vector::X,
            planet_radius: 10.0,
            min_height: 10.0,
            max_height: 10.0,
        };
        let region = ProjectionRegion::new(Face::Px, 0, 1, 0, 1, &query);
        let actual_distance2 = (20.0_f64 / 3.0_f64.sqrt()).powi(2);
        assert!(
            region.lower_distance2 <= actual_distance2,
            "pruning bound {} exceeds closest triangle interior {actual_distance2}",
            region.lower_distance2
        );
    }

    #[test]
    fn projection_region_distance_bounds_enclose_sampled_triangle_interiors() {
        // Test every face, full-face and offset subrectangles, variable height
        // envelopes, several query radii, and many barycentric points. This
        // is independent of the nearest-point search and probes its pruning
        // bound directly, including interior points hidden from vertex bounds.
        let axes = [
            Vector::X,
            -Vector::X,
            Vector::Y,
            Vector::Z,
            Vector::new(1.0, 2.0, -3.0).normalize(),
        ];
        for face in Face::iter() {
            for resolution in [1, 2, 3, 4] {
                for (min_h, max_h) in [(-9.0, -2.0), (-9.0, 15.0), (0.0, 10.0), (10.0, 10.0)] {
                    for (x0, x1, y0, y1) in [
                        (0, resolution, 0, resolution),
                        (0, 1, 0, 1),
                        (resolution - 1, resolution, resolution - 1, resolution),
                    ] {
                        let mut mesh_points = Vec::new();
                        for y in y0..y1 {
                            for x in x0..x1 {
                                let coords = Coords { x, y, face };
                                let patch = Patch::new(&coords, resolution);
                                let samples: Vec<_> = (0..9)
                                    .map(|i| {
                                        if (i + x + 2 * y + u32::from(face as u8)) % 3 == 0 {
                                            min_h as f32
                                        } else {
                                            max_h as f32
                                        }
                                    })
                                    .collect();
                                for quad in patch.quads(3) {
                                    for tri in quad.displace(10.0, 3, &samples).triangles() {
                                        for weights in [
                                            [1.0, 0.0, 0.0],
                                            [0.0, 1.0, 0.0],
                                            [0.0, 0.0, 1.0],
                                            [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
                                            [0.8, 0.1, 0.1],
                                        ] {
                                            mesh_points.push(
                                                tri.a * weights[0]
                                                    + tri.b * weights[1]
                                                    + tri.c * weights[2],
                                            );
                                        }
                                    }
                                }
                            }
                        }
                        for axis in axes {
                            for query_radius in [0.0, 3.0, 10.0, 25.0] {
                                let query = ProjectionQuery {
                                    resolution,
                                    query_axis: axis,
                                    query_radius,
                                    planet_radius: 10.0,
                                    min_height: min_h,
                                    max_height: max_h,
                                };
                                let region = ProjectionRegion::new(face, x0, x1, y0, y1, &query);
                                let query_point = axis * query_radius;
                                for &point in &mesh_points {
                                    let d2 = point.distance_squared(query_point);
                                    assert!(
                                        region.lower_distance2 <= d2 + 1.0e-9,
                                        "region bound {} > point distance {d2}; \
face={face:?}, res={resolution}, rect=({x0},{x1},{y0},{y1}), \
heights=({min_h},{max_h}), q={query_point:?}, p={point:?}",
                                        region.lower_distance2
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn projection_finds_cliff_across_adjacent_patch_and_cube_face() {
        let near = Coords {
            x: 1,
            y: 1,
            face: Face::Pz,
        };
        let across_face = Coords {
            x: 3,
            y: 1,
            face: Face::Pz,
        };
        for (low, high, uv) in [
            (near, Coords { x: 2, ..near }, [0.98, 0.5]),
            (across_face, across_face.neighbors(4)[Edge::Px], [0.98, 0.5]),
        ] {
            let source = Arc::new(ProjectionTerrain {
                face_resolution: 4,
                peak: Some(high),
                patterned: false,
                samples: std::sync::atomic::AtomicUsize::new(0),
            });
            let planet = Planet::new(source, 1, 10.0).unwrap();
            let direction = low.direction(4, &na::Point2::from(uv)).into_inner();
            let query = Vector::new(direction.x, direction.y, direction.z) * 15.0;
            assert_eq!(Coords::from_vector(4, &na3f32(query)), low);
            let (expected_dist2, expected_pt, winning_patch) =
                brute_force_projection(&planet, query);
            assert_eq!(winning_patch, high);
            let (actual, _) = planet.project_local_point_and_get_feature(query);
            assert_relative_eq!(actual.point, expected_pt, epsilon = 1.0e-9);
            assert_relative_eq!(
                actual.point.distance_squared(query),
                expected_dist2,
                epsilon = 1.0e-9
            );
            let slot = actual.subshape / planet.triangles_per_chunk();
            let cache = planet.cache.lock().unwrap();
            assert_eq!(
                cache
                    .slots
                    .iter()
                    .find(|(id, _)| *id == slot)
                    .unwrap()
                    .1
                    .coords,
                winning_patch
            );
        }
    }

    #[test]
    fn projection_matches_exhaustive_mesh_at_boundaries_and_varied_radii() {
        let source = Arc::new(ProjectionTerrain {
            face_resolution: 3,
            peak: None,
            patterned: true,
            samples: std::sync::atomic::AtomicUsize::new(0),
        });
        let planet = Planet::new(source, 3, 10.0).unwrap();
        let mut points = vec![Vector::ZERO];
        for face in Face::iter() {
            for uv in [[0.0, 0.0], [0.99, 0.01], [0.51, 0.50], [1.0, 1.0]] {
                let coords = Coords { x: 1, y: 1, face };
                let dir = coords.direction(3, &na::Point2::from(uv)).into_inner();
                let axis = Vector::new(dir.x, dir.y, dir.z);
                for distance in [5.0, 11.0, 16.0] {
                    points.push(axis * distance);
                }
            }
        }
        // Deterministic, off-axis queries prevent the exhaustive reference
        // from accidentally sharing the same symmetry or face-edge behavior.
        let mut state = 0x00c0_ffee_5eed_1234_u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 32) as u32) as f64 / f64::from(u32::MAX)
        };
        for _ in 0..128 {
            let axis =
                Vector::new(next() * 2.0 - 1.0, next() * 2.0 - 1.0, next() * 2.0 - 1.0).normalize();
            points.push(axis * (2.0 + next() * 20.0));
        }
        for (index, point) in points.into_iter().enumerate() {
            let (expected_dist2, _, _) = brute_force_projection(&planet, point);
            let (actual, _) = planet.project_local_point_and_get_feature(point);
            assert!(
                (actual.point.distance_squared(point) - expected_dist2).abs() < 1.0e-9,
                "query index {index}: {point:?}, expected {expected_dist2}, actual {:?}",
                actual.point.distance_squared(point)
            );
            // Multiple triangles can be equally close at different points;
            // compare the global minimum distance, not a particular tie-break.
            assert!(actual.point.is_finite());
        }
    }

    #[test]
    fn projection_handles_negative_radial_interpolation() {
        struct InwardTerrain;
        impl Terrain for InwardTerrain {
            fn sample(&self, coords: &Coords, out: &mut [f32]) {
                out.fill(if coords.face == Face::Px { -9.0 } else { 0.0 });
            }
            fn face_resolution(&self) -> u32 {
                1
            }
            fn chunk_resolution(&self) -> u32 {
                3
            }
            fn min_height(&self) -> f32 {
                -9.0
            }
            fn max_height(&self) -> f32 {
                0.0
            }
        }
        let planet = Planet::new(Arc::new(InwardTerrain), 1, 10.0).unwrap();
        // The coarse +X patch has corners at radius one, but its interpolation
        // can have length < 0.9 and an inward height displaces it past zero.
        // Such triangles cannot be pruned using a positive-radius-only cone.
        for point in [
            Vector::ZERO,
            Vector::X,
            -Vector::X,
            Vector::Y * 5.0,
            Vector::new(0.8, -0.9, 0.4),
            Vector::new(8.0, 2.0, 1.0),
        ] {
            let (expected_dist2, _, _) = brute_force_projection(&planet, point);
            let projected = planet.project_local_point(point, false);
            assert_relative_eq!(
                projected.point.distance_squared(point),
                expected_dist2,
                epsilon = 1.0e-9
            );
        }
    }

    #[test]
    fn projection_vertex_cache_preserves_quad_geometry_exactly() {
        let source = Arc::new(ProjectionTerrain {
            face_resolution: 3,
            peak: None,
            patterned: true,
            samples: std::sync::atomic::AtomicUsize::new(0),
        });
        let planet = Planet::new(source, 2, 10.0).unwrap();
        for coords in [
            Coords {
                face: Face::Pz,
                x: 0,
                y: 2,
            },
            Coords {
                face: Face::Nx,
                x: 2,
                y: 0,
            },
            Coords {
                face: Face::Py,
                x: 1,
                y: 1,
            },
        ] {
            let (_, data) = planet.cache.lock().unwrap().get(&planet, &coords);
            let vertices = data.projection_vertices(&planet);
            assert_eq!(
                vertices.as_ptr(),
                data.projection_vertices(&planet).as_ptr()
            );
            assert_eq!(
                vertices.len(),
                (planet.chunk_resolution * planet.chunk_resolution) as usize
            );
            let patch = Patch::new(&coords, planet.terrain.face_resolution());
            for quad in patch.quads(planet.chunk_resolution) {
                let p = quad.position;
                let idx = (p.y * planet.chunk_resolution + p.x) as usize;
                let width = planet.chunk_resolution as usize;
                let actual = [
                    vertices[idx],
                    vertices[idx + 1],
                    vertices[idx + width],
                    vertices[idx + width + 1],
                ];
                let expected = quad
                    .displace(planet.radius, planet.chunk_resolution, &data.samples)
                    .corners;
                assert_eq!(
                    actual, expected,
                    "cached geometry must match ray and contact geometry for {coords:?}"
                );
            }
        }
    }

    #[test]
    fn large_projection_grids_do_not_allocate_persistent_meshes() {
        let planet = Planet::new(Arc::new(FlatTerrain::new(1, 257).unwrap()), 1, 10.0).unwrap();
        let query = Vector::new(11.0, 0.0, 0.0);
        let projection = planet.project_local_point(query, false);
        // With one patch per cube face, the chordal +X surface is the plane
        // x=radius/sqrt(3), not the radius-10 sphere.
        assert_relative_eq!(
            projection.point.distance(query),
            11.0 - 10.0 / 3.0_f64.sqrt(),
            epsilon = 1.0e-8
        );
        let cache = planet.cache.lock().unwrap();
        let (_, data) = cache.slots.iter().next().unwrap();
        assert!(
            data.projection_vertices.get().is_none(),
            "large tile geometry must be transient"
        );
    }

    #[test]
    fn projection_near_surface_avoids_scanning_entire_planet() {
        let source = Arc::new(ProjectionTerrain {
            face_resolution: 4096,
            peak: None,
            patterned: false,
            samples: std::sync::atomic::AtomicUsize::new(0),
        });
        let planet = Planet::new(Arc::clone(&source) as Arc<dyn Terrain>, 8, 100.0).unwrap();
        let query = Vector::new(100.1, 0.2, 0.1);
        let projected = planet.project_local_point(query, false);
        assert!(projected.point.distance(query) < 0.3);
        let visited = source.samples.load(std::sync::atomic::Ordering::Relaxed);
        assert!(
            visited < 1000,
            "searched {visited} patches for near-surface query"
        );
    }

    #[test]
    fn deeply_negative_height_encloses_inverted_geometry_for_all_queries() {
        struct NegativeTerrain;
        impl Terrain for NegativeTerrain {
            fn sample(&self, _: &Coords, out: &mut [f32]) {
                out.fill(-9.0);
            }
            fn face_resolution(&self) -> u32 {
                1
            }
            fn chunk_resolution(&self) -> u32 {
                3
            }
            fn min_height(&self) -> f32 {
                -9.0
            }
            fn max_height(&self) -> f32 {
                -9.0
            }
        }
        let planet = Planet::new(Arc::new(NegativeTerrain), 1, 10.0).unwrap();
        let envelope = planet.max_radius();
        assert_eq!(
            envelope, 9.0,
            "inverted vertices must extend past R+max_height=1"
        );
        let aabb = planet.compute_local_aabb();
        let mut inverted = 0;
        for face in Face::iter() {
            let coords = Coords { face, x: 0, y: 0 };
            let samples = planet.sample(&coords);
            for (idx, triangle) in Patch::new(&coords, 1).triangles(10.0, 3, &samples) {
                for vertex in [triangle.a, triangle.b, triangle.c] {
                    assert!(
                        vertex.length() <= envelope,
                        "triangle vertex outside global bounding sphere: {vertex:?}"
                    );
                    assert!(vertex.cmpge(aabb.mins).all() && vertex.cmple(aabb.maxs).all());
                }
                let point = (triangle.a + triangle.b + triangle.c) / 3.0;
                let outward = coords.center::<f64>(1).into_inner();
                if point.dot(Vector::new(outward.x, outward.y, outward.z)) < 0.0 {
                    inverted += 1;
                    let (expected_dist2, _, _) = brute_force_projection(&planet, point);
                    let projected = planet.project_surface_point_and_get_feature(point).0;
                    assert!(
                        (projected.point.distance_squared(point) - expected_dist2).abs() < 1.0e-9,
                        "projection missed inverted triangle {idx} on {face:?}"
                    );
                    let normal = (triangle.b - triangle.a)
                        .cross(triangle.c - triangle.a)
                        .normalize();
                    let ray = Ray {
                        origin: point + normal * 0.3,
                        dir: -normal,
                    };
                    let hit = planet.cast_local_ray_and_get_normal(&ray, 0.6, true);
                    assert!(
                        hit.is_some(),
                        "raycast missed inward-flipped triangle {idx}, {face:?}"
                    );
                }
            }
        }
        assert!(inverted > 0);
    }

    #[test]
    fn first_collision_query_does_not_build_whole_chunk_mesh() {
        let planet = Planet::new(Arc::new(FlatTerrain::new(1, 17).unwrap()), 1, 10.0).unwrap();
        let coords = Coords {
            face: Face::Px,
            x: 0,
            y: 0,
        };
        let patch = Patch::new(&coords, 1);
        let samples = planet.sample(&coords);
        let triangle = Quad::new(&patch, 16, UVec2::new(8, 8))
            .displace(10.0, 17, &samples)
            .triangles()[0];
        let center = (triangle.a + triangle.b + triangle.c) / 3.0;
        let aabb = Aabb::from_half_extents(center, Vector::splat(0.01));
        let mut matched = false;
        planet.map_elements_in_local_sphere(&aabb.bounding_sphere(), &aabb, |&at, _, _, _| {
            matched |= at == coords;
            true
        });
        assert!(
            matched,
            "the cold query must still emit the actual surface triangle"
        );
        let cache = planet.cache.lock().unwrap();
        for (_, data) in cache.slots.iter() {
            assert!(
                data.quad_bounds.get().is_none(),
                "single cold query should not build all quad bounding boxes"
            );
            assert!(
                data.projection_vertices.get().is_none(),
                "single cold query should not build the whole displaced vertex grid"
            );
        }
        drop(cache);
        // The first three calls use sparse traversal. Repeated collision
        // queries on a still-resident chunk graduate to the cached index.
        for _ in 0..3 {
            planet.map_elements_in_local_sphere(&aabb.bounding_sphere(), &aabb, |_, _, _, _| true);
        }
        let cache = planet.cache.lock().unwrap();
        assert!(
            cache
                .slots
                .iter()
                .any(|(_, data)| data.quad_bounds.get().is_some()),
            "reused chunks must graduate to the existing warm index"
        );
    }

    #[test]
    fn sparse_region_bounds_cover_every_displaced_sample_in_nonuniform_terrain() {
        // Includes diagonal crossings, non-power-of-two grids, reversed
        // radial directions, six faces, and both small/planetary radii.
        let mut seed = 0xa69e_3d94_1718_31c5_u64;
        let mut next = || {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (seed >> 32) as u32
        };
        for (radius, radius_f32) in [(10.0, 10.0_f32), (6_371_000.0, 6_371_000.0_f32)] {
            for resolution in [3, 4, 5, 9, 17] {
                for face in Face::iter() {
                    let coords = Coords { face, x: 0, y: 0 };
                    let patch = Patch::new(&coords, 1);
                    let mut samples = vec![0.0_f32; (resolution * resolution) as usize];
                    for (index, height) in samples.iter_mut().enumerate() {
                        let modulus = (index * 37 + usize::from(face as u8) * 13) % 101;
                        let fraction = f32::from(u8::try_from(modulus).unwrap()) / 100.0;
                        *height = fraction.mul_add(1.4, -0.9) * radius_f32;
                    }
                    let height_abs = samples
                        .iter()
                        .map(|height| f64::from(*height).abs())
                        .fold(0.0, f64::max);
                    let query = Aabb::from_half_extents(Vector::ZERO, Vector::splat(3.0 * radius));
                    let search = SparseQuadSearch {
                        patch: &patch,
                        samples: &samples,
                        query: &query,
                        radius,
                        height_abs,
                        resolution,
                    };
                    let side = resolution - 1;
                    for case in 0..64 {
                        let x0 = next() % side;
                        let x1 = x0 + 1 + next() % (side - x0);
                        let y0 = next() % side;
                        let y1 = y0 + 1 + next() % (side - y0);
                        let region = QuadCellRegion { x0, x1, y0, y1 };
                        let bound = search.conservative_bounds(region);
                        for row in y0..y1 {
                            for col in x0..x1 {
                                let quad = Quad::new(&patch, side, UVec2::new(col, row));
                                for vertex in quad.displace(radius, resolution, &samples).corners {
                                    assert!(vertex.cmpge(bound.mins).all() && vertex.cmple(bound.maxs).all(),
                                        "radius={radius},res={resolution},face={face:?},case={case},region={region:?},vertex={vertex:?},bound={bound:?}");
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn cold_collision_broadphase_covers_variable_heights_with_inverted_samples() {
        // Different heights at every *vertex*, rather than only one height
        // per chunk, including negative values that reverse radial vectors.
        struct VaryingHeight {
            face_resolution: u32,
            chunk_resolution: u32,
        }
        impl Terrain for VaryingHeight {
            fn sample(&self, coords: &Coords, out: &mut [f32]) {
                for (index, sample) in out.iter_mut().enumerate() {
                    let modulus = (index * 31
                        + usize::try_from(coords.x).unwrap() * 13
                        + usize::try_from(coords.y).unwrap() * 11
                        + usize::from(coords.face as u8) * 17)
                        % 101;
                    *sample = f32::from(u8::try_from(modulus).unwrap()).mul_add(0.14, -9.0);
                }
            }
            fn face_resolution(&self) -> u32 {
                self.face_resolution
            }
            fn chunk_resolution(&self) -> u32 {
                self.chunk_resolution
            }
            fn min_height(&self) -> f32 {
                -9.0
            }
            fn max_height(&self) -> f32 {
                5.0
            }
        }
        for (face_resolution, resolution) in [(1, 9), (2, 5), (3, 4)] {
            let planet = Planet::new(
                Arc::new(VaryingHeight {
                    face_resolution,
                    chunk_resolution: resolution,
                }),
                1,
                10.0,
            )
            .unwrap();
            let mut triangles = Vec::new();
            for face in Face::iter() {
                for y in 0..face_resolution {
                    for x in 0..face_resolution {
                        let coords = Coords { face, x, y };
                        let samples = planet.sample(&coords);
                        let patch = Patch::new(&coords, face_resolution);
                        for (index, triangle) in patch.triangles(10.0, resolution, &samples) {
                            triangles.push((coords, index, triangle));
                        }
                    }
                }
            }
            for case in 0..72 {
                let (_, _, seed) = triangles[(case * 101 + 7) % triangles.len()];
                let center = seed.a * 0.2 + seed.b * 0.3 + seed.c * 0.5;
                let extent = [0.02, 0.2, 1.0, 4.0][case % 4];
                let aabb = Aabb::from_half_extents(center, Vector::splat(extent));
                let mut found = std::collections::HashSet::new();
                planet.map_elements_in_local_sphere(
                    &aabb.bounding_sphere(),
                    &aabb,
                    |&coords, _, index, _| {
                        found.insert((coords, index));
                        true
                    },
                );
                let cuboid = Cuboid::new(Vector::splat(extent));
                let pos12 = Pose::translation(center.x, center.y, center.z);
                let missing: Vec<_> = triangles
                    .iter()
                    .filter(|(coords, index, triangle)| {
                        triangle.compute_local_aabb().intersects(&aabb)
                            && !found.contains(&(*coords, *index))
                            && DefaultQueryDispatcher
                                .intersection_test(&pos12, triangle, &cuboid)
                                .unwrap()
                                .intersecting
                    })
                    .map(|(coords, index, _)| (*coords, *index))
                    .collect();
                assert!(missing.is_empty(), "face_resolution={face_resolution},resolution={resolution},case={case},center={center:?},missing={missing:?}");
            }
        }
    }

    #[test]
    fn cached_displaced_quad_boxes_match_actual_quad_geometry() {
        let planet = Planet::new(Arc::new(FlatTerrain::new(3, 17).unwrap()), 1, 10.0).unwrap();
        for face in Face::iter() {
            let coords = Coords { face, x: 1, y: 1 };
            let (_, data) = planet.cache.lock().unwrap().get(&planet, &coords);
            assert!(data.quad_bounds.get().is_none());
            let boxes = data.quad_bounds(&planet);
            assert_eq!(boxes.quads.len(), 256);
            assert_eq!(boxes.rows.len(), 16);
            assert_eq!(boxes.blocks.len(), 32);
            let patch = Patch::new(&coords, 3);
            for quad in patch.quads(17) {
                let [a, b, c, d] = quad.displace(10.0, 17, &data.samples).corners;
                let expected = Aabb::new(a.min(b).min(c).min(d), a.max(b).max(c).max(d));
                let actual = &boxes.quads[quad.index(17) as usize];
                assert_eq!(actual.mins, expected.mins);
                assert_eq!(actual.maxs, expected.maxs);
            }
        }
    }

    #[test]
    fn broadphase_handles_volume_centered_on_planet_origin() {
        let planet = Planet::new(Arc::new(FlatTerrain::new(2, 3).unwrap()), 1, 10.0).unwrap();
        let aabb = Aabb::from_half_extents(Vector::ZERO, Vector::splat(15.0));
        let mut found = std::collections::HashSet::new();
        planet.map_elements_in_local_sphere(&aabb.bounding_sphere(), &aabb, |&coords, _, _, _| {
            found.insert(coords.face);
            true
        });
        assert_eq!(
            found.len(),
            6,
            "a volume enclosing the planet must visit all six faces"
        );
    }

    #[test]
    fn broadphase_inward_displacement_reaches_opposite_direction() {
        struct Inward;
        impl Terrain for Inward {
            fn sample(&self, coords: &Coords, out: &mut [f32]) {
                out.fill(if coords.face == Face::Px { -9.0 } else { 0.0 });
            }
            fn face_resolution(&self) -> u32 {
                1
            }
            fn chunk_resolution(&self) -> u32 {
                3
            }
            fn min_height(&self) -> f32 {
                -9.0
            }
            fn max_height(&self) -> f32 {
                0.0
            }
        }
        let planet = Planet::new(Arc::new(Inward), 1, 10.0).unwrap();
        let target = Coords {
            face: Face::Px,
            x: 0,
            y: 0,
        };
        let patch = Patch::new(&target, 1);
        let samples = planet.sample(&target);
        let mut inverted = 0;
        for (idx, tri) in patch.triangles(10.0, 3, &samples) {
            let point = (tri.a + tri.b + tri.c) / 3.0;
            if point.x >= 0.0 {
                continue;
            }
            inverted += 1;
            let aabb = Aabb::from_half_extents(point, Vector::splat(0.01));
            let mut matched = false;
            planet.map_elements_in_local_sphere(
                &aabb.bounding_sphere(),
                &aabb,
                |&coords, _, tri_idx, _| {
                    matched |= coords == target && tri_idx == idx;
                    true
                },
            );
            assert!(
                matched,
                "reversed radial surface from +X face missing at {point:?}, triangle {idx}"
            );
        }
        assert!(
            inverted > 0,
            "terrain displacement must invert a triangle centroid"
        );
    }

    #[test]
    fn broadphase_includes_planetary_radius_face_seams_and_quad_corners() {
        const RES: u32 = 4096;
        const RADIUS: f64 = 6_371_000.0;
        let planet = Planet::new(Arc::new(FlatTerrain::new(RES, 17).unwrap()), 1, RADIUS).unwrap();
        let mut case = 0;
        for face in Face::iter() {
            for x in [0, 1, RES / 2, RES - 2, RES - 1] {
                for y in [0, 1, RES / 2, RES - 2, RES - 1] {
                    let coords = Coords { face, x, y };
                    let patch = Patch::new(&coords, RES);
                    let samples = planet.sample(&coords);
                    for xy in [UVec2::new(0, 0), UVec2::new(7, 7), UVec2::new(15, 15)] {
                        let quad = Quad::new(&patch, 16, xy);
                        for (i, triangle) in quad
                            .displace(RADIUS, 17, &samples)
                            .triangles()
                            .into_iter()
                            .enumerate()
                        {
                            let index = (quad.index(17) << 1) | u32::from(i == 1);
                            for weights in [[0.23, 0.42, 0.35], [0.0001, 0.5, 0.4999]] {
                                let center = triangle.a * weights[0]
                                    + triangle.b * weights[1]
                                    + triangle.c * weights[2];
                                let aabb = Aabb::from_half_extents(center, Vector::splat(0.05));
                                let mut visited = false;
                                planet.map_elements_in_local_sphere(
                                    &aabb.bounding_sphere(),
                                    &aabb,
                                    |&at, _, idx, _| {
                                        visited |= at == coords && idx == index;
                                        true
                                    },
                                );
                                assert!(
                                    visited,
                                    "case={case} missing {coords:?}/{index} at {center:?}"
                                );
                                case += 1;
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(case, 1800);
    }

    #[test]
    fn broadphase_covers_randomly_oriented_boxes_and_variable_heights() {
        let mut rng = 0x9173_14bd_61cc_493f_u64;
        let mut rand = || {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((rng >> 32) as u32) as f64 / f64::from(u32::MAX)
        };
        for resolution in [1, 2, 3, 4] {
            let source = Arc::new(ProjectionTerrain {
                face_resolution: resolution,
                peak: None,
                patterned: true,
                samples: std::sync::atomic::AtomicUsize::new(0),
            });
            let planet = Planet::new(source, 1, 10.0).unwrap();
            let mut exhaustive = Vec::new();
            for face in Face::iter() {
                for y in 0..resolution {
                    for x in 0..resolution {
                        let coords = Coords { face, x, y };
                        let samples = planet.sample(&coords);
                        let patch = Patch::new(&coords, resolution);
                        for (idx, tri) in patch.triangles(10.0, 3, &samples) {
                            exhaustive.push((coords, idx, tri));
                        }
                    }
                }
            }
            for case in 0..192 {
                let center = Vector::new(
                    rand() * 28.0 - 14.0,
                    rand() * 28.0 - 14.0,
                    rand() * 28.0 - 14.0,
                );
                let magnitude = if case % 4 == 0 {
                    0.05
                } else if case % 4 == 1 {
                    0.5
                } else if case % 4 == 2 {
                    3.0
                } else {
                    8.0
                };
                let half = Vector::new(
                    (0.2 + rand()) * magnitude,
                    (0.2 + rand()) * magnitude,
                    (0.2 + rand()) * magnitude,
                );
                let aabb = Aabb::from_half_extents(center, half);
                let mut actual = std::collections::HashSet::new();
                planet.map_elements_in_local_sphere(
                    &aabb.bounding_sphere(),
                    &aabb,
                    |&coords, _, idx, _| {
                        actual.insert((coords, idx));
                        true
                    },
                );
                // An overlap of the triangle's AABB alone is not an
                // intersection with the actual triangle, so use Parry's
                // independent exact convex-triangle box intersection here.
                let cuboid = Cuboid::new(half);
                let pos12 = Pose::translation(center.x, center.y, center.z);
                let missing: Vec<_> = exhaustive
                    .iter()
                    .filter(|(coords, idx, triangle)| {
                        !actual.contains(&(*coords, *idx))
                            && triangle.compute_local_aabb().intersects(&aabb)
                            && DefaultQueryDispatcher
                                .intersection_test(&pos12, triangle, &cuboid)
                                .expect("convex triangle/AABB intersection is supported")
                                .intersecting
                    })
                    .map(|(coords, idx, _)| (*coords, *idx))
                    .collect();
                assert!(missing.is_empty(), "resolution={resolution}, case={case}, center={center:?}, half={half:?}, missing={missing:?}");
            }
        }
    }

    #[test]
    fn broadphase_reports_every_triangle_whose_aabb_intersects_query() {
        // Independently enumerate the actual triangles in all six faces.
        // The broadphase is allowed false positives, but never false
        // negatives even at shared patch/cube-face edges and corners.
        for resolution in [1, 2, 3] {
            let planet =
                Planet::new(Arc::new(FlatTerrain::new(resolution, 5).unwrap()), 1, 10.0).unwrap();
            let mut query_count = 0;
            for face in Face::iter() {
                for x in 0..resolution {
                    for y in 0..resolution {
                        let coords = Coords { face, x, y };
                        let patch = Patch::new(&coords, resolution);
                        let samples = planet.sample(&coords);
                        for tri in patch
                            .triangles(planet.radius, planet.chunk_resolution, &samples)
                            .map(|(_, tri)| tri)
                            .step_by(5)
                        {
                            let point = tri.a * 0.5 + tri.b * 0.3 + tri.c * 0.2;
                            for center in [point, tri.a, (tri.b + tri.c) * 0.5] {
                                let aabb = Aabb::from_half_extents(center, Vector::splat(0.05));
                                let sphere = aabb.bounding_sphere();
                                let mut actual = std::collections::HashSet::new();
                                planet.map_elements_in_local_sphere(
                                    &sphere,
                                    &aabb,
                                    |&at, _, idx, _| {
                                        actual.insert((at, idx));
                                        true
                                    },
                                );
                                let mut missing = Vec::new();
                                for test_face in Face::iter() {
                                    for yy in 0..resolution {
                                        for xx in 0..resolution {
                                            let at = Coords {
                                                face: test_face,
                                                x: xx,
                                                y: yy,
                                            };
                                            let samples = planet.sample(&at);
                                            let patch = Patch::new(&at, resolution);
                                            for (idx, triangle) in patch.triangles(
                                                planet.radius,
                                                planet.chunk_resolution,
                                                &samples,
                                            ) {
                                                if triangle.compute_local_aabb().intersects(&aabb)
                                                    && !actual.contains(&(at, idx))
                                                {
                                                    missing.push((at, idx));
                                                }
                                            }
                                        }
                                    }
                                }
                                assert!(missing.is_empty(),
                                    "resolution={resolution}, query={query_count}, center={center:?}, missing={missing:?}, emitted={} ", actual.len());
                                query_count += 1;
                            }
                        }
                    }
                }
            }
            assert!(query_count > 50);
        }
    }

    #[test]
    fn collider_callback_can_reenter_cache_and_evict_current_chunk() {
        let planet = Planet::new(Arc::new(FlatTerrain::new(1, 5).unwrap()), 1, 10.0).unwrap();
        let center = Vector::new(10.0, 0.0, 0.0);
        let bounds = BoundingSphere::new(center, 7.0);
        let aabb = Aabb::from_half_extents(center, Vector::splat(7.0));
        let mut seen = 0;
        planet.map_elements_in_local_sphere(&bounds, &aabb, |&coords, _, _, _| {
            // Previously the callback was invoked while holding this mutex.
            assert!(planet.cache.try_lock().is_ok());
            // Reenter a different cubemap face with capacity 1 to ensure the
            // outer Arc snapshot remains valid after its LRU slot is evicted.
            let mut cache = planet.cache.lock().unwrap();
            let other = Coords {
                x: 0,
                y: 0,
                face: Face::Py,
            };
            let _ = cache.get(&planet, &other);
            drop(cache);
            assert_eq!(coords.face, Face::Px);
            seen += 1;
            true
        });
        assert!(
            seen > 0,
            "expected collision callback to visit terrain triangles"
        );
    }

    #[test]
    fn terrain_config_rejects_invalid_dimensions() {
        assert!(matches!(
            FlatTerrain::new(0, 2),
            Err(PlanetConfigError::FaceResolution)
        ));
        assert!(matches!(
            FlatTerrain::new(1, 0),
            Err(PlanetConfigError::ChunkResolution)
        ));
        assert!(matches!(
            FlatTerrain::new(1, 1),
            Err(PlanetConfigError::ChunkResolution)
        ));
        assert!(matches!(
            FlatTerrain::new(1, u32::MAX),
            Err(PlanetConfigError::SampleGridTooLarge)
        ));
        assert!(FlatTerrain::new(3, 3).is_ok());
    }

    #[test]
    fn planet_config_rejects_invalid_geometry_heights_and_cache_size() {
        let make = |face, chunk, min, max| -> Arc<dyn Terrain> {
            Arc::new(TestTerrain {
                face,
                chunk,
                min,
                max,
            })
        };
        assert!(matches!(
            Planet::new(make(0, 2, 0.0, 0.0), 1, 10.0),
            Err(PlanetConfigError::FaceResolution)
        ));
        assert!(matches!(
            Planet::new(make(1, 1, 0.0, 0.0), 1, 10.0),
            Err(PlanetConfigError::ChunkResolution)
        ));
        assert!(matches!(
            Planet::new(make(1, u32::MAX, 0.0, 0.0), 1, 10.0),
            Err(PlanetConfigError::SampleGridTooLarge)
        ));
        assert!(matches!(
            Planet::new(make(1, 2, 0.0, 0.0), 0, 10.0),
            Err(PlanetConfigError::CacheSize)
        ));
        for radius in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, -1.0] {
            assert!(matches!(
                Planet::new(make(1, 2, 0.0, 0.0), 1, radius),
                Err(PlanetConfigError::Radius)
            ));
        }
        assert!(matches!(
            Planet::new(make(1, 2, -11.0, 0.0), 1, 10.0),
            Err(PlanetConfigError::Radius)
        ));
        assert!(matches!(
            Planet::new(make(1, 2, f32::NAN, 0.0), 1, 10.0),
            Err(PlanetConfigError::HeightRange)
        ));
        assert!(matches!(
            Planet::new(make(1, 2, 0.0, f32::INFINITY), 1, 10.0),
            Err(PlanetConfigError::HeightRange)
        ));
        assert!(matches!(
            Planet::new(make(1, 2, 3.0, -1.0), 1, 10.0),
            Err(PlanetConfigError::HeightRange)
        ));
        assert!(matches!(
            Planet::new(make(1, 65_536, 0.0, 0.0), 1, 10.0),
            Err(PlanetConfigError::SubshapeIdOverflow)
        ));
        assert!(matches!(
            Planet::new(make(1, 2, 0.0, 0.0), u32::MAX, 10.0),
            Err(PlanetConfigError::SubshapeIdOverflow)
        ));
    }

    #[test]
    fn planet_subshape_ids_fit_exactly_and_cache_capacity_is_lazy() {
        // Two triangles per chunk; the full 32-bit ID range fits exactly.
        let terrain = Arc::new(FlatTerrain::new(1, 2).unwrap());
        let planet = Planet::new(terrain, 1 << 31, 1.0)
            .expect("maximum slot count is representable without preallocating all slots");
        let mut cache = planet.cache.lock().unwrap();
        assert_eq!(cache.slots.len(), 0);
        let (first, _) = cache.get(
            &planet,
            &Coords {
                x: 0,
                y: 0,
                face: Face::Px,
            },
        );
        assert_eq!(first, 0);
        assert_eq!(planet.subshape_id(first, 1), 1);
        assert!(cache.slots.capacity() < 32);
        drop(cache);
        // Ordinary LRU eviction must preserve the configured slot bound.
        let planet = Planet::new(Arc::new(FlatTerrain::new(1, 2).unwrap()), 1, 1.0).unwrap();
        let mut cache = planet.cache.lock().unwrap();
        let (a, _) = cache.get(
            &planet,
            &Coords {
                x: 0,
                y: 0,
                face: Face::Px,
            },
        );
        let (b, _) = cache.get(
            &planet,
            &Coords {
                x: 0,
                y: 0,
                face: Face::Py,
            },
        );
        assert_eq!(a, b);
        assert_eq!(cache.slots.len(), 1);
        assert_eq!(cache.index.len(), 1);
    }

    #[test]
    fn triangles() {
        let planet = Planet::new(
            Arc::new(FlatTerrain::new(1, 2).expect("valid test terrain")),
            32,
            1.0,
        )
        .expect("valid test planet");
        let coords = Coords {
            x: 0,
            y: 0,
            face: Face::Pz,
        };
        let samples = planet.sample(&coords);
        let patch = Patch::new(&coords, planet.terrain.face_resolution());
        assert_eq!(
            patch
                .triangles(planet.radius, planet.chunk_resolution, &samples)
                .count(),
            2
        );
        let expected = 1.0 / 3.0f64.sqrt();
        for (_, tri) in patch.triangles(planet.radius, planet.chunk_resolution, &samples) {
            assert!(tri.normal().unwrap().z > 0.0);
            for vert in &[tri.a, tri.b, tri.c] {
                assert!(vert.z > 0.0);
                for coord in vert.to_array() {
                    assert_eq!(coord.abs(), expected);
                }
                assert_relative_eq!(vert.length(), 1.0);
            }
        }
    }

    fn ball_contacts(planet: &Planet, pos: Vector, radius: Real) -> usize {
        let dispatcher = PlanetDispatcher.chain(DefaultQueryDispatcher);
        let ball = Ball { radius };
        let mut manifolds = Vec::<ContactManifold<(), ()>>::new();
        let mut workspace = None;
        dispatcher
            .contact_manifolds(
                &Pose::translation(pos.x, pos.y, pos.z),
                planet,
                &ball,
                0.0,
                &mut manifolds,
                &mut workspace,
            )
            .unwrap();
        manifolds.iter().map(|m| m.contacts().len()).sum()
    }

    #[test]
    fn end_to_end() {
        const PLANET_RADIUS: f64 = 6371e3;

        let planet = Planet::new(
            Arc::new(FlatTerrain::new(2u32.pow(12), 17).expect("valid test terrain")),
            32,
            PLANET_RADIUS,
        )
        .expect("valid test planet");

        // We add 0.1 to PLANET_RADIUS in positive tests below to hack around the issue in
        // https://github.com/dimforge/parry/pull/148. Can be removed once fix is released.
        assert!(ball_contacts(&planet, Vector::new(2.0, PLANET_RADIUS + 0.1, 0.0), 1.0) >= 2,
                "a ball lying on an axis of a planet with an even number of chunks per face overlaps with at least four triangles");
        assert_eq!(
            ball_contacts(&planet, Vector::new(0.0, PLANET_RADIUS + 2.0, 0.0), 1.0),
            0
        );
        assert!(ball_contacts(&planet, Vector::new(-1.0, PLANET_RADIUS + 0.1, 0.0), 1.0) > 0);

        for i in 0..10 {
            use std::f64;
            let rot =
                Rotation::from_axis_angle(Vector::Z, (i as f64 / 1000.0) * f64::consts::PI * 1e-4);
            let pos = rot * Vector::new(0.0, PLANET_RADIUS + 0.1, 0.0);
            assert!(ball_contacts(&planet, pos, 1.0) > 0);
        }
    }

    // Ensure absence of a collision hole arising from mistakenly considering chunk centers *not* to
    // be offset by 0.5 / face_resolution from edges of cubemap faces.
    #[test]
    fn coordinate_center_regression() {
        const PLANET_RADIUS: f64 = 6371e3;
        const BALL_RADIUS: f64 = 50.0;
        let planet = Planet::new(
            Arc::new(FlatTerrain::new(2u32.pow(12), 17).expect("valid test terrain")),
            32,
            PLANET_RADIUS,
        )
        .expect("valid test planet");

        let pos =
            Vector::new(-5_195_083.148, 3_582_099.812, -877_091.267).normalize() * PLANET_RADIUS;

        assert!(ball_contacts(&planet, pos, BALL_RADIUS) > 0);
    }

    #[test]
    fn cast_shape_smoke() {
        const PLANET_RADIUS: f64 = 6371e3;
        const DISTANCE: f64 = 10.0;
        let ball = Ball { radius: 1.0 };
        let planet = Planet::new(
            Arc::new(FlatTerrain::new(2u32.pow(12), 17).expect("valid test terrain")),
            32,
            PLANET_RADIUS,
        )
        .expect("valid test planet");

        let impact = PlanetDispatcher
            .cast_shapes(
                &Pose::translation(PLANET_RADIUS + DISTANCE, 0.0, 0.0),
                Vector::new(-1.0, 0.0, 0.0),
                &planet,
                &ball,
                ShapeCastOptions {
                    max_time_of_impact: 100.0,
                    stop_at_penetration: false,
                    ..Default::default()
                },
            )
            .unwrap()
            .expect("toi not found");
        assert_eq!(impact.status, ShapeCastStatus::Converged);
        assert_relative_eq!(impact.time_of_impact, DISTANCE - ball.radius);
        assert_relative_eq!(impact.witness1, Vector::new(PLANET_RADIUS, 0.0, 0.0));
        assert_relative_eq!(impact.witness2, Vector::new(-ball.radius, 0.0, 0.0));
        assert_relative_eq!(impact.normal1, Vector::X);
        assert_relative_eq!(impact.normal2, -Vector::X);
    }

    #[test]
    fn patch_triangle_indices_are_dense_and_bounded_by_chunk_stride() {
        let coords = Coords {
            x: 0,
            y: 0,
            face: Face::Pz,
        };
        let patch = Patch::new(&coords, 1);
        for resolution in [2, 3, 5, 17] {
            let samples = vec![0.0; (resolution * resolution) as usize];
            let ids: Vec<_> = patch
                .triangles(10.0, resolution, &samples)
                .map(|(index, _)| index)
                .collect();
            let count = 2 * (resolution - 1) * (resolution - 1);
            assert_eq!(ids, (0..count).collect::<Vec<_>>());
        }
    }

    #[test]
    fn subshape_id_unique() {
        let planet = Planet::new(
            Arc::new(FlatTerrain::new(2u32.pow(12), 17).expect("valid test terrain")),
            32,
            1.0,
        )
        .expect("valid test planet");
        let stride = planet.triangles_per_chunk();
        assert_eq!(planet.subshape_id(1, 0) - planet.subshape_id(0, 0), stride);
        // The last triangle of one chunk must not alias the first triangle of the next.
        assert!(planet.subshape_id(0, stride - 1) < planet.subshape_id(1, 0));
    }

    #[test]
    fn subshapes_consistent() {
        const PLANET_RADIUS: f64 = 6371e3;
        const DISTANCE: f64 = 10.0;
        let planet = Planet::new(
            Arc::new(FlatTerrain::new(2u32.pow(12), 17).expect("valid test terrain")),
            32,
            PLANET_RADIUS,
        )
        .expect("valid test planet");
        // A point well within a single triangle, away from chunk and triangle edges.
        let point = Vector::new(PLANET_RADIUS, 123.0, 456.0);
        let (proj, _) = planet.project_local_point_and_get_feature(point);
        let hit = planet
            .cast_local_ray_and_get_normal(
                &Ray {
                    origin: point + Vector::X * DISTANCE,
                    dir: -Vector::X,
                },
                100.0,
                true,
            )
            .expect("hit not found");
        assert_eq!(hit.subshape, proj.subshape);
        assert!(hit.subshape < planet.triangles_per_chunk() * 32);
    }

    #[test]
    fn intersection_subshapes() {
        const PLANET_RADIUS: f64 = 6371e3;
        let planet = Planet::new(
            Arc::new(FlatTerrain::new(2u32.pow(12), 17).expect("valid test terrain")),
            32,
            PLANET_RADIUS,
        )
        .expect("valid test planet");
        // The first triangle hovers above the surface; the second pierces it.
        let mesh = TriMesh::new(
            vec![
                Vector::new(PLANET_RADIUS + 500.0, -10.0, -10.0),
                Vector::new(PLANET_RADIUS + 500.0, 10.0, -10.0),
                Vector::new(PLANET_RADIUS + 500.0, 0.0, 10.0),
                Vector::new(PLANET_RADIUS - 10.0, -10.0, -10.0),
                Vector::new(PLANET_RADIUS - 10.0, 10.0, -10.0),
                Vector::new(PLANET_RADIUS + 10.0, 0.0, 0.0),
            ],
            vec![[0, 1, 2], [3, 4, 5]],
        )
        .unwrap();

        let planet_first = PlanetDispatcher
            .intersection_test(&Pose::identity(), &planet, &mesh)
            .unwrap();
        assert!(planet_first.intersecting);
        // The mesh's own subshape must survive when the planet's is filled in.
        assert_eq!(planet_first.subshape2, 1);
        assert!(planet_first.subshape1 < planet.triangles_per_chunk() * 32);

        let mesh_first = PlanetDispatcher
            .intersection_test(&Pose::identity(), &mesh, &planet)
            .unwrap();
        assert!(mesh_first.intersecting);
        assert_eq!(mesh_first.subshape1, 1);
        assert!(mesh_first.subshape2 < planet.triangles_per_chunk() * 32);
    }

    #[test]
    fn swapped_composite_manifolds_use_mesh_part_ids() {
        const RADIUS: f64 = 6371e3;
        let planet = Planet::new(
            Arc::new(FlatTerrain::new(4096, 17).expect("valid test terrain")),
            32,
            RADIUS,
        )
        .expect("valid test planet");
        let mesh = TriMesh::new(
            vec![
                Vector::new(RADIUS - 10.0, -10.0, -10.0),
                Vector::new(RADIUS + 10.0, 10.0, -10.0),
                Vector::new(RADIUS + 10.0, 0.0, 10.0),
            ],
            vec![[0, 1, 2]],
        )
        .unwrap();
        let mut workspace = None;
        let mut manifolds = Vec::<ContactManifold<(), ()>>::new();
        PlanetDispatcher
            .contact_manifolds(
                &Pose::identity(),
                &mesh,
                &planet,
                1.0,
                &mut manifolds,
                &mut workspace,
            )
            .unwrap();
        assert!(manifolds
            .iter()
            .any(|manifold| !manifold.contacts().is_empty()));
        assert!(manifolds.iter().all(|manifold| manifold.subshape1 == 0));
    }

    #[test]
    fn nonlinear_cast_swapped_order_preserves_shape_orientation() {
        const RADIUS: f64 = 6371e3;
        let planet = Planet::new(
            Arc::new(FlatTerrain::new(4096, 17).expect("valid test terrain")),
            32,
            RADIUS,
        )
        .expect("valid test planet");
        let ball = Ball { radius: 1.0 };
        let moving = NonlinearRigidMotion {
            start: Pose::translation(RADIUS + 1.5, 0.0, 0.0),
            local_center: Vector::ZERO,
            linvel: -Vector::X,
            angvel: Vector::ZERO,
        };
        let stationary = NonlinearRigidMotion::constant_position(Pose::identity());
        let hit = PlanetDispatcher
            .cast_shapes_nonlinear(&moving, &ball, &stationary, &planet, 0.0, 1.0, true)
            .unwrap()
            .expect("swapped nonlinear cast should hit");
        assert_relative_eq!(hit.time_of_impact, 0.5);
        assert_relative_eq!(hit.witness1, Vector::new(-1.0, 0.0, 0.0));
        assert_relative_eq!(hit.witness2, Vector::new(RADIUS, 0.0, 0.0));
        assert_relative_eq!(hit.normal1, -Vector::X);
        assert_relative_eq!(hit.normal2, Vector::X);
    }

    /// Reference for collision correctness; deliberately scans all six faces
    /// and all triangles, independent of the ray's origin and movement.
    fn exhaustive_ray_hit(planet: &Planet, ray: &Ray, max_toi: f64) -> Option<RayIntersection> {
        let mut closest: Option<RayIntersection> = None;
        for face in Face::iter() {
            for y in 0..planet.terrain.face_resolution() {
                for x in 0..planet.terrain.face_resolution() {
                    let coords = Coords { x, y, face };
                    let samples = planet.sample(&coords);
                    let patch = Patch::new(&coords, planet.terrain.face_resolution());
                    for (_, triangle) in
                        patch.triangles(planet.radius, planet.chunk_resolution, &samples)
                    {
                        if let Some(hit) = triangle
                            .cast_local_ray_and_get_normal(ray, max_toi, true)
                            .or_else(|| cast_triangle_ray_near_edge(&triangle, ray, max_toi))
                        {
                            if closest
                                .as_ref()
                                .is_none_or(|old| hit.time_of_impact < old.time_of_impact)
                            {
                                closest = Some(hit);
                            }
                        }
                    }
                }
            }
        }
        closest
    }

    #[test]
    fn raycast_matches_full_mesh_for_crossings_and_grazing_cases() {
        let source = Arc::new(ProjectionTerrain {
            face_resolution: 3,
            peak: None,
            patterned: true,
            samples: std::sync::atomic::AtomicUsize::new(0),
        });
        let planet = Planet::new(source, 3, 10.0).unwrap();
        let rays = [
            (Vector::new(20.0, 0.0, 0.0), -Vector::X),
            (Vector::new(-20.0, 0.0, 0.0), Vector::X),
            (
                Vector::new(20.0, 20.0, 0.0),
                Vector::new(-1.0, -1.0, 0.0).normalize(),
            ),
            (
                Vector::new(20.0, 20.0, 0.0),
                Vector::new(-1.0, -1.0, 0.0001).normalize(),
            ),
            (Vector::new(0.0, 0.0, 0.0), Vector::X),
            (Vector::new(0.01, 0.0, 0.0), -Vector::X),
            (Vector::new(0.01, 0.01, 0.01), -Vector::X),
            (
                Vector::new(1.0, 2.0, 3.0),
                -Vector::new(1.0, 2.0, 3.0).normalize(),
            ),
            (Vector::new(20.0, 0.0, 0.0), Vector::X),
            (
                Vector::new(0.01, -0.01, 0.01),
                Vector::new(1.0, -2.0, 3.0).normalize(),
            ),
            (Vector::new(5.0, 5.0, 5.0), Vector::X),
            (Vector::new(15.0, 0.0, 0.0), Vector::Y),
            (Vector::new(15.0, 0.0, 0.0), -Vector::Y),
            (Vector::new(0.0, 0.0, 20.0), -Vector::Z),
        ];
        let mut state = 0xdead_beef_cafe_f00d_u64;
        let mut next = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 32) as u32) as f64 / f64::from(u32::MAX)
        };
        let random_rays = (0..100).map(|_| {
            let origin = Vector::new(
                next() * 40.0 - 20.0,
                next() * 40.0 - 20.0,
                next() * 40.0 - 20.0,
            );
            let direction =
                Vector::new(next() * 2.0 - 1.0, next() * 2.0 - 1.0, next() * 2.0 - 1.0).normalize();
            Ray {
                origin,
                dir: direction,
            }
        });
        for (i, ray) in rays
            .into_iter()
            .map(|(origin, dir)| Ray { origin, dir })
            .chain(random_rays)
            .enumerate()
        {
            let reference = exhaustive_ray_hit(&planet, &ray, 100.0);
            let actual = planet.cast_local_ray_and_get_normal(&ray, 100.0, true);
            assert_eq!(
                actual.is_some(),
                reference.is_some(),
                "ray {i}: {ray:?}, expected {reference:?}, actual {actual:?}"
            );
            if let (Some(a), Some(b)) = (actual, reference) {
                assert!(
                    (a.time_of_impact - b.time_of_impact).abs() < 1.0e-7,
                    "ray {i}: {ray:?}, expected {}, actual {}",
                    b.time_of_impact,
                    a.time_of_impact
                );
            }
        }
    }

    #[test]
    fn raycast_matches_full_mesh_for_targeted_triangle_interiors() {
        let source = Arc::new(ProjectionTerrain {
            face_resolution: 3,
            peak: None,
            patterned: true,
            samples: std::sync::atomic::AtomicUsize::new(0),
        });
        let planet = Planet::new(source, 1, 10.0).unwrap();
        let mut count = 0;
        for face in Face::iter() {
            for y in 0..3 {
                for x in 0..3 {
                    let coords = Coords { x, y, face };
                    let samples = planet.sample(&coords);
                    let patch = Patch::new(&coords, 3);
                    for quad in patch.quads(3) {
                        for triangle in quad.displace(10.0, 3, &samples).triangles() {
                            for weights in [[0.33, 0.33, 0.34], [0.000001, 0.499999, 0.5]] {
                                let point = triangle.a * weights[0]
                                    + triangle.b * weights[1]
                                    + triangle.c * weights[2];
                                let outward = point.normalize();
                                for origin in
                                    [point + outward * 10.0, point + Vector::new(8.0, -11.0, 9.0)]
                                {
                                    let ray = Ray {
                                        origin,
                                        dir: (point - origin).normalize(),
                                    };
                                    let expected = exhaustive_ray_hit(&planet, &ray, 50.0);
                                    let actual =
                                        planet.cast_local_ray_and_get_normal(&ray, 50.0, true);
                                    assert_eq!(
                                        actual.is_some(),
                                        expected.is_some(),
                                        "target ray {count}: {ray:?}"
                                    );
                                    if let (Some(a), Some(b)) = (actual, expected) {
                                        assert!((a.time_of_impact - b.time_of_impact).abs() < 1.0e-7,
                                            "target ray {count}: {ray:?}, actual={a:?}, expected={b:?}");
                                    }
                                    count += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(count, 1728);
    }

    #[test]
    fn raycast_deep_inward_displacements_match_exhaustive_reference() {
        // Negative terrain height can make the root face cone's radial
        // lower bound nonpositive even when every sampled vertex is valid.
        // That region must use the full enclosing cube, not an unsafe cone.
        struct InwardTerrain;
        impl Terrain for InwardTerrain {
            fn sample(&self, coords: &Coords, out: &mut [f32]) {
                out.fill(if coords.face == Face::Px { -9.0 } else { 0.0 });
            }
            fn face_resolution(&self) -> u32 {
                2
            }
            fn chunk_resolution(&self) -> u32 {
                3
            }
            fn min_height(&self) -> f32 {
                -9.0
            }
            fn max_height(&self) -> f32 {
                0.0
            }
        }
        let planet = Planet::new(Arc::new(InwardTerrain), 1, 10.0).unwrap();
        for origin in [
            Vector::ZERO,
            Vector::X * 2.0,
            Vector::Y * 12.0,
            Vector::new(20.0, -4.0, 2.0),
            Vector::new(-14.0, 6.0, 0.1),
        ] {
            for dir in [
                Vector::X,
                -Vector::X,
                Vector::Y,
                -Vector::Y,
                Vector::new(-1.0, 0.5, 0.25).normalize(),
            ] {
                let ray = Ray { origin, dir };
                let expected = exhaustive_ray_hit(&planet, &ray, 100.0);
                let actual = planet.cast_local_ray_and_get_normal(&ray, 100.0, true);
                assert_eq!(actual.is_some(), expected.is_some(), "ray {ray:?}");
                if let (Some(a), Some(b)) = (actual, expected) {
                    assert!(
                        (a.time_of_impact - b.time_of_impact).abs() < 1.0e-8,
                        "ray {ray:?}, expected {b:?}, got {a:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn raycast_certified_quad_fast_path_only_applies_to_contained_segments() {
        const RADIUS: f64 = 6_371_000.0;
        let planet = Planet::new(Arc::new(FlatTerrain::new(4096, 17).unwrap()), 2, RADIUS).unwrap();
        let direct = Ray {
            origin: Vector::new(RADIUS + 10.0, 1.0, 1.0),
            dir: -Vector::X,
        };
        let fast_hit = planet
            .cast_ray_inside_one_quad(&direct, 100.0, true)
            .expect("radial ray entirely inside a single quad should use the fast path");
        assert_relative_eq!(fast_hit.time_of_impact, 10.0, epsilon = 1.0e-3);
        let actual = planet
            .cast_local_ray_and_get_normal(&direct, 100.0, true)
            .unwrap();
        assert_relative_eq!(actual.time_of_impact, fast_hit.time_of_impact);
        // A segment crossing the entire planet center cannot remain in one
        // quad cone even when its starting direction lies near its middle.
        let crossing = Ray {
            origin: Vector::new(RADIUS + 10.0, 1.0, 1.0),
            dir: -Vector::X,
        };
        assert!(planet
            .cast_ray_inside_one_quad(&crossing, 2.0 * RADIUS + 40.0, true)
            .is_none());
        // Ray starting near a cube corner, for which the original walk
        // missed a known triangle, must still be handled globally.
        let seam = Ray {
            origin: Vector::new(3678337.188936222, 3678238.6096734814, -3678337.1887390064),
            dir: Vector::new(-0.5773554268532486, -0.577339953755108, 0.5773554268222935),
        };
        let global_hit = planet
            .cast_local_ray_and_get_normal(&seam, 20.0, true)
            .expect("cube-corner triangle must be found");
        assert_relative_eq!(global_hit.time_of_impact, 10.0, epsilon = 1.0e-6);
    }

    #[test]
    fn raycast_hits_targeted_face_seams_on_planetary_radius() {
        const RADIUS: f64 = 6_371_000.0;
        const RES: u32 = 4096;
        let planet = Planet::new(Arc::new(FlatTerrain::new(RES, 17).unwrap()), 2, RADIUS).unwrap();
        let mut count = 0;
        for face in Face::iter() {
            for x in [0, 1, RES / 2, RES - 2, RES - 1] {
                for y in [0, 1, RES / 2, RES - 2, RES - 1] {
                    let coords = Coords { x, y, face };
                    let samples = planet.sample(&coords);
                    let patch = Patch::new(&coords, RES);
                    for xy in [UVec2::new(0, 0), UVec2::new(7, 7), UVec2::new(15, 15)] {
                        for tri in Quad::new(&patch, 16, xy)
                            .displace(RADIUS, 17, &samples)
                            .triangles()
                        {
                            for weights in [[0.3, 0.3, 0.4], [0.000001, 0.499999, 0.5]] {
                                let point =
                                    tri.a * weights[0] + tri.b * weights[1] + tri.c * weights[2];
                                let direction = (tri.b - tri.a).cross(tri.c - tri.a).normalize();
                                for outward in [point.normalize(), direction] {
                                    let origin = point + outward * 10.0;
                                    let ray = Ray {
                                        origin,
                                        dir: -outward,
                                    };
                                    let direct = tri
                                        .cast_local_ray_and_get_normal(&ray, 20.0, true)
                                        .or_else(|| cast_triangle_ray_near_edge(&tri, &ray, 20.0))
                                        .expect("ray must intersect targeted triangle");
                                    let actual =
                                        planet.cast_local_ray_and_get_normal(&ray, 20.0, true);
                                    assert!(actual.is_some_and(|hit| hit.time_of_impact <= direct.time_of_impact + 0.0001),
                                        "missed target {count}: face={face:?}, coords={coords:?}, ray={ray:?}, direct={direct:?}, planet={actual:?}");
                                    count += 1;
                                }
                            }
                        }
                    }
                }
            }
        }
        assert_eq!(count, 3600);
    }

    #[test]
    fn raycast_toi_is_relative_to_original_origin_across_patches() {
        let planet = Planet::new(
            Arc::new(FlatTerrain::new(8, 5).expect("valid test terrain")),
            128,
            10.0,
        )
        .expect("valid test planet");
        let ray = Ray {
            origin: Vector::new(20.0, 5.0, 0.25),
            dir: -Vector::X,
        };
        let hit = planet
            .cast_local_ray_and_get_normal(&ray, 30.0, true)
            .expect("ray crosses multiple patches before hitting planet");
        let point = ray.origin + ray.dir * hit.time_of_impact;
        assert!(
            (point.length() - 10.0).abs() < 0.1,
            "reported toi {} points to a radius of {} rather than the surface",
            hit.time_of_impact,
            point.length()
        );
        assert!(planet
            .cast_local_ray_and_get_normal(&ray, hit.time_of_impact * 0.5, true)
            .is_none());
    }

    #[test]
    fn ray_direct() {
        const PLANET_RADIUS: f64 = 6371e3;
        const DISTANCE: f64 = 10.0;
        let planet = Planet::new(
            Arc::new(FlatTerrain::new(2u32.pow(12), 17).expect("valid test terrain")),
            32,
            PLANET_RADIUS,
        )
        .expect("valid test planet");
        let hit = planet
            .cast_local_ray_and_get_normal(
                &Ray {
                    origin: Vector::new(PLANET_RADIUS + DISTANCE, 1.0, 1.0),
                    dir: -Vector::X,
                },
                100.0,
                true,
            )
            .expect("hit not found");
        assert_relative_eq!(hit.time_of_impact, DISTANCE, epsilon = 1e-3);
        assert_relative_eq!(hit.normal, Vector::X, epsilon = 1e-3);

        let hit = planet.cast_local_ray_and_get_normal(
            &Ray {
                origin: Vector::new(PLANET_RADIUS + DISTANCE, 1.0, 1.0),
                dir: Vector::X,
            },
            100.0,
            true,
        );
        assert!(hit.is_none());
    }

    #[test]
    fn ray_perp() {
        const PLANET_RADIUS: f64 = 6371e3;
        const DISTANCE: f64 = 10.0;
        let planet = Planet::new(
            Arc::new(FlatTerrain::new(2u32.pow(12), 17).expect("valid test terrain")),
            32,
            PLANET_RADIUS,
        )
        .expect("valid test planet");

        for &dir in [Vector::X, Vector::Y, -Vector::X, -Vector::Y].iter() {
            let hit = planet.cast_local_ray_and_get_normal(
                &Ray {
                    origin: Vector::new(1.0, 1.0, PLANET_RADIUS + DISTANCE),
                    dir,
                },
                10000.0,
                true,
            );
            assert!(hit.is_none());
        }
    }

    #[test]
    fn ray_glancing() {
        const PLANET_RADIUS: f64 = 6371e3;
        const DISTANCE: f64 = 1000.0;
        let planet = Planet::new(
            Arc::new(FlatTerrain::new(2u32.pow(12), 17).expect("valid test terrain")),
            32,
            PLANET_RADIUS,
        )
        .expect("valid test planet");
        let ray = Ray {
            origin: Vector::new(1.0, 1.0, PLANET_RADIUS + DISTANCE),
            dir: Vector::new(1.5, 1.5, -1.0).normalize(),
        };
        let hit = planet
            .cast_local_ray_and_get_normal(&ray, 1e5, true)
            .expect("hit not found");
        assert!(hit.time_of_impact > 0.0 && hit.time_of_impact < 1e5);
        let surface_point = ray.origin + ray.dir * hit.time_of_impact;
        assert!((surface_point.length() - PLANET_RADIUS).abs() < 5.0);
    }

    #[test]
    fn intersects_smoke() {
        const PLANET_RADIUS: f64 = 6371e3;
        let ball = Ball { radius: 1.0 };
        let planet = Planet::new(
            Arc::new(FlatTerrain::new(2u32.pow(12), 17).expect("valid test terrain")),
            32,
            PLANET_RADIUS,
        )
        .expect("valid test planet");

        assert!(
            PlanetDispatcher
                .intersection_test(&Pose::translation(PLANET_RADIUS, 0.0, 0.0), &planet, &ball,)
                .unwrap()
                .intersecting
        );
        assert!(
            !PlanetDispatcher
                .intersection_test(
                    &Pose::translation(PLANET_RADIUS + ball.radius * 2.0, 0.0, 0.0),
                    &planet,
                    &ball,
                )
                .unwrap()
                .intersecting
        );
    }

    #[test]
    fn cast_shape_nonlinear_smoke() {
        const PLANET_RADIUS: f64 = 6371e3;
        let ball = Ball { radius: 1.0 };
        let planet = Planet::new(
            Arc::new(FlatTerrain::new(2u32.pow(12), 17).expect("valid test terrain")),
            32,
            PLANET_RADIUS,
        )
        .expect("valid test planet");

        let toi = PlanetDispatcher
            .cast_shapes_nonlinear(
                &NonlinearRigidMotion::constant_position(Pose::identity()),
                &planet,
                &NonlinearRigidMotion {
                    start: Pose::translation(PLANET_RADIUS + ball.radius + 0.5, 0.0, 0.0),
                    local_center: Vector::ZERO,
                    linvel: -Vector::X,
                    angvel: Vector::ZERO,
                },
                &ball,
                0.0,
                1.0,
                true,
            )
            .unwrap()
            .expect("no hit");
        assert_eq!(toi.status, ShapeCastStatus::Converged);
        assert_relative_eq!(toi.time_of_impact, 0.5);
        assert_relative_eq!(toi.witness1, Vector::new(PLANET_RADIUS, 0.0, 0.0));
        assert_relative_eq!(toi.witness2, Vector::new(-ball.radius, 0.0, 0.0));
        assert_relative_eq!(toi.normal1, Vector::X);
        assert_relative_eq!(toi.normal2, -Vector::X);

        // Same configuration as above, but too far to hit within the allotted time
        let toi = PlanetDispatcher
            .cast_shapes_nonlinear(
                &NonlinearRigidMotion::constant_position(Pose::identity()),
                &planet,
                &NonlinearRigidMotion {
                    start: Pose::translation(PLANET_RADIUS + ball.radius + 1.5, 0.0, 0.0),
                    local_center: Vector::ZERO,
                    linvel: -Vector::X,
                    angvel: Vector::ZERO,
                },
                &ball,
                0.0,
                1.0,
                true,
            )
            .unwrap();
        assert!(toi.is_none());
    }

    #[test]
    fn patch_interpolation() {
        let res = 2u32.pow(12);
        let chunk = Coords {
            x: 12,
            y: 47,
            face: Face::Px,
        };
        let patch = Patch::new(&chunk, res);

        // Verify that the corners are consistent. All other points won't be, since
        // `Coords::direction` interpolates on the sphere.
        for coords in [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]] {
            // Exact equality is intended here, as a prerequisite for neighboring patches to be
            // seamless.
            assert_eq!(patch.get(&Vec2::from(coords)), corner(&chunk, res, coords));
        }
    }

    #[test]
    fn patch_projection() {
        let res = 2u32.pow(12);
        let chunk = Coords {
            x: 12,
            y: 47,
            face: Face::Px,
        };
        let patch = Patch::new(&chunk, res);

        let coords = [0.1, 0.4];
        assert_abs_diff_eq!(
            patch.project(&patch.get(&Vec2::from(coords))),
            Vec2::from(coords),
            epsilon = 1e-4
        );

        let coords = [0.9, 0.7];
        assert_abs_diff_eq!(
            patch.project(&patch.get(&Vec2::from(coords))),
            Vec2::from(coords),
            epsilon = 1e-4
        );
    }

    #[test]
    fn patch_projection_2() {
        // Regression test for a case that needs the second quadratic solution
        let patch = Patch {
            a: [1.0, 1.1, 0.0].into(),
            b: [1.0, 1.2, 1.3].into(),
            c: [1.0, -0.1, -0.1].into(),
            d: [1.0, 0.0, 1.0].into(),
        };

        let p = patch.project(&Vector::new(1.0, 0.1, 0.1));
        assert!(p.x >= 0.0 && p.x <= 1.0);
        assert!(p.y >= 0.0 && p.y <= 1.0);
    }

    #[test]
    fn patch_projection_3() {
        // Regression test for a case that needs the second quadratic solution
        let patch = Patch {
            a: [1.0, 2.0, 1.0].into(),
            b: [1.0, 2.0, 2.0].into(),
            c: [1.0, 1.0, 1.0].into(),
            d: [1.0, 1.0, 2.0].into(),
        };

        assert_abs_diff_eq!(
            patch.project(&Vector::new(1.0, 1.1, 1.1)),
            Vec2::new(0.1, 0.9)
        );
        assert_abs_diff_eq!(patch.get(&Vec2::new(0.1, 0.9)), Vector::new(1.0, 1.1, 1.1));
    }
}
