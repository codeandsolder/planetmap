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
    shape::{
        CompositeShape, FeatureId, HalfSpace, Shape, ShapeType, SubShapeId, Triangle, TypedShape,
    },
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

    /// Search an uncached chunk without constructing its full displaced
    /// vertex grid or quad index. Candidate leaves are checked against the
    /// same exact world-space quad AABBs as the warmed index.
    fn map_sparse_collision_quads(
        &self,
        patch: &Patch,
        data: &ChunkData,
        aabb: &Aabb,
        emit: &mut impl FnMut(Quad) -> bool,
    ) -> bool {
        let sparse = SparseQuadSearch {
            patch,
            samples: &data.samples,
            query: aabb,
            radius: self.radius,
            height_abs: f64::from(data.min).abs().max(f64::from(data.max).abs()),
            resolution: self.chunk_resolution,
        };
        sparse.visit(
            QuadCellRegion {
                x0: 0,
                x1: self.chunk_resolution - 1,
                y0: 0,
                y1: self.chunk_resolution - 1,
            },
            emit,
        )
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
        if data.samples.len() <= MAX_CACHED_PROJECTION_VERTICES {
            // A warm chunk takes the original cached-index path after one
            // relaxed counter load; avoid repeated OnceLock checks and an
            // extra chunk-size branch in the steady-state collision path.
            if data.broadphase_queries.load(AtomicOrdering::Relaxed)
                < QUAD_INDEX_BUILD_AFTER_QUERIES
                && data.quad_bounds.get().is_none()
                && data
                    .broadphase_queries
                    .fetch_add(1, AtomicOrdering::Relaxed)
                    < QUAD_INDEX_BUILD_AFTER_QUERIES
            {
                return self.map_sparse_collision_quads(&patch, &data, aabb, &mut emit);
            }
            // Chunks receiving repeat queries retain the exact row/block
            // index; extremely cold chunks do not allocate that index.
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
                            && !emit(Quad::new(&patch, side, na::Vector2::new(x, y)))
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

/// Conservative local AABB of a displaced angular cone, shared by ray and pair traversal.
fn displaced_cone_aabb(
    axis: Vector,
    min_cos: f64,
    radial_min: f64,
    radial_max: f64,
    pad: Vector,
) -> Aabb {
    if radial_min <= 0.0 {
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
    }
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
        let aabb = displaced_cone_aabb(axis, min_cos, radial_min, radial_max, pad);
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
        let address = na::Vector2::new(
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
    /// lies component-wise within `max(abs(min_height), abs(max_height))`.
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
                na::Vector2::new(region.x0, region.y0),
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

impl Planet {
    /// Seed proximity from the perspective quad closest to the other shape
    /// center, evaluating just two real triangles. Any valid candidate is a
    /// global *upper* distance bound, never an assumption about which patch
    /// wins; all closer faces are still visited by the proven hierarchy.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "Validated normalized patch positions are clamped to the u32 quad grid"
    )]
    #[expect(
        clippy::cast_sign_loss,
        reason = "Clamped normalized patch coordinates are never negative"
    )]
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
        let resolution = self.terrain.face_resolution();
        let coords = Coords::from_vector(resolution, &na3f32(axis));
        let patch = Patch::new(&coords, resolution);
        let side = self.chunk_resolution - 1;
        let uv = patch.project(&axis) * f64::from(side);
        let quad = Quad::new(
            &patch,
            side,
            na::Vector2::new(
                (uv.x.floor() as u32).min(side - 1),
                (uv.y.floor() as u32).min(side - 1),
            ),
        );
        let (_, data) = self
            .cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(self, &coords);
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
    #[expect(
        clippy::cast_possible_truncation,
        reason = "Only finite small angular cones are cast to f32, with 2e-6 radians padding"
    )]
    fn visit_proximity_cap(
        &self,
        bounds: &BoundingSphere,
        other_aabb: &Aabb,
        max_distance: &mut Real,
        visitor: &mut impl FnMut(Coords, u32, &Triangle, &mut Real) -> bool,
    ) -> Option<bool> {
        let distance = bounds.center().length();
        let cap_radius = bounds.radius() + *max_distance;
        let inscribed = self.radius / 3.0_f64.sqrt() + f64::from(self.terrain.min_height());
        if inscribed <= 64.0 * f64::EPSILON * self.radius
            || !distance.is_finite()
            || distance <= cap_radius
            || !cap_radius.is_finite()
        {
            return None;
        }
        let theta = (cap_radius / distance).asin();
        if (theta + 2.0e-6) * f64::from(self.terrain.face_resolution()) > 2.0 {
            return None;
        }
        let axis = na3f32(bounds.center() / distance);
        let face_resolution = self.terrain.face_resolution();
        for coords in Coords::neighborhood(face_resolution, axis, (theta + 2.0e-6) as f32) {
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
            let patch = Patch::new(&coords, self.terrain.face_resolution());
            let (_, data) = self
                .cache
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .get(self, &coords);
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
            resolution: self.terrain.face_resolution(),
            query_axis: if center_radius > 0.0 {
                center / center_radius
            } else {
                Vector::X
            },
            query_radius: center_radius,
            planet_radius: self.radius,
            min_height: f64::from(self.terrain.min_height()),
            max_height: f64::from(self.terrain.max_height()),
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
        for face in crate::cubemap::Face::iter() {
            regions.push(ProjectionRegion::new(
                face,
                0,
                query.resolution,
                0,
                query.resolution,
                &query,
            ));
        }
        while let Some(region) = regions.pop() {
            let threshold = bounds.radius() + *max_distance + roundoff;
            if region.lower_distance2 > threshold * threshold {
                break;
            }
            if region.x1 - region.x0 == 1 && region.y1 - region.y0 == 1 {
                let coords = Coords {
                    face: region.face,
                    x: region.x0,
                    y: region.y0,
                };
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
                    let next = ProjectionRegion::new(region.face, x0, x1, y0, y1, &query);
                    if next.lower_distance2 <= threshold * threshold {
                        regions.push(next);
                    }
                }
            }
        }
    }

    /// Return the terrain subshape ID only once the winner is known; visiting
    /// other chunks can recycle LRU slots, especially with capacity one.
    fn stable_triangle_id(&self, coords: Coords, triangle: u32) -> SubShapeId {
        let (slot, _) = self
            .cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .get(self, &coords);
        self.subshape_id(slot, triangle)
    }
}

/// A best-first cubemap region ordered by the smallest possible vertex
/// projection along a given plane normal. The bounds include all triangle
/// interiors, and remain safe if large negative heights invert their rays.
#[derive(Clone, Copy, Debug)]
struct PlaneSupportRegion {
    region: ProjectionRegion,
    lower_dot: f64,
}

impl PlaneSupportRegion {
    fn new(
        planet: &Planet,
        face: crate::cubemap::Face,
        x0: u32,
        x1: u32,
        y0: u32,
        y1: u32,
        normal: Vector,
    ) -> Self {
        let resolution = planet.terrain.face_resolution();
        let (axis, min_cos) = ProjectionRegion::cone(face, x0, x1, y0, y1, resolution);
        let min_height = f64::from(planet.terrain.min_height());
        let min_height_projection = if min_height >= 0.0 {
            min_height * min_cos
        } else {
            min_height
        };
        let min_radius = planet.radius.mul_add(min_cos, min_height_projection);
        let max_radius = planet.max_radius();
        let cos_center = axis.dot(normal).clamp(-1.0, 1.0);
        let min_direction_dot = if -cos_center >= min_cos {
            -1.0
        } else {
            let transverse = cos_center.mul_add(-cos_center, 1.0).max(0.0).sqrt();
            let aperture = min_cos.mul_add(-min_cos, 1.0).max(0.0).sqrt();
            cos_center
                .mul_add(min_cos, -transverse * aperture)
                .max(-1.0)
        };
        let minimum = if min_radius <= 0.0 {
            // An inward-flipped triangle is no longer confined to this
            // positive angular cone. The enclosing planet sphere remains
            // a conservative bound on every possible projected component.
            -max_radius
        } else if min_direction_dot >= 0.0 {
            min_direction_dot * min_radius
        } else {
            min_direction_dot * max_radius
        };
        let slack = 512.0 * f64::EPSILON * (max_radius + planet.radius);
        Self {
            region: ProjectionRegion {
                face,
                x0,
                x1,
                y0,
                y1,
                lower_distance2: 0.0,
            },
            lower_dot: minimum - slack,
        }
    }
}
impl PartialEq for PlaneSupportRegion {
    fn eq(&self, other: &Self) -> bool {
        self.lower_dot.total_cmp(&other.lower_dot).is_eq() && self.region == other.region
    }
}
impl Eq for PlaneSupportRegion {}
impl PartialOrd for PlaneSupportRegion {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for PlaneSupportRegion {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other
            .lower_dot
            .total_cmp(&self.lower_dot)
            .then_with(|| self.region.cmp(&other.region))
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
        let resolution = self.terrain.face_resolution();
        for face in crate::cubemap::Face::iter() {
            regions.push(PlaneSupportRegion::new(
                self, face, 0, resolution, 0, resolution, normal,
            ));
        }
        while let Some(region) = regions.pop() {
            if winner
                .as_ref()
                .is_some_and(|best| region.lower_dot > best.dot)
            {
                break;
            }
            if region.region.x1 - region.region.x0 == 1 && region.region.y1 - region.region.y0 == 1
            {
                let coords = Coords {
                    face: region.region.face,
                    x: region.region.x0,
                    y: region.region.y0,
                };
                let (_, data) = self
                    .cache
                    .lock()
                    .unwrap_or_else(std::sync::PoisonError::into_inner)
                    .get(self, &coords);
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
            let mid_x = region.region.x0 + (region.region.x1 - region.region.x0) / 2;
            let mid_y = region.region.y0 + (region.region.y1 - region.region.y0) / 2;
            for (x0, x1) in [(region.region.x0, mid_x), (mid_x, region.region.x1)] {
                if x0 == x1 {
                    continue;
                }
                for (y0, y1) in [(region.region.y0, mid_y), (mid_y, region.region.y1)] {
                    if y0 == y1 {
                        continue;
                    }
                    let candidate =
                        PlaneSupportRegion::new(self, region.region.face, x0, x1, y0, y1, normal);
                    if winner
                        .as_ref()
                        .is_none_or(|best| candidate.lower_dot <= best.dot)
                    {
                        regions.push(candidate);
                    }
                }
            }
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
    fn new(planet: &Planet, pose: &Pose, plane: &HalfSpace) -> Option<Self> {
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

/// One cubemap rectangle and its conservative world-space AABB.
/// The bounds enclose displaced triangle interiors, including seams and
/// deeply negative terrain; they are the same cone bounds as the ray BVH.
#[derive(Clone, Copy, Debug)]
struct PairPatch {
    region: ProjectionRegion,
    aabb: Aabb,
    // Shape1 frame; the cone contains all displaced triangle interiors.
    origin: Vector,
    cone_axis: Vector,
    cone_cos: f64,
    radial_min: f64,
    radial_max: f64,
    sphere_center: Vector,
    sphere_radius: f64,
    padding: f64,
}

impl PairPatch {
    fn new(
        planet: &Planet,
        pose: &Pose,
        face: crate::cubemap::Face,
        x0: u32,
        x1: u32,
        y0: u32,
        y1: u32,
    ) -> Self {
        let (axis, min_cos) =
            ProjectionRegion::cone(face, x0, x1, y0, y1, planet.terrain.face_resolution());
        let min_height = f64::from(planet.terrain.min_height());
        let max_height = f64::from(planet.terrain.max_height());
        let min_projection = if min_height >= 0.0 {
            min_height * min_cos
        } else {
            min_height
        };
        let radial_min = planet.radius.mul_add(min_cos, min_projection);
        let radial_max = outer_displaced_radius(planet.radius, min_height, max_height);
        let pad = Vector::splat(
            512.0
                * f64::EPSILON
                * (radial_max + planet.radius + pose.translation.abs().max_element()),
        );
        // Convexity of the squared distance to a point on the cone axis
        // makes the extremal radial value occur at an endpoint.
        let cone_axis = pose.rotation * axis;
        let (sphere_center, sphere_radius) = if radial_min <= 0.0 {
            (pose.translation, radial_max + pad.x)
        } else {
            let center_radius = radial_min.midpoint(radial_max);
            let radius_at = |radial: f64| {
                radial
                    .mul_add(
                        radial,
                        center_radius
                            .mul_add(center_radius, -2.0 * radial * center_radius * min_cos),
                    )
                    .max(0.0)
                    .sqrt()
            };
            (
                pose.translation + cone_axis * center_radius,
                radius_at(radial_min).max(radius_at(radial_max)) + pad.x,
            )
        };
        let aabb =
            displaced_cone_aabb(axis, min_cos, radial_min, radial_max, pad).transform_by(pose);
        Self {
            region: ProjectionRegion {
                face,
                x0,
                x1,
                y0,
                y1,
                lower_distance2: 0.0,
            },
            aabb,
            origin: pose.translation,
            cone_axis,
            cone_cos: min_cos,
            radial_min,
            radial_max,
            sphere_center,
            sphere_radius,
            padding: pad.x,
        }
    }

    /// Signed support interval enclosing every displaced triangle point
    /// along a unit world direction. Inverted cones use the outer sphere.
    fn support_interval(&self, direction: Vector) -> (f64, f64) {
        let shift = self.origin.dot(direction);
        if self.radial_min <= 0.0 {
            return (
                shift - self.radial_max - self.padding,
                shift + self.radial_max + self.padding,
            );
        }
        let cos_center = self.cone_axis.dot(direction).clamp(-1.0, 1.0);
        let transverse = cos_center.mul_add(-cos_center, 1.0).max(0.0).sqrt()
            * self.cone_cos.mul_add(-self.cone_cos, 1.0).max(0.0).sqrt();
        let min_direction = if -cos_center >= self.cone_cos {
            -1.0
        } else {
            cos_center.mul_add(self.cone_cos, -transverse).max(-1.0)
        };
        let max_direction = if cos_center >= self.cone_cos {
            1.0
        } else {
            cos_center.mul_add(self.cone_cos, transverse).min(1.0)
        };
        let low = min_direction
            * if min_direction >= 0.0 {
                self.radial_min
            } else {
                self.radial_max
            };
        let high = max_direction
            * if max_direction >= 0.0 {
                self.radial_max
            } else {
                self.radial_min
            };
        (shift + low - self.padding, shift + high + self.padding)
    }

    const fn is_leaf(&self) -> bool {
        self.region.x1 - self.region.x0 == 1 && self.region.y1 - self.region.y0 == 1
    }

    /// Generate ready-to-use child bounds, dropping empty rectangles
    /// before evaluating costly angular cone geometry.
    fn children<'a>(&self, planet: &'a Planet, pose: &'a Pose) -> impl Iterator<Item = Self> + 'a {
        let r = self.region;
        let mid_x = r.x0 + (r.x1 - r.x0) / 2;
        let mid_y = r.y0 + (r.y1 - r.y0) / 2;
        [
            (r.x0, mid_x, r.y0, mid_y),
            (mid_x, r.x1, r.y0, mid_y),
            (r.x0, mid_x, mid_y, r.y1),
            (mid_x, r.x1, mid_y, r.y1),
        ]
        .into_iter()
        .filter(|(x0, x1, y0, y1)| x0 < x1 && y0 < y1)
        .map(move |(x0, x1, y0, y1)| Self::new(planet, pose, r.face, x0, x1, y0, y1))
    }

    fn extent_squared(&self) -> f64 {
        (self.aabb.maxs - self.aabb.mins).length_squared()
    }
}

#[derive(Clone, Copy, Debug)]
struct PlanetPairRegion {
    first: PairPatch,
    second: PairPatch,
    lower: f64,
}

impl PlanetPairRegion {
    fn new(first: PairPatch, second: PairPatch) -> Self {
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
    let res1 = first.terrain.face_resolution();
    let res2 = second.terrain.face_resolution();
    for f1 in crate::cubemap::Face::iter() {
        let a = PairPatch::new(first, &Pose::IDENTITY, f1, 0, res1, 0, res1);
        for f2 in crate::cubemap::Face::iter() {
            let b = PairPatch::new(second, pos12, f2, 0, res2, 0, res2);
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
        if candidate.first.is_leaf() && candidate.second.is_leaf() {
            let a = candidate.first.region;
            let b = candidate.second.region;
            let coords1 = Coords {
                face: a.face,
                x: a.x0,
                y: a.y0,
            };
            let coords2 = Coords {
                face: b.face,
                x: b.x0,
                y: b.y0,
            };
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
        let mut cache = first
            .cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if at1 != at2 && cache.max_size < 2 {
            return Err(Unsupported);
        }
        let (slot1, _) = cache.get(first, &at1);
        let slot2 = if at1 == at2 {
            slot1
        } else {
            cache.get(first, &at2).0
        };
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

fn planet_shape_contact(
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

impl QueryDispatcher for PlanetDispatcher {
    fn intersection_test(
        &self,
        pos12: &Pose,
        g1: &dyn Shape,
        g2: &dyn Shape,
    ) -> Result<ShapeIntersection, Unsupported> {
        if let (Some(first), Some(second)) = (g1.as_shape::<Planet>(), g2.as_shape::<Planet>()) {
            return planet_planet_intersection(pos12, first, second);
        }
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
        pos12: &Pose,
        g1: &dyn Shape,
        g2: &dyn Shape,
    ) -> Result<ShapeDistance, Unsupported> {
        if let (Some(first), Some(second)) = (g1.as_shape::<Planet>(), g2.as_shape::<Planet>()) {
            return planet_planet_distance(pos12, first, second);
        }
        if let Some(planet) = g1.as_shape::<Planet>() {
            return planet_shape_distance(pos12, planet, g2);
        }
        if let Some(planet) = g2.as_shape::<Planet>() {
            return planet_shape_distance(&pos12.inverse(), planet, g1).map(ShapeDistance::swapped);
        }
        Err(Unsupported)
    }

    fn contact(
        &self,
        pos12: &Pose,
        g1: &dyn Shape,
        g2: &dyn Shape,
        prediction: Real,
    ) -> Result<Option<Contact>, Unsupported> {
        if let (Some(first), Some(second)) = (g1.as_shape::<Planet>(), g2.as_shape::<Planet>()) {
            return planet_planet_contact(pos12, first, second, prediction);
        }
        if let Some(planet) = g1.as_shape::<Planet>() {
            return planet_shape_contact(pos12, planet, g2, prediction);
        }
        if let Some(planet) = g2.as_shape::<Planet>() {
            return planet_shape_contact(&pos12.inverse(), planet, g1, prediction)
                .map(|contact| contact.map(Contact::flipped));
        }
        Err(Unsupported)
    }

    fn closest_points(
        &self,
        pos12: &Pose,
        g1: &dyn Shape,
        g2: &dyn Shape,
        max_dist: Real,
    ) -> Result<ClosestPoints, Unsupported> {
        if let (Some(first), Some(second)) = (g1.as_shape::<Planet>(), g2.as_shape::<Planet>()) {
            return planet_planet_closest_points(pos12, first, second, max_dist);
        }
        if let Some(planet) = g1.as_shape::<Planet>() {
            return planet_shape_closest_points(pos12, planet, g2, max_dist);
        }
        if let Some(planet) = g2.as_shape::<Planet>() {
            return planet_shape_closest_points(&pos12.inverse(), planet, g1, max_dist)
                .map(|points| points.flipped());
        }
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
        // The single-planet CCD adapter delegates to Parry's
        // Triangle/other-shape primitive; Planet/Planet is not one of
        // those primitives. Report Unsupported instead of a false miss.
        if g1.as_shape::<Planet>().is_some() && g2.as_shape::<Planet>().is_some() {
            return Err(Unsupported);
        }
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
        // The single-planet CCD adapter delegates to Parry's
        // Triangle/other-shape primitive; Planet/Planet is not one of
        // those primitives. Report Unsupported instead of a false miss.
        if g1.as_shape::<Planet>().is_some() && g2.as_shape::<Planet>().is_some() {
            return Err(Unsupported);
        }
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
        // Do not report a successful empty manifold when the convex
        // per-triangle dispatcher cannot accept another custom Planet.
        if g1.as_shape::<Planet>().is_some() && g2.as_shape::<Planet>().is_some() {
            return Err(Unsupported);
        }
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
            (0..quad_resolution)
                .map(move |x| Quad::new(self, quad_resolution, na::Vector2::new(x, y)))
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
    position: na::Vector2<u32>,
}

impl Quad {
    fn new(patch: &Patch, resolution: u32, position: na::Vector2<u32>) -> Self {
        let offsets = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let position_f = Vec2::new(f64::from(position.x), f64::from(position.y));
        Self {
            corners: offsets
                .map(|x| patch.get(&((position_f + Vec2::from(x)) / f64::from(resolution)))),
            position,
        }
    }

    fn index(&self, chunk_resolution: u32) -> u32 {
        self.position.y * (chunk_resolution - 1) + self.position.x
    }

    fn displace(&self, radius: f64, chunk_resolution: u32, chunk_samples: &[f32]) -> DisplacedQuad {
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
                let corners = Quad::new(&Patch::new(&coords, 1), 1, na::Vector2::zeros())
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
                    for position in [na::Vector2::new(0, 0), na::Vector2::new(3, 3)] {
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
        let triangle = Quad::new(&patch, 16, na::Vector2::new(8, 8))
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
                                let quad = Quad::new(&patch, side, na::Vector2::new(col, row));
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
                    for xy in [
                        na::Vector2::new(0, 0),
                        na::Vector2::new(7, 7),
                        na::Vector2::new(15, 15),
                    ] {
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
    fn planet_pair_proximity_is_supported_in_both_orders() {
        let first = Planet::new(Arc::new(FlatTerrain::new(1, 3).unwrap()), 1, 10.0).unwrap();
        let second = Planet::new(Arc::new(FlatTerrain::new(1, 3).unwrap()), 1, 7.0).unwrap();
        let pose = Pose::translation(15.0, 0.2, -0.3);
        let dispatcher = PlanetDispatcher;
        let d1 = dispatcher
            .distance(&pose, &first, &second)
            .expect("Planet/Planet distance should be supported");
        let d2 = dispatcher
            .distance(&pose.inverse(), &second, &first)
            .expect("Planet/Planet reversed distance should be supported");
        assert!(d1.distance > 0.0);
        assert!((d1.distance - d2.distance).abs() < 1.0e-10);
        let pts = dispatcher
            .closest_points(&pose, &first, &second, 20.0)
            .expect("Planet/Planet closest points should be supported");
        let reversed = dispatcher
            .closest_points(&pose.inverse(), &second, &first, 20.0)
            .expect("Planet/Planet reversed closest points should be supported");
        // Flat parallel faces have infinitely many equally valid witness
        // pairs; reversal need not choose the same point on the face.
        for pair in [pts, reversed.flipped()] {
            let ClosestPoints::WithinMargin(a, b) = pair else {
                panic!("expected separated witnesses: {pair:?}");
            };
            assert!((a.distance(pose * b) - d1.distance).abs() < 1.0e-10);
        }
        assert!(dispatcher
            .contact(&pose, &first, &second, 20.0)
            .expect("Planet/Planet predicted contact should be supported")
            .is_some());
    }

    /// A reference that uses neither the dual-tree hierarchy nor the
    /// chunk collision broadphase: enumerate both meshes independently
    /// and evaluate every triangle pair with the stock Parry dispatcher.
    fn brute_force_planet_pair(
        first: &Planet,
        second: &Planet,
        pos12: &Pose,
        prediction: f64,
    ) -> (f64, ClosestPoints, Option<Contact>, bool) {
        let all = |planet: &Planet| {
            let mut result = Vec::new();
            for face in Face::iter() {
                for y in 0..planet.terrain.face_resolution() {
                    for x in 0..planet.terrain.face_resolution() {
                        let coords = Coords { face, x, y };
                        let samples = planet.sample(&coords);
                        let patch = Patch::new(&coords, planet.terrain.face_resolution());
                        result.extend(
                            patch
                                .triangles(planet.radius, planet.chunk_resolution, &samples)
                                .map(|(_, triangle)| triangle),
                        );
                    }
                }
            }
            result
        };
        let first_triangles = all(first);
        let second_triangles = all(second);
        let dispatcher = DefaultQueryDispatcher;
        let mut distance = f64::INFINITY;
        let mut pts = ClosestPoints::Disjoint;
        let mut best_points_dist = f64::INFINITY;
        let mut contact: Option<Contact> = None;
        let mut intersects = false;
        for first_triangle in &first_triangles {
            for second_triangle in &second_triangles {
                let result = dispatcher
                    .distance(pos12, first_triangle, second_triangle)
                    .unwrap();
                distance = distance.min(result.distance);
                let this_pts = dispatcher
                    .closest_points(pos12, first_triangle, second_triangle, 100.0)
                    .unwrap();
                match this_pts {
                    ClosestPoints::Intersecting => pts = ClosestPoints::Intersecting,
                    ClosestPoints::WithinMargin(a, b)
                        if !matches!(pts, ClosestPoints::Intersecting) =>
                    {
                        let d = a.distance(*pos12 * b);
                        if d < best_points_dist {
                            best_points_dist = d;
                            pts = this_pts;
                        }
                    }
                    _ => {}
                }
                if let Some(c) = dispatcher
                    .contact(pos12, first_triangle, second_triangle, prediction)
                    .unwrap()
                {
                    if contact.is_none_or(|best| c.dist < best.dist) {
                        contact = Some(c);
                    }
                }
                intersects |= dispatcher
                    .intersection_test(pos12, first_triangle, second_triangle)
                    .unwrap()
                    .intersecting;
            }
        }
        (distance, pts, contact, intersects)
    }

    #[test]
    fn planet_pair_queries_match_exhaustive_six_face_triangle_pairs() {
        let first = Planet::new(Arc::new(FlatTerrain::new(1, 3).unwrap()), 1, 10.0).unwrap();
        let second = Planet::new(
            Arc::new(ProjectionTerrain {
                face_resolution: 2,
                peak: None,
                patterned: true,
                samples: std::sync::atomic::AtomicUsize::new(0),
            }),
            1,
            7.0,
        )
        .unwrap();
        let poses = [
            Pose::translation(15.0, 0.1, -0.4),
            Pose::translation(9.0, 0.4, 0.3),
            Pose::translation(12.0, 12.0, -0.5),
            Pose::translation(-11.0, 3.0, 0.5),
            Pose::translation(0.0, 0.0, 0.0),
            Pose::translation(14.0, 0.3, -0.1)
                * Pose::from_rotation(Rotation::from_axis_angle(Vector::Y, 0.37)),
        ];
        for (case, pose) in poses.into_iter().enumerate() {
            let (expected_dist, expected_pts, expected_contact, expected_hit) =
                brute_force_planet_pair(&first, &second, &pose, 1.8);
            let actual = PlanetDispatcher.distance(&pose, &first, &second).unwrap();
            assert!(
                (actual.distance - expected_dist).abs() < 1.0e-8,
                "case={case},dist={actual:?},expected={expected_dist}"
            );
            let reversed = PlanetDispatcher
                .distance(&pose.inverse(), &second, &first)
                .unwrap();
            assert!((reversed.distance - expected_dist).abs() < 1.0e-8);
            let points = PlanetDispatcher
                .closest_points(&pose, &first, &second, 100.0)
                .unwrap();
            let points_swapped = PlanetDispatcher
                .closest_points(&pose.inverse(), &second, &first, 100.0)
                .unwrap()
                .flipped();
            for witness in [points, points_swapped] {
                match (expected_pts, witness) {
                    (ClosestPoints::Intersecting, ClosestPoints::Intersecting) => {}
                    (ClosestPoints::WithinMargin(_, _), ClosestPoints::WithinMargin(a, b)) => {
                        assert!(
                            (a.distance(pose * b) - expected_dist).abs() < 1.0e-8,
                            "case={case},witness={witness:?},expected={expected_pts:?}"
                        );
                    }
                    _ => panic!(
                        "case={case}, witness mismatch: {witness:?}, expected={expected_pts:?}"
                    ),
                }
            }
            let cutoff = (expected_dist - 0.2).max(0.0);
            if expected_dist > cutoff {
                assert_eq!(
                    PlanetDispatcher
                        .closest_points(&pose, &first, &second, cutoff)
                        .unwrap(),
                    ClosestPoints::Disjoint
                );
            }
            let contact = PlanetDispatcher
                .contact(&pose, &first, &second, 1.8)
                .unwrap();
            assert_eq!(contact.is_some(), expected_contact.is_some(), "case={case}");
            if let (Some(a), Some(b)) = (contact, expected_contact) {
                assert!(
                    (a.dist - b.dist).abs() < 1.0e-8,
                    "case={case},contact={a:?},reference={b:?}"
                );
            }
            let intersection = PlanetDispatcher
                .intersection_test(&pose, &first, &second)
                .unwrap();
            assert_eq!(intersection.intersecting, expected_hit, "case={case}");
        }
    }

    #[test]
    fn planet_pair_deeply_inward_heights_match_exhaustive_meshes() {
        struct InwardTerrain;
        impl Terrain for InwardTerrain {
            fn sample(&self, coords: &Coords, out: &mut [f32]) {
                let level = match coords.face {
                    Face::Px => -9.0,
                    Face::Nz => -5.0,
                    _ => 0.0,
                };
                out.fill(level);
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
        let first = Planet::new(Arc::new(InwardTerrain), 1, 10.0).unwrap();
        let second = Planet::new(Arc::new(FlatTerrain::new(1, 3).unwrap()), 1, 7.0).unwrap();
        for (case, pose) in [
            Pose::translation(-11.0, 0.0, 0.0),
            Pose::translation(4.0, 3.0, 2.0),
            Pose::translation(18.0, 1.0, -0.5),
            Pose::translation(-1.0, -2.0, 1.0)
                * Pose::from_rotation(Rotation::from_axis_angle(Vector::Z, 0.7)),
        ]
        .into_iter()
        .enumerate()
        {
            let (expected_dist, expected_pts, expected_contact, expected_hit) =
                brute_force_planet_pair(&first, &second, &pose, 2.0);
            let actual = PlanetDispatcher.distance(&pose, &first, &second).unwrap();
            assert!(
                (actual.distance - expected_dist).abs() < 1.0e-8,
                "inverted case={case}, actual={actual:?}, reference={expected_dist}"
            );
            let points = PlanetDispatcher
                .closest_points(&pose, &first, &second, 40.0)
                .unwrap();
            match (expected_pts, points) {
                (ClosestPoints::Intersecting, ClosestPoints::Intersecting) => {}
                (ClosestPoints::WithinMargin(_, _), ClosestPoints::WithinMargin(a, b)) => {
                    assert!(
                        (a.distance(pose * b) - expected_dist).abs() < 1.0e-8,
                        "inverted case={case}, points={points:?}"
                    );
                }
                _ => panic!("inverted case={case}, points={points:?}, expected={expected_pts:?}"),
            }
            let c = PlanetDispatcher
                .contact(&pose, &first, &second, 2.0)
                .unwrap();
            assert_eq!(c.is_some(), expected_contact.is_some());
            if let (Some(a), Some(b)) = (c, expected_contact) {
                assert!((a.dist - b.dist).abs() < 1.0e-8);
            }
            assert_eq!(
                PlanetDispatcher
                    .intersection_test(&pose, &first, &second)
                    .unwrap()
                    .intersecting,
                expected_hit,
                "inverted case={case}"
            );
        }
    }

    #[test]
    fn planetary_radius_pair_queries_prune_global_cross_face_regions() {
        const R1: f64 = 6_371_000.0;
        const R2: f64 = 500_000.0;
        let first = Planet::new(Arc::new(FlatTerrain::new(4096, 17).unwrap()), 1, R1).unwrap();
        let second = Planet::new(Arc::new(FlatTerrain::new(4096, 17).unwrap()), 1, R2).unwrap();
        for (case, (axis, gap)) in [
            (Vector::X, 2.5),
            (-Vector::X, 10.0),
            (Vector::new(1.0, 1.0, 0.0).normalize(), 20.0),
        ]
        .into_iter()
        .enumerate()
        {
            let center = axis * (R1 + R2 + gap);
            let pose = Pose::translation(center.x, center.y, center.z);
            let distance = PlanetDispatcher.distance(&pose, &first, &second).unwrap();
            assert!(
                distance.distance >= gap - 1.0e-4 && distance.distance <= gap + 5.0,
                "case={case},distance={distance:?},lower={gap}"
            );
            let reverse = PlanetDispatcher
                .distance(&pose.inverse(), &second, &first)
                .unwrap();
            assert!((distance.distance - reverse.distance).abs() < 1.0e-6);
            let pair = PlanetDispatcher
                .closest_points(&pose, &first, &second, gap + 10.0)
                .unwrap();
            let ClosestPoints::WithinMargin(a, b) = pair else {
                panic!("case={case},planetary witnesses={pair:?}");
            };
            assert!((a.distance(pose * b) - distance.distance).abs() < 1.0e-5);
            assert_eq!(
                PlanetDispatcher
                    .closest_points(&pose, &first, &second, gap * 0.5)
                    .unwrap(),
                ClosestPoints::Disjoint
            );
            assert!(
                !PlanetDispatcher
                    .intersection_test(&pose, &first, &second)
                    .unwrap()
                    .intersecting
            );
            // Per-shape IDs must refer to live cache slots immediately
            // after each query, even with capacity one on both planets.
            for (planet, id) in [(&first, distance.subshape1), (&second, distance.subshape2)] {
                let slot = id / planet.triangles_per_chunk();
                assert!(planet
                    .cache
                    .lock()
                    .unwrap()
                    .slots
                    .iter()
                    .any(|(i, _)| i == slot));
            }
        }
    }

    #[test]
    fn pair_region_bounds_enclose_every_triangle_in_variable_height_terrain() {
        struct VaryingTerrain {
            low: f32,
            high: f32,
        }
        impl Terrain for VaryingTerrain {
            fn sample(&self, at: &Coords, out: &mut [f32]) {
                let values = [self.low, self.low * 0.5, 0.0, self.high * 0.5, self.high];
                for (index, sample) in out.iter_mut().enumerate() {
                    let hash = (at.face as u32)
                        .wrapping_mul(17)
                        .wrapping_add(at.x.wrapping_mul(7))
                        .wrapping_add(at.y.wrapping_mul(11))
                        .wrapping_add(u32::try_from(index).unwrap().wrapping_mul(23));
                    *sample = values[(hash % 5) as usize];
                }
            }
            fn face_resolution(&self) -> u32 {
                2
            }
            fn chunk_resolution(&self) -> u32 {
                3
            }
            fn min_height(&self) -> f32 {
                self.low
            }
            fn max_height(&self) -> f32 {
                self.high
            }
        }
        let first = Planet::new(
            Arc::new(VaryingTerrain {
                low: -9.0,
                high: 4.0,
            }),
            1,
            10.0,
        )
        .unwrap();
        let second = Planet::new(
            Arc::new(VaryingTerrain {
                low: -6.0,
                high: 5.0,
            }),
            1,
            7.0,
        )
        .unwrap();
        let poses = [
            Pose::translation(17.5, 1.3, -2.0),
            Pose::translation(9.0, 8.5, 0.2)
                * Pose::from_rotation(Rotation::from_axis_angle(Vector::Y, 0.43)),
        ];
        let directions = [
            Vector::X,
            Vector::Y,
            Vector::Z,
            Vector::new(1.0, 1.0, 0.0).normalize(),
            Vector::new(-0.3, 0.5, -0.8).normalize(),
        ];
        let tile_mesh = |planet: &Planet| {
            let mut tiles = Vec::new();
            for face in Face::iter() {
                for y in 0..2 {
                    for x in 0..2 {
                        let coords = Coords { face, x, y };
                        let patch = Patch::new(&coords, 2);
                        let samples = planet.sample(&coords);
                        let tris: Vec<_> = patch
                            .triangles(planet.radius, 3, &samples)
                            .map(|(_, triangle)| triangle)
                            .collect();
                        tiles.push((coords, tris));
                    }
                }
            }
            tiles
        };
        let first_tiles = tile_mesh(&first);
        let second_tiles = tile_mesh(&second);
        for (planet, pose, tiles) in [
            (&first, Pose::IDENTITY, &first_tiles),
            (&second, poses[1], &second_tiles),
        ] {
            for face in Face::iter() {
                for (x0, x1, y0, y1) in [
                    (0, 2, 0, 2),
                    (0, 1, 0, 1),
                    (1, 2, 0, 1),
                    (0, 1, 1, 2),
                    (1, 2, 1, 2),
                ] {
                    let region = PairPatch::new(planet, &pose, face, x0, x1, y0, y1);
                    for (coords, tris) in tiles {
                        if coords.face != face
                            || coords.x < x0
                            || coords.x >= x1
                            || coords.y < y0
                            || coords.y >= y1
                        {
                            continue;
                        }
                        for tri in tris {
                            for local in [tri.a, tri.b, tri.c, (tri.a + tri.b + tri.c) / 3.0] {
                                let pt = pose * local;
                                let inside_box = pt.x >= region.aabb.mins.x - 1.0e-9
                                    && pt.x <= region.aabb.maxs.x + 1.0e-9
                                    && pt.y >= region.aabb.mins.y - 1.0e-9
                                    && pt.y <= region.aabb.maxs.y + 1.0e-9
                                    && pt.z >= region.aabb.mins.z - 1.0e-9
                                    && pt.z <= region.aabb.maxs.z + 1.0e-9;
                                assert!(
                                    inside_box,
                                    "region {face:?} {x0}:{x1},{y0}:{y1}: AABB misses {pt:?}"
                                );
                                assert!(
                                    pt.distance(region.sphere_center)
                                        <= region.sphere_radius + 1.0e-9,
                                    "region {face:?} {x0}:{x1},{y0}:{y1}: sphere misses {pt:?}"
                                );
                                for direction in directions {
                                    let (low, high) = region.support_interval(direction);
                                    let dot = pt.dot(direction);
                                    assert!(dot>=low-1.0e-9 && dot<=high+1.0e-9,
                                        "region {face:?} {x0}:{x1},{y0}:{y1}: dir={direction:?}, dot={dot}, bounds={low}..{high}");
                                }
                            }
                        }
                    }
                }
            }
        }
        let dispatcher = DefaultQueryDispatcher;
        for pose in poses {
            for (at1, tris1) in &first_tiles {
                let a = PairPatch::new(
                    &first,
                    &Pose::IDENTITY,
                    at1.face,
                    at1.x,
                    at1.x + 1,
                    at1.y,
                    at1.y + 1,
                );
                for (at2, tris2) in &second_tiles {
                    let b = PairPatch::new(
                        &second,
                        &pose,
                        at2.face,
                        at2.x,
                        at2.x + 1,
                        at2.y,
                        at2.y + 1,
                    );
                    let lower = PlanetPairRegion::new(a, b).lower;
                    let mut exact = f64::INFINITY;
                    for tri1 in tris1 {
                        for tri2 in tris2 {
                            exact =
                                exact.min(dispatcher.distance(&pose, tri1, tri2).unwrap().distance);
                        }
                    }
                    assert!(
                        lower <= exact + 1.0e-8,
                        "pair {at1:?} {at2:?}: lower={lower}, exact={exact}"
                    );
                }
            }
        }
    }

    #[test]
    fn planet_pair_same_instance_cache_ids_never_alias_evicted_chunks() {
        let one = Planet::new(Arc::new(FlatTerrain::new(1, 3).unwrap()), 1, 10.0).unwrap();
        let two = Planet::new(Arc::new(FlatTerrain::new(1, 3).unwrap()), 2, 10.0).unwrap();
        let pose = Pose::translation(12.5, 0.1, 0.2);
        // The first shape is one planet physically compared with its
        // translated copy; the winning triangles lie on opposite faces.
        // Capacity one cannot hold both subshapes' live IDs.
        assert!(PlanetDispatcher.distance(&pose, &one, &one).is_err());
        assert!(PlanetDispatcher.contact(&pose, &one, &one, 3.0).is_err());
        assert!(matches!(
            PlanetDispatcher
                .closest_points(&pose, &one, &one, 3.0)
                .unwrap(),
            ClosestPoints::WithinMargin(_, _)
        ));
        let distance = PlanetDispatcher.distance(&pose, &two, &two).unwrap();
        let n = two.triangles_per_chunk();
        assert_ne!(distance.subshape1 / n, distance.subshape2 / n);
        let cache = two.cache.lock().unwrap();
        for slot in [distance.subshape1 / n, distance.subshape2 / n] {
            assert!(cache.slots.iter().any(|(id, _)| id == slot));
        }
        drop(cache);
        let contact = PlanetDispatcher
            .contact(&pose, &two, &two, 3.0)
            .unwrap()
            .unwrap();
        assert_ne!(contact.subshape1 / n, contact.subshape2 / n);
    }

    #[test]
    fn unsupported_planet_pair_ccd_and_persistent_manifolds_report_errors() {
        let first = Planet::new(Arc::new(FlatTerrain::new(1, 3).unwrap()), 1, 10.0).unwrap();
        let second = Planet::new(Arc::new(FlatTerrain::new(1, 3).unwrap()), 1, 7.0).unwrap();
        let pose = Pose::translation(9.0, 0.2, -0.1);
        assert!(PlanetDispatcher
            .cast_shapes(
                &pose,
                Vector::ZERO,
                &first,
                &second,
                ShapeCastOptions::default()
            )
            .is_err());
        let motion1 =
            NonlinearRigidMotion::new(Pose::IDENTITY, Vector::ZERO, Vector::ZERO, Vector::ZERO);
        let motion2 = NonlinearRigidMotion::new(pose, Vector::ZERO, Vector::ZERO, Vector::ZERO);
        assert!(PlanetDispatcher
            .cast_shapes_nonlinear(&motion1, &first, &motion2, &second, 0.0, 1.0, true)
            .is_err());
        let mut manifolds = Vec::<ContactManifold<(), ()>>::new();
        let mut workspace = None;
        assert!(PlanetDispatcher
            .contact_manifolds(&pose, &first, &second, 0.0, &mut manifolds, &mut workspace)
            .is_err());
    }

    #[test]
    fn planet_dispatcher_supports_three_proximity_queries_in_both_orders() {
        let planet = Planet::new(Arc::new(FlatTerrain::new(1, 3).unwrap()), 1, 10.0).unwrap();
        let ball = Ball::new(0.3);
        let center = 10.0 / 3.0_f64.sqrt() + 1.3;
        let pose = Pose::translation(center, 0.0, 0.0);
        let dispatcher = PlanetDispatcher;
        let distance = dispatcher
            .distance(&pose, &planet, &ball)
            .expect("Planet/Ball distance must be supported");
        assert!(distance.distance > 0.0);
        let reverse_distance = dispatcher
            .distance(&pose.inverse(), &ball, &planet)
            .expect("Ball/Planet distance must be supported");
        assert!((distance.distance - reverse_distance.distance).abs() < 1.0e-9);
        let closest = dispatcher
            .closest_points(&pose, &planet, &ball, 3.0)
            .expect("Planet/Ball closest points must be supported");
        assert!(matches!(closest, ClosestPoints::WithinMargin(_, _)));
        let reversed = dispatcher
            .closest_points(&pose.inverse(), &ball, &planet, 3.0)
            .expect("Ball/Planet closest points must be supported");
        assert!(matches!(reversed, ClosestPoints::WithinMargin(_, _)));
        let contact = dispatcher
            .contact(&pose, &planet, &ball, 3.0)
            .expect("Planet/Ball contact must be supported");
        assert!(contact.is_some());
        let reverse_contact = dispatcher
            .contact(&pose.inverse(), &ball, &planet, 3.0)
            .expect("Ball/Planet contact must be supported");
        assert!(reverse_contact.is_some());
    }

    /// Independent exact reference over all six cube faces, without reusing
    /// the dispatcher traversal or any of its spatial pruning.
    fn exhaustive_proximity_reference(
        planet: &Planet,
        other: &dyn Shape,
        pos12: &Pose,
        prediction: f64,
    ) -> (f64, ClosestPoints, Option<Contact>) {
        let dispatcher = DefaultQueryDispatcher;
        let mut min_dist = f64::INFINITY;
        let mut closest = ClosestPoints::Disjoint;
        let mut contact: Option<Contact> = None;
        for face in Face::iter() {
            for y in 0..planet.terrain.face_resolution() {
                for x in 0..planet.terrain.face_resolution() {
                    let coords = Coords { face, x, y };
                    let samples = planet.sample(&coords);
                    let patch = Patch::new(&coords, planet.terrain.face_resolution());
                    for (_, triangle) in
                        patch.triangles(planet.radius, planet.chunk_resolution, &samples)
                    {
                        let dist = dispatcher
                            .distance(pos12, &triangle, other)
                            .unwrap()
                            .distance;
                        min_dist = min_dist.min(dist);
                        match dispatcher.closest_points(pos12, &triangle, other, f64::MAX) {
                            Ok(ClosestPoints::Intersecting) => {
                                closest = ClosestPoints::Intersecting;
                            }
                            Ok(ClosestPoints::WithinMargin(p1, p2))
                                if !matches!(closest, ClosestPoints::Intersecting) =>
                            {
                                let d = p1.distance(*pos12 * p2);
                                let existing = match closest {
                                    ClosestPoints::WithinMargin(a, b) => a.distance(*pos12 * b),
                                    _ => f64::INFINITY,
                                };
                                if d < existing {
                                    closest = ClosestPoints::WithinMargin(p1, p2);
                                }
                            }
                            _ => {}
                        }
                        if let Some(candidate) = dispatcher
                            .contact(pos12, &triangle, other, prediction)
                            .unwrap()
                        {
                            if contact.is_none_or(|previous| candidate.dist < previous.dist) {
                                contact = Some(candidate);
                            }
                        }
                    }
                }
            }
        }
        (min_dist, closest, contact)
    }

    #[test]
    fn proximity_queries_match_exhaustive_all_face_references() {
        for patterned in [false, true] {
            let terrain = Arc::new(ProjectionTerrain {
                face_resolution: 2,
                peak: None,
                patterned,
                samples: std::sync::atomic::AtomicUsize::new(0),
            });
            let planet = Planet::new(terrain, 1, 10.0).unwrap();
            let ball = Ball::new(0.4);
            let positions = [
                Vector::X * 5.2,
                Vector::X * 6.2,
                Vector::X * 12.0,
                Vector::new(5.1, 4.9, -5.3),
                Vector::new(-7.0, 7.4, 0.1),
                Vector::new(2.0, -8.3, 9.5),
                Vector::ZERO,
                Vector::new(-20.0, 4.0, 3.0),
            ];
            for (case, center) in positions.into_iter().enumerate() {
                let pose = Pose::translation(center.x, center.y, center.z);
                let (expected_dist, expected_pts, expected_contact) =
                    exhaustive_proximity_reference(&planet, &ball, &pose, 3.0);
                let distance = PlanetDispatcher.distance(&pose, &planet, &ball).unwrap();
                assert!(
                    (distance.distance - expected_dist).abs() < 1.0e-8,
                    "patterned={patterned},case={case},dist={distance:?},expected={expected_dist}"
                );
                let swapped = PlanetDispatcher
                    .distance(&pose.inverse(), &ball, &planet)
                    .unwrap();
                assert!((swapped.distance - expected_dist).abs() < 1.0e-8);
                assert_eq!(distance.subshape1, swapped.subshape2);
                let points = PlanetDispatcher
                    .closest_points(&pose, &planet, &ball, 100.0)
                    .unwrap();
                let reverse_points = PlanetDispatcher
                    .closest_points(&pose.inverse(), &ball, &planet, 100.0)
                    .unwrap();
                assert_eq!(points, reverse_points.flipped());
                if matches!(expected_pts, ClosestPoints::Intersecting) {
                    assert_eq!(points, ClosestPoints::Intersecting);
                } else if let ClosestPoints::WithinMargin(p1, p2) = points {
                    assert!((p1.distance(pose*p2)-expected_dist).abs() < 1.0e-8,
                        "patterned={patterned},case={case},closest={points:?},expected={expected_pts:?}");
                } else {
                    panic!("missing witness points: {points:?}");
                }
                let contact = PlanetDispatcher
                    .contact(&pose, &planet, &ball, 3.0)
                    .unwrap();
                assert_eq!(contact.is_some(), expected_contact.is_some());
                if let (Some(actual), Some(expected)) = (contact, expected_contact) {
                    assert!(
                        (actual.dist - expected.dist).abs() < 1.0e-8,
                        "patterned={patterned},case={case},actual={actual:?},expected={expected:?}"
                    );
                }
                let reverse = PlanetDispatcher
                    .contact(&pose.inverse(), &ball, &planet, 3.0)
                    .unwrap();
                assert_eq!(reverse.is_some(), contact.is_some());
                if let (Some(a), Some(b)) = (contact, reverse) {
                    assert!((a.dist - b.dist).abs() < 1.0e-8);
                    assert_eq!(a.subshape1, b.subshape2);
                }
                // The cache has one slot: the final terrain subshape ID
                // must describe the resident chunk after the whole search.
                let triangle_count = planet.triangles_per_chunk();
                let slot = distance.subshape1 / triangle_count;
                assert!(
                    planet
                        .cache
                        .lock()
                        .unwrap()
                        .slots
                        .iter()
                        .any(|(id, _)| id == slot),
                    "winner was already evicted"
                );
                let cutoff = (expected_dist - 0.1).max(0.0);
                let result = PlanetDispatcher
                    .closest_points(&pose, &planet, &ball, cutoff)
                    .unwrap();
                if expected_dist > cutoff {
                    assert_eq!(result, ClosestPoints::Disjoint);
                }
            }
        }
    }

    #[test]
    fn proximity_queries_match_exhaustive_rotated_cuboid() {
        let planet = Planet::new(Arc::new(FlatTerrain::new(2, 4).unwrap()), 1, 10.0).unwrap();
        let box_shape = Cuboid::new(Vector::new(0.3, 0.5, 0.8));
        for (case, center) in [
            Vector::new(7.0, 0.3, 0.2),
            Vector::new(5.2, 0.5, -0.7),
            Vector::new(7.2, 7.2, 0.1),
            Vector::new(-9.0, 4.0, -2.0),
        ]
        .into_iter()
        .enumerate()
        {
            let angle = 0.37 * (f64::from(u32::try_from(case).unwrap()) + 1.0);
            let pose = Pose::translation(center.x, center.y, center.z)
                * Pose::from_rotation(Rotation::from_axis_angle(Vector::Y, angle));
            let (expected_dist, _, expected_contact) =
                exhaustive_proximity_reference(&planet, &box_shape, &pose, 2.0);
            let actual = PlanetDispatcher
                .distance(&pose, &planet, &box_shape)
                .unwrap();
            assert!((actual.distance - expected_dist).abs() < 1.0e-8);
            let points = PlanetDispatcher
                .closest_points(&pose, &planet, &box_shape, 20.0)
                .unwrap();
            if let ClosestPoints::WithinMargin(a, b) = points {
                assert!((a.distance(pose * b) - expected_dist).abs() < 1.0e-8);
            } else {
                assert_eq!(points, ClosestPoints::Intersecting);
            }
            let actual_contact = PlanetDispatcher
                .contact(&pose, &planet, &box_shape, 2.0)
                .unwrap();
            assert_eq!(actual_contact.is_some(), expected_contact.is_some());
            if let (Some(a), Some(b)) = (actual_contact, expected_contact) {
                assert!((a.dist - b.dist).abs() < 1.0e-8);
            }
        }
    }

    #[test]
    fn planetary_radius_proximity_matches_independent_surface_projection() {
        const RADIUS: f64 = 6_371_000.0;
        let planet = Planet::new(Arc::new(FlatTerrain::new(4096, 17).unwrap()), 1, RADIUS).unwrap();
        let ball = Ball::new(0.3);
        for (case, axis) in [
            Vector::X,
            Vector::new(1.0, 1.0, 0.0002).normalize(),
            Vector::new(0.0001, -1.0, 1.0).normalize(),
            Vector::new(-1.0, -1.0, -1.0).normalize(),
        ]
        .into_iter()
        .enumerate()
        {
            for offset in [-0.4, 0.3, 1.7, 8.0] {
                let point = axis * (RADIUS + offset);
                let pose = Pose::translation(point.x, point.y, point.z);
                let projection = planet.project_local_point(point, false);
                let expected = (projection.point.distance(point) - ball.radius).max(0.0);
                let result = PlanetDispatcher.distance(&pose, &planet, &ball).unwrap();
                assert!(
                    (result.distance - expected).abs() < 1.0e-5,
                    "case={case},offset={offset},distance={result:?},expected={expected}"
                );
                let closest = PlanetDispatcher
                    .closest_points(&pose, &planet, &ball, 12.0)
                    .unwrap();
                if expected > 0.0 {
                    let ClosestPoints::WithinMargin(a, b) = closest else {
                        panic!("missing planetary closest point: {case}, {offset}: {closest:?}");
                    };
                    assert!((a.distance(pose * b) - expected).abs() < 1.0e-5);
                } else {
                    assert_eq!(closest, ClosestPoints::Intersecting);
                }
                if expected > 0.1 {
                    let cutoff = expected * 0.5;
                    let disjoint = PlanetDispatcher
                        .closest_points(&pose, &planet, &ball, cutoff)
                        .unwrap();
                    assert_eq!(disjoint, ClosestPoints::Disjoint);
                }
            }
        }
    }

    #[test]
    fn contact_normals_and_subshape_ids_follow_shape_local_coordinates() {
        let planet = Planet::new(Arc::new(FlatTerrain::new(3, 4).unwrap()), 1, 10.0).unwrap();
        let ball = Ball::new(0.55);
        for center in [
            Vector::new(9.7, 0.2, -0.1),
            Vector::new(10.2, -0.3, 0.2),
            Vector::new(-0.1, -0.2, 10.1),
        ] {
            let pose = Pose::translation(center.x, center.y, center.z);
            let actual = PlanetDispatcher
                .contact(&pose, &planet, &ball, 1.5)
                .unwrap()
                .unwrap();
            let reverse = PlanetDispatcher
                .contact(&pose.inverse(), &ball, &planet, 1.5)
                .unwrap()
                .unwrap();
            assert!((actual.dist - reverse.dist).abs() < 1.0e-10);
            assert!((actual.point1 - reverse.point2).length() < 1.0e-10);
            assert!((actual.point2 - reverse.point1).length() < 1.0e-10);
            assert!((actual.normal1 - reverse.normal2).length() < 1.0e-10);
            assert!((actual.normal2 - reverse.normal1).length() < 1.0e-10);
            assert_eq!(actual.subshape1, reverse.subshape2);
            assert_eq!(actual.subshape2, reverse.subshape1);
            assert!((actual.normal1.length() - 1.0).abs() < 1.0e-9);
            assert!((actual.normal2.length() - 1.0).abs() < 1.0e-9);
            assert!((actual.normal1 + pose.rotation * actual.normal2).length() < 1.0e-9);
            // Subshape IDs are meaningful immediately on return, even if
            // the global search evicted the geometrically best chunk.
            let slot = actual.subshape1 / planet.triangles_per_chunk();
            assert!(planet
                .cache
                .lock()
                .unwrap()
                .slots
                .iter()
                .any(|(id, _)| id == slot));
        }
    }

    #[test]
    fn proximity_queries_preserve_other_composite_part_ids() {
        use parry3d_f64::shape::{Compound, SharedShape};
        let planet = Planet::new(Arc::new(FlatTerrain::new(1, 3).unwrap()), 1, 10.0).unwrap();
        let other = Compound::new(vec![
            (Pose::translation(-1.0, 0.0, 0.0), SharedShape::ball(0.5)),
            (Pose::translation(2.0, 0.0, 0.0), SharedShape::ball(0.25)),
        ]);
        let pose = Pose::translation(7.5, 0.1, -0.2);
        let (expected, _, expected_contact) =
            exhaustive_proximity_reference(&planet, &other, &pose, 2.0);
        let distance = PlanetDispatcher.distance(&pose, &planet, &other).unwrap();
        assert!((distance.distance - expected).abs() < 1.0e-9);
        assert_eq!(distance.subshape2, 0);
        let reversed = PlanetDispatcher
            .distance(&pose.inverse(), &other, &planet)
            .unwrap();
        assert_eq!(reversed.subshape1, 0);
        assert!((reversed.distance - expected).abs() < 1.0e-9);
        let closest = PlanetDispatcher
            .closest_points(&pose, &planet, &other, 5.0)
            .unwrap();
        let ClosestPoints::WithinMargin(a, b) = closest else {
            panic!("expected composite witness pair, got {closest:?}");
        };
        assert!((a.distance(pose * b) - expected).abs() < 1.0e-9);
        let contact = PlanetDispatcher
            .contact(&pose, &planet, &other, 2.0)
            .unwrap();
        assert_eq!(contact.is_some(), expected_contact.is_some());
        if let Some(contact) = contact {
            assert_eq!(contact.subshape2, 0);
            assert!((contact.dist - expected_contact.unwrap().dist).abs() < 1.0e-9);
        }
    }

    #[test]
    fn infinite_halfspace_proximity_matches_exhaustive_surface_support() {
        for patterned in [false, true] {
            let terrain = Arc::new(ProjectionTerrain {
                face_resolution: 2,
                peak: None,
                patterned,
                samples: std::sync::atomic::AtomicUsize::new(0),
            });
            let planet = Planet::new(terrain, 1, 10.0).unwrap();
            let plane = HalfSpace::new(Vector::X);
            for (case, (center, angle)) in [
                (Vector::new(-12.0, 0.0, 0.0), 0.0),
                (Vector::new(12.0, 0.0, 0.0), 0.0),
                (Vector::new(-4.0, 3.0, 2.0), 0.47),
                (Vector::new(0.0, -6.0, 0.0), -0.9),
                (Vector::ZERO, 1.1),
            ]
            .into_iter()
            .enumerate()
            {
                let pose = Pose::translation(center.x, center.y, center.z)
                    * Pose::from_rotation(Rotation::from_axis_angle(Vector::Z, angle));
                let (min_dist, expected_pts, _) =
                    exhaustive_proximity_reference(&planet, &plane, &pose, 40.0);
                // The exact supporting-plane reference must use real
                // displaced geometry. Parry 0.31 triangle/HalfSpace contact
                // may report a signed depth inconsistent with its points
                // when the HalfSpace is translated; do not perpetuate that.
                let normal = pose.rotation * plane.normal;
                let mut expected_signed = f64::INFINITY;
                for face in Face::iter() {
                    for y in 0..planet.terrain.face_resolution() {
                        for x in 0..planet.terrain.face_resolution() {
                            let coords = Coords { face, x, y };
                            let samples = planet.sample(&coords);
                            let patch = Patch::new(&coords, planet.terrain.face_resolution());
                            for (_, triangle) in
                                patch.triangles(planet.radius, planet.chunk_resolution, &samples)
                            {
                                for vertex in [triangle.a, triangle.b, triangle.c] {
                                    expected_signed =
                                        expected_signed.min(normal.dot(vertex - pose.translation));
                                }
                            }
                        }
                    }
                }
                let dist = PlanetDispatcher.distance(&pose, &planet, &plane).unwrap();
                assert!(
                    (dist.distance - min_dist).abs() < 1.0e-9,
                    "patterned={patterned},case={case},distance={dist:?},expected={min_dist}"
                );
                let reverse = PlanetDispatcher
                    .distance(&pose.inverse(), &plane, &planet)
                    .unwrap();
                assert!((reverse.distance - min_dist).abs() < 1.0e-9);
                assert_eq!(dist.subshape1, reverse.subshape2);
                let closest = PlanetDispatcher
                    .closest_points(&pose, &planet, &plane, 40.0)
                    .unwrap();
                let reversed = PlanetDispatcher
                    .closest_points(&pose.inverse(), &plane, &planet, 40.0)
                    .unwrap();
                assert_eq!(closest, reversed.flipped());
                if matches!(expected_pts, ClosestPoints::Intersecting) {
                    assert_eq!(closest, ClosestPoints::Intersecting);
                } else {
                    let ClosestPoints::WithinMargin(a, b) = closest else {
                        panic!("expected separated witness points: {closest:?}");
                    };
                    assert!((a.distance(pose * b) - min_dist).abs() < 1.0e-9);
                }
                let actual = PlanetDispatcher
                    .contact(&pose, &planet, &plane, 40.0)
                    .unwrap();
                assert_eq!(actual.is_some(), expected_signed <= 40.0);
                if let Some(a) = actual {
                    assert!(
                        (a.dist - expected_signed).abs() < 1.0e-9,
                        "patterned={patterned},case={case},contact={a:?},reference={expected_signed}"
                    );
                    let flipped = PlanetDispatcher
                        .contact(&pose.inverse(), &plane, &planet, 40.0)
                        .unwrap()
                        .unwrap();
                    assert!((a.dist - flipped.dist).abs() < 1.0e-9);
                    assert_eq!(a.subshape1, flipped.subshape2);
                    assert!((a.point1 - flipped.point2).length() < 1.0e-9);
                    assert!((a.normal1 - flipped.normal2).length() < 1.0e-9);
                }
                if min_dist > 0.1 {
                    let disjoint = PlanetDispatcher
                        .closest_points(&pose, &planet, &plane, min_dist / 2.0)
                        .unwrap();
                    assert_eq!(disjoint, ClosestPoints::Disjoint);
                }
            }
        }
    }

    #[test]
    fn inward_inverted_halfspace_support_matches_exhaustive_vertices() {
        struct InvertedHeight;
        impl Terrain for InvertedHeight {
            fn sample(&self, coords: &Coords, out: &mut [f32]) {
                let level = if coords.face == Face::Px {
                    -9.0
                } else if coords.face == Face::Nz {
                    -5.0
                } else {
                    0.0
                };
                out.fill(level);
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
        let planet = Planet::new(Arc::new(InvertedHeight), 1, 10.0).unwrap();
        let plane = HalfSpace::new(Vector::X);
        for (case, (center, axis)) in [
            (Vector::new(-0.4, 0.2, 0.0), Vector::X),
            (
                Vector::new(-4.0, 2.0, 3.0),
                Vector::new(-1.0, 1.0, 1.0).normalize(),
            ),
            (
                Vector::new(8.0, -2.0, 3.0),
                Vector::new(0.1, -1.0, 0.4).normalize(),
            ),
        ]
        .into_iter()
        .enumerate()
        {
            let rotation = Rotation::from_rotation_arc(Vector::X, axis);
            let pose =
                Pose::translation(center.x, center.y, center.z) * Pose::from_rotation(rotation);
            let normal = pose.rotation * plane.normal;
            let mut minimum = f64::INFINITY;
            for face in Face::iter() {
                for y in 0..2 {
                    for x in 0..2 {
                        let coords = Coords { face, x, y };
                        let samples = planet.sample(&coords);
                        let patch = Patch::new(&coords, 2);
                        for (_, triangle) in patch.triangles(10.0, 3, &samples) {
                            for vertex in [triangle.a, triangle.b, triangle.c] {
                                minimum = minimum.min(normal.dot(vertex - pose.translation));
                            }
                        }
                    }
                }
            }
            let actual = PlanetDispatcher
                .contact(&pose, &planet, &plane, 20.0)
                .unwrap()
                .unwrap();
            assert!(
                (actual.dist - minimum).abs() < 1.0e-9,
                "inversion case={case},expected={minimum},actual={actual:?}"
            );
            assert_eq!(
                PlanetDispatcher
                    .distance(&pose, &planet, &plane)
                    .unwrap()
                    .distance,
                minimum.max(0.0)
            );
        }
    }

    #[test]
    fn large_planet_halfspace_distance_prunes_all_six_faces() {
        const RADIUS: f64 = 6_371_000.0;
        let planet = Planet::new(Arc::new(FlatTerrain::new(4096, 17).unwrap()), 1, RADIUS).unwrap();
        let plane = HalfSpace::new(Vector::X);
        let pose = Pose::translation(-RADIUS - 2.5, 0.0, 0.0);
        let result = PlanetDispatcher.distance(&pose, &planet, &plane).unwrap();
        assert!((result.distance - 2.5).abs() < 1.0e-5, "{result:?}");
        let closest = PlanetDispatcher
            .closest_points(&pose, &planet, &plane, 3.0)
            .unwrap();
        let ClosestPoints::WithinMargin(point1, point2) = closest else {
            panic!("unexpected closest points to halfspace: {closest:?}");
        };
        assert!((point1.distance(pose * point2) - 2.5).abs() < 1.0e-5);
        let contact = PlanetDispatcher
            .contact(&pose, &planet, &plane, 3.0)
            .unwrap()
            .unwrap();
        assert!((contact.dist - 2.5).abs() < 1.0e-5);
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
                    for xy in [
                        na::Vector2::new(0, 0),
                        na::Vector2::new(7, 7),
                        na::Vector2::new(15, 15),
                    ] {
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
