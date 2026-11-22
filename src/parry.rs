//! Collision detection for radial heightmaps
//!
//! Implement [`Terrain`] for your heightmap, then create colliders using it with a [`Planet`]. A
//! query dispatcher that handles both planets and standard shapes can be constructed with
//! `PlanetDispatcher.chain(DefaultQueryDispatcher)`.

mod bounds;
mod broadphase;
mod cast;
mod discrete;
mod geometry;
mod manifold;
mod projection;
mod ray;
mod tile_cache;

#[cfg(test)]
mod tests;

pub use manifold::Workspace;

use crate::cubemap::Coords;
use parry3d_f64::bounding_volume::{Aabb, BoundingSphere};
use parry3d_f64::mass_properties::MassProperties;
use parry3d_f64::math::{Real, Vector};
use parry3d_f64::shape::{Shape, ShapeType, SubShapeId, TypedShape};
use std::sync::{Arc, Mutex};
use tile_cache::{Cache, ChunkData};

/// Convert between the collision engine and cubemap vector representations.
const fn cube_vector(v: Vector) -> na::Vector3<f64> {
    na::Vector3::new(v.x, v.y, v.z)
}

/// Height data source for `Planet`.
///
/// Geometry and bounds are fixed for each Planet. Samples for a coordinate must
/// be deterministic, finite, and inside the advertised inclusive height range.
/// Concurrent cache misses may sample the same coordinate independently.
pub trait Terrain: Send + Sync + 'static {
    /// Generate a `chunk_resolution() * chunk_resolution()` grid of heights wrt. sea level.
    /// The address's resolution is equal to `face_resolution()`.
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
    face_resolution: u32,
    min_height: f32,
    max_height: f32,
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
        let face_resolution = terrain.face_resolution();
        validate_terrain_geometry(face_resolution, chunk_resolution)?;
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
        let triangles_per_chunk = quads_per_edge
            .checked_mul(quads_per_edge)
            .and_then(|quads| quads.checked_mul(2))
            .ok_or(PlanetConfigError::SubshapeIdOverflow)?;
        if triangles_per_chunk > (u64::from(u32::MAX) + 1) / u64::from(cache_size) {
            return Err(PlanetConfigError::SubshapeIdOverflow);
        }
        Ok(Self {
            chunk_resolution,
            face_resolution,
            min_height: min,
            max_height: max,
            terrain,
            radius,
            cache: Mutex::new(Cache::new(cache_size)),
        })
    }

    fn max_radius(&self) -> f64 {
        outer_displaced_radius(
            self.radius,
            f64::from(self.min_height),
            f64::from(self.max_height),
        )
    }

    fn sample(&self, coords: &Coords) -> Box<[f32]> {
        let mut samples =
            vec![0.0; self.chunk_resolution as usize * self.chunk_resolution as usize]
                .into_boxed_slice();
        self.terrain.sample(coords, &mut samples[..]);
        samples
    }

    fn lock_cache(&self) -> std::sync::MutexGuard<'_, Cache> {
        self.cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    fn cached_chunk(&self, coords: &Coords) -> (u32, Arc<ChunkData>) {
        let cached = self.lock_cache().lookup(coords);
        if let Some(entry) = cached {
            return entry;
        }
        // Application code runs without the mutex. A concurrent caller may
        // populate the same coordinate first; deterministic samples make
        // either result equivalent.
        let data = Arc::new(ChunkData::new(*coords, self.sample(coords)));
        self.lock_cache().insert(data)
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

    /// Return the terrain subshape ID only once the winner is known; visiting
    /// other chunks can recycle LRU slots, especially with capacity one.
    fn stable_triangle_id(&self, coords: Coords, triangle: u32) -> SubShapeId {
        let (slot, _) = self.cached_chunk(&coords);
        self.subshape_id(slot, triangle)
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

/// A [`PersistentQueryDispatcher`] that handles `Planet` shapes
pub struct PlanetDispatcher;
