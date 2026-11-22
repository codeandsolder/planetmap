use super::Planet;
use super::geometry::Patch;
use crate::cubemap::Coords;
use hashbrown::HashMap;
use lru_slab::LruSlab;
use parry3d_f64::bounding_volume::{Aabb, BoundingVolume};
use parry3d_f64::math::{Vec2, Vector};
use std::sync::atomic::AtomicU8;
use std::sync::{Arc, OnceLock};

/// At most 1.5 MiB of f64 vertices per tile is retained for warm queries.
/// Larger grids use the same projection algorithm with transient geometry.
pub(super) const MAX_CACHED_PROJECTION_VERTICES: usize = 65_536;

#[derive(Clone)]
pub(super) struct Cache {
    pub(super) slots: LruSlab<Arc<ChunkData>>,
    pub(super) index: HashMap<Coords, u32>,
    pub(super) max_size: u32,
}

impl Cache {
    pub(super) fn new(capacity: u32) -> Self {
        Self {
            // LruSlab grows on demand. Do not preallocate potentially
            // enormous capacity merely because it fits the identifier space.
            slots: LruSlab::new(),
            index: HashMap::new(),
            max_size: capacity,
        }
    }

    pub(super) fn lookup(&mut self, coords: &Coords) -> Option<(u32, Arc<ChunkData>)> {
        let slot = *self.index.get(coords)?;
        Some((slot, Arc::clone(self.slots.get_mut(slot))))
    }

    pub(super) fn insert(&mut self, data: Arc<ChunkData>) -> (u32, Arc<ChunkData>) {
        if let Some(entry) = self.lookup(&data.coords) {
            return entry;
        }
        if self.slots.len() == self.max_size
            && let Some(lru) = self.slots.lru()
        {
            let old = self.slots.remove(lru);
            self.index.remove(&old.coords);
        }
        let coords = data.coords;
        let slot = self.slots.insert(Arc::clone(&data));
        self.index.insert(coords, slot);
        (slot, data)
    }
}

/// Row-major displaced-quad boxes with compact row and block summaries.
/// Query cost is proportional to nearby rows/blocks rather than the number
/// of quads in the whole chunk; every summary is a union of exact leaf AABBs.
#[derive(Debug)]
pub(super) struct QuadBoundsIndex {
    pub(super) quads: Box<[Aabb]>,
    pub(super) rows: Box<[Aabb]>,
    pub(super) blocks: Box<[Aabb]>,
}

pub(super) const QUAD_BLOCK_WIDTH: u32 = 8;
/// Index on fourth collision query for a still-resident chunk; a single
/// collision query should not build every vertex and bounding box.
pub(super) const QUAD_INDEX_BUILD_AFTER_QUERIES: u8 = 3;

#[derive(Debug)]
pub(super) struct ChunkData {
    pub(super) samples: Box<[f32]>,
    pub(super) coords: Coords,
    pub(super) min: f32,
    pub(super) max: f32,
    /// Lazily constructed displaced mesh vertices shared across point queries.
    pub(super) projection_vertices: OnceLock<Box<[Vector]>>,
    /// Exact AABBs of displaced quads in row-major order; initialized only
    /// by collision broadphase queries and shared by the chunk LRU's Arc.
    pub(super) quad_bounds: OnceLock<QuadBoundsIndex>,
    pub(super) broadphase_queries: AtomicU8,
}

impl ChunkData {
    pub(super) fn new(coords: Coords, samples: Box<[f32]>) -> Self {
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
    pub(super) fn quad_bounds(&self, planet: &Planet) -> &QuadBoundsIndex {
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
    pub(super) fn projection_vertices(&self, planet: &Planet) -> &[Vector] {
        self.projection_vertices.get_or_init(|| {
            let patch = Patch::new(&self.coords);
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
