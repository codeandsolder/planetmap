use hashbrown::HashMap;
use std::ops::{Index, IndexMut};
use std::sync::atomic::{AtomicU64, Ordering};

use slab::Slab;

use crate::cubemap::{Edge, Face};
use crate::Chunk;

struct Slot {
    chunk: Chunk,
    generation: u64,
    /// Whether the slot is ready for reading
    ready: bool,
    /// Sequence number of the most recent frame this slot was visited in
    in_frame: u64,
    /// Whether this slot belongs to the current visible render cut
    drawn: bool,
}

#[derive(Debug, Clone)]
pub struct Config {
    /// Maximum depth of quad-tree traversal
    pub(crate) max_depth: u8,
    /// Upper bound for `Neighborhood` fields
    ///
    /// Should be set to the base 2 logarithm of the number of quads along the edge of a chunk to
    /// reduce stitching artifacts across extreme `LoD` boundaries.
    pub(crate) max_neighbor_delta: u8,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            max_depth: 12,
            max_neighbor_delta: u8::MAX,
        }
    }
}

/// Failure to create an addressable terrain cache configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CacheConfigError {
    /// The quadtree depth exceeds the range supported by 32-bit grid coordinates.
    Depth(u8),
    /// A slot count is zero or cannot be represented as 32-bit cache indices.
    Slots(usize),
}

impl std::fmt::Display for CacheConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Depth(depth) => write!(f, "cache depth {depth} exceeds maximum 31"),
            Self::Slots(slots) => write!(f, "cache slots {slots} must fit in 1..=u32::MAX"),
        }
    }
}
impl std::error::Error for CacheConfigError {}

impl Config {
    /// Construct an addressable terrain cache configuration.
    ///
    /// # Errors
    ///
    /// Rejects a maximum depth that cannot be represented as a `u32` face grid.
    pub const fn new(max_depth: u8, max_neighbor_delta: u8) -> Result<Self, CacheConfigError> {
        if max_depth > Chunk::MAX_DEPTH {
            return Err(CacheConfigError::Depth(max_depth));
        }
        Ok(Self {
            max_depth,
            max_neighbor_delta,
        })
    }

    /// Maximum quadtree depth that can be selected for a viewpoint.
    #[must_use]
    pub const fn max_depth(&self) -> u8 {
        self.max_depth
    }

    /// Maximum amount of `LoD` coarsening reported across a shared edge.
    #[must_use]
    pub const fn max_neighbor_delta(&self) -> u8 {
        self.max_neighbor_delta
    }

    /// Number of slots needed by this config to represent maximum detail for a worst-case viewpoint
    #[must_use]
    pub fn slots_needed(&self) -> Option<usize> {
        Face::iter()
            .map(Chunk::root)
            .try_fold(0_usize, |sum, chunk| {
                sum.checked_add(self.slots_needed_inner(&chunk)?)
            })
    }

    fn slots_needed_inner(&self, chunk: &Chunk) -> Option<usize> {
        let viewpoint = na::Point3::from(na::Vector3::new(1.0, 1.0, 1.0).normalize());
        if chunk.depth == self.max_depth || !needs_subdivision(chunk, &viewpoint) {
            return Some(1);
        }
        let children = chunk.children()?;
        children.iter().try_fold(1_usize, |sum, child| {
            sum.checked_add(self.slots_needed_inner(child)?)
        })
    }
}

/// Identifies one allocation, not just its reusable physical cache slot.
/// Retain this token until the asynchronous transfer finishes or is cancelled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Allocation {
    slot: u32,
    generation: u64,
}

impl Allocation {
    /// Physical slot to which the caller writes the terrain data.
    #[must_use]
    pub const fn slot(self) -> u32 {
        self.slot
    }
}

// Unique even across managers; ordering with other memory is unnecessary.
static NEXT_GENERATION: AtomicU64 = AtomicU64::new(0);

/// Failure to complete or cancel a pending allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AllocationError {
    /// This allocation no longer exists or belongs to a different manager.
    InvalidAllocation(Allocation),
    /// This allocation has already completed.
    CompletedAllocation(Allocation),
}

impl std::fmt::Display for AllocationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidAllocation(allocation) => {
                write!(f, "invalid cache allocation {allocation:?}")
            }
            Self::CompletedAllocation(allocation) => {
                write!(f, "cache allocation {allocation:?} is already ready")
            }
        }
    }
}
impl std::error::Error for AllocationError {}

/// Helper for streaming `Chunk`-oriented `LoD` into a fixed-size cache
pub struct Manager {
    chunks: Slab<Slot>,
    max_slots: usize,
    index: HashMap<Chunk, u32>,
    config: Config,
    /// Frame counter used to update `Slot.in_frame`
    frame: u64,
    render: Vec<(Chunk, Neighborhood, u32)>,
}

impl Manager {
    /// Create a manager for a cache of `slots` slots.
    ///
    /// # Errors
    ///
    /// Rejects zero slots, slot counts exceeding `u32::MAX` and invalid
    /// depth configurations. Storage is grown on demand, not preallocated.
    pub fn new(slots: usize, config: Config) -> Result<Self, CacheConfigError> {
        if slots == 0 || slots > u32::MAX as usize {
            return Err(CacheConfigError::Slots(slots));
        }
        if config.max_depth > Chunk::MAX_DEPTH {
            return Err(CacheConfigError::Depth(config.max_depth));
        }
        Ok(Self {
            chunks: Slab::new(),
            max_slots: slots,
            index: HashMap::new(),
            config,
            frame: 0,
            render: Vec::new(),
        })
    }

    /// Compute slots to render for a given set of viewpoints, and to load for improved detail in
    /// future passes.
    ///
    /// Viewpoints should be positioned with regard to the sphere's origin, and scaled such
    /// viewpoints on the surface are 1.0 units from the sphere's origin.
    ///
    /// Writes unavailable chunks needed for target quality to `transfer`,
    /// ordered by increasing depth so coarse coverage can load first.
    pub fn update(&mut self, viewpoints: &[na::Point3<f64>], transfer: &mut Vec<Chunk>) {
        self.update_inner(viewpoints, transfer, None);
    }

    /// Updates the render cut and reports *every* chunk selected by the
    /// current viewpoint traversal, including already allocated and pending
    /// chunks. Applications can use `visited` to cancel obsolete in-flight
    /// requests without guessing which parts of the quadtree are needed.
    ///
    /// Both vectors are cleared before use. The visited list contains no
    /// duplicate chunks for a single update.
    pub fn update_with_visited(
        &mut self,
        viewpoints: &[na::Point3<f64>],
        transfer: &mut Vec<Chunk>,
        visited: &mut Vec<Chunk>,
    ) {
        self.update_inner(viewpoints, transfer, Some(visited));
    }

    fn update_inner(
        &mut self,
        viewpoints: &[na::Point3<f64>],
        transfer: &mut Vec<Chunk>,
        mut visited: Option<&mut Vec<Chunk>>,
    ) {
        transfer.clear();
        if let Some(list) = visited.as_deref_mut() {
            list.clear();
        }
        for &(_, _, slot) in &self.render {
            self.chunks[slot as usize].drawn = false;
        }
        self.render.clear();
        self.walk(viewpoints, transfer, &mut visited);
        // Resolve ancestors before descendants, across all faces. A small
        // cache must not fill with one deep subtree before other roots load.
        transfer.sort_unstable_by_key(Chunk::depth);

        // Make room for transfers by discarding chunks that we don't currently need.
        let mut available = self.max_slots - self.chunks.len();
        for idx in 0..self.chunks.capacity() {
            if available >= transfer.len() {
                break;
            }
            if !self.chunks.contains(idx)
                || !self.chunks[idx].ready
                || self.chunks[idx].in_frame == self.frame
            {
                continue;
            }
            let old = self.chunks.remove(idx);
            self.index.remove(&old.chunk);
            available += 1;
        }
        transfer.truncate(available);
        if let Some(next) = self.frame.checked_add(1) {
            self.frame = next;
        } else {
            // Reset visit marks before the frame counter can alias an old one.
            for (_, slot) in &mut self.chunks {
                slot.in_frame = u64::MAX;
            }
            self.frame = 0;
        }
    }

    /// Chunks that can be rendered immediately, with their `LoD` neighborhood and slot index
    #[inline]
    #[must_use]
    pub fn renderable(&self) -> &[(Chunk, Neighborhood, u32)] {
        &self.render
    }

    /// Allocate a slot for writing
    ///
    /// Determines where in the cache to transfer chunk data to. Returns `None` if the cache is
    /// full or the chunk already has a slot (pending or ready).
    pub fn allocate(&mut self, chunk: Chunk) -> Option<Allocation> {
        if self.index.contains_key(&chunk) || self.chunks.len() == self.max_slots {
            return None;
        }
        let entry = self.chunks.vacant_entry();
        let slot = u32::try_from(entry.key()).ok()?;
        let generation = NEXT_GENERATION
            .try_update(Ordering::Relaxed, Ordering::Relaxed, |id| id.checked_add(1))
            .ok()?;
        entry.insert(Slot {
            chunk,
            generation,
            ready: false,
            in_frame: self.frame.wrapping_sub(1),
            drawn: false,
        });
        let old = self.index.insert(chunk, slot);
        debug_assert!(
            old.is_none(),
            "a slot has already been allocated for this chunk"
        );
        Some(Allocation { slot, generation })
    }

    /// Revoke an unfinished allocation and make its slot immediately reusable.
    ///
    /// The caller must stop asynchronous writes before recycling the physical
    /// slot. Late completion notifications cannot complete a new allocation.
    ///
    /// # Errors
    /// Returns an allocation error if the token is stale, foreign, or already ready.
    pub fn cancel(&mut self, allocation: Allocation) -> Result<Chunk, AllocationError> {
        self.pending_slot(allocation)?;
        let chunk = self.chunks.remove(allocation.slot as usize).chunk;
        let previous = self.index.remove(&chunk);
        debug_assert_eq!(previous, Some(allocation.slot));
        Ok(chunk)
    }

    /// Mark the allocation's terrain data ready for reading.
    ///
    /// # Errors
    /// Returns an allocation error if the token is stale, foreign, or already ready.
    pub fn release(&mut self, allocation: Allocation) -> Result<(), AllocationError> {
        self.pending_slot(allocation)?.ready = true;
        Ok(())
    }

    fn pending_slot(&mut self, allocation: Allocation) -> Result<&mut Slot, AllocationError> {
        let slot = self
            .chunks
            .get_mut(allocation.slot as usize)
            .filter(|slot| slot.generation == allocation.generation)
            .ok_or(AllocationError::InvalidAllocation(allocation))?;
        if slot.ready {
            return Err(AllocationError::CompletedAllocation(allocation));
        }
        Ok(slot)
    }

    fn get(&self, chunk: &Chunk) -> Option<u32> {
        self.index.get(chunk).copied()
    }

    fn walk(
        &mut self,
        viewpoints: &[na::Point3<f64>],
        transfers: &mut Vec<Chunk>,
        visited: &mut Option<&mut Vec<Chunk>>,
    ) {
        // Gather the set of chunks we can should render and want to transfer
        for chunk in Face::iter().map(Chunk::root) {
            let slot = self.get(&chunk);
            // Kick off the loop for each face's quadtree
            self.walk_inner(
                viewpoints,
                transfers,
                visited,
                &ChunkState {
                    chunk,
                    slot,
                    renderable: slot.is_some_and(|idx| self.chunks[idx as usize].ready),
                },
            );
        }

        // Only actually drawn chunks can be stitching neighbors. Resident
        // parents of fully resolved child groups are not part of this cut.
        for &(_, _, slot) in &self.render {
            self.chunks[slot as usize].drawn = true;
        }

        // Compute the neighborhood of each rendered chunk.
        for &mut (chunk, ref mut neighborhood, _) in &mut self.render {
            for (edge, neighbor) in Edge::iter().zip(chunk.neighbors().iter()) {
                for x in neighbor.path() {
                    let slot = match self.index.get(&x) {
                        None => continue,
                        Some(&x) => x,
                    };
                    let state = &self.chunks[slot as usize];
                    if state.drawn {
                        neighborhood[edge] =
                            (chunk.depth - x.depth).min(self.config.max_neighbor_delta);
                        break;
                    }
                }
            }
        }
    }

    /// Walk the quadtree below `chunk`, recording chunks to render and transfer.
    fn walk_inner(
        &mut self,
        viewpoints: &[na::Point3<f64>],
        transfers: &mut Vec<Chunk>,
        visited: &mut Option<&mut Vec<Chunk>>,
        chunk: &ChunkState,
    ) {
        if let Some(list) = visited.as_deref_mut() {
            list.push(chunk.chunk);
        }
        // If this chunk is already associated with a cache slot, preserve that slot; otherwise,
        // tell the caller we want it.
        if let Some(idx) = chunk.slot {
            self.chunks[idx as usize].in_frame = self.frame;
        } else {
            transfers.push(chunk.chunk);
        }

        let subdivide = chunk.chunk.depth < self.config.max_depth
            && viewpoints
                .iter()
                .any(|v| needs_subdivision(&chunk.chunk, v));
        if !subdivide {
            if let Some(slot) = chunk.slot.filter(|_| chunk.renderable) {
                self.render
                    .push((chunk.chunk, Neighborhood::default(), slot));
            }
            return;
        }

        let Some(children) = chunk.chunk.children() else {
            return;
        };
        let child_slots = [
            self.get(&children[Edge::Nx]),
            self.get(&children[Edge::Ny]),
            self.get(&children[Edge::Px]),
            self.get(&children[Edge::Py]),
        ];
        // The children of this chunk might be rendered if:
        let children_renderable = chunk.renderable // this subtree should be rendered at all, and
            && child_slots                         // every child is already resident in the cache
            .iter()
            .all(|slot| slot.is_some_and(|x| self.chunks[x as usize].ready));
        // If this subtree should be rendered and the children can't be rendered, this chunk must be rendered.
        if !children_renderable {
            if let Some(slot) = chunk.slot.filter(|_| chunk.renderable) {
                self.render
                    .push((chunk.chunk, Neighborhood::default(), slot));
            }
        }
        // Recurse into the children
        for (&child, &slot) in children.iter().zip(child_slots.iter()) {
            self.walk_inner(
                viewpoints,
                transfers,
                visited,
                &ChunkState {
                    chunk: child,
                    renderable: children_renderable,
                    slot,
                },
            );
        }
    }
}

/// For each edge of a particular chunk, this represents the number of `LoD` levels higher that chunk
/// is than its neighbor on that edge.
///
/// Smoothly interpolating across chunk boundaries requires careful attention to these values. In
/// particular, any visualization of chunk data should take care to be continuous at the edge with
/// regard to an adjacent lower-detail level if discontinuities are undesirable. For example,
/// terrain using heightmapped tiles should weld together a subset of the vertices on an edge shared
/// with a lower-detail chunk.
///
/// Note that increases in `LoD` are not represented here; it is always the responsibility of the
/// higher-detail chunk to account for neighboring lower-detail chunks.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Default)]
pub struct Neighborhood {
    /// Decrease in `LoD` in the local -X direction
    pub nx: u8,
    /// Decrease in `LoD` in the local -Y direction
    pub ny: u8,
    /// Decrease in `LoD` in the local +X direction
    pub px: u8,
    /// Decrease in `LoD` in the local +Y direction
    pub py: u8,
}

impl Index<Edge> for Neighborhood {
    type Output = u8;
    fn index(&self, edge: Edge) -> &u8 {
        use Edge::{Nx, Ny, Px, Py};
        match edge {
            Nx => &self.nx,
            Ny => &self.ny,
            Px => &self.px,
            Py => &self.py,
        }
    }
}

impl IndexMut<Edge> for Neighborhood {
    fn index_mut(&mut self, edge: Edge) -> &mut u8 {
        use Edge::{Nx, Ny, Px, Py};
        match edge {
            Nx => &mut self.nx,
            Ny => &mut self.ny,
            Px => &mut self.px,
            Py => &mut self.py,
        }
    }
}

struct ChunkState {
    chunk: Chunk,
    /// Cache slot associated with this chunk, whether or not it's ready
    slot: Option<u32>,
    /// Whether the subtree at this chunk will be rendered
    renderable: bool,
}

fn needs_subdivision(chunk: &Chunk, viewpoint: &na::Point3<f64>) -> bool {
    // Half-angle of the cone whose edges are simultaneously tangent to the edges of a pair of
    // circles inscribed on a chunk at depth D and a neighbor of that chunk at depth D+1. Setting a
    // threshold larger than this leads to `LoD` deltas greater than 1 across edges.
    let max_half_angle = 1.0f64.atan2(10.0f64.sqrt());

    let center = na::Point3::from(chunk.origin().into_inner());
    if center.coords.dot(&viewpoint.coords) < 0.0 {
        return false;
    }
    let distance = na::distance(&center, viewpoint);
    // Half-angle of the circular cone from the camera containing an inscribed sphere
    let half_angle = (chunk.edge_length::<f64>() * 0.5).atan2(distance);
    half_angle >= max_half_angle
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn cache_rejects_invalid_depth_or_capacity_and_allocates_lazily() {
        assert!(matches!(
            Config::new(32, 1),
            Err(CacheConfigError::Depth(32))
        ));
        let config = Config::new(31, u8::MAX).unwrap();
        assert_eq!(config.max_depth(), 31);
        assert_eq!(config.max_neighbor_delta(), u8::MAX);
        assert!(matches!(
            Manager::new(0, config.clone()),
            Err(CacheConfigError::Slots(0))
        ));
        if let Ok(too_many) = usize::try_from(u64::from(u32::MAX) + 1) {
            assert!(matches!(
                Manager::new(too_many, config.clone()),
                Err(CacheConfigError::Slots(_))
            ));
        }
        let mut manager = Manager::new(100_000_000, config).unwrap();
        assert_eq!(manager.chunks.capacity(), 0);
        let slot = manager.allocate(Chunk::root(Face::Px)).unwrap();
        assert_eq!(slot.slot(), 0);
        assert!(manager.chunks.capacity() < 32);
    }

    #[test]
    fn duplicate_allocation_preserves_cache_slot_identity() {
        let mut manager =
            Manager::new(2, Config::default()).expect("valid test cache configuration");
        let first = Chunk::root(Face::Px);
        let second = Chunk::root(Face::Py);
        let slot = manager.allocate(first).unwrap();
        assert_eq!(manager.allocate(first), None);
        let other_slot = manager.allocate(second).unwrap();
        assert_ne!(slot, other_slot);
        assert_eq!(manager.cancel(slot), Ok(first));
        assert_eq!(manager.cancel(other_slot), Ok(second));
        let reused = manager.allocate(first).unwrap();
        manager.release(reused).unwrap();
        assert_eq!(
            manager.cancel(reused),
            Err(AllocationError::CompletedAllocation(reused))
        );
        assert_eq!(manager.allocate(first), None);
    }

    #[test]
    fn cancellation_releases_only_unfinished_slots_and_can_be_reallocated() {
        let mut mgr = Manager::new(
            6,
            Config {
                max_depth: 1,
                max_neighbor_delta: 1,
            },
        )
        .expect("valid test cache configuration");
        let root = Chunk::root(Face::Px);
        let slot = mgr.allocate(root).expect("room for first patch");
        assert_eq!(mgr.cancel(slot), Ok(root));
        assert_eq!(
            mgr.cancel(slot),
            Err(AllocationError::InvalidAllocation(slot))
        );
        let reused = mgr.allocate(root).expect("cancel frees cache slot");
        assert_eq!(reused.slot(), slot.slot());
        assert_ne!(reused, slot);
        mgr.release(reused).unwrap();
        assert_eq!(
            mgr.cancel(reused),
            Err(AllocationError::CompletedAllocation(reused))
        );
    }

    #[test]
    fn stale_and_foreign_completion_cannot_publish_recycled_slot() {
        let mut manager = Manager::new(1, Config::default()).unwrap();
        let mut other = Manager::new(1, Config::default()).unwrap();
        let root = Chunk::root(Face::Px);
        let first = manager.allocate(root).unwrap();
        manager.cancel(first).unwrap();
        let second = manager.allocate(root).unwrap();
        let foreign = other.allocate(root).unwrap();
        assert_eq!(first.slot(), second.slot());
        for wrong in [first, foreign] {
            assert_eq!(
                manager.release(wrong),
                Err(AllocationError::InvalidAllocation(wrong))
            );
            assert_eq!(
                manager.cancel(wrong),
                Err(AllocationError::InvalidAllocation(wrong))
            );
        }
        assert!(!manager.chunks[second.slot() as usize].ready);
        manager.release(second).unwrap();
        assert_eq!(
            manager.release(second),
            Err(AllocationError::CompletedAllocation(second))
        );
    }

    #[test]
    fn constrained_cache_loads_all_roots_before_descendants() {
        let mut manager = Manager::new(6, Config::new(8, 1).unwrap()).unwrap();
        let mut transfer = Vec::new();
        let viewpoint = na::Point3::new(1.0, 0.0, 0.0);
        manager.update(&[viewpoint], &mut transfer);
        assert_eq!(transfer.len(), 6);
        assert!(transfer.iter().all(|chunk| chunk.depth() == 0));
        for chunk in transfer.drain(..) {
            let allocation = manager.allocate(chunk).unwrap();
            manager.release(allocation).unwrap();
        }
        manager.update(&[viewpoint], &mut transfer);
        assert_eq!(manager.renderable().len(), 6);
    }

    #[test]
    fn visited_reports_current_viewpoint_tree_including_pending_chunks() {
        use std::collections::HashSet;

        let mut mgr = Manager::new(
            2048,
            Config {
                max_depth: 5,
                max_neighbor_delta: 1,
            },
        )
        .expect("valid test cache configuration");
        let east = na::Point3::new(1.001, 0.0, 0.0);
        let west = na::Point3::new(-1.001, 0.0, 0.0);
        let mut transfer = Vec::new();
        let mut visited = Vec::new();
        mgr.update_with_visited(&[east], &mut transfer, &mut visited);
        assert!(!transfer.is_empty());
        assert!(transfer.iter().all(|chunk| visited.contains(chunk)));
        assert_eq!(visited.len(), visited.iter().collect::<HashSet<_>>().len());
        let pending = *transfer
            .iter()
            .find(|chunk| chunk.depth >= 3)
            .expect("near-viewpoint deep patch");
        let slot = mgr.allocate(pending).unwrap();
        mgr.update_with_visited(&[west], &mut transfer, &mut visited);
        assert!(
            !visited.contains(&pending),
            "opposite hemisphere should not need old deep patch"
        );
        assert_eq!(mgr.cancel(slot), Ok(pending));
        mgr.update_with_visited(&[east], &mut transfer, &mut visited);
        assert!(
            transfer.contains(&pending),
            "cancelled tile must be requested again"
        );
    }

    #[test]
    fn cancellation_preserves_completed_parent_fallback() {
        let mut mgr = Manager::new(
            128,
            Config {
                max_depth: 2,
                max_neighbor_delta: 1,
            },
        )
        .expect("valid test cache configuration");
        for face in Face::iter() {
            let slot = mgr.allocate(Chunk::root(face)).unwrap();
            mgr.release(slot).unwrap();
        }
        let viewpoint = na::Point3::new(1.001, 0.0, 0.0);
        let mut transfer = Vec::new();
        let mut visited = Vec::new();
        mgr.update_with_visited(&[viewpoint], &mut transfer, &mut visited);
        let pending = *transfer.iter().find(|c| c.depth == 1).unwrap();
        let slot = mgr.allocate(pending).unwrap();
        assert_eq!(mgr.cancel(slot), Ok(pending));
        mgr.update_with_visited(&[viewpoint], &mut transfer, &mut visited);
        assert!(transfer.contains(&pending));
        assert!(
            mgr.renderable().contains(&(
                Chunk::root(pending.coords.face),
                Neighborhood::default(),
                mgr.index[&Chunk::root(pending.coords.face)],
            )),
            "root must stay rendered during incomplete child split"
        );
    }

    #[test]
    fn transfer_completeness() {
        let mut mgr =
            Manager::new(2048, Config::default()).expect("valid test cache configuration");
        let mut transfers = Vec::new();
        mgr.update(&[na::Point3::from(na::Vector3::z())], &mut transfers);
        assert_eq!(mgr.renderable().len(), 0);
        for &transfer in &transfers {
            let slot = mgr.allocate(transfer).unwrap();
            mgr.release(slot).unwrap();
        }
        mgr.update(&[na::Point3::from(na::Vector3::z())], &mut transfers);
        assert_eq!(transfers.len(), 0);
        assert_ne!(mgr.renderable().len(), 0);
    }

    #[test]
    fn slots_needed() {
        let viewpoint = na::Point3::from(na::Vector3::new(1.0, 1.0, 1.0).normalize());
        for max_depth in 0..12 {
            let config = Config {
                max_depth,
                ..Config::default()
            };
            let needed = config.slots_needed().unwrap();
            let mut mgr = Manager::new(2048, config).expect("valid test cache configuration");
            let mut transfers = Vec::new();
            mgr.update(&[viewpoint], &mut transfers);
            assert_eq!(transfers.len(), needed);
        }
    }

    #[test]
    fn neighborhood() {
        use crate::cubemap::Coords;
        let mut mgr =
            Manager::new(2048, Config::default()).expect("valid test cache configuration");
        let viewpoint = na::Point3::from(na::Vector3::new(1.0, 1.0, 1.0).normalize());
        let mut transfers = Vec::new();
        mgr.update(&[viewpoint], &mut transfers);
        assert_eq!(mgr.renderable().len(), 0);
        // Get +X to `LoD` 1, +Y to `LoD` 0
        for &chunk in &[
            Chunk {
                coords: Coords {
                    x: 0,
                    y: 0,
                    face: Face::Px,
                },
                depth: 0,
            },
            Chunk {
                coords: Coords {
                    x: 0,
                    y: 0,
                    face: Face::Py,
                },
                depth: 0,
            },
            Chunk {
                coords: Coords {
                    x: 0,
                    y: 0,
                    face: Face::Px,
                },
                depth: 1,
            },
            Chunk {
                coords: Coords {
                    x: 0,
                    y: 1,
                    face: Face::Px,
                },
                depth: 1,
            },
            Chunk {
                coords: Coords {
                    x: 1,
                    y: 0,
                    face: Face::Px,
                },
                depth: 1,
            },
            Chunk {
                coords: Coords {
                    x: 1,
                    y: 1,
                    face: Face::Px,
                },
                depth: 1,
            },
        ] {
            assert!(transfers.contains(&chunk));
            let slot = mgr.allocate(chunk).unwrap();
            mgr.release(slot).unwrap();
        }

        // Validate output
        let mut transfers = Vec::new();
        mgr.update(&[viewpoint], &mut transfers);
        assert_eq!(mgr.renderable().len(), 5);
        use std::collections::HashMap;
        let neighbors = mgr
            .renderable()
            .iter()
            .map(|&(chunk, neighbors, _slot)| (chunk, neighbors))
            .collect::<HashMap<_, _>>();
        let neighborhood = *neighbors
            .get(&Chunk {
                coords: Coords {
                    x: 0,
                    y: 0,
                    face: Face::Px,
                },
                depth: 1,
            })
            .unwrap();
        assert_ne!(
            neighborhood,
            Neighborhood {
                nx: 0,
                ny: 0,
                px: 0,
                py: 0
            }
        );
    }

    #[test]
    fn incremental_transfer() {
        let mut mgr =
            Manager::new(2048, Config::default()).expect("valid test cache configuration");
        let viewpoint = na::Point3::from(na::Vector3::new(0.0, 1.0, 0.0));
        let mut transfers = Vec::new();
        mgr.update(&[viewpoint], &mut transfers);
        let expected = transfers.len();
        const BATCH_SIZE: usize = 17;
        let mut actual = 0;
        while !transfers.is_empty() {
            for &chunk in transfers.iter().take(BATCH_SIZE) {
                actual += 1;
                let slot = mgr.allocate(chunk).unwrap();
                mgr.release(slot).unwrap();
            }
            transfers.clear();
            mgr.update(&[viewpoint], &mut transfers);
            assert_eq!(transfers.len(), expected - actual);
        }
    }

    #[test]
    fn lod_neighborhood_refers_only_to_actually_rendered_chunks() {
        use std::collections::HashSet;

        let mut manager = Manager::new(
            4096,
            Config {
                max_depth: 4,
                max_neighbor_delta: u8::MAX,
            },
        )
        .expect("valid test cache configuration");
        let viewpoints = [
            na::Point3::new(1.001, 0.5, 0.5),
            na::Point3::new(-0.75, 0.5, -1.0),
        ];
        let mut transfers = Vec::new();
        manager.update(&viewpoints, &mut transfers);
        // Fully resolve coarse levels, with only selected complete groups of
        // deeper children. This models partial splits and asymmetric streaming.
        for chunk in transfers {
            if chunk.depth <= 2
                || (chunk.depth == 3 && (chunk.coords.x / 2 + chunk.coords.y / 2) % 2 == 0)
            {
                let slot = manager.allocate(chunk).unwrap();
                manager.release(slot).unwrap();
            }
        }
        manager.update(&viewpoints, &mut Vec::new());
        let rendered = manager.renderable();
        assert!(!rendered.is_empty());
        let visible: HashSet<_> = rendered.iter().map(|(chunk, _, _)| *chunk).collect();
        for (chunk, neighborhood, _) in rendered {
            for (edge, neighbor) in Edge::iter().zip(chunk.neighbors()) {
                let expected = neighbor
                    .path()
                    .find(|candidate| visible.contains(candidate))
                    .map_or(0, |candidate| chunk.depth - candidate.depth);
                assert_eq!(
                    neighborhood[edge], expected,
                    "chunk {chunk:?} edge {edge:?}"
                );
            }
        }
    }

    #[test]
    fn lod_neighborhood_invariant_across_incremental_teleports() {
        use std::collections::HashSet;
        let mut manager = Manager::new(
            4096,
            Config {
                max_depth: 6,
                max_neighbor_delta: u8::MAX,
            },
        )
        .expect("valid test cache configuration");
        let viewpoints = [
            na::Point3::new(1.001, 0.1, 0.1),
            na::Point3::new(-1.001, 0.4, 0.2),
            na::Point3::new(0.2, 1.001, 0.2),
            na::Point3::new(0.1, -1.001, -0.2),
            na::Point3::new(0.3, 0.1, 1.001),
            na::Point3::new(-0.2, 0.1, -1.001),
        ];
        let mut transfers = Vec::new();
        for frame in 0..120 {
            let viewpoint = viewpoints[(frame / 10) % viewpoints.len()];
            manager.update(&[viewpoint], &mut transfers);
            for (idx, &chunk) in transfers.iter().enumerate() {
                if chunk.depth == 0 || idx % 3 == frame % 3 {
                    if let Some(slot) = manager.allocate(chunk) {
                        manager.release(slot).unwrap();
                    }
                }
            }
            manager.update(&[viewpoint], &mut transfers);
            let visible: HashSet<_> = manager
                .renderable()
                .iter()
                .map(|(chunk, _, _)| *chunk)
                .collect();
            for (chunk, neighborhood, _) in manager.renderable() {
                for (edge, neighbor) in Edge::iter().zip(chunk.neighbors()) {
                    let expected = neighbor
                        .path()
                        .find(|candidate| visible.contains(candidate))
                        .map_or(0, |candidate| chunk.depth - candidate.depth);
                    assert_eq!(
                        neighborhood[edge], expected,
                        "frame {frame} chunk {chunk:?} edge {edge:?}"
                    );
                }
            }
        }
    }

    #[test]
    fn subdivision_sanity() {
        let chunk = Chunk {
            depth: 10,
            coords: crate::cubemap::Coords {
                x: 12,
                y: 34,
                face: Face::Px,
            },
        };
        // Verify that a chunk containing the viewpoint gets subdivided
        assert!(needs_subdivision(
            &chunk,
            &na::Point3::from(chunk.direction(&[0.5; 2].into()).into_inner())
        ));
    }
}
