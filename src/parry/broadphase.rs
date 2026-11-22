use super::bounds::{DirectionCone, FaceRegion, GridRect, PatchBounds};
use super::geometry::{Patch, Quad};
use super::tile_cache::{
    ChunkData, MAX_CACHED_PROJECTION_VERTICES, QUAD_BLOCK_WIDTH, QUAD_INDEX_BUILD_AFTER_QUERIES,
};
use super::{Planet, cube_vector, outer_displaced_radius};
use crate::cubemap::Coords;
use parry3d_f64::bounding_volume::{Aabb, BoundingSphere, BoundingVolume};
use parry3d_f64::math::{Pose, Vec2, Vector};
use parry3d_f64::shape::Triangle;
use std::sync::atomic::Ordering as AtomicOrdering;

impl Planet {
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
            GridRect {
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
    pub(super) fn map_elements_in_chunk(
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
        let (slot, data) = self.cached_chunk(&chunk_coords);
        // Both triangle interiors and displaced vertices lie in this
        // patch's angular cone; unlike the sample vertex radii, its
        // projected radial minimum is conservative for triangle chords.
        let cone = DirectionCone::from_region(FaceRegion::from_coords(chunk_coords));
        let min_height = f64::from(data.min);
        let minimum_radius = cone.minimum_radius(self.radius, min_height);
        if outer_displaced_radius(self.radius, f64::from(data.min), f64::from(data.max))
            + bounds.radius()
            < distance
            || minimum_radius - bounds.radius() > distance
        {
            return true;
        }
        let patch = Patch::new(&chunk_coords);
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
    /// discovery for small caps. Wide caps and inward-displaced triangles
    /// instead use conservative hierarchical region bounds.
    pub fn map_elements_in_local_sphere(
        &self,
        bounds: &BoundingSphere,
        aabb: &Aabb,
        mut f: impl FnMut(&Coords, u32, u32, &Triangle) -> bool,
    ) {
        let dir = bounds.center();
        let distance = dir.length();
        let min_height = f64::from(self.min_height);
        // A full cube face has minimum axis projection 1/sqrt(3).
        // This bound covers triangle *interiors*, not only vertices.
        let inscribed = self.radius / 3.0_f64.sqrt()
            + if min_height >= 0.0 {
                min_height / 3.0_f64.sqrt()
            } else {
                min_height
            };
        let direction = cube_vector(dir);
        if inscribed > 256.0 * f64::EPSILON * self.radius
            && distance.is_finite()
            && distance > 0.0
            && direction.iter().all(|v| v.is_finite())
            && direction.norm_squared() > 0.0
        {
            // A query bounding sphere at distance d subtends asin(r/d);
            // Coords::neighborhood includes numerical boundary slack.
            let theta = if distance > bounds.radius() {
                (bounds.radius() / distance).asin()
            } else {
                std::f64::consts::PI
            };
            if theta * f64::from(self.face_resolution) <= MAX_DIRECT_CAP_CELLS {
                for coords in Coords::neighborhood(self.face_resolution, direction, theta) {
                    if !self.map_elements_in_chunk(coords, bounds, aabb, &mut f) {
                        return;
                    }
                }
                return;
            }
        }

        // A broad angular cap may cover billions of tiles, even if the
        // query AABB touches only a narrow stripe or none of the surface.
        // Inward terrain also invalidates angular address clipping. Both
        // cases use the same conservative hierarchy as swept queries.
        self.visit_bounded_triangles(bounds, aabb, f);
    }
}

/// Use direct angular enumeration only for caps comparable to a single tile.
pub(super) const MAX_DIRECT_CAP_CELLS: f64 = 2.0;

/// Sparse, proof-backed search for quads in a cold terrain chunk. It uses
/// only four to six evaluations of the *undisplaced* piecewise-affine patch
/// for each visited region, and displaces geometry only at candidate leaves.
pub(super) struct SparseQuadSearch<'a> {
    pub(super) patch: &'a Patch,
    pub(super) samples: &'a [f32],
    pub(super) query: &'a Aabb,
    pub(super) radius: f64,
    pub(super) height_abs: f64,
    pub(super) resolution: u32,
}

impl SparseQuadSearch<'_> {
    /// `Patch::get(u,v)` is affine on each side of `u=v`, and continuous
    /// along that diagonal. The extrema of its coordinates over a rectangle
    /// occur at the four corners or the diagonal's boundary intersections.
    /// A displaced vertex is R*v + h*normalize(v), whose height contribution
    /// lies component-wise within `max(abs(min_height), abs(max_height))`.
    /// Convex combinations preserve this AABB for complete triangle interiors.
    pub(super) fn conservative_bounds(&self, region: GridRect) -> Aabb {
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
    fn visit(&self, region: GridRect, emit: &mut impl FnMut(Quad) -> bool) -> bool {
        if !self.conservative_bounds(region).intersects(self.query) {
            return true;
        }
        if region.is_leaf() {
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
        region.children().all(|child| self.visit(child, emit))
    }
}

impl Planet {
    /// Traverse the same geometrically certified cubemap region bounds for
    /// broad static queries and swept shapes. Both the AABB and sphere
    /// must be disjoint-safe; their intersection encloses the other shape.
    pub(super) fn visit_bounded_triangles(
        &self,
        bounds: &BoundingSphere,
        aabb: &Aabb,
        mut visit: impl FnMut(&Coords, u32, u32, &Triangle) -> bool,
    ) {
        let mut stack: Vec<_> = FaceRegion::roots(self.face_resolution)
            .map(|region| PatchBounds::new(self, &Pose::IDENTITY, region))
            .collect();
        while let Some(patch) = stack.pop() {
            if !patch.aabb.intersects(aabb) {
                continue;
            }
            let distance = patch.sphere_center.distance(bounds.center());
            let allowance = patch.sphere_radius + bounds.radius();
            let rounding = 128.0
                * f64::EPSILON
                * (patch.sphere_center.abs().max_element()
                    + bounds.center().abs().max_element()
                    + allowance);
            if distance.is_finite() && distance > allowance + rounding {
                continue;
            }
            if let Some(coords) = patch.region.leaf() {
                if !self.map_elements_in_chunk(coords, bounds, aabb, &mut visit) {
                    return;
                }
            } else {
                stack.extend(patch.children(self, &Pose::IDENTITY));
            }
        }
    }
}
