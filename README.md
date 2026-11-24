# Planetmap

Planetmap is a library for processing lazily evaluated radial
functions with dynamic level-of-detail.

## Screenshots

![high orbit](https://user-images.githubusercontent.com/3484507/65808207-c6e2ce80-e18a-11e9-9625-6ad85e5d54d5.png)
![low orbit](https://user-images.githubusercontent.com/3484507/65808209-c77b6500-e18a-11e9-84d3-9efa44f2f902.png)
![close](https://user-images.githubusercontent.com/3484507/65808208-c6e2ce80-e18a-11e9-83e1-100f0863473d.png)

## Terrain Rendering

Planetmap is motivated by the desire to model to-scale planetary
terrain in real time on commodity hardware. To accomplish this, it
addresses multiple challenges:

- Dynamic level of detail
- Streaming larger-than-memory or procedural heightmap data
- Addressing the surface of the sphere without singularities
- Maintaining numerical stability over massive distances

This is accomplished by organizing data into six virtual quadtrees,
each corresponding to one face of a cubemap. Each node in a quadtree,
referred to as a `Chunk`, can be associated with a square grid of
height samples, each of which represents the altitude relative to sea
level along a ray originating at the center of a planet. Streaming and
level of detail can then be implemented with a `CacheManager`
associated with GPU-resident buffers.

The application supplies a deterministic refinement criterion to
`CacheManager::update(refine, &mut transfers, visited)`. For ordinary
view-distance selection, use `cache::viewpoint_refinement(&viewpoints)`.
Compose it with local surface feature or geometric error criteria as needed.
`cache::Config::slots_needed(refine)` counts the same desired chunks,
including ancestors and roots. Both operations obey the configured maximum
depth. `Chunk::angular_radius_bound()` conservatively encloses each
warped chunk in a spherical cap, useful for broad-phase feature queries.

The renderable cut contains named `RenderableChunk` records with the chunk,
edge-stitching neighborhood and ready physical slot.

A transfer allocation returns an opaque generation-qualified token.
Use its `slot()` to locate the GPU buffer region, then pass the token to
`release` on successful completion or `cancel` on abandonment. An abandoned
transfer must stop writing before the slot is recycled. Stale completion
notifications are rejected even if the physical slot has been reused.

When generating geometry to be displaced by the heightmap, care must
be taken to ensure numerical stability and avoid hairline cracks
between neighboring chunks. One effective tactic is to upload the
corners of each chunk from the CPU, then find interior points using
interpolation in a vertex shader.

## Features

- `parry` exposes helpers for rapier-compatible collision
  detection against the surface of a radial heightmap
- `fearless_simd` exposes `fearless_simd`-based iterators for data-parallel
  computation of sample coordinates within a `Chunk` or
  `cubemap::Coords`, useful for fast noise sampling

## Coordinate precision

Cubemap direction addressing, dense-map iteration, and angular neighborhood
queries use f64 throughout so the entire supported u32 grid remains
addressable. Scalar and SIMD sampling share the same exact cell positions;
the SIMD interface narrows positions to f32 only at the final sampling stage.
Use f64 scalar samples when distinct directions at the deepest quadtree
levels are required.

## Coordinate addressing

`Coords::new(resolution, face, x, y)` validates a cell address. Its resolution
is part of its identity, and the indices and face are available through
`x()`, `y()`, and `face()`. `Coords::from_vector` returns `None` for an
invalid direction or empty grid. Face-level lookup (`Face::from_vector` and
`Face::coords`) also rejects zero and nonfinite directions. Direction, center, neighbor, and sampling
methods use the address's own resolution; callers supply only local offsets
or the desired sample count.

Use `Chunk::try_new(depth, face, x, y)` for checked quadtree construction.
`CubeMap::get` and `get_mut` reject addresses from a different grid
resolution, while `get_direction` and `get_direction_mut` provide fallible
direction-based lookup. Indexing with a coordinate of the wrong resolution
panics. An empty cubemap has no valid coordinate addresses.

## Collision query contracts

Terrain dimensions and declared height bounds are fixed per planet. Sampling
must be deterministic, finite, and within those bounds; independent cache
misses may invoke the sampler concurrently. Subshape IDs identify the current
collision-cache slots, not permanent world-space triangles. Broadphase
callbacks receive a `TerrainTriangle` containing its validated
address, local triangle index and geometry rather than a recyclable cache
slot. Queries returning multiple live IDs require enough resident slots to represent those chunks
simultaneously; persistent manifolds report an unsupported query when that is
not possible. Planet-versus-planet continuous casts and persistent contact
manifolds are not implemented.

The minimum supported compiler is Rust 1.100. CI runs the all-feature tests,
strict source policy, duplicate-code regression, and focused cubemap memory
safety tests under a dated Miri nightly without the optional collision and
native SIMD features.
