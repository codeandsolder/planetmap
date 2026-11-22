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

## Collision query contracts

Terrain dimensions and declared height bounds are fixed per planet. Sampling
must be deterministic, finite, and within those bounds; independent cache
misses may invoke the sampler concurrently. Subshape IDs identify the current
collision-cache slots, not permanent world-space triangles. Queries returning
multiple live IDs require enough resident slots to represent those chunks
simultaneously; persistent manifolds report an unsupported query when that is
not possible. Planet-versus-planet continuous casts and persistent contact
manifolds are not implemented.

The minimum supported compiler is Rust 1.100.
