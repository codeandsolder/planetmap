use super::TerrainTriangle;
use super::bounds::{FaceRegion, GridRect, PatchBounds, ProjectionQuery, RankedRegion};
use super::broadphase::SparseQuadSearch;
use super::cast::nonlinear_swept_bounds;
use super::discrete::PlanetPairRegion;
use super::geometry::{Patch, Quad, corner};
use super::manifold::Workspace;
use super::ray::cast_triangle_ray_near_edge;
use super::{FlatTerrain, Planet, PlanetConfigError, PlanetDispatcher, Terrain, cube_vector};
use crate::cubemap::Coords;
use parry3d_f64::bounding_volume::{Aabb, BoundingSphere, BoundingVolume};
use parry3d_f64::math::{Pose, Real, Vec2, Vector};
use parry3d_f64::query::{
    ClosestPoints, Contact, ContactManifold, DefaultQueryDispatcher, NonlinearRigidMotion,
    PersistentQueryDispatcher, PointQuery, QueryDispatcher, Ray, RayCast, RayIntersection,
    ShapeCastHit, ShapeCastOptions, ShapeCastStatus,
};
use parry3d_f64::shape::{HalfSpace, Shape};
use std::sync::Arc;
use std::sync::atomic::Ordering as AtomicOrdering;

use approx::{assert_abs_diff_eq, assert_relative_eq};
use parry3d_f64::{
    math::Rotation,
    shape::{Ball, Cuboid, TriMesh},
};

use crate::cubemap::{Edge, Face};

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
            let hash = (coords.face() as u32)
                .wrapping_mul(7)
                .wrapping_add(coords.x().wrapping_mul(13))
                .wrapping_add(coords.y().wrapping_mul(17));
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
        if self.patterned { -1.0 } else { 0.0 }
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
        for y in 0..planet.face_resolution {
            for x in 0..planet.face_resolution {
                let coords = Coords::new(planet.face_resolution, face, x, y).unwrap();
                let samples = planet.sample(&coords);
                let patch = Patch::new(&coords);
                for (_, tri) in patch.triangles(planet.radius, planet.chunk_resolution, &samples) {
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
                let axis = Vector::new(next() * 2.0 - 1.0, next() * 2.0 - 1.0, next() * 2.0 - 1.0)
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
                assert_eq!(
                    planet.contains_local_point(point),
                    expected_inside,
                    "face_resolution={face_resolution}, patterned={patterned}, index={index}, radius={near_point}, boundary={boundary}, point={point:?}"
                );
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
            let coords = Coords::root(face);
            let corners = Quad::new(&Patch::new(&coords), 1, na::Vector2::zeros())
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
                assert!(
                    planet.contains_local_point(surface),
                    "on-surface cube edge/corner must be contained: face={face:?}, point={surface:?}"
                );
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
                let coords = Coords::new(3, face, x, y).unwrap();
                let patch = Patch::new(&coords);
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
                            assert_eq!(
                                solid.is_inside, inside,
                                "radius={radius}, coords={coords:?}, query={query:?}, point={point:?}"
                            );
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
    let depressed = Coords::new(4, Face::Pz, 2, 1).unwrap();
    let low = Coords::new(4, Face::Pz, 1, 1).unwrap();
    let planet = Planet::new(Arc::new(Depression(depressed)), 1, 10.0).unwrap();
    let direction = low.direction(&na::Point2::new(0.999, 0.5)).into_inner();
    let query = Vector::new(direction.x, direction.y, direction.z) * 3.0;
    assert_eq!(Coords::from_vector(4, &cube_vector(query)), Some(low));
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
fn shared_grid_partition_covers_odd_and_narrow_rectangles_exactly_once() {
    for width in 1..=17 {
        for height in 1..=17 {
            let mut stack = vec![GridRect::new(u32::MAX - width, u32::MAX, 0, height)];
            let mut leaves = std::collections::HashSet::new();
            while let Some(rect) = stack.pop() {
                if rect.is_leaf() {
                    assert!(leaves.insert((rect.x0, rect.y0)));
                    assert_eq!(rect.children().count(), 0);
                } else {
                    stack.extend(rect.children());
                }
            }
            assert_eq!(leaves.len(), (width * height) as usize);
            for x in u32::MAX - width..u32::MAX {
                for y in 0..height {
                    assert!(leaves.contains(&(x, y)));
                }
            }
        }
    }
}

#[test]
fn deep_patch_bounds_remain_tight_and_enclose_real_geometry() {
    const RADIUS: f64 = 6_371_000.0;
    const RESOLUTION: u32 = 1 << 31;
    let planet = Planet::new(
        Arc::new(FlatTerrain::new(RESOLUTION, 3).unwrap()),
        1,
        RADIUS,
    )
    .unwrap();
    for face in Face::iter() {
        for (x, y) in [
            (0, 0),
            (RESOLUTION / 2, RESOLUTION / 2),
            (RESOLUTION - 1, RESOLUTION - 1),
        ] {
            let coords = Coords::new(RESOLUTION, face, x, y).unwrap();
            let bounds =
                PatchBounds::new(&planet, &Pose::IDENTITY, FaceRegion::from_coords(coords));
            assert!(
                (bounds.aabb.maxs - bounds.aabb.mins).length() < 0.03,
                "{bounds:?}"
            );
            assert!(bounds.sphere_radius < 0.02, "{bounds:?}");
            let patch = Patch::new(&coords);
            for (_, triangle) in patch.triangles(RADIUS, 3, &[0.0; 9]) {
                for point in [
                    triangle.a,
                    triangle.b,
                    triangle.c,
                    (triangle.a + triangle.b + triangle.c) / 3.0,
                ] {
                    assert!(
                        bounds.aabb.mins.cmple(point).all() && bounds.aabb.maxs.cmpge(point).all(),
                        "{point:?} outside {bounds:?}"
                    );
                    assert!(bounds.sphere_center.distance(point) <= bounds.sphere_radius);
                    for direction in [
                        Vector::X,
                        Vector::Y,
                        Vector::Z,
                        Vector::new(1.0, 2.0, -3.0).normalize(),
                    ] {
                        let (lo, hi) = bounds.support_interval(direction);
                        let projected = direction.dot(point);
                        assert!(
                            lo <= projected && projected <= hi,
                            "{lo} <= {projected} <= {hi}"
                        );
                    }
                }
            }
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
    let region = RankedRegion::projection(FaceRegion::new(1, Face::Px, 0, 1, 0, 1), &query);
    let actual_distance2 = (20.0_f64 / 3.0_f64.sqrt()).powi(2);
    assert!(
        region.lower_bound <= actual_distance2,
        "pruning bound {} exceeds closest triangle interior {actual_distance2}",
        region.lower_bound
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
                            let coords = Coords::new(resolution, face, x, y).unwrap();
                            let patch = Patch::new(&coords);
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
                            let region = RankedRegion::projection(
                                FaceRegion::new(resolution, face, x0, x1, y0, y1),
                                &query,
                            );
                            let query_point = axis * query_radius;
                            for &point in &mesh_points {
                                let d2 = point.distance_squared(query_point);
                                assert!(
                                    region.lower_bound <= d2 + 1.0e-9,
                                    "region bound {} > point distance {d2}; \
face={face:?}, res={resolution}, rect=({x0},{x1},{y0},{y1}), \
heights=({min_h},{max_h}), q={query_point:?}, p={point:?}",
                                    region.lower_bound
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
    let near = Coords::new(4, Face::Pz, 1, 1).unwrap();
    let across_face = Coords::new(4, Face::Pz, 3, 1).unwrap();
    for (low, high, uv) in [
        (
            near,
            Coords::new(4, near.face(), 2, near.y()).unwrap(),
            [0.98, 0.5],
        ),
        (across_face, across_face.neighbors()[Edge::Px], [0.98, 0.5]),
    ] {
        let source = Arc::new(ProjectionTerrain {
            face_resolution: 4,
            peak: Some(high),
            patterned: false,
            samples: std::sync::atomic::AtomicUsize::new(0),
        });
        let planet = Planet::new(source, 1, 10.0).unwrap();
        let direction = low.direction(&na::Point2::from(uv)).into_inner();
        let query = Vector::new(direction.x, direction.y, direction.z) * 15.0;
        assert_eq!(Coords::from_vector(4, &cube_vector(query)), Some(low));
        let (expected_dist2, expected_pt, winning_patch) = brute_force_projection(&planet, query);
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
            let coords = Coords::new(3, face, 1, 1).unwrap();
            let dir = coords.direction(&na::Point2::from(uv)).into_inner();
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
            out.fill(if coords.face() == Face::Px { -9.0 } else { 0.0 });
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
        Coords::new(3, Face::Pz, 0, 2).unwrap(),
        Coords::new(3, Face::Nx, 2, 0).unwrap(),
        Coords::new(3, Face::Py, 1, 1).unwrap(),
    ] {
        let (_, data) = planet.cached_chunk(&coords);
        let vertices = data.projection_vertices(&planet);
        assert_eq!(
            vertices.as_ptr(),
            data.projection_vertices(&planet).as_ptr()
        );
        assert_eq!(
            vertices.len(),
            (planet.chunk_resolution * planet.chunk_resolution) as usize
        );
        let patch = Patch::new(&coords);
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
        let coords = Coords::root(face);
        let samples = planet.sample(&coords);
        for (idx, triangle) in Patch::new(&coords).triangles(10.0, 3, &samples) {
            for vertex in [triangle.a, triangle.b, triangle.c] {
                assert!(
                    vertex.length() <= envelope,
                    "triangle vertex outside global bounding sphere: {vertex:?}"
                );
                assert!(vertex.cmpge(aabb.mins).all() && vertex.cmple(aabb.maxs).all());
            }
            let point = (triangle.a + triangle.b + triangle.c) / 3.0;
            let outward = coords.center::<f64>().into_inner();
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
    let coords = Coords::root(Face::Px);
    let patch = Patch::new(&coords);
    let samples = planet.sample(&coords);
    let triangle = Quad::new(&patch, 16, na::Vector2::new(8, 8))
        .displace(10.0, 17, &samples)
        .triangles()[0];
    let center = (triangle.a + triangle.b + triangle.c) / 3.0;
    let aabb = Aabb::from_half_extents(center, Vector::splat(0.01));
    let mut matched = false;
    planet.map_elements_in_local_sphere(
        &aabb.bounding_sphere(),
        &aabb,
        |TerrainTriangle { coords: at, .. }| {
            matched |= at == coords;
            true
        },
    );
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
        planet.map_elements_in_local_sphere(&aabb.bounding_sphere(), &aabb, |_| true);
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
                let coords = Coords::root(face);
                let patch = Patch::new(&coords);
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
                    let region = GridRect { x0, x1, y0, y1 };
                    let bound = search.conservative_bounds(region);
                    for row in y0..y1 {
                        for col in x0..x1 {
                            let quad = Quad::new(&patch, side, na::Vector2::new(col, row));
                            for vertex in quad.displace(radius, resolution, &samples).corners {
                                assert!(
                                    vertex.cmpge(bound.mins).all()
                                        && vertex.cmple(bound.maxs).all(),
                                    "radius={radius},res={resolution},face={face:?},case={case},region={region:?},vertex={vertex:?},bound={bound:?}"
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
                    + usize::try_from(coords.x()).unwrap() * 13
                    + usize::try_from(coords.y()).unwrap() * 11
                    + usize::from(coords.face() as u8) * 17)
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
                    let coords = Coords::new(face_resolution, face, x, y).unwrap();
                    let samples = planet.sample(&coords);
                    let patch = Patch::new(&coords);
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
                |TerrainTriangle {
                     coords,
                     triangle_index: index,
                     ..
                 }| {
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
            assert!(
                missing.is_empty(),
                "face_resolution={face_resolution},resolution={resolution},case={case},center={center:?},missing={missing:?}"
            );
        }
    }
}

#[test]
fn cached_displaced_quad_boxes_match_actual_quad_geometry() {
    let planet = Planet::new(Arc::new(FlatTerrain::new(3, 17).unwrap()), 1, 10.0).unwrap();
    for face in Face::iter() {
        let coords = Coords::new(3, face, 1, 1).unwrap();
        let (_, data) = planet.cached_chunk(&coords);
        assert!(data.quad_bounds.get().is_none());
        let boxes = data.quad_bounds(&planet);
        assert_eq!(boxes.quads.len(), 256);
        assert_eq!(boxes.rows.len(), 16);
        assert_eq!(boxes.blocks.len(), 32);
        let patch = Patch::new(&coords);
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
fn broadphase_wide_caps_prune_before_sampling_high_resolution_terrain() {
    const RADIUS: f64 = 6_371_000.0;
    let source = Arc::new(ProjectionTerrain {
        face_resolution: 1 << 24,
        peak: None,
        patterned: false,
        samples: std::sync::atomic::AtomicUsize::new(0),
    });
    let planet = Planet::new(source.clone(), 8, RADIUS).unwrap();

    // Off-center sphere covers the origin; its AABB is still
    // well inside the shell. No chunk should need sampling.
    let inside = Aabb::from_half_extents(Vector::new(100.0, 0.0, 0.0), Vector::splat(1000.0));
    planet.map_elements_in_local_sphere(&inside.bounding_sphere(), &inside, |_| {
        panic!("interior volume cannot intersect flat terrain")
    });
    assert_eq!(source.samples.load(AtomicOrdering::Relaxed), 0);

    // The bounding sphere covers the planet, but the AABB is a
    // millimetre-thin stripe intersecting only the two axis crossings.
    let stripe = Aabb::from_half_extents(
        Vector::new(0.01, 0.0, 0.0),
        Vector::new(RADIUS + 10.0, 0.001, 0.001),
    );
    let mut faces = std::collections::HashSet::new();
    planet.map_elements_in_local_sphere(
        &stripe.bounding_sphere(),
        &stripe,
        |TerrainTriangle { coords, .. }| {
            faces.insert(coords.face());
            true
        },
    );
    assert_eq!(faces, [Face::Px, Face::Nx].into_iter().collect());
    let samples = source.samples.load(AtomicOrdering::Relaxed);
    assert!(samples < 64, "thin stripe sampled {samples} patches");
}

#[test]
fn broadphase_handles_volume_centered_on_planet_origin() {
    let planet = Planet::new(Arc::new(FlatTerrain::new(2, 3).unwrap()), 1, 10.0).unwrap();
    let aabb = Aabb::from_half_extents(Vector::ZERO, Vector::splat(15.0));
    let mut found = std::collections::HashSet::new();
    planet.map_elements_in_local_sphere(
        &aabb.bounding_sphere(),
        &aabb,
        |TerrainTriangle { coords, .. }| {
            found.insert(coords.face());
            true
        },
    );
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
            out.fill(if coords.face() == Face::Px { -9.0 } else { 0.0 });
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
    let target = Coords::root(Face::Px);
    let patch = Patch::new(&target);
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
            |TerrainTriangle {
                 coords,
                 triangle_index: tri_idx,
                 ..
             }| {
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
                let coords = Coords::new(RES, face, x, y).unwrap();
                let patch = Patch::new(&coords);
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
                                |TerrainTriangle {
                                     coords: at,
                                     triangle_index: idx,
                                     ..
                                 }| {
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
                    let coords = Coords::new(resolution, face, x, y).unwrap();
                    let samples = planet.sample(&coords);
                    let patch = Patch::new(&coords);
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
                |TerrainTriangle {
                     coords,
                     triangle_index: idx,
                     ..
                 }| {
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
            assert!(
                missing.is_empty(),
                "resolution={resolution}, case={case}, center={center:?}, half={half:?}, missing={missing:?}"
            );
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
                    let coords = Coords::new(resolution, face, x, y).unwrap();
                    let patch = Patch::new(&coords);
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
                                |TerrainTriangle {
                                     coords: at,
                                     triangle_index: idx,
                                     ..
                                 }| {
                                    actual.insert((at, idx));
                                    true
                                },
                            );
                            let mut missing = Vec::new();
                            for test_face in Face::iter() {
                                for yy in 0..resolution {
                                    for xx in 0..resolution {
                                        let at =
                                            Coords::new(resolution, test_face, xx, yy).unwrap();
                                        let samples = planet.sample(&at);
                                        let patch = Patch::new(&at);
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
                            assert!(
                                missing.is_empty(),
                                "resolution={resolution}, query={query_count}, center={center:?}, missing={missing:?}, emitted={} ",
                                actual.len()
                            );
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
    assert!(
        dispatcher
            .contact(&pose, &first, &second, 20.0)
            .expect("Planet/Planet predicted contact should be supported")
            .is_some()
    );
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
            for y in 0..planet.face_resolution {
                for x in 0..planet.face_resolution {
                    let coords = Coords::new(planet.face_resolution, face, x, y).unwrap();
                    let samples = planet.sample(&coords);
                    let patch = Patch::new(&coords);
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
                _ => {
                    panic!("case={case}, witness mismatch: {witness:?}, expected={expected_pts:?}")
                }
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
            let level = match coords.face() {
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
            assert!(
                planet
                    .cache
                    .lock()
                    .unwrap()
                    .slots
                    .iter()
                    .any(|(i, _)| i == slot)
            );
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
                let hash = (at.face() as u32)
                    .wrapping_mul(17)
                    .wrapping_add(at.x().wrapping_mul(7))
                    .wrapping_add(at.y().wrapping_mul(11))
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
                    let coords = Coords::new(2, face, x, y).unwrap();
                    let patch = Patch::new(&coords);
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
                let region =
                    PatchBounds::new(planet, &pose, FaceRegion::new(2, face, x0, x1, y0, y1));
                for (coords, tris) in tiles {
                    if coords.face() != face
                        || coords.x() < x0
                        || coords.x() >= x1
                        || coords.y() < y0
                        || coords.y() >= y1
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
                                pt.distance(region.sphere_center) <= region.sphere_radius + 1.0e-9,
                                "region {face:?} {x0}:{x1},{y0}:{y1}: sphere misses {pt:?}"
                            );
                            for direction in directions {
                                let (low, high) = region.support_interval(direction);
                                let dot = pt.dot(direction);
                                assert!(
                                    dot >= low - 1.0e-9 && dot <= high + 1.0e-9,
                                    "region {face:?} {x0}:{x1},{y0}:{y1}: dir={direction:?}, dot={dot}, bounds={low}..{high}"
                                );
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
            let a = PatchBounds::new(&first, &Pose::IDENTITY, FaceRegion::from_coords(*at1));
            for (at2, tris2) in &second_tiles {
                let b = PatchBounds::new(&second, &pose, FaceRegion::from_coords(*at2));
                let lower = PlanetPairRegion::new(a, b).lower;
                let mut exact = f64::INFINITY;
                for tri1 in tris1 {
                    for tri2 in tris2 {
                        exact = exact.min(dispatcher.distance(&pose, tri1, tri2).unwrap().distance);
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
    assert!(
        PlanetDispatcher
            .cast_shapes(
                &pose,
                Vector::ZERO,
                &first,
                &second,
                ShapeCastOptions::default()
            )
            .is_err()
    );
    let motion1 =
        NonlinearRigidMotion::new(Pose::IDENTITY, Vector::ZERO, Vector::ZERO, Vector::ZERO);
    let motion2 = NonlinearRigidMotion::new(pose, Vector::ZERO, Vector::ZERO, Vector::ZERO);
    assert!(
        PlanetDispatcher
            .cast_shapes_nonlinear(&motion1, &first, &motion2, &second, 0.0, 1.0, true)
            .is_err()
    );
    let mut manifolds = Vec::<ContactManifold<(), ()>>::new();
    let mut workspace = None;
    assert!(
        PlanetDispatcher
            .contact_manifolds(&pose, &first, &second, 0.0, &mut manifolds, &mut workspace)
            .is_err()
    );
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
        for y in 0..planet.face_resolution {
            for x in 0..planet.face_resolution {
                let coords = Coords::new(planet.face_resolution, face, x, y).unwrap();
                let samples = planet.sample(&coords);
                let patch = Patch::new(&coords);
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
                assert!(
                    (p1.distance(pose * p2) - expected_dist).abs() < 1.0e-8,
                    "patterned={patterned},case={case},closest={points:?},expected={expected_pts:?}"
                );
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
        assert!(
            planet
                .cache
                .lock()
                .unwrap()
                .slots
                .iter()
                .any(|(id, _)| id == slot)
        );
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
                for y in 0..planet.face_resolution {
                    for x in 0..planet.face_resolution {
                        let coords = Coords::new(planet.face_resolution, face, x, y).unwrap();
                        let samples = planet.sample(&coords);
                        let patch = Patch::new(&coords);
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
            let level = if coords.face() == Face::Px {
                -9.0
            } else if coords.face() == Face::Nz {
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
        let pose = Pose::translation(center.x, center.y, center.z) * Pose::from_rotation(rotation);
        let normal = pose.rotation * plane.normal;
        let mut minimum = f64::INFINITY;
        for face in Face::iter() {
            for y in 0..2 {
                for x in 0..2 {
                    let coords = Coords::new(2, face, x, y).unwrap();
                    let samples = planet.sample(&coords);
                    let patch = Patch::new(&coords);
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
    planet.map_elements_in_local_sphere(&bounds, &aabb, |TerrainTriangle { coords, .. }| {
        // Previously the callback was invoked while holding this mutex.
        assert!(planet.cache.try_lock().is_ok());
        // Reenter a different cubemap face with capacity 1 to ensure the
        // outer Arc snapshot remains valid after its LRU slot is evicted.
        let other = Coords::root(Face::Py);
        let _ = planet.cached_chunk(&other);
        assert_eq!(coords.face(), Face::Px);
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
fn planet_rejects_overflowing_subshape_geometry() {
    let chunk = 100_000;
    if let Ok(terrain) = FlatTerrain::new(1, chunk) {
        assert!(matches!(
            Planet::new(Arc::new(terrain), u32::MAX, 10.0),
            Err(PlanetConfigError::SubshapeIdOverflow)
        ));
    }
}

#[test]
fn terrain_sample_panic_does_not_invalidate_existing_cache_entry() {
    struct PanickingTerrain;
    impl Terrain for PanickingTerrain {
        fn face_resolution(&self) -> u32 {
            1
        }
        fn chunk_resolution(&self) -> u32 {
            2
        }
        fn min_height(&self) -> f32 {
            0.0
        }
        fn max_height(&self) -> f32 {
            0.0
        }
        fn sample(&self, coords: &Coords, out: &mut [f32]) {
            assert_ne!(coords.face(), Face::Nx, "intentional sample panic");
            out.fill(0.0);
        }
    }
    let planet = Planet::new(Arc::new(PanickingTerrain), 1, 10.0).unwrap();
    let first = Coords::root(Face::Px);
    let failed = Coords::root(Face::Nx);
    let (slot, data) = planet.cached_chunk(&first);
    let failure = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        planet.cached_chunk(&failed);
    }));
    assert!(failure.is_err());
    assert!(!planet.cache.is_poisoned());
    let (recovered_slot, recovered_data) = planet.cached_chunk(&first);
    assert_eq!(slot, recovered_slot);
    assert!(Arc::ptr_eq(&data, &recovered_data));
    let cache = planet.cache.lock().unwrap();
    assert_eq!(cache.slots.len(), 1);
    assert_eq!(cache.index.len(), 1);
}

#[test]
fn poisoned_collision_cache_discards_partial_mutation_before_reuse() {
    let planet = Planet::new(Arc::new(FlatTerrain::new(1, 2).unwrap()), 1, 10.0).unwrap();
    let first = Coords::root(Face::Px);
    let second = Coords::root(Face::Ny);
    let (_, original) = planet.cached_chunk(&first);
    let failure = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut cache = planet.cache.lock().unwrap();
        // Model an interrupted update of the lookup index, leaving the
        // resident LRU slot without its corresponding index entry.
        cache.index.clear();
        panic!("intentional cache mutation failure");
    }));
    assert!(failure.is_err());
    assert!(planet.cache.is_poisoned());
    let cloned = planet.clone();
    assert!(!planet.cache.is_poisoned());
    assert!(cloned.lock_cache().index.is_empty());
    let (_, recovered) = planet.cached_chunk(&first);
    assert!(!Arc::ptr_eq(&original, &recovered));
    let (slot, _) = planet.cached_chunk(&second);
    let cache = planet.lock_cache();
    assert_eq!(cache.max_size, 1);
    assert_eq!(cache.slots.len(), 1);
    assert_eq!(cache.index.len(), 1);
    assert_eq!(cache.index[&second], slot);
    assert!(!cache.index.contains_key(&first));
}

#[test]
fn planet_subshape_ids_fit_exactly_and_cache_capacity_is_lazy() {
    // Two triangles per chunk; the full 32-bit ID range fits exactly.
    let terrain = Arc::new(FlatTerrain::new(1, 2).unwrap());
    let planet = Planet::new(terrain, 1 << 31, 1.0)
        .expect("maximum slot count is representable without preallocating all slots");
    assert_eq!(planet.lock_cache().slots.len(), 0);
    let (first, _) = planet.cached_chunk(&Coords::root(Face::Px));
    assert_eq!(first, 0);
    assert_eq!(planet.subshape_id(first, 1), 1);
    assert!(planet.lock_cache().slots.capacity() < 32);
    // Ordinary LRU eviction must preserve the configured slot bound.
    let planet = Planet::new(Arc::new(FlatTerrain::new(1, 2).unwrap()), 1, 1.0).unwrap();
    let (a, _) = planet.cached_chunk(&Coords::root(Face::Px));
    let (b, _) = planet.cached_chunk(&Coords::root(Face::Py));
    assert_eq!(a, b);
    let cache = planet.lock_cache();
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
    let coords = Coords::new(planet.face_resolution, Face::Pz, 0, 0).unwrap();
    let samples = planet.sample(&coords);
    let patch = Patch::new(&coords);
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
    assert!(
        ball_contacts(&planet, Vector::new(2.0, PLANET_RADIUS + 0.1, 0.0), 1.0) >= 2,
        "a ball lying on an axis of a planet with an even number of chunks per face overlaps with at least four triangles"
    );
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

    let pos = Vector::new(-5_195_083.148, 3_582_099.812, -877_091.267).normalize() * PLANET_RADIUS;

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
fn linear_cast_target_distance_and_reversed_rotated_frame() {
    let planet = Planet::new(Arc::new(FlatTerrain::new(4, 3).unwrap()), 4, 10.0).unwrap();
    let ball = Ball { radius: 1.0 };
    let pose = Pose::new(
        Vector::new(13.0, 0.0, 0.0),
        Vector::Z * std::f64::consts::FRAC_PI_2,
    );
    let options = ShapeCastOptions {
        max_time_of_impact: 1.5,
        target_distance: 1.0,
        ..Default::default()
    };
    let forward = PlanetDispatcher
        .cast_shapes(&pose, -Vector::X, &planet, &ball, options)
        .unwrap()
        .unwrap();
    let reverse = PlanetDispatcher
        .cast_shapes(
            &pose.inverse(),
            pose.rotation.inverse() * Vector::X,
            &ball,
            &planet,
            options,
        )
        .unwrap()
        .unwrap();
    assert_relative_eq!(forward.time_of_impact, 1.0, epsilon = 1e-6);
    assert_relative_eq!(
        forward.time_of_impact,
        reverse.time_of_impact,
        epsilon = 1e-6
    );
    assert_relative_eq!(forward.witness1, reverse.witness2, epsilon = 1e-6);
    assert_relative_eq!(forward.witness2, reverse.witness1, epsilon = 1e-6);
}

#[test]
fn stationary_cast_can_hit_within_target_distance() {
    let planet = Planet::new(Arc::new(FlatTerrain::new(4, 3).unwrap()), 4, 10.0).unwrap();
    let ball = Ball { radius: 1.0 };
    let hit = PlanetDispatcher
        .cast_shapes(
            &Pose::translation(11.5, 0.0, 0.0),
            Vector::ZERO,
            &planet,
            &ball,
            ShapeCastOptions {
                target_distance: 0.75,
                ..Default::default()
            },
        )
        .unwrap()
        .unwrap();
    assert_eq!(hit.time_of_impact, 0.0);
}

#[test]
fn planetary_unbounded_cast_prunes_long_thin_sweep() {
    const RADIUS: f64 = 6_371_000.0;
    let source = Arc::new(ProjectionTerrain {
        face_resolution: 4096,
        peak: None,
        patterned: false,
        samples: std::sync::atomic::AtomicUsize::new(0),
    });
    let planet = Planet::new(source.clone(), 8, RADIUS).unwrap();
    let hit = PlanetDispatcher
        .cast_shapes(
            &Pose::translation(RADIUS + 10.0, 0.0, 0.0),
            -Vector::X,
            &planet,
            &Ball { radius: 1.0 },
            ShapeCastOptions::default(),
        )
        .unwrap()
        .unwrap();
    assert_relative_eq!(hit.time_of_impact, 9.0, epsilon = 1e-5);
    let samples = source.samples.load(AtomicOrdering::Relaxed);
    assert!(
        samples < 128,
        "narrow planetary sweep sampled {samples} patches"
    );
}

#[test]
fn nonlinear_envelope_contains_rotating_shape_at_intermediate_times() {
    let shape = Cuboid::new(Vector::new(1.0, 2.0, 3.0));
    let planet_motion = NonlinearRigidMotion::new(
        Pose::translation(1.0, -2.0, 0.0),
        Vector::new(2.0, 0.0, -1.0),
        Vector::new(1.0, 0.2, 0.0),
        Vector::new(0.0, 0.0, 0.6),
    );
    let other_motion = NonlinearRigidMotion::new(
        Pose::translation(14.0, 5.0, 3.0),
        Vector::new(0.0, -5.0, 1.0),
        Vector::new(-0.1, 5.0, 0.3),
        Vector::new(0.3, 0.6, 2.0),
    );
    for (p, o) in [(planet_motion, other_motion), (other_motion, planet_motion)] {
        let bound = nonlinear_swept_bounds(&p, &o, &shape, -0.5, 1.2);
        for step in 0..=100 {
            let time = f64::from(step).mul_add(0.017, -0.5);
            let pose = p.position_at_time(time).inverse() * o.position_at_time(time);
            let exact = shape.compute_aabb(&pose);
            assert!(
                bound.mins.cmple(exact.mins).all() && bound.maxs.cmpge(exact.maxs).all(),
                "time={time}: {exact:?} outside {bound:?}"
            );
        }
    }
}

#[test]
fn nonlinear_cast_checks_rotational_arc_not_just_endpoints() {
    let planet = Planet::new(Arc::new(FlatTerrain::new(3, 3).unwrap()), 4, 10.0).unwrap();
    let ball = Ball { radius: 1.0 };
    let stationary = NonlinearRigidMotion::constant_position(Pose::IDENTITY);
    let moving = NonlinearRigidMotion::new(
        Pose::translation(14.0, 5.0, 0.0),
        Vector::new(0.0, -5.0, 0.0),
        Vector::ZERO,
        Vector::Z * std::f64::consts::PI,
    );
    let broad = nonlinear_swept_bounds(&stationary, &moving, &ball, 0.0, 1.0);
    assert!(broad.mins.x <= 8.0 && broad.maxs.x >= 15.0);
    let actual = PlanetDispatcher
        .cast_shapes_nonlinear(&stationary, &planet, &moving, &ball, 0.0, 1.0, true)
        .unwrap();
    let mut reference: Option<ShapeCastHit> = None;
    for face in Face::iter() {
        for y in 0..3 {
            for x in 0..3 {
                let coords = Coords::new(3, face, x, y).unwrap();
                let patch = Patch::new(&coords);
                let samples = planet.sample(&coords);
                for (_, triangle) in patch.triangles(10.0, 3, &samples) {
                    if let Some(hit) = DefaultQueryDispatcher
                        .cast_shapes_nonlinear(
                            &stationary,
                            &triangle,
                            &moving,
                            &ball,
                            0.0,
                            1.0,
                            true,
                        )
                        .unwrap()
                    {
                        if reference.is_none_or(|old| hit.time_of_impact < old.time_of_impact) {
                            reference = Some(hit);
                        }
                    }
                }
            }
        }
    }
    let reference = reference.expect("arc crosses the planet");
    assert_relative_eq!(
        actual.unwrap().time_of_impact,
        reference.time_of_impact,
        epsilon = 1e-8
    );
}

#[test]
fn patch_triangle_indices_are_dense_and_bounded_by_chunk_stride() {
    let coords = Coords::root(Face::Pz);
    let patch = Patch::new(&coords);
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
fn manifold_workspace_survives_eviction_and_external_reset() {
    let planet = Planet::new(Arc::new(FlatTerrain::new(4, 3).unwrap()), 8, 10.0).unwrap();
    let ball = Ball { radius: 1.0 };
    let pose = Pose::translation(10.5, 0.0, 0.0);
    let mut workspace = None;
    let mut manifolds = Vec::<ContactManifold<(), ()>>::new();
    for reset in [false, true] {
        PlanetDispatcher
            .contact_manifolds(&pose, &planet, &ball, 0.0, &mut manifolds, &mut workspace)
            .unwrap();
        let state = workspace
            .as_ref()
            .unwrap()
            .0
            .downcast_ref::<Workspace>()
            .unwrap();
        assert!(state.state.len() > 1);
        let cache = planet.lock_cache();
        let mut ids = std::collections::HashSet::new();
        for (key, &index) in &state.state {
            let id = manifolds[index].subshape1;
            assert!(ids.insert(id));
            assert_eq!(cache.index[&key.coords], id / planet.triangles_per_chunk());
            assert_eq!(key.triangle, id % planet.triangles_per_chunk());
        }
        drop(cache);
        if reset {
            manifolds.clear();
        }
        for x in 0..4 {
            for y in 0..4 {
                planet.cached_chunk(&Coords::new(4, Face::Nx, x, y).unwrap());
            }
        }
    }
    PlanetDispatcher
        .contact_manifolds(&pose, &planet, &ball, 0.0, &mut manifolds, &mut workspace)
        .unwrap();
    assert!(!manifolds.is_empty());
}

#[test]
fn manifold_query_rejects_unrepresentable_live_ids() {
    let planet = Planet::new(Arc::new(FlatTerrain::new(4, 3).unwrap()), 1, 10.0).unwrap();
    let mut workspace = None;
    let mut manifolds = Vec::<ContactManifold<(), ()>>::new();
    let result = PlanetDispatcher.contact_manifolds(
        &Pose::translation(10.5, 0.0, 0.0),
        &planet,
        &Ball { radius: 1.0 },
        0.0,
        &mut manifolds,
        &mut workspace,
    );
    assert!(result.is_err());
    assert!(manifolds.is_empty());
    assert!(workspace.is_none());
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
    assert!(
        manifolds
            .iter()
            .any(|manifold| !manifold.contacts().is_empty())
    );
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
        for y in 0..planet.face_resolution {
            for x in 0..planet.face_resolution {
                let coords = Coords::new(planet.face_resolution, face, x, y).unwrap();
                let samples = planet.sample(&coords);
                let patch = Patch::new(&coords);
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
                let coords = Coords::new(3, face, x, y).unwrap();
                let samples = planet.sample(&coords);
                let patch = Patch::new(&coords);
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
                                let actual = planet.cast_local_ray_and_get_normal(&ray, 50.0, true);
                                assert_eq!(
                                    actual.is_some(),
                                    expected.is_some(),
                                    "target ray {count}: {ray:?}"
                                );
                                if let (Some(a), Some(b)) = (actual, expected) {
                                    assert!(
                                        (a.time_of_impact - b.time_of_impact).abs() < 1.0e-7,
                                        "target ray {count}: {ray:?}, actual={a:?}, expected={b:?}"
                                    );
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
            out.fill(if coords.face() == Face::Px { -9.0 } else { 0.0 });
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
    assert!(
        planet
            .cast_ray_inside_one_quad(&crossing, 2.0 * RADIUS + 40.0, true)
            .is_none()
    );
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
                let coords = Coords::new(RES, face, x, y).unwrap();
                let samples = planet.sample(&coords);
                let patch = Patch::new(&coords);
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
                                let actual = planet.cast_local_ray_and_get_normal(&ray, 20.0, true);
                                assert!(
                                    actual
                                        .is_some_and(|hit| hit.time_of_impact
                                            <= direct.time_of_impact + 0.0001),
                                    "missed target {count}: face={face:?}, coords={coords:?}, ray={ray:?}, direct={direct:?}, planet={actual:?}"
                                );
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
    assert!(
        planet
            .cast_local_ray_and_get_normal(&ray, hit.time_of_impact * 0.5, true)
            .is_none()
    );
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
    let chunk = Coords::new(res, Face::Px, 12, 47).unwrap();
    let patch = Patch::new(&chunk);

    // Verify that the corners are consistent. All other points won't be, since
    // `Coords::direction` interpolates on the sphere.
    for coords in [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]] {
        // Exact equality is intended here, as a prerequisite for neighboring patches to be
        // seamless.
        assert_eq!(patch.get(&Vec2::from(coords)), corner(&chunk, coords));
    }
}

#[test]
fn patch_projection() {
    let res = 2u32.pow(12);
    let chunk = Coords::new(res, Face::Px, 12, 47).unwrap();
    let patch = Patch::new(&chunk);

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
