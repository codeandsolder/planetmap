use criterion::{Criterion, criterion_group, criterion_main};
#[cfg(feature = "parry")]
use parry3d_f64::{
    math::{Pose, Vector},
    query::{PointQuery, QueryDispatcher, Ray, RayCast},
    shape::{Ball, HalfSpace},
};

#[cfg(feature = "parry")]
use planetmap::parry::{FlatTerrain, Planet, PlanetDispatcher};
use planetmap::*;

#[cfg(feature = "parry")]
criterion_group!(benches, cache, face_coords, collision);
#[cfg(not(feature = "parry"))]
criterion_group!(benches, cache, face_coords);
criterion_main!(benches);

fn face_coords(c: &mut Criterion) {
    c.bench_function("get vector coordinates on face", |b| {
        b.iter(|| {
            let v = na::Vector3::new(0.42, 0.17, 0.12);
            std::hint::black_box(cubemap::Face::coords(&v));
        });
    });
}

fn cache(c: &mut Criterion) {
    let mut mgr = CacheManager::new(2048, Default::default()).expect("valid benchmark cache");
    let mut transfers = Vec::with_capacity(2048);
    mgr.update(&[na::Point3::from(na::Vector3::z())], &mut transfers);
    resolve_transfers(&mut mgr, &transfers);
    c.bench_function("best-case update", move |b| {
        let mut transfers = Vec::with_capacity(2048);
        b.iter(|| mgr.update(&[na::Point3::from(na::Vector3::z())], &mut transfers));
    });

    let mut mgr = CacheManager::new(2048, Default::default()).expect("valid benchmark cache");
    mgr.update(&[na::Point3::from(na::Vector3::z())], &mut transfers);
    resolve_transfers(&mut mgr, &transfers);
    let diagonal1 = na::Point3::from(-na::Vector3::new(1.0, 1.0, 1.0).normalize());
    let diagonal2 = na::Point3::from(na::Vector3::new(1.0, 1.0, 1.0).normalize());
    c.bench_function("worst-case update", move |b| {
        let mut which = true;
        b.iter(|| {
            if which {
                mgr.update(&[diagonal1], &mut transfers);
            } else {
                mgr.update(&[diagonal2], &mut transfers);
            }
            resolve_transfers(&mut mgr, &transfers);
            which = !which;
        });
    });
}

fn resolve_transfers(mgr: &mut CacheManager, transfers: &[Chunk]) {
    for &transfer in transfers {
        let slot = mgr.allocate(transfer).unwrap();
        mgr.release(slot).unwrap();
    }
}

#[cfg(feature = "parry")]
fn collision(c: &mut Criterion) {
    use std::sync::Arc;

    const PLANET_RADIUS: f64 = 6371e3;
    let ball = Ball { radius: 1.0 };
    let planet = Planet::new(
        Arc::new(FlatTerrain::new(2u32.pow(12), 17).expect("valid benchmark terrain")),
        32,
        PLANET_RADIUS,
    )
    .expect("valid benchmark planet");

    c.bench_function("intersect", |b| {
        b.iter(|| {
            assert!(
                PlanetDispatcher
                    .intersection_test(&Pose::translation(PLANET_RADIUS, 0.0, 0.0), &planet, &ball,)
                    .unwrap()
                    .intersecting
            );
        });
    });

    c.bench_function("collision broadphase near seam", |b| {
        use parry3d_f64::bounding_volume::Aabb;
        let center = Vector::new(0.0, PLANET_RADIUS + 0.1, 0.0);
        let aabb = Aabb::from_half_extents(center, Vector::splat(1.0));
        let sphere = aabb.bounding_sphere();
        b.iter(|| {
            let mut count = 0;
            planet.map_elements_in_local_sphere(&sphere, &aabb, |_, _, _, _| {
                count += 1;
                true
            });
            std::hint::black_box(count)
        });
    });

    // Alternating opposite faces with a one-slot cache measures the cost
    // of sampling/building new collision chunks, not only warm traversal.
    let churn = Planet::new(
        Arc::new(FlatTerrain::new(2u32.pow(12), 17).expect("valid benchmark terrain")),
        1,
        PLANET_RADIUS,
    )
    .expect("valid benchmark planet");
    let churn_boxes = [
        parry3d_f64::bounding_volume::Aabb::from_half_extents(
            Vector::new(PLANET_RADIUS + 0.1, 1.0, 1.0),
            Vector::splat(1.0),
        ),
        parry3d_f64::bounding_volume::Aabb::from_half_extents(
            Vector::new(-PLANET_RADIUS - 0.1, 1.0, 1.0),
            Vector::splat(1.0),
        ),
    ];
    let churn_spheres = churn_boxes.map(|aabb| aabb.bounding_sphere());
    let mut next_face = 0;
    c.bench_function("collision broadphase cache churn", |b| {
        b.iter(|| {
            let index = next_face;
            next_face ^= 1;
            let mut count = 0;
            churn.map_elements_in_local_sphere(
                &churn_spheres[index],
                &churn_boxes[index],
                |_, _, _, _| {
                    count += 1;
                    true
                },
            );
            std::hint::black_box(count)
        });
    });

    // Shape-to-planet minimum-separation queries cover the full cubemap
    // hierarchy rather than only the collision AABB's current chunk.
    let proximity_pose = Pose::translation(PLANET_RADIUS + 2.5, 0.2, -0.4);
    c.bench_function("planet distance ball", |b| {
        b.iter(|| {
            std::hint::black_box(
                PlanetDispatcher
                    .distance(&proximity_pose, &planet, &ball)
                    .unwrap(),
            )
        });
    });
    c.bench_function("planet closest points ball", |b| {
        b.iter(|| {
            std::hint::black_box(
                PlanetDispatcher
                    .closest_points(&proximity_pose, &planet, &ball, 10.0)
                    .unwrap(),
            )
        });
    });
    c.bench_function("planet contact ball", |b| {
        b.iter(|| {
            std::hint::black_box(
                PlanetDispatcher
                    .contact(&proximity_pose, &planet, &ball, 3.0)
                    .unwrap(),
            )
        });
    });

    // Infinite supporting planes have no finite bounding sphere; the
    // directional region hierarchy must prune globally without scanning
    // every tile on the 4096x4096 cubemap face.
    let halfspace = HalfSpace::new(Vector::X);
    let plane_pose = Pose::translation(-PLANET_RADIUS - 2.5, 0.0, 0.0);
    // Dual cubemap branch-and-bound over both independently streamed
    // planetary triangle meshes, without materializing either full mesh.
    let moon = Planet::new(
        Arc::new(FlatTerrain::new(2u32.pow(12), 17).expect("valid benchmark terrain")),
        32,
        500_000.0,
    )
    .expect("valid benchmark moon");
    let moon_pose = Pose::translation(PLANET_RADIUS + 500_000.0 + 2.5, 0.0, 0.0);
    c.bench_function("planet pair distance near axis", |b| {
        b.iter(|| {
            std::hint::black_box(
                PlanetDispatcher
                    .distance(&moon_pose, &planet, &moon)
                    .unwrap(),
            )
        });
    });
    c.bench_function("planet pair closest points near axis", |b| {
        b.iter(|| {
            std::hint::black_box(
                PlanetDispatcher
                    .closest_points(&moon_pose, &planet, &moon, 10.0)
                    .unwrap(),
            )
        });
    });
    c.bench_function("planet pair contact near axis", |b| {
        b.iter(|| {
            std::hint::black_box(
                PlanetDispatcher
                    .contact(&moon_pose, &planet, &moon, 3.0)
                    .unwrap(),
            )
        });
    });
    let distant_moon_pose = Pose::translation(9_000_000.0, 0.0, 0.0);
    c.bench_function("planet pair distance far separated", |b| {
        b.iter(|| {
            std::hint::black_box(
                PlanetDispatcher
                    .distance(&distant_moon_pose, &planet, &moon)
                    .unwrap(),
            )
        });
    });

    c.bench_function("planet distance halfspace", |b| {
        b.iter(|| {
            std::hint::black_box(
                PlanetDispatcher
                    .distance(&plane_pose, &planet, &halfspace)
                    .unwrap(),
            )
        });
    });

    c.bench_function("short raycast", |b| {
        b.iter(|| {
            planet.cast_local_ray(
                &Ray {
                    origin: Vector::new(PLANET_RADIUS + 1.0, 0.0, 0.0),
                    dir: Vector::Y,
                },
                1e1,
                true,
            );
        });
    });

    c.bench_function("long raycast", |b| {
        b.iter(|| {
            planet.cast_local_ray(
                &Ray {
                    origin: Vector::new(PLANET_RADIUS + 1.0, 0.0, 0.0),
                    dir: Vector::Y,
                },
                1e3,
                true,
            );
        });
    });

    c.bench_function("hit raycast", |b| {
        let ray = Ray {
            origin: Vector::new(PLANET_RADIUS + 10.0, 1.0, 1.0),
            dir: -Vector::X,
        };
        b.iter(|| planet.cast_local_ray(&ray, 100.0, true));
    });

    // A confirmed regression case at the intersection of three cube faces.
    // The original origin-directed quad walk missed this existing triangle.
    c.bench_function("corner seam raycast", |b| {
        let ray = Ray {
            origin: Vector::new(3678337.188936222, 3678238.6096734814, -3678337.1887390064),
            dir: Vector::new(-0.5773554268532486, -0.577339953755108, 0.5773554268222935),
        };
        b.iter(|| planet.cast_local_ray(&ray, 20.0, true));
    });

    // Containment is used by contact and signed-distance queries; measure
    // exact near-surface classification separately from guaranteed-core hits.
    let near_inside = Vector::new(PLANET_RADIUS - 1.0, 0.5, 0.5);
    let near_outside = Vector::new(PLANET_RADIUS + 1.0, 0.5, 0.5);
    let core = Vector::new(PLANET_RADIUS * 0.5, 0.0, 0.0);
    c.bench_function("point contains near inside", |b| {
        b.iter(|| planet.contains_local_point(near_inside));
    });
    c.bench_function("point contains near outside", |b| {
        b.iter(|| planet.contains_local_point(near_outside));
    });
    c.bench_function("point contains core", |b| {
        b.iter(|| planet.contains_local_point(core));
    });
    c.bench_function("point solid near inside", |b| {
        b.iter(|| planet.project_local_point(near_inside, true));
    });
    c.bench_function("point hollow near inside", |b| {
        b.iter(|| planet.project_local_point(near_inside, false));
    });
    c.bench_function("point solid near outside", |b| {
        b.iter(|| planet.project_local_point(near_outside, true));
    });

    c.bench_function("project point", |b| {
        b.iter(|| {
            planet.project_point(
                &Pose::identity(),
                Vector::new(PLANET_RADIUS + 1.0, 0.0, 0.0),
                true,
            )
        });
    });
}
