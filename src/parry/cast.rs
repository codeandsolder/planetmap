use super::Planet;
use crate::cubemap::Coords;
use parry3d_f64::bounding_volume::{Aabb, BoundingVolume};
use parry3d_f64::math::{Pose, Real, Vector};
use parry3d_f64::query::{
    DefaultQueryDispatcher, NonlinearRigidMotion, QueryDispatcher, ShapeCastHit, ShapeCastOptions,
    ShapeCastStatus, Unsupported,
};
use parry3d_f64::shape::{Shape, Triangle};

/// Intersect the translational path with the Minkowski AABB of the shape
/// and the entire planet. This also bounds casts with an unlimited horizon.
fn swept_translation_bounds(
    planet: &Planet,
    start: Aabb,
    velocity: Vector,
    options: ShapeCastOptions,
) -> Option<Aabb> {
    let scale = planet
        .max_radius()
        .max(start.mins.abs().max_element())
        .max(start.maxs.abs().max_element());
    let margin = (128.0 * f64::EPSILON).mul_add(scale, options.target_distance);
    let outer = planet.compute_local_aabb().loosened(margin);
    let mut lo = 0.0_f64;
    let mut hi = options.max_time_of_impact.min(f64::MAX);
    for axis in 0..3 {
        let lower = outer.mins[axis] - start.maxs[axis];
        let upper = outer.maxs[axis] - start.mins[axis];
        let speed = velocity[axis];
        if speed == 0.0 {
            if lower > 0.0 || upper < 0.0 {
                return None;
            }
        } else {
            let a = lower / speed;
            let b = upper / speed;
            lo = lo.max(a.min(b));
            hi = hi.min(a.max(b));
            if lo > hi {
                return None;
            }
        }
    }
    let first = start.transform_by(&Pose::from_translation(velocity * lo));
    // Multiplying an infinite horizon by zero would generate NaNs. A
    // stationary shape needs only its original bounds.
    let last = if velocity == Vector::ZERO {
        first
    } else {
        start.transform_by(&Pose::from_translation(velocity * hi))
    };
    Some(first.merged(&last).loosened(margin))
}

/// Select the earliest narrow-phase result, retaining its geometric address
/// until all visits have finished. Cache slots may be recycled mid-search.
fn earliest_cast(
    planet: &Planet,
    aabb: &Aabb,
    mut cast: impl FnMut(&Triangle) -> Result<Option<ShapeCastHit>, Unsupported>,
) -> Result<Option<ShapeCastHit>, Unsupported> {
    let mut closest: Option<(Coords, u32, ShapeCastHit)> = None;
    let mut unsupported = false;
    planet.visit_bounded_triangles(
        &aabb.bounding_sphere(),
        aabb,
        |&coords, _, index, triangle| match cast(triangle) {
            Ok(Some(hit)) => {
                if closest.is_none_or(|(_, _, old)| hit.time_of_impact < old.time_of_impact) {
                    closest = Some((coords, index, hit));
                }
                true
            }
            Ok(None) => true,
            Err(_) => {
                unsupported = true;
                false
            }
        },
    );
    if unsupported {
        return Err(Unsupported);
    }
    Ok(closest.map(|(coords, index, mut hit)| {
        hit.subshape1 = planet.stable_triangle_id(coords, index);
        hit
    }))
}

pub(super) fn compute_toi(
    pos12: &Pose,
    vel12: Vector,
    planet: &Planet,
    other: &dyn Shape,
    options: ShapeCastOptions,
) -> Result<Option<ShapeCastHit>, Unsupported> {
    if options.max_time_of_impact.is_nan()
        || options.max_time_of_impact < 0.0
        || !options.target_distance.is_finite()
        || options.target_distance < 0.0
        || !vel12.is_finite()
    {
        return Err(Unsupported);
    }
    if vel12 == Vector::ZERO {
        if !options.stop_at_penetration {
            return Ok(None);
        }
        return planet_shape_contact(pos12, planet, other, options.target_distance).map(
            |contact| {
                contact.map(|contact| ShapeCastHit {
                    time_of_impact: 0.0,
                    witness1: contact.point1,
                    witness2: contact.point2,
                    normal1: contact.normal1,
                    normal2: contact.normal2,
                    status: if contact.dist < options.target_distance {
                        ShapeCastStatus::PenetratingOrWithinTargetDist
                    } else {
                        ShapeCastStatus::Converged
                    },
                    subshape1: contact.subshape1,
                    subshape2: contact.subshape2,
                })
            },
        );
    }
    let start = other.compute_aabb(pos12);
    if !start.mins.is_finite() || !start.maxs.is_finite() {
        return Err(Unsupported);
    }
    let Some(aabb) = swept_translation_bounds(planet, start, vel12, options) else {
        return Ok(None);
    };
    if !aabb.mins.is_finite() || !aabb.maxs.is_finite() {
        return Err(Unsupported);
    }
    earliest_cast(planet, &aabb, |triangle| {
        DefaultQueryDispatcher.cast_shapes(pos12, vel12, triangle, other, options)
    })
}

/// Bound the entire relative motion, not just the endpoint poses.
/// World-space rotation centers move linearly. In the planet frame the
/// interpolated endpoint positions can differ from the true path by at
/// most the rotation of their center separation and of the shape's
/// offset from its own rotation center (each chord is bounded by its arc
/// length and by twice the radius).
pub(super) fn nonlinear_swept_bounds(
    motion_planet: &NonlinearRigidMotion,
    motion_other: &NonlinearRigidMotion,
    other: &dyn Shape,
    start_time: Real,
    end_time: Real,
) -> Aabb {
    let start_pos = motion_planet.position_at_time(start_time).inverse()
        * motion_other.position_at_time(start_time);
    let end_pos = motion_planet.position_at_time(end_time).inverse()
        * motion_other.position_at_time(end_time);
    let endpoints = other
        .compute_aabb(&start_pos)
        .merged(&other.compute_aabb(&end_pos));
    let planet_rate = motion_planet.angvel.length();
    let other_rate = motion_other.angvel.length();
    if planet_rate == 0.0 && other_rate == 0.0 {
        return endpoints;
    }
    let duration = end_time - start_time;
    let relative_center = |time| {
        (motion_other.position_at_time(time) * motion_other.local_center)
            - (motion_planet.position_at_time(time) * motion_planet.local_center)
    };
    let max_separation = relative_center(start_time)
        .length()
        .max(relative_center(end_time).length());
    let sphere = other.compute_local_bounding_sphere();
    let radius = (sphere.center() - motion_other.local_center).length() + sphere.radius();
    // Convex interpolation of endpoint bounds: 2*u*(1-u) <= 1/2.
    let pad = radius.mul_add(
        ((planet_rate + other_rate) * duration * 0.5).min(2.0),
        max_separation * (planet_rate * duration * 0.5).min(2.0),
    );
    endpoints.loosened(pad)
}

#[expect(
    clippy::too_many_arguments,
    reason = "Parry dispatch requires separate motion, shape, time-bound and orientation arguments"
)]
pub(super) fn compute_nonlinear_toi(
    motion_planet: &NonlinearRigidMotion,
    planet: &Planet,
    motion_other: &NonlinearRigidMotion,
    other: &dyn Shape,
    start_time: Real,
    end_time: Real,
    stop_at_penetration: bool,
    flipped: bool,
) -> Result<Option<ShapeCastHit>, Unsupported> {
    if !start_time.is_finite() || !end_time.is_finite() || end_time < start_time {
        return Err(Unsupported);
    }
    let aabb = nonlinear_swept_bounds(motion_planet, motion_other, other, start_time, end_time);
    if !aabb.mins.is_finite() || !aabb.maxs.is_finite() {
        return Err(Unsupported);
    }
    let hit = earliest_cast(planet, &aabb, |triangle| {
        if flipped {
            DefaultQueryDispatcher
                .cast_shapes_nonlinear(
                    motion_other,
                    other,
                    motion_planet,
                    triangle,
                    start_time,
                    end_time,
                    stop_at_penetration,
                )
                .map(|hit| hit.map(ShapeCastHit::swapped))
        } else {
            DefaultQueryDispatcher.cast_shapes_nonlinear(
                motion_planet,
                triangle,
                motion_other,
                other,
                start_time,
                end_time,
                stop_at_penetration,
            )
        }
    })?;
    Ok(if flipped {
        hit.map(ShapeCastHit::swapped)
    } else {
        hit
    })
}
