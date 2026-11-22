use super::TerrainTriangle;
use super::{Planet, PlanetDispatcher};
use crate::cubemap::Coords;
use hashbrown::HashMap;
use parry3d_f64::bounding_volume::{Aabb, BoundingSphere, BoundingVolume};
use parry3d_f64::math::{Pose, Real};
use parry3d_f64::query::details::NormalConstraints;
use parry3d_f64::query::{
    ContactManifold, ContactManifoldsWorkspace, DefaultQueryDispatcher, PersistentQueryDispatcher,
    TypedWorkspaceData, Unsupported, WorkspaceData,
};
use parry3d_f64::shape::{Shape, Triangle};
use parry3d_f64::utils::PoseOpt;

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
        let (planet, other, pose, flipped) =
            match (g1.as_shape::<Planet>(), g2.as_shape::<Planet>()) {
                (Some(planet), None) => (planet, g2, *pos12, false),
                (None, Some(planet)) => (planet, g1, pos12.inverse(), true),
                _ => return Err(Unsupported),
            };
        compute_manifolds(
            &pose, planet, other, prediction, manifolds, workspace, flipped,
        )
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

/// The persistent key identifies geometry rather than its temporary LRU slot.
#[derive(Eq, PartialEq, Hash, Copy, Clone)]
pub(super) struct ManifoldKey {
    pub(super) coords: Coords,
    pub(super) triangle: u32,
    pub(super) other_part: Option<u32>,
}

/// Narrow-phase state associated with one shape pair.
#[derive(Default, Clone)]
pub struct Workspace {
    pub(super) state: HashMap<ManifoldKey, usize>,
    pub(super) flipped: bool,
}

impl WorkspaceData for Workspace {
    fn as_typed_workspace_data(&self) -> TypedWorkspaceData<'_> {
        TypedWorkspaceData::Custom
    }

    fn clone_dyn(&self) -> Box<dyn WorkspaceData> {
        Box::new(self.clone())
    }
}

fn assign_manifold_subshapes<ManifoldData, ContactData>(
    planet: &Planet,
    keys: &[ManifoldKey],
    manifolds: &mut [ContactManifold<ManifoldData, ContactData>],
    flipped: bool,
) -> Result<(), Unsupported> {
    // Resolve every active chunk together, *after* traversal; otherwise
    // a later chunk can recycle a slot used by an earlier manifold.
    let mut chunks = HashMap::new();
    let capacity = planet.lock_cache().max_size as usize;
    for key in keys {
        let full = chunks.len() == capacity;
        if let hashbrown::hash_map::Entry::Vacant(entry) = chunks.entry(key.coords) {
            if full {
                // All live subshape IDs must fit simultaneously.
                return Err(Unsupported);
            }
            entry.insert(planet.cached_chunk(&key.coords).1);
        }
    }
    let mut slots = HashMap::with_capacity(chunks.len());
    let mut cache = planet.lock_cache();
    for (coords, data) in chunks {
        let (slot, _) = cache.insert(data);
        slots.insert(coords, slot);
    }
    // Keep every newly assigned slot resident until publication is complete.
    for (key, manifold) in keys.iter().zip(manifolds) {
        let slot = *slots.get(&key.coords).ok_or(Unsupported)?;
        let subshape = planet.subshape_id(slot, key.triangle);
        if flipped {
            manifold.subshape2 = subshape;
        } else {
            manifold.subshape1 = subshape;
        }
    }
    drop(cache);
    Ok(())
}

fn manifold_bounds(
    other: &dyn Shape,
    pose: &Pose,
    prediction: Real,
) -> Result<(Aabb, BoundingSphere), Unsupported> {
    if !prediction.is_finite() || prediction < 0.0 {
        return Err(Unsupported);
    }
    let aabb = other.compute_aabb(pose).loosened(prediction);
    if !aabb.mins.is_finite() || !aabb.maxs.is_finite() {
        return Err(Unsupported);
    }
    let bounds = aabb.bounding_sphere();
    if !bounds.radius().is_finite() {
        return Err(Unsupported);
    }
    Ok((aabb, bounds))
}

/// Descend into the other shape's BVH only for parts that can meet the
/// current terrain triangle. The visitor is shared with the simple-shape path.
fn visit_manifold_parts(
    other: &dyn Shape,
    pos21: &Pose,
    prediction: Real,
    mut key: ManifoldKey,
    triangle: &Triangle,
    visit: &mut impl FnMut(
        ManifoldKey,
        &Triangle,
        &dyn Shape,
        Option<&Pose>,
        Option<&dyn NormalConstraints>,
    ) -> bool,
) -> bool {
    let Some(composite) = other.as_composite_shape() else {
        return visit(key, triangle, other, None, None);
    };
    let tri_aabb = triangle.compute_aabb(pos21).loosened(prediction);
    for part in composite.bvh().intersect_aabb(&tri_aabb) {
        key.other_part = Some(part);
        let mut keep_going = true;
        composite.map_part_at(part, &mut |pose, shape, constraints| {
            keep_going = keep_going && visit(key, triangle, shape, pose, constraints);
        });
        if !keep_going {
            return false;
        }
    }
    true
}

fn compute_manifolds<ManifoldData, ContactData>(
    pos12: &Pose,
    planet: &Planet,
    other: &dyn Shape,
    prediction: Real,
    manifolds: &mut Vec<ContactManifold<ManifoldData, ContactData>>,
    workspace: &mut Option<ContactManifoldsWorkspace>,
    flipped: bool,
) -> Result<(), Unsupported>
where
    ManifoldData: Default + Clone + Send + Sync,
    ContactData: Default + Copy + Send + Sync,
{
    let old = workspace
        .as_ref()
        .and_then(|workspace| workspace.0.downcast_ref::<Workspace>())
        .filter(|workspace| workspace.flipped == flipped);
    let (aabb, bounds) = manifold_bounds(other, pos12, prediction)?;
    let pos21 = pos12.inverse();
    let mut next = Vec::new();
    let mut keys = Vec::new();
    let mut next_state = Workspace {
        state: HashMap::new(),
        flipped,
    };
    let mut status = Ok(());
    let mut visit = |key: ManifoldKey,
                     triangle: &Triangle,
                     part: &dyn Shape,
                     part_pos: Option<&Pose>,
                     constraints: Option<&dyn NormalConstraints>| {
        if next_state.state.contains_key(&key) {
            return true;
        }
        // An external caller can reset the vector without resetting the
        // workspace. Missing historical entries simply start cold.
        let mut manifold = old
            .and_then(|old| old.state.get(&key))
            .and_then(|&index| manifolds.get(index))
            .cloned()
            .unwrap_or_else(ContactManifold::new);
        let other_id = key.other_part.unwrap_or(0);
        let result = if flipped {
            manifold.subshape1 = other_id;
            manifold.set_subshape_pos1(part_pos.copied());
            manifold.set_subshape_pos2(None);
            DefaultQueryDispatcher.contact_manifold_convex_convex(
                &part_pos.inv_mul(&pos21),
                part,
                triangle,
                constraints,
                None,
                prediction,
                &mut manifold,
            )
        } else {
            manifold.subshape2 = other_id;
            manifold.set_subshape_pos1(None);
            manifold.set_subshape_pos2(part_pos.copied());
            DefaultQueryDispatcher.contact_manifold_convex_convex(
                &part_pos.prepend_to(pos12),
                triangle,
                part,
                None,
                constraints,
                prediction,
                &mut manifold,
            )
        };
        if let Err(error) = result {
            status = Err(error);
            return false;
        }
        if !manifold.contacts().is_empty() {
            next_state.state.insert(key, next.len());
            keys.push(key);
            next.push(manifold);
        }
        true
    };
    planet.map_elements_in_local_sphere(
        &bounds,
        &aabb,
        |TerrainTriangle {
             coords,
             triangle_index,
             triangle,
         }| {
            visit_manifold_parts(
                other,
                &pos21,
                prediction,
                ManifoldKey {
                    coords,
                    triangle: triangle_index,
                    other_part: None,
                },
                triangle,
                &mut visit,
            )
        },
    );
    status?;

    assign_manifold_subshapes(planet, &keys, &mut next, flipped)?;
    *manifolds = next;
    *workspace = Some(ContactManifoldsWorkspace(Box::new(next_state)));
    Ok(())
}
