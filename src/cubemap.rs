use std::cmp::Ordering;
use std::marker::PhantomData;
use std::ops::{Index, IndexMut, Neg};
use std::{alloc, fmt, mem, ptr};

#[cfg(feature = "fearless_simd")]
use fearless_simd::{Simd, SimdBase, SimdElement, SimdFloat};
use na::{RealField, SimdRealField};

/// A dense, fixed-resolution, warped cube map
///
/// Useful for storing and manipulating moderate-resolution samplings of radial functions such as
/// spherical heightmaps. Warped to improve spacing of slots compared to a naive cube map, based on
/// the 5th order polynomial method described in "Cube-to-sphere Projections for Procedural
/// Texturing and Beyond" (Zucker, Higashi).
///
/// For addressing purposes, slots along the edges and at the corners of a face do *not* overlap
/// with their neighbors. Note that `Coords::samples` nonetheless *does* produce samples that will
/// overlap along the edges of neighboring `Coords`.
#[derive(Debug)]
#[repr(C)]
pub struct CubeMap<T> {
    resolution: u32,
    data: [T],
}

/// Owns the partly initialized DST allocation until it can safely become a Box.
struct InitializationGuard<T> {
    base: *mut u8,
    data: *mut T,
    layout: alloc::Layout,
    initialized: usize,
}

impl<T> Drop for InitializationGuard<T> {
    fn drop(&mut self) {
        // SAFETY: the prefix of initialized T values is fully valid, and the
        // header/payload allocation was made with exactly self.layout.
        unsafe {
            ptr::drop_in_place(ptr::slice_from_raw_parts_mut(self.data, self.initialized));
            alloc::dealloc(self.base, self.layout);
        }
    }
}

impl<T> CubeMap<T> {
    /// Access a cubemap in existing memory
    ///
    /// # Safety
    ///
    /// `ptr` must point to the aligned start of a fully initialized `#[repr(C)]`
    /// `CubeMap<T>` allocation, with a valid `u32` resolution and exactly
    /// `6 * resolution * resolution` initialized `T` values. The payload follows
    /// the header at the alignment required by `T` (and may have padding).
    /// The entire DST must be valid for shared reads for `'a`, with no mutable
    /// borrows during that lifetime. The metadata length must fit in `usize`.
    #[must_use]
    pub const unsafe fn from_raw<'a>(ptr: *const u32) -> &'a Self {
        // SAFETY: caller guarantees a valid, aligned u32 header.
        let dim = unsafe { ptr.read() } as usize;
        let len = dim * dim * 6;
        // SAFETY: the caller guarantees a valid, aligned allocation for the entire DST.
        unsafe { &*(ptr::slice_from_raw_parts(ptr, len) as *const Self) }
    }

    /// Uniquely access a cubemap in existing memory
    ///
    /// # Safety
    ///
    /// `ptr` must point to the aligned start of a fully initialized `#[repr(C)]`
    /// `CubeMap<T>` allocation, with a valid `u32` resolution and exactly
    /// `6 * resolution * resolution` initialized `T` values. The payload follows
    /// the header at the alignment required by `T` (and may have padding).
    /// The entire DST must be exclusively accessible for `'a`. The metadata
    /// length must fit in `usize`, and no other references may alias the data.
    #[must_use]
    pub const unsafe fn from_raw_mut<'a>(ptr: *mut u32) -> &'a mut Self {
        // SAFETY: caller guarantees a valid, aligned u32 header.
        let dim = unsafe { ptr.read() } as usize;
        let len = dim * dim * 6;
        // SAFETY: the caller guarantees an exclusive, aligned allocation for the entire DST.
        unsafe { &mut *(ptr::slice_from_raw_parts_mut(ptr, len) as *mut Self) }
    }

    fn payload_len(resolution: u32) -> Option<usize> {
        let dimension = usize::try_from(resolution).ok()?;
        dimension.checked_mul(dimension)?.checked_mul(6)
    }

    // The backing allocation contains a u32 header followed by len elements of T.
    // Layout::extend determines the correct padding and alignment for any T.
    #[expect(
        clippy::cast_ptr_alignment,
        reason = "Layout::extend guarantees the header, payload and DST are properly aligned"
    )]
    fn allocate_with(
        resolution: u32,
        mut initialize: impl FnMut(usize) -> Option<T>,
    ) -> Option<Box<Self>> {
        let len = Self::payload_len(resolution)?;
        let payload_layout = alloc::Layout::array::<T>(len).ok()?;
        let (layout, offset) = alloc::Layout::new::<u32>().extend(payload_layout).ok()?;
        let layout = layout.pad_to_align();

        // SAFETY: Layout::extend ensures both the u32 header and the T payload are
        // aligned and fit in the allocated region. InitializationGuard owns the
        // allocation until all T elements are initialized, even if initialize
        // returns None or panics. Afterwards Box owns the exact DST layout.
        unsafe {
            let base = alloc::alloc(layout);
            if base.is_null() {
                return None;
            }
            let mut guard = InitializationGuard {
                base,
                data: base.add(offset).cast::<T>(),
                layout,
                initialized: 0,
            };
            base.cast::<u32>().write(resolution);
            for i in 0..len {
                let value = initialize(i)?;
                guard.data.add(i).write(value);
                guard.initialized += 1;
            }

            let raw = ptr::slice_from_raw_parts_mut(base, len) as *mut Self;
            mem::forget(guard);
            Some(Box::from_raw(raw))
        }
    }

    /// Construct a cube map with faces containing `resolution * resolution` slots, each
    /// initialized to `value`. Returns `None` if dimensions or allocation cannot be represented.
    #[must_use]
    pub fn new(resolution: u32, value: T) -> Option<Box<Self>>
    where
        T: Clone,
    {
        Self::allocate_with(resolution, |_| Some(value.clone()))
    }

    /// Copy a cube map from a contiguous slice of `resolution` by `resolution` data, in +X,
    /// -X, +Y, -Y, +Z, -Z order.
    ///
    /// Returns `None` for mismatched data length, unrepresentable dimensions or allocation failure.
    #[must_use]
    pub fn from_slice(resolution: u32, data: &[T]) -> Option<Box<Self>>
    where
        T: Copy,
    {
        if data.len() != Self::payload_len(resolution)? {
            return None;
        }
        Self::allocate_with(resolution, |i| data.get(i).copied())
    }

    /// Compute a cube map based on the direction of each slot. Returns `None`
    /// if the dimensions cannot be represented or allocation fails.
    #[must_use]
    pub fn from_fn(
        resolution: u32,
        mut f: impl FnMut(na::Unit<na::Vector3<f64>>) -> T,
    ) -> Option<Box<Self>> {
        Self::allocate_with(resolution, |i| get_dir(resolution, i).map(&mut f))
    }

    #[must_use]
    pub const fn resolution(&self) -> u32 {
        self.resolution
    }

    #[must_use]
    pub fn iter(&self) -> Iter<'_, T> {
        self.into_iter()
    }

    #[must_use]
    pub fn iter_mut(&mut self) -> IterMut<'_, T> {
        self.into_iter()
    }
}

impl<T> AsRef<[T]> for CubeMap<T> {
    fn as_ref(&self) -> &[T] {
        &self.data
    }
}

impl<T> AsMut<[T]> for CubeMap<T> {
    fn as_mut(&mut self) -> &mut [T] {
        &mut self.data
    }
}

impl<T> Index<Face> for CubeMap<T> {
    type Output = [T];
    fn index(&self, face: Face) -> &[T] {
        let face_size = self.resolution as usize * self.resolution as usize;
        let offset = face_size * face as usize;
        &self.data[offset..offset + face_size]
    }
}

impl<T> IndexMut<Face> for CubeMap<T> {
    fn index_mut(&mut self, face: Face) -> &mut [T] {
        let face_size = self.resolution as usize * self.resolution as usize;
        let offset = face_size * face as usize;
        &mut self.data[offset..offset + face_size]
    }
}

impl<T> Index<Coords> for CubeMap<T> {
    type Output = T;
    fn index(&self, coord: Coords) -> &T {
        assert!(
            coord.x < self.resolution && coord.y < self.resolution,
            "cube map cell coordinates out of bounds"
        );
        let face_size = self.resolution as usize * self.resolution as usize;
        let offset = face_size * coord.face as usize;
        &self.data[offset + self.resolution as usize * coord.y as usize + coord.x as usize]
    }
}

impl<T> IndexMut<Coords> for CubeMap<T> {
    fn index_mut(&mut self, coord: Coords) -> &mut T {
        assert!(
            coord.x < self.resolution && coord.y < self.resolution,
            "cube map cell coordinates out of bounds"
        );
        let face_size = self.resolution as usize * self.resolution as usize;
        let offset = face_size * coord.face as usize;
        &mut self.data[offset + self.resolution as usize * coord.y as usize + coord.x as usize]
    }
}

impl<'a, T> Index<&'a na::Vector3<f64>> for CubeMap<T> {
    type Output = T;
    fn index(&self, x: &'a na::Vector3<f64>) -> &T {
        &self.data[index(self.resolution, x)]
    }
}

impl<'a, T> IndexMut<&'a na::Vector3<f64>> for CubeMap<T> {
    fn index_mut(&mut self, x: &'a na::Vector3<f64>) -> &mut T {
        &mut self.data[index(self.resolution, x)]
    }
}

fn index(resolution: u32, direction: &na::Vector3<f64>) -> usize {
    let Coords { x, y, face } = Coords::from_vector(resolution, direction);
    let dimension = resolution as usize;
    face as usize * dimension * dimension + y as usize * dimension + x as usize
}

impl<'a, T> IntoIterator for &'a CubeMap<T> {
    type Item = (na::Unit<na::Vector3<f64>>, &'a T);
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        Iter {
            resolution: self.resolution,
            data: &self.data,
            index: 0,
        }
    }
}

pub struct Iter<'a, T> {
    resolution: u32,
    data: &'a [T],
    index: usize,
}

impl<'a, T> Iterator for Iter<'a, T> {
    type Item = (na::Unit<na::Vector3<f64>>, &'a T);
    fn next(&mut self) -> Option<Self::Item> {
        let dir = get_dir(self.resolution, self.index)?;
        let value = &self.data[self.index];
        self.index += 1;
        Some((dir, value))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.data.len().saturating_sub(self.index);
        (remaining, Some(remaining))
    }
}

impl<T> ExactSizeIterator for Iter<'_, T> {
    fn len(&self) -> usize {
        self.data.len().saturating_sub(self.index)
    }
}

impl<'a, T> IntoIterator for &'a mut CubeMap<T> {
    type Item = (na::Unit<na::Vector3<f64>>, &'a mut T);
    type IntoIter = IterMut<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        IterMut {
            resolution: self.resolution,
            data: self.data.iter_mut().enumerate(),
        }
    }
}

pub struct IterMut<'a, T> {
    resolution: u32,
    data: std::iter::Enumerate<std::slice::IterMut<'a, T>>,
}

impl<'a, T> Iterator for IterMut<'a, T> {
    type Item = (na::Unit<na::Vector3<f64>>, &'a mut T);
    fn next(&mut self) -> Option<Self::Item> {
        let (index, value) = self.data.next()?;
        Some((get_dir(self.resolution, index)?, value))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.data.size_hint()
    }
}

impl<T> ExactSizeIterator for IterMut<'_, T> {
    fn len(&self) -> usize {
        self.data.len()
    }
}

fn get_dir(resolution: u32, index: usize) -> Option<na::Unit<na::Vector3<f64>>> {
    let dimension = usize::try_from(resolution).ok()?;
    let face_size = dimension.checked_mul(dimension)?;
    let total = face_size.checked_mul(6)?;
    if face_size == 0 || index >= total {
        return None;
    }
    let face = [Face::Px, Face::Nx, Face::Py, Face::Ny, Face::Pz, Face::Nz][index / face_size];
    let rem = index % face_size;
    let y = u32::try_from(rem / dimension).ok()?;
    let x = u32::try_from(rem % dimension).ok()?;
    Some(Coords { x, y, face }.center(resolution))
}

/// Face of a cube map, identified by direction
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
#[repr(u8)]
pub enum Face {
    /// The face in the +X direction
    Px,
    /// The face in the -X direction
    Nx,
    /// The face in the +Y direction
    Py,
    /// The face in the -Y direction
    Ny,
    /// The face in the +Z direction
    Pz,
    /// The face in the -Z direction
    Nz,
}

impl fmt::Display for Face {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use self::Face::{Nx, Ny, Nz, Px, Py, Pz};
        let s = match *self {
            Px => "+X",
            Nx => "-X",
            Py => "+Y",
            Ny => "-Y",
            Pz => "+Z",
            Nz => "-Z",
        };
        f.write_str(s)
    }
}

impl Neg for Face {
    type Output = Self;
    fn neg(self) -> Self {
        use self::Face::{Nx, Ny, Nz, Px, Py, Pz};
        match self {
            Px => Nx,
            Py => Ny,
            Pz => Nz,
            Nx => Px,
            Ny => Py,
            Nz => Pz,
        }
    }
}

impl Face {
    /// Find the face that intersects a vector originating at the center of a cube
    #[must_use]
    pub fn from_vector<N: RealField + PartialOrd + Copy>(x: &na::Vector3<N>) -> Self {
        // Seed from X: the three-component vector cannot be empty. On ties
        // and incomparable values, prefer the later axis, as max_by does.
        let (mut value, mut axis) = (x.x, Self::Px);
        for (candidate, candidate_axis) in [(x.y, Self::Py), (x.z, Self::Pz)] {
            let order = value
                .abs()
                .partial_cmp(&candidate.abs())
                .unwrap_or(Ordering::Less);
            if order != Ordering::Greater {
                value = candidate;
                axis = candidate_axis;
            }
        }
        if value.is_sign_negative() {
            -axis
        } else {
            axis
        }
    }

    /// Compute which `Face` a vector intersects, and where the intersection lies in [-1..1]^2
    #[must_use]
    pub fn coords<N: RealField + Copy>(x: &na::Vector3<N>) -> (Self, na::Point2<N>) {
        let face = Self::from_vector(x);
        let wrt_face = face.basis().inverse_transform_vector(x);
        (
            face,
            na::Point2::from(wrt_face.xy() / wrt_face.z).map(unwarp),
        )
    }

    /// Transform from face space (facing +Z) to sphere space (facing the named axis).
    #[must_use]
    pub fn basis<N: RealField>(self) -> na::Rotation3<N> {
        use self::Face::{Nx, Ny, Nz, Px, Py, Pz};
        let (x, y, z) = match self {
            Px => (na::Vector3::z(), -na::Vector3::y(), na::Vector3::x()),
            Nx => (-na::Vector3::z(), -na::Vector3::y(), -na::Vector3::x()),
            Py => (na::Vector3::x(), -na::Vector3::z(), na::Vector3::y()),
            Ny => (na::Vector3::x(), na::Vector3::z(), -na::Vector3::y()),
            Pz => (na::Vector3::x(), na::Vector3::y(), na::Vector3::z()),
            Nz => (-na::Vector3::x(), na::Vector3::y(), -na::Vector3::z()),
        };
        na::Rotation3::from_basis_unchecked(&(x, y, z).into())
    }

    /// Iterator over all `Face`s
    pub fn iter() -> impl Iterator<Item = Self> {
        [Self::Px, Self::Nx, Self::Py, Self::Ny, Self::Pz, Self::Nz].into_iter()
    }

    /// Neighboring faces wrt. local axes [-x, -y, +x, +y].
    ///
    /// Returns the neighboring face, the edge of that face, and whether the axis shared with that face is parallel or antiparallel.
    ///
    /// Index by `sign << 1 | axis`.
    #[must_use]
    pub(crate) const fn neighbors(self) -> &'static [(Self, Edge, bool); 4] {
        use self::Face::{Nx, Ny, Nz, Px, Py, Pz};
        match self {
            Px => &[
                (Nz, Edge::Nx, false),
                (Py, Edge::Px, false),
                (Pz, Edge::Px, false),
                (Ny, Edge::Px, true),
            ],
            Nx => &[
                (Pz, Edge::Nx, false),
                (Py, Edge::Nx, true),
                (Nz, Edge::Px, false),
                (Ny, Edge::Nx, false),
            ],
            Py => &[
                (Nx, Edge::Ny, true),
                (Pz, Edge::Py, true),
                (Px, Edge::Ny, false),
                (Nz, Edge::Py, false),
            ],
            Ny => &[
                (Nx, Edge::Py, false),
                (Nz, Edge::Ny, false),
                (Px, Edge::Py, true),
                (Pz, Edge::Ny, true),
            ],
            Pz => &[
                (Nx, Edge::Nx, false),
                (Ny, Edge::Py, true),
                (Px, Edge::Px, false),
                (Py, Edge::Ny, true),
            ],
            Nz => &[
                (Px, Edge::Nx, false),
                (Ny, Edge::Ny, false),
                (Nx, Edge::Px, false),
                (Py, Edge::Py, false),
            ],
        }
    }

    /// Compute the direction identified by a [-1..1]^2 vector on this face
    #[must_use]
    pub fn direction<N: RealField + Copy>(
        self,
        coords: &na::Point2<N>,
    ) -> na::Unit<na::Vector3<N>> {
        let coords = coords.map(warp);
        let dir_z = na::Unit::new_normalize(na::Vector3::new(coords.x, coords.y, N::one()));
        self.basis() * dir_z
    }
}

// Conservative rounding allowance in unit-direction and unit-grid space.
const ANGULAR_ROUNDOFF: f64 = 64.0 * f64::EPSILON;

fn face_to_grid(x: f64) -> f64 {
    unwarp(x.clamp(-1.0, 1.0)).midpoint(1.0).clamp(0.0, 1.0)
}

// Range of one component of a unit vector inside an angular cap.
// The extrema may be at the +/- cardinal directions, not the rim.
fn cap_range(axis: f64, transverse: f64, cos: f64, sin: f64) -> (f64, f64) {
    let perpendicular = transverse * sin;
    let min = if -axis >= cos {
        -1.0
    } else {
        axis.mul_add(cos, -perpendicular)
    };
    let max = if axis >= cos {
        1.0
    } else {
        axis.mul_add(cos, perpendicular)
    };
    (min, max)
}

// Exact maximum angular dot product for points in the face's
// projective square. The maximum is attained at an interior point,
// a stationary point on one of its four edges, or a corner.
fn face_max_dot(direction: na::Vector3<f64>) -> f64 {
    if direction.z >= direction.x.abs() && direction.z >= direction.y.abs() {
        return 1.0;
    }
    let mut best = f64::NEG_INFINITY;
    let mut consider = |x: f64, y: f64| {
        let dot = direction.x.mul_add(x, direction.y.mul_add(y, direction.z));
        let length = x.mul_add(x, y.mul_add(y, 1.0)).sqrt();
        best = best.max(dot / length);
    };
    for edge in [-1.0, 1.0] {
        for corner in [-1.0, 1.0] {
            consider(edge, corner);
        }
        let x_edge = direction.x.mul_add(edge, direction.z);
        if x_edge > 0.0 {
            consider(edge, (2.0 * direction.y / x_edge).clamp(-1.0, 1.0));
        }
        let y_edge = direction.y.mul_add(edge, direction.z);
        if y_edge > 0.0 {
            consider((2.0 * direction.x / y_edge).clamp(-1.0, 1.0), edge);
        }
    }
    best
}

/// Coordinates in a discretized cubemap
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct Coords {
    pub x: u32,
    pub y: u32,
    pub face: Face,
}

impl Coords {
    #[must_use]
    pub fn from_vector(resolution: u32, vector: &na::Vector3<f64>) -> Self {
        Self::from_vector_with_coords(resolution, vector).0
    }

    /// Compute the coordinates of the cell containing a finite nonzero vector,
    /// and its [0..1]^2 coordinates within that cell.
    #[must_use]
    pub fn from_vector_with_coords(
        resolution: u32,
        vector: &na::Vector3<f64>,
    ) -> (Self, na::Point2<f64>) {
        let (face, coords) = Face::coords(vector);
        let coords = coords * 0.5 + na::Vector2::new(0.5, 0.5);
        let (x, y) = discretize(resolution, coords);
        let scale = f64::from(resolution);
        let local = na::Point2::new(
            coords.x.mul_add(scale, -f64::from(x)).clamp(0.0, 1.0),
            coords.y.mul_add(scale, -f64::from(y)).clamp(0.0, 1.0),
        );
        (Self { x, y, face }, local)
    }

    #[must_use]
    pub fn neighbors(&self, resolution: u32) -> [Self; 4] {
        let Self { x, y, face } = *self;
        let max = resolution - 1;
        let neighbor_chunk = |face: Face, edge: Edge| {
            let (neighboring_face, neighbor_edge, parallel_axis) = face.neighbors()[edge];
            let other = match edge {
                Edge::Nx | Edge::Px => y,
                Edge::Ny | Edge::Py => x,
            };
            let other = if parallel_axis { other } else { max - other };
            let (x, y) = match neighbor_edge {
                Edge::Nx => (0, other),
                Edge::Ny => (other, 0),
                Edge::Px => (max, other),
                Edge::Py => (other, max),
            };
            Self {
                x,
                y,
                face: neighboring_face,
            }
        };
        [
            if x == 0 {
                neighbor_chunk(face, Edge::Nx)
            } else {
                Self { x: x - 1, y, face }
            },
            if y == 0 {
                neighbor_chunk(face, Edge::Ny)
            } else {
                Self { x, y: y - 1, face }
            },
            if x == max {
                neighbor_chunk(face, Edge::Px)
            } else {
                Self { x: x + 1, y, face }
            },
            if y == max {
                neighbor_chunk(face, Edge::Py)
            } else {
                Self { x, y: y + 1, face }
            },
        ]
    }

    /// Conservatively select cells intersecting a spherical cone around `direction`.
    ///
    /// The bounds are computed in 3D before projecting to each cube face, so
    /// wide cones and cones that cross the horizon of a face remain covered.
    /// Invalid directions, negative/non-finite angles, and zero resolution
    /// produce an empty iterator.
    pub fn neighborhood(
        resolution: u32,
        direction: na::Vector3<f64>,
        theta: f64,
    ) -> impl Iterator<Item = Self> {
        let valid = resolution != 0 && theta.is_finite() && theta >= 0.0;
        // Scale first: normalization must also work for finite vectors whose
        // squared norm would overflow or underflow.
        let scale = direction.amax();
        let axis = if direction.iter().all(|x| x.is_finite()) && scale > 0.0 {
            (direction / scale).try_normalize(0.0)
        } else {
            None
        };
        // Roundoff at the cap/face or cap/cell boundary must not discard
        // geometrically touching cells.
        let angle = (theta + ANGULAR_ROUNDOFF).min(std::f64::consts::PI);
        let (sin, cos) = angle.sin_cos();
        Face::iter()
            .filter_map(move |face| {
                if !valid {
                    return None;
                }
                let axis = axis?;
                let local = face.basis::<f64>().inverse_transform_vector(&axis);
                // Reject faces that the cone cannot physically reach. Without
                // this check, a tiny cone near a face's horizon could scan
                // every texel of that face due to the 1/z projection.
                if face_max_dot(local) < cos - ANGULAR_ROUNDOFF {
                    return None;
                }
                let (lo_x, hi_x) = cap_range(local.x, local.y.hypot(local.z), cos, sin);
                let (lo_y, hi_y) = cap_range(local.y, local.x.hypot(local.z), cos, sin);
                let (lo_z, hi_z) = cap_range(local.z, local.x.hypot(local.y), cos, sin);
                if hi_z < -ANGULAR_ROUNDOFF {
                    return None;
                }
                let ((x_min, y_min), (x_max, y_max)) = if lo_z <= ANGULAR_ROUNDOFF {
                    // The cone touches the projection horizon; even finite
                    // component bounds can correspond to unbounded x/z,y/z.
                    ((-1.0, -1.0), (1.0, 1.0))
                } else {
                    let ratio_min = |lo: f64| if lo < 0.0 { lo / lo_z } else { lo / hi_z };
                    let ratio_max = |hi: f64| if hi > 0.0 { hi / lo_z } else { hi / hi_z };
                    (
                        (ratio_min(lo_x), ratio_min(lo_y)),
                        (ratio_max(hi_x), ratio_max(hi_y)),
                    )
                };
                if x_min > 1.0 + ANGULAR_ROUNDOFF
                    || y_min > 1.0 + ANGULAR_ROUNDOFF
                    || x_max < -1.0 - ANGULAR_ROUNDOFF
                    || y_max < -1.0 - ANGULAR_ROUNDOFF
                {
                    return None;
                }
                let (x0, y0) = discretize(
                    resolution,
                    na::Point2::new(
                        face_to_grid(x_min) - ANGULAR_ROUNDOFF,
                        face_to_grid(y_min) - ANGULAR_ROUNDOFF,
                    ),
                );
                let (x1, y1) = discretize(
                    resolution,
                    na::Point2::new(
                        face_to_grid(x_max) + ANGULAR_ROUNDOFF,
                        face_to_grid(y_max) + ANGULAR_ROUNDOFF,
                    ),
                );
                Some((face, x0, x1, y0, y1))
            })
            .flat_map(move |(face, x0, x1, y0, y1)| {
                (y0..=y1).flat_map(move |y| (x0..=x1).map(move |x| Self { x, y, face }))
            })
    }

    /// The direction through the center of this cell.
    #[must_use]
    pub fn center<N: RealField + Copy>(&self, resolution: u32) -> na::Unit<na::Vector3<N>> {
        self.face.direction(
            &self
                .face_position(resolution, na::Point2::new(0.5, 0.5))
                .cast(),
        )
    }

    fn face_position(&self, resolution: u32, local: na::Point2<f64>) -> na::Point2<f64> {
        let scale = f64::from(resolution);
        let centered =
            |index, uv: f64| uv.mul_add(2.0, f64::from(index).mul_add(2.0, -scale)) / scale;
        na::Point2::new(centered(self.x, local.x), centered(self.y, local.y))
    }

    /// Compute the direction identified by a point in the [0..1]^2 area covered by these
    /// coordinates. Address arithmetic is performed in f64 before conversion to the output type.
    #[must_use]
    pub fn direction<N: RealField + Copy + Into<f64>>(
        &self,
        resolution: u32,
        coords: &na::Point2<N>,
    ) -> na::Unit<na::Vector3<N>> {
        self.face.direction(
            &self
                .face_position(resolution, coords.map(Into::into))
                .cast(),
        )
    }

    /// Length of an edge of the bounding square of the cubemap-space area covered by a coordinate
    /// in a cubemap with a particular `resolution`
    #[must_use]
    pub fn edge_length<N: RealField>(resolution: u32) -> N {
        na::convert::<_, N>(2.0) / na::convert::<_, N>(f64::from(resolution))
    }

    /// Returns a grid of resolution^2 directions contained by these coords, in scan-line order
    ///
    /// - `face_resolution` represents the number of coordinates along an edge of the cubemap
    /// - `chunk_resolution` represents the number of samples along an edge of this specific coordinate
    ///
    /// Edge/corner samples lie *directly* on the edge/corner, and hence are *not* the centers of
    /// traditionally addressed texels. In other words, the first sample has position (0, 0), not
    /// (0.5/w, 0.5/h), and the last has position (1, 1), not (1 - 0.5/w, 1 - 0.5/h). This allows
    /// for seamless interpolation in the neighborhood of chunk edges/corners without needing access
    /// to data for neighboring chunks.
    #[must_use]
    pub fn samples<N: RealField + Copy>(
        &self,
        face_resolution: u32,
        chunk_resolution: u32,
    ) -> SampleIter<N> {
        SampleIter {
            coords: *self,
            face_resolution,
            chunk_resolution,
            index: 0,
            _marker: PhantomData,
        }
    }

    /// SIMD variant of `samples`.
    ///
    /// Because this returns data in batches of `S::VF32_WIDTH`, a few excess values will be
    /// computed at the end for any `resolution` whose square is not a multiple of the batch size.
    #[cfg(feature = "fearless_simd")]
    #[must_use]
    pub const fn samples_ps<S: Simd>(
        &self,
        simd: S,
        face_resolution: u32,
        chunk_resolution: u32,
    ) -> SampleIterSimd<S> {
        SampleIterSimd {
            coords: *self,
            face_resolution,
            chunk_resolution,
            index: 0,
            simd,
        }
    }
}

/// Distort a [-1..1] cube face coordinate such that, when projected onto a sphere, area is close
/// to equal
pub(crate) fn warp<N: SimdRealField + Copy>(x: N) -> N {
    // 5th-order polynomial from "Cube-to-sphere Projections for Procedural Texturing and Beyond"
    // (Zucker, Higashi).
    let x2 = x * x;
    (na::convert::<_, N>(0.745_558_715_593)
        + (na::convert::<_, N>(0.130_546_850_193) + na::convert::<_, N>(0.123_894_434_214) * x2)
            * x2)
        * x
}

/// See `warp`
#[cfg(feature = "fearless_simd")]
fn warp_ps<S: Simd, N: RealField + SimdElement, F: SimdFloat<S, Element = N>>(x: F) -> F {
    let x2 = x * x;
    x * ((x2 * na::convert::<_, N>(0.123_894_434_214) + na::convert::<_, N>(0.130_546_850_193))
        * x2
        + na::convert::<_, N>(0.745_558_715_593))
}

/// Derivative of `warp`
fn dwarp<N: RealField + Copy>(x: N) -> N {
    let x2 = x * x;
    (na::convert::<_, N>(0.619_472_171_07) * x2 + na::convert::<_, N>(0.391_640_550_579)) * x2
        + na::convert::<_, N>(0.745_558_715_593)
}

/// Inverse of `warp`
pub(crate) fn unwarp<N: RealField + Copy>(x: N) -> N {
    let x2 = x * x;

    // Approximate inverse using a numerically fit curve
    let mut estimate = (na::convert::<_, N>(1.343_182_295_52)
        + (na::convert::<_, N>(-0.486_514_066_449) + na::convert::<_, N>(0.143_331_770_927) * x2)
            * x2)
        * x;

    // Refine estimate with newton's method
    for _ in 0..4 {
        estimate -= (warp(estimate) - x) / dwarp(estimate);
    }

    estimate
}

/// Edge of a quad
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
#[repr(u8)]
pub enum Edge {
    Nx = 0,
    Ny = 1,
    Px = 2,
    Py = 3,
}

impl Edge {
    /// Iterator over all `Edge`s
    pub fn iter() -> impl Iterator<Item = Self> {
        [Self::Nx, Self::Ny, Self::Px, Self::Py].into_iter()
    }

    /// Unit vector across the edge
    #[must_use]
    pub fn direction<N: RealField>(self) -> na::UnitVector2<N> {
        let x = match self {
            Self::Nx => [-1.0, 0.0],
            Self::Ny => [0.0, -1.0],
            Self::Px => [1.0, 0.0],
            Self::Py => [0.0, 1.0],
        };
        na::Unit::new_unchecked(na::Vector2::from(x).cast())
    }
}

impl Neg for Edge {
    type Output = Self;
    fn neg(self) -> Self {
        use self::Edge::{Nx, Ny, Px, Py};
        match self {
            Px => Nx,
            Py => Ny,
            Nx => Px,
            Ny => Py,
        }
    }
}

impl<T> Index<Edge> for [T] {
    type Output = T;
    fn index(&self, edge: Edge) -> &T {
        &self[edge as usize]
    }
}

impl<T> IndexMut<Edge> for [T] {
    fn index_mut(&mut self, edge: Edge) -> &mut T {
        &mut self[edge as usize]
    }
}

fn sample_position(index: u64, resolution: u32) -> Option<na::Point2<f64>> {
    if resolution == 1 {
        return Some(na::Point2::new(0.5, 0.5));
    }
    let edge = u64::from(resolution);
    let x = u32::try_from(index % edge).ok()?;
    let y = u32::try_from(index / edge).ok()?;
    let max = f64::from(resolution - 1);
    Some(na::Point2::new(f64::from(x) / max, f64::from(y) / max))
}

/// Iterator over sample points distributed in a regular grid across a chunk, including its edges
#[derive(Debug)]
pub struct SampleIter<N> {
    coords: Coords,
    face_resolution: u32,
    chunk_resolution: u32,
    // A full u32-by-u32 grid needs u64 indexing.
    index: u64,
    _marker: PhantomData<fn() -> N>,
}

impl<N: RealField + Copy> Iterator for SampleIter<N> {
    type Item = na::Unit<na::Vector3<N>>;
    fn next(&mut self) -> Option<Self::Item> {
        let edge = u64::from(self.chunk_resolution);
        if self.index >= edge * edge {
            return None;
        }
        let local = sample_position(self.index, self.chunk_resolution)?;
        let pos = self.coords.face_position(self.face_resolution, local);
        let dir = self.coords.face.direction(&pos.cast());
        self.index += 1;
        Some(dir)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let edge = u64::from(self.chunk_resolution);
        let remaining = (edge * edge).saturating_sub(self.index);
        usize::try_from(remaining).map_or((usize::MAX, None), |count| (count, Some(count)))
    }
}

// Only 64-bit targets can represent the full count of an unrestricted u32 x u32
// grid in usize. On narrower targets the iterator remains correct but not ExactSize.
#[cfg(target_pointer_width = "64")]
impl<N: RealField + Copy> ExactSizeIterator for SampleIter<N> {
    fn len(&self) -> usize {
        self.size_hint().0
    }
}

/// Iterator over sample points distributed in a regular grid across a chunk, including its edges
///
/// Hand-vectorized, returning batches of each dimension in a separate register.
#[derive(Debug)]
#[cfg(feature = "fearless_simd")]
pub struct SampleIterSimd<S> {
    coords: Coords,
    face_resolution: u32,
    chunk_resolution: u32,
    index: u64,
    simd: S,
}

#[cfg(feature = "fearless_simd")]
impl<S: Simd> Iterator for SampleIterSimd<S> {
    type Item = [S::f32s; 3];
    #[expect(
        clippy::cast_possible_truncation,
        reason = "The SIMD interface deliberately returns f32 samples; normalized face positions are narrowed only after f64 addressing"
    )]
    fn next(&mut self) -> Option<Self::Item> {
        let edge = u64::from(self.chunk_resolution);
        let total = edge * edge;
        if self.index >= total {
            return None;
        }
        let mut pos_on_face_x = S::f32s::splat(self.simd, 0.0);
        let mut pos_on_face_y = S::f32s::splat(self.simd, 0.0);
        for (lane, (x, y)) in pos_on_face_x
            .as_mut_slice()
            .iter_mut()
            .zip(pos_on_face_y.as_mut_slice())
            .enumerate()
        {
            let lane = u64::try_from(lane).ok()?;
            let sample = self.index.saturating_add(lane).min(total - 1);
            let position = self.coords.face_position(
                self.face_resolution,
                sample_position(sample, self.chunk_resolution)?,
            );
            *x = position.x as f32;
            *y = position.y as f32;
        }

        let warped_x = warp_ps(pos_on_face_x);
        let warped_y = warp_ps(pos_on_face_y);

        let len = warped_y
            .mul_add(warped_y, warped_x.mul_add(warped_x, 1.0))
            .sqrt();
        let dir_x = warped_x / len;
        let dir_y = warped_y / len;
        let dir_z = S::f32s::splat(self.simd, 1.0) / len;

        let basis = self.coords.face.basis::<f32>();
        let basis = basis.matrix();
        let x = S::f32s::splat(self.simd, basis.m11).mul_add(
            dir_x,
            S::f32s::splat(self.simd, basis.m12).mul_add(dir_y, dir_z * basis.m13),
        );
        let y = S::f32s::splat(self.simd, basis.m21).mul_add(
            dir_x,
            S::f32s::splat(self.simd, basis.m22).mul_add(dir_y, dir_z * basis.m23),
        );
        let z = S::f32s::splat(self.simd, basis.m31).mul_add(
            dir_x,
            S::f32s::splat(self.simd, basis.m32).mul_add(dir_y, dir_z * basis.m33),
        );

        let lanes = u64::try_from(S::f32s::LEN).ok()?;
        self.index = self.index.saturating_add(lanes);
        Some([x, y, z])
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let edge = u64::from(self.chunk_resolution);
        let remaining = (edge * edge).saturating_sub(self.index);
        let lanes = u64::try_from(S::f32s::LEN).unwrap_or(u64::MAX).max(1);
        let batches = remaining.div_ceil(lanes);
        usize::try_from(batches).map_or((usize::MAX, None), |count| (count, Some(count)))
    }
}

#[cfg(all(feature = "fearless_simd", target_pointer_width = "64"))]
impl<S: Simd> ExactSizeIterator for SampleIterSimd<S> {
    fn len(&self) -> usize {
        self.size_hint().0
    }
}

/// Map real coordinates in [0, 1)^2 to integer coordinates in [0, n)^2 such that each integer
/// covers exactly the same distance
#[expect(
    clippy::cast_possible_truncation,
    reason = "The normalized f64 grid position is bounded before converting to a u32 cell index"
)]
#[expect(
    clippy::cast_sign_loss,
    reason = "Negative and non-finite coordinates are normalized before converting to a u32 cell index"
)]
pub(crate) fn discretize(resolution: u32, texcoords: na::Point2<f64>) -> (u32, u32) {
    if resolution == 0 {
        return (0, 0);
    }
    let last = resolution - 1;
    let grid = f64::from(resolution);
    let to_grid = |coordinate: f64| {
        let normalized = if coordinate.is_finite() {
            coordinate.clamp(0.0, 1.0)
        } else {
            0.0
        };
        (normalized * grid).floor() as u32
    };
    (
        to_grid(texcoords.x).min(last),
        to_grid(texcoords.y).min(last),
    )
}

#[cfg(test)]
mod test {
    use super::*;
    use approx::*;

    #[test]
    fn index_sanity() {
        const RES: u32 = 2048;
        assert_eq!(
            index(RES, &na::Vector3::x_axis()) as u32,
            (RES / 2) * (RES + 1)
        );
    }

    #[test]
    fn iter() {
        for res in 0..8 {
            let map = CubeMap::new(res, 0).expect("small map dimensions must fit");
            assert_eq!(map.iter().count() as u32, res * res * 6);
        }
    }

    #[test]
    fn cubemap_iter_size_hints_cover_all_faces() {
        for resolution in 0..8 {
            let mut map = CubeMap::new(resolution, 0u8).unwrap();
            let count = usize::try_from(resolution * resolution * 6).unwrap();

            let mut iter = map.iter();
            assert_eq!(iter.size_hint(), (count, Some(count)));
            assert_eq!(iter.len(), count);
            if resolution > 0 {
                iter.next();
                assert_eq!(iter.len(), count - 1);
            }
            assert_eq!(
                iter.count(),
                count.saturating_sub(usize::from(resolution > 0))
            );

            let mut iter_mut = map.iter_mut();
            assert_eq!(iter_mut.size_hint(), (count, Some(count)));
            assert_eq!(iter_mut.len(), count);
            if resolution > 0 {
                iter_mut.next();
                assert_eq!(iter_mut.len(), count - 1);
            }
            assert_eq!(
                iter_mut.count(),
                count.saturating_sub(usize::from(resolution > 0))
            );
        }
    }

    #[test]
    fn mutable_iterator_retains_disjoint_references() {
        let mut map = CubeMap::new(2, 0_u32).unwrap();
        let mut refs: Vec<_> = map.iter_mut().map(|(_, value)| value).collect();
        for (index, value) in refs.iter_mut().enumerate() {
            **value = u32::try_from(index).unwrap();
        }
        assert!(
            map.iter()
                .enumerate()
                .all(|(index, (_, value))| { usize::try_from(*value) == Ok(index) })
        );
    }

    #[test]
    fn invalid_coordinate_never_aliases_adjacent_cells_or_faces() {
        let mut map = CubeMap::new(2, 0_u32).unwrap();
        let valid = Coords {
            x: 1,
            y: 0,
            face: Face::Px,
        };
        map[valid] = 42;
        assert_eq!(map[valid], 42);
        for (x, y) in [(2, 0), (0, 2), (u32::MAX, 0)] {
            let invalid = Coords {
                x,
                y,
                face: Face::Px,
            };
            assert!(
                std::panic::catch_unwind(|| {
                    let _ = map[invalid];
                })
                .is_err()
            );
            assert!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    map[invalid] = 99;
                }))
                .is_err()
            );
        }
        assert_eq!(map[valid], 42);
        assert_eq!(
            map[Coords {
                x: 0,
                y: 0,
                face: Face::Nx
            }],
            0
        );
    }

    #[test]
    fn cubemap_aligned_elements() {
        #[repr(align(64))]
        #[derive(Clone)]
        struct Aligned(u64);

        let map = CubeMap::new(2, Aligned(7)).unwrap();
        assert_eq!(map.as_ref().as_ref().len(), 24);
        for element in map.as_ref().as_ref() {
            assert_eq!(element.0, 7);
            assert_eq!((std::ptr::from_ref(element) as usize) % 64, 0);
        }
    }

    #[test]
    fn allocation_guard_drops_initialized_prefix_on_panic() {
        use std::cell::Cell;
        use std::rc::Rc;

        struct Tracked(Rc<Cell<usize>>);
        impl Drop for Tracked {
            fn drop(&mut self) {
                self.0.set(self.0.get() + 1);
            }
        }

        let dropped = Rc::new(Cell::new(0));
        let counter = Rc::clone(&dropped);
        let initialized = Cell::new(0usize);
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            CubeMap::from_fn(2, |_| {
                if initialized.get() == 3 {
                    panic!("intentional initializer panic");
                }
                initialized.set(initialized.get() + 1);
                Tracked(Rc::clone(&counter))
            })
        }));
        assert!(result.is_err());
        assert_eq!(initialized.get(), 3);
        assert_eq!(dropped.get(), 3);
    }

    #[test]
    fn invalid_cubemap_dimensions_rejected() {
        assert!(CubeMap::<u32>::new(u32::MAX, 0).is_none());
        assert!(CubeMap::<u32>::from_slice(2, &[0; 6]).is_none());
        assert!(CubeMap::<u32>::from_fn(u32::MAX, |_| 0).is_none());
    }

    #[test]
    fn addressing_roundtrip() {
        const RES: u32 = 2049; // must be odd for there to be a point exactly on the axes
        for dir in &[
            na::Vector3::x_axis(),
            na::Vector3::y_axis(),
            na::Vector3::z_axis(),
            -na::Vector3::x_axis(),
            -na::Vector3::y_axis(),
            -na::Vector3::z_axis(),
        ] {
            let index = index(RES, dir);
            let out = get_dir(RES, index).unwrap();
            assert_eq!(dir, &out);
        }
        // A cube corner lies on a cell boundary, not at a dense cell center.
        let corner = na::Unit::new_normalize(na::Vector3::new(1.0, 1.0, 1.0));
        let cell = index(RES, &corner);
        assert_eq!(index(RES, &get_dir(RES, cell).unwrap()), cell);
    }

    #[test]
    fn every_dense_cell_center_roundtrips_without_face_aliasing() {
        for resolution in [1, 2, 3, 8, 19] {
            let count = (6 * resolution * resolution) as usize;
            for i in 0..count {
                let direction = get_dir(resolution, i).unwrap();
                assert_eq!(index(resolution, &direction), i);
                let (coords, local) = Coords::from_vector_with_coords(resolution, &direction);
                assert!((local.x - 0.5).abs() < 1.0e-5);
                assert!((local.y - 0.5).abs() < 1.0e-5);
                assert_abs_diff_eq!(coords.center::<f64>(resolution), direction);
            }
        }
    }

    #[test]
    fn deepest_grid_addresses_roundtrip_across_every_face() {
        for resolution in [1 << 24, 1 << 31, u32::MAX] {
            let mut random = 0x85f2_4c97_u32;
            let mut next = || {
                random ^= random << 13;
                random ^= random >> 17;
                random ^= random << 5;
                random % resolution
            };
            for face in Face::iter() {
                for (x, y) in [
                    (0, 0),
                    (resolution - 1, resolution - 1),
                    (resolution / 2, resolution / 2),
                    (resolution / 2 - 1, resolution / 2 + 1),
                ]
                .into_iter()
                .chain((0..128).map(|_| (next(), next())))
                {
                    let coords = Coords { x, y, face };
                    for local in [na::Point2::new(0.5, 0.5), na::Point2::new(0.2, 0.8)] {
                        let direction = coords.direction(resolution, &local);
                        let (actual, uv) = Coords::from_vector_with_coords(resolution, &direction);
                        assert_eq!(actual, coords);
                        assert_abs_diff_eq!(uv, local, epsilon = 1e-5);
                        let nearby: Vec<_> =
                            Coords::neighborhood(resolution, *direction, 0.0).collect();
                        assert!(nearby.contains(&coords), "missed {coords:?} in {nearby:?}");
                        assert!(
                            nearby.len() <= 4,
                            "unnecessarily broad zero-angle cone: {nearby:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn cone_addressing_is_invariant_to_extreme_direction_magnitude() {
        let direction = na::Vector3::new(1.0, 0.5, -0.25);
        let expected: Vec<_> = Coords::neighborhood(128, direction, 0.01).collect();
        for scale in [1e-310, 1e300] {
            let scaled = direction * scale;
            let actual: Vec<_> = Coords::neighborhood(128, scaled, 0.01).collect();
            assert_eq!(actual, expected);
            assert_eq!(
                Coords::from_vector(128, &scaled),
                Coords::from_vector(128, &direction)
            );
        }
    }

    #[test]
    fn adjacent_samples_match_at_non_power_of_two_and_deep_resolutions() {
        for resolution in [3, 11, 1 << 24, 1 << 31, u32::MAX] {
            for x in [0, resolution / 2, resolution - 2] {
                let left = Coords {
                    x,
                    y: resolution / 2,
                    face: Face::Pz,
                };
                let right = Coords { x: x + 1, ..left };
                for samples in [2, 5, 8, 17] {
                    let a: Vec<_> = left.samples::<f64>(resolution, samples).collect();
                    let b: Vec<_> = right.samples::<f64>(resolution, samples).collect();
                    for row in 0..samples as usize {
                        assert_eq!(
                            a[row * samples as usize + samples as usize - 1],
                            b[row * samples as usize]
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn vector_lookup_returns_local_coordinates_within_selected_cell() {
        for face in Face::iter() {
            let coords = Coords { x: 2, y: 5, face };
            let local = na::Point2::new(0.2_f64, 0.8);
            let direction = coords.direction(8, &local);
            let (actual, uv) = Coords::from_vector_with_coords(8, &direction);
            assert_eq!(coords, actual);
            assert_abs_diff_eq!(uv, local, epsilon = 1.0e-5);
        }
    }

    #[test]
    fn face_neighbors() {
        // Opposite faces are incident to opposite edges
        assert_eq!(Face::Px.neighbors()[0b00].1, -Face::Nx.neighbors()[0b10].1);
        assert_eq!(Face::Px.neighbors()[0b01].1, -Face::Nx.neighbors()[0b01].1);
        assert_eq!(Face::Px.neighbors()[0b10].1, -Face::Nx.neighbors()[0b00].1);
        assert_eq!(Face::Px.neighbors()[0b11].1, -Face::Nx.neighbors()[0b11].1);

        assert_eq!(Face::Py.neighbors()[0b00].1, -Face::Ny.neighbors()[0b00].1);
        assert_eq!(Face::Py.neighbors()[0b01].1, -Face::Ny.neighbors()[0b11].1);
        assert_eq!(Face::Py.neighbors()[0b10].1, -Face::Ny.neighbors()[0b10].1);
        assert_eq!(Face::Py.neighbors()[0b11].1, -Face::Ny.neighbors()[0b01].1);

        assert_eq!(Face::Pz.neighbors()[0b00].1, -Face::Nz.neighbors()[0b10].1);
        assert_eq!(Face::Pz.neighbors()[0b01].1, -Face::Nz.neighbors()[0b01].1);
        assert_eq!(Face::Pz.neighbors()[0b10].1, -Face::Nz.neighbors()[0b00].1);
        assert_eq!(Face::Pz.neighbors()[0b11].1, -Face::Nz.neighbors()[0b11].1);
    }

    #[test]
    fn face_neighbor_axes() {
        // Neighboring faces correctly track whether the axes they intersect on in their local
        // reference frames are parallel or antiparallel
        for face in Face::iter() {
            for (edge, &(neighbor, neighbor_edge, parallel)) in Edge::iter().zip(face.neighbors()) {
                let local = face.basis()
                    * match edge {
                        Edge::Px | Edge::Nx => na::Vector3::y(),
                        Edge::Py | Edge::Ny => na::Vector3::x(),
                    };
                let neighbor = neighbor.basis()
                    * match neighbor_edge {
                        Edge::Px | Edge::Nx => na::Vector3::y(),
                        Edge::Py | Edge::Ny => na::Vector3::x(),
                    };
                let sign = if parallel { 1.0 } else { -1.0 };
                assert_eq!(local, sign * neighbor);
            }
        }
    }

    #[test]
    fn face_coord_sanity() {
        let o = na::Point2::<f32>::origin();
        assert_eq!(Face::coords(&na::Vector3::x()), (Face::Px, o));
        assert_eq!(Face::coords(&na::Vector3::y()), (Face::Py, o));
        assert_eq!(Face::coords(&na::Vector3::z()), (Face::Pz, o));
        assert_eq!(Face::coords(&-na::Vector3::x()), (Face::Nx, o));
        assert_eq!(Face::coords(&-na::Vector3::y()), (Face::Ny, o));
        assert_eq!(Face::coords(&-na::Vector3::z()), (Face::Nz, o));
    }

    #[test]
    fn coord_neighborhood() {
        use Face::*;
        for face in Face::iter() {
            let center = face.basis() * na::Vector3::z();
            assert_eq!(
                Coords::neighborhood(1, center, 0.1).collect::<Vec<_>>(),
                vec![Coords { x: 0, y: 0, face }]
            );
            assert_eq!(
                Coords::neighborhood(3, center, 0.1).collect::<Vec<_>>(),
                vec![Coords { x: 1, y: 1, face }]
            );
            let xs = Coords::neighborhood(2, center, 0.1).collect::<Vec<_>>();
            assert_eq!(xs.len(), 4);
            for expected in (0..2).flat_map(|y| (0..2).map(move |x| Coords { x, y, face })) {
                assert!(xs.contains(&expected));
            }
        }
        assert_eq!(
            Coords::neighborhood(1, na::Vector3::new(1.0, 1.0, 1.0), 0.1).collect::<Vec<_>>(),
            vec![
                Coords {
                    x: 0,
                    y: 0,
                    face: Px
                },
                Coords {
                    x: 0,
                    y: 0,
                    face: Py
                },
                Coords {
                    x: 0,
                    y: 0,
                    face: Pz
                }
            ]
        );
        assert_eq!(
            Coords::neighborhood(1, na::Vector3::new(1.0, 1.0, 0.0), 0.1).collect::<Vec<_>>(),
            vec![
                Coords {
                    x: 0,
                    y: 0,
                    face: Px
                },
                Coords {
                    x: 0,
                    y: 0,
                    face: Py
                },
            ]
        );
        assert_eq!(
            Coords::neighborhood(5, na::Vector3::new(1.0, 1.0, 1.0), 0.1).count(),
            3
        );
    }

    #[test]
    fn discretize_sanity() {
        assert_eq!(discretize(100, na::Point2::new(1.0, 1.0)), (99, 99));
        assert_eq!(discretize(100, na::Point2::new(0.0, 0.0)), (0, 0));
        assert_eq!(discretize(100, na::Point2::new(0.99, 0.99)), (99, 99));
        assert_eq!(discretize(100, na::Point2::new(0.989, 0.989)), (98, 98));
        assert_eq!(discretize(100, na::Point2::new(0.01, 0.01)), (1, 1));
        assert_eq!(discretize(100, na::Point2::new(0.009, 0.009)), (0, 0));

        assert_eq!(discretize(2, na::Point2::new(0.49, 0.49)), (0, 0));
        assert_eq!(discretize(2, na::Point2::new(0.50, 0.50)), (1, 1));
    }

    #[test]
    fn from_raw() {
        let data = [1, 0, 1, 2, 3, 4, 5];
        let x = unsafe { CubeMap::<u32>::from_raw(data.as_ptr()) };
        assert_eq!(x.resolution(), 1);
        assert_eq!(x.as_ref().len(), 6);
        for i in 0..6 {
            assert_eq!(x.as_ref()[i], data[i + 1]);
        }
    }

    #[test]
    fn samples_sanity() {
        const COORDS: Coords = Coords {
            x: 0,
            y: 0,
            face: Face::Py,
        };
        assert_abs_diff_eq!(
            COORDS.samples::<f32>(1, 1).next().unwrap(),
            na::Vector3::y_axis()
        );
        let corners = COORDS
            .samples(1, 2)
            .map(|x| x.into_inner())
            .collect::<Vec<_>>();
        let corner = na::Unit::new_normalize(na::Vector3::new(1.0, 1.0, 1.0));
        assert_abs_diff_eq!(
            corners[..],
            [
                na::Vector3::new(-corner.x, corner.y, corner.z),
                na::Vector3::new(corner.x, corner.y, corner.z),
                na::Vector3::new(-corner.x, corner.y, -corner.z),
                na::Vector3::new(corner.x, corner.y, -corner.z),
            ][..]
        );
    }

    #[test]
    fn neighboring_samples_align() {
        const LEFT: Coords = Coords {
            x: 0,
            y: 0,
            face: Face::Pz,
        };
        const RIGHT: Coords = Coords {
            x: 1,
            y: 0,
            face: Face::Pz,
        };
        let left = LEFT.samples::<f32>(2, 2).collect::<Vec<_>>();
        let right = RIGHT.samples::<f32>(2, 2).collect::<Vec<_>>();
        assert_abs_diff_eq!(left[1], right[0]);
        assert_abs_diff_eq!(left[3], right[2]);
    }

    #[test]
    #[cfg(feature = "fearless_simd")]
    fn simd_samples_consistent() {
        use fearless_simd::Fallback;

        const COORDS: Coords = Coords {
            x: 0,
            y: 0,
            face: Face::Py,
        };
        const FACE_RES: u32 = 1;
        const CHUNK_RES: u32 = 17;
        let mut scalar = COORDS.samples(FACE_RES, CHUNK_RES);
        let simd = COORDS.samples_ps(Fallback::new(), FACE_RES, CHUNK_RES);
        assert_eq!(
            simd.len(),
            scalar.len().div_ceil(<Fallback as Simd>::f32s::LEN)
        );
        for vs in simd {
            for i in 0..<Fallback as Simd>::f32s::LEN {
                let [x, y, z] = vs.map(|c| c.as_slice()[i]);
                let Some(scalar) = scalar.next() else {
                    break;
                };
                let simd = na::Unit::new_unchecked(na::Vector3::new(x, y, z));
                assert_abs_diff_eq!(scalar, simd);
            }
        }
    }

    #[test]
    fn warp_roundtrip() {
        for i in -100..100 {
            let x = i as f32 / 100.0;
            assert_abs_diff_eq!(x, unwarp(warp(x)));
        }
    }

    #[test]
    fn direction_roundtrip() {
        for face in Face::iter() {
            for y in -10..10 {
                for x in -10..10 {
                    let p = na::Point2::new(x as f32 / 11.0, y as f32 / 11.0);
                    let v = face.direction(&p);
                    let (actual_face, actual_p) = Face::coords(&v);
                    assert_eq!(face, actual_face);
                    assert_abs_diff_eq!(p, actual_p);
                }
            }
        }
    }

    #[test]
    fn neighborhood_contains_every_sample_direction_within_cone() {
        use std::collections::HashSet;
        const RESOLUTION: u32 = 8;
        let directions = [
            na::Vector3::new(0.1, 1.0, 0.0),
            na::Vector3::new(1.0, 0.1, 0.0),
            na::Vector3::new(-0.2, 0.3, 1.0),
            na::Vector3::new(0.4, 0.8, -1.0),
            na::Vector3::new(-0.8, -0.2, -0.3),
            na::Vector3::new(0.5, -1.0, 0.25),
        ];
        for direction in directions {
            let axis = direction.normalize();
            for theta in [0.01_f64, 0.2, 0.5, 0.8, 1.2, 1.8, std::f64::consts::PI] {
                let actual: HashSet<_> =
                    Coords::neighborhood(RESOLUTION, direction, theta).collect();
                for face in Face::iter() {
                    for y in 0..RESOLUTION {
                        for x in 0..RESOLUTION {
                            let coords = Coords { x, y, face };
                            let sample = coords.center::<f64>(RESOLUTION);
                            if axis.dot(sample.as_ref()) > theta.cos() + 1e-4 {
                                assert!(
                                    actual.contains(&coords),
                                    "missed {coords:?} for direction {direction:?} theta {theta}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn neighborhood_consistency() {
        let dir = na::Vector3::new(-5_195_083.148, 3_582_099.8, -877_091.25);
        assert_eq!(
            Coords::from_vector(4096, &dir),
            Coords::neighborhood(4096, dir, 0.0).next().unwrap()
        );
    }
}
