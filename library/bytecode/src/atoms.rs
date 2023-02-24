use core::fmt::{self, Debug, Display};
use core::hash::Hash;
use core::ops::DerefMut;
use core::slice;
use core::str;

/// The [`Atom`] trait represents the type of the handle used to refer to entries in an [`AtomTable`].
///
/// It is implemented for `*const str` by default to provide a simple exchange medium for atoms.
pub trait Atom: Copy + PartialEq + Eq + PartialOrd + Ord + Hash + Debug + Display {
    /// This type is the internal representation of the `Atom` entry in the table
    ///
    /// This representation is stored in the bytecode format
    type Repr: Copy;

    /// Get the raw data backing this atom
    fn as_bytes(&self) -> &[u8];
    /// Extracts a pointer to the table entry to which this atom refers
    fn unpack(self) -> Self::Repr;
    /// Produces an atom from a pointer to its corresponding table entry
    fn pack(raw: Self::Repr) -> Self;
    /// Extracts the raw components of the internal representation
    fn into_raw_parts(data: Self::Repr) -> (*const u8, usize);
    /// Creates a new internal representation from raw components
    unsafe fn from_raw_parts(ptr: *const u8, size: usize) -> Self::Repr;
}

/// A type of [`Atom`] whose representation matches that of `firefly_rt::term::Atom`
#[derive(Copy, Clone, Hash)]
#[repr(C)]
pub struct AtomicStr {
    pub ptr: *const u8,
    pub size: usize,
}
impl Debug for AtomicStr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", str::from_utf8(self.as_bytes()).unwrap())
    }
}
impl Display for AtomicStr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", str::from_utf8(self.as_bytes()).unwrap())
    }
}
impl PartialEq for AtomicStr {
    fn eq(&self, other: &Self) -> bool {
        if self.size != other.size {
            return false;
        }
        self.as_bytes() == other.as_bytes()
    }
}
impl Eq for AtomicStr {}
impl From<&'static str> for AtomicStr {
    fn from(s: &'static str) -> Self {
        let bytes = s.as_bytes();
        Self {
            ptr: bytes.as_ptr(),
            size: bytes.len(),
        }
    }
}
impl PartialOrd for AtomicStr {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for AtomicStr {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.as_bytes().cmp(other.as_bytes())
    }
}
impl Atom for AtomicStr {
    type Repr = Self;

    #[inline]
    fn as_bytes(&self) -> &[u8] {
        unsafe { slice::from_raw_parts(self.ptr, self.size) }
    }
    #[inline]
    fn unpack(self) -> Self::Repr {
        self
    }
    #[inline]
    fn pack(raw: Self::Repr) -> Self {
        raw
    }
    #[inline]
    fn into_raw_parts(data: Self::Repr) -> (*const u8, usize) {
        (data.ptr, data.size)
    }
    #[inline]
    unsafe fn from_raw_parts(ptr: *const u8, size: usize) -> Self::Repr {
        Self { ptr, size }
    }
}

/// Represents an atom table through which atoms will be fetched and stored
///
/// We use this trait so that we can use different tables in different contexts.
///
/// When creating bytecode modules, we want local tables which are self-contained
/// and can be freed when the bytecode is written to disk.
///
/// When loading bytecode modules though, we may also want local tables if we're
/// performing analysis/merging; or we may want to load directly into a global table
/// if we're loading into the emulator where we want the atoms to be loaded in such a
/// state that they can be used immediately.
pub trait AtomTable {
    /// The type equivalent to `firefly_rt::term::Atom`
    type Atom: Atom;
    /// The type to use when there is an error creating an atom from string/bytes
    type AtomError: Debug;
    /// The type of the guard required for bulk insert into the table
    type Guard: AtomTable<Atom = Self::Atom, AtomError = Self::AtomError>;

    /// Returns the number of atoms in the table
    fn len(&self) -> usize;

    /// Returns an iterator over the atoms in the table
    fn iter<'a>(&'a self) -> impl Iterator<Item = Self::Atom> + 'a;

    /// Get an `Atom` corresponding to the given string from the table, creating
    /// a new one if the string hasn't been inserted in the table yet.
    fn get_or_insert(&mut self, s: &str) -> Result<Self::Atom, Self::AtomError>;

    /// Apply changes to the table in bulk, using the provided callback.
    ///
    /// The callback takes a `Guard`, which is used to lock the table for exclusive access.
    fn change<F, T>(&mut self, callback: F) -> T
    where
        F: FnOnce(&mut Self::Guard) -> T;
}

impl<T, U> AtomTable for U
where
    for<'a> T: AtomTable + 'a,
    U: DerefMut<Target = T>,
{
    type Atom = <T as AtomTable>::Atom;
    type AtomError = <T as AtomTable>::AtomError;
    type Guard = <T as AtomTable>::Guard;

    #[inline]
    fn len(&self) -> usize {
        self.deref().len()
    }

    #[inline]
    fn iter<'a>(&'a self) -> impl Iterator<Item = Self::Atom> + 'a {
        self.deref().iter()
    }

    #[inline]
    fn get_or_insert(&mut self, s: &str) -> Result<Self::Atom, Self::AtomError> {
        self.deref_mut().get_or_insert(s)
    }

    #[inline]
    fn change<F, R>(&mut self, callback: F) -> R
    where
        F: FnOnce(&mut Self::Guard) -> R,
    {
        self.deref_mut().change(callback)
    }
}
