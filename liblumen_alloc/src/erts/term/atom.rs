use core::cmp;
use core::convert::TryFrom;
use core::fmt::{self, Debug, Display, Write};
use core::mem;

use core::ptr;
use core::slice;
use core::str::{self, Utf8Error};

use alloc::vec::Vec;

use hashbrown::HashMap;
use lazy_static::lazy_static;
use thiserror::Error;

use liblumen_arena::DroplessArena;

use liblumen_core::locks::RwLock;

use super::prelude::{Term, TypeError, TypedTerm};

/// The maximum number of atoms allowed
pub const MAX_ATOMS: usize = super::arch::MAX_ATOM_ID - 1;

/// The maximum length of an atom (255)
pub const MAX_ATOM_LENGTH: usize = u16::max_value() as usize;

lazy_static! {
    /// The atom table used by the runtime system
    static ref ATOMS: RwLock<AtomTable> = Default::default();
}

/// An interned string, represented in memory as a integer ID.
///
/// This struct is simply a transparent wrapper around the ID.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct Atom(usize);
impl Atom {
    pub const SIZE_IN_WORDS: usize = 1;

    /// Gets the identifier associated with this atom
    #[inline(always)]
    pub fn id(&self) -> usize {
        self.0
    }

    /// Returns the string representation of this atom
    #[inline]
    pub fn name(&self) -> &'static str {
        ATOMS.read().get_name(self.0).unwrap()
    }

    /// Creates a new atom from a slice of bytes interpreted as Latin-1.
    ///
    /// Returns `Err` if the atom name is invalid or the table overflows
    #[inline]
    pub fn try_from_latin1_bytes(name: &[u8]) -> Result<Self, AtomError> {
        Self::try_from_str(str::from_utf8(name).unwrap())
    }

    /// Like `try_from_latin1_bytes`, but requires that the atom already exists
    ///
    /// Returns `Err` if the atom does not exist
    #[inline]
    pub fn try_from_latin1_bytes_existing(name: &[u8]) -> Result<Self, AtomError> {
        Self::try_from_str_existing(str::from_utf8(name).unwrap())
    }

    /// For convenience, this function takes a `str`, creates an atom
    /// from it, and immediately encodes the resulting `Atom` as a `Term`
    ///
    /// Panics if the name is invalid, the table overflows, or term encoding fails
    #[inline]
    pub fn str_to_term<S: AsRef<str>>(s: S) -> Term {
        use crate::erts::term::prelude::Encode;
       
        Self::from_str(s)
            .encode()
            .unwrap()
    }

    /// Creates a new atom from a `str`.
    ///
    /// Panics if the name is invalid or the table overflows
    #[inline]
    pub fn from_str<S: AsRef<str>>(s: S) -> Self {
        Self::try_from_str(s).unwrap()
    }

    /// Creates a new atom from a `str`.
    ///
    /// Returns `Err` if the atom name is invalid or the table overflows
    #[inline]
    pub fn try_from_str<S: AsRef<str>>(s: S) -> Result<Self, AtomError> {
        let name = s.as_ref();
        Self::validate(name)?;
        if let Some(id) = ATOMS.read().get_id(name) {
            return Ok(Atom(id));
        }
        let id = ATOMS.write().get_id_or_insert(name)?;
        Ok(Atom(id))
    }

    /// Creates a new atom from a `str`, but only if the atom already exists
    ///
    /// Returns `Err` if the atom does not exist
    #[inline]
    pub fn try_from_str_existing<S: AsRef<str>>(s: S) -> Result<Self, AtomError> {
        let name = s.as_ref();
        Self::validate(name)?;
        if let Some(id) = ATOMS.read().get_id(name) {
            return Ok(Atom(id));
        }
        Err(AtomError(AtomErrorKind::NonExistent))
    }

    /// Creates a new atom from its id.
    ///
    /// # Safety
    ///
    /// This function is unsafe because creating an `Atom`
    /// with an id that doesn't exist will result in undefined
    /// behavior. This should only be used by `Term` when converting
    /// to `TypedTerm`
    /// ```
    #[inline]
    pub unsafe fn from_id(id: usize) -> Self {
        Self(id)
    }

    fn validate(name: &str) -> Result<(), AtomError> {
        let len = name.len();
        if len > MAX_ATOM_LENGTH {
            return Err(AtomError(AtomErrorKind::InvalidLength(len)));
        }
        Ok(())
    }
}

impl From<bool> for Atom {
    #[inline]
    fn from(b: bool) -> Self {
        // NOTE: We can make these assumptions because the AtomTable
        // is initialized in a deterministic way - it is critical that
        // if the initialization changes that these values get updated
        if b {
            unsafe { Atom::from_id(0) }
        } else {
            unsafe { Atom::from_id(1) }
        }
    }
}

impl Display for Atom {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str(":'")?;
        self.name()
            .chars()
            .flat_map(char::escape_default)
            .try_for_each(|c| f.write_char(c))?;
        f.write_char('\'')
    }
}

impl Debug for Atom {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if let Some(name) = ATOMS.read().get_name(self.0) {
            f.write_str(":\"")?;
            name.chars()
                .flat_map(char::escape_default)
                .try_for_each(|c| f.write_char(c))?;
            f.write_char('\"')
        } else {
            f.debug_tuple("Atom").field(&self.0).finish()
        }
    }
}

impl PartialOrd for Atom {
    #[inline]
    fn partial_cmp(&self, other: &Atom) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Atom {
    #[inline]
    fn cmp(&self, other: &Atom) -> cmp::Ordering {
        use cmp::Ordering;

        if self.0 == other.0 {
            return Ordering::Equal;
        }
        self.name().cmp(other.name())
    }
}

impl TryFrom<TypedTerm> for Atom {
    type Error = TypeError;

    fn try_from(typed_term: TypedTerm) -> Result<Self, Self::Error> {
        match typed_term {
            TypedTerm::Atom(atom) => Ok(atom),
            _ => Err(TypeError),
        }
    }
}

/// Produced by operations which create atoms
#[derive(Error, Debug)]
pub enum AtomError {
    #[error("exceeded system limit: maximum number of atoms ({})", MAX_ATOMS)]
    TooManyAtoms,
    #[error("invalid atom, length is {}, maximum length is {}", .0, MAX_ATOM_LENGTH)]
    InvalidLength(usize),
    #[error("tried to convert to an atom that doesn't exist")]
    NonExistent,
    #[error("invalid utf-8 bytes: {}", .0)]
    InvalidString(#[from] Utf8Error),
}
impl Eq for AtomError {}
impl PartialEq for AtomError {
    fn eq(&self, other: &AtomError) -> bool {
        mem::discriminant(self) == mem::discriminant(other)
    }
}

struct AtomTable {
    ids: HashMap<&'static str, usize>,
    names: Vec<&'static str>,
    arena: DroplessArena,
}
impl AtomTable {
    fn new(names: &[&'static str]) -> Self {
        let len = names.len();
        let mut table = Self {
            ids: HashMap::with_capacity(len),
            names: Vec::with_capacity(len),
            arena: DroplessArena::default(),
        };
        let interned_names = &mut table.names;
        for name in names {
            table.ids.entry(name).or_insert_with(|| {
                let id = interned_names.len();
                interned_names.push(name);
                id
            });
        }
        table
    }

    fn get_id(&self, name: &str) -> Option<usize> {
        self.ids.get(name).cloned()
    }

    fn get_name(&self, id: usize) -> Option<&'static str> {
        self.names.get(id).cloned()
    }

    fn get_id_or_insert(&mut self, name: &str) -> Result<usize, AtomError> {
        match self.get_id(name) {
            Some(existing_id) => Ok(existing_id),
            None => unsafe { self.insert(name) },
        }
    }

    // Unsafe because `name` should already have been checked as not existing while holding a
    // `mut reference`.
    unsafe fn insert(&mut self, name: &str) -> Result<usize, AtomError> {
        let id = self.names.len();
        if id > MAX_ATOMS {
            return Err(AtomError(AtomErrorKind::TooManyAtoms));
        }

        let size = name.len();

        let s = if size > 0 {
            // Copy string into arena
            let ptr = self.arena.alloc_raw(size, mem::align_of::<u8>());
            ptr::copy_nonoverlapping(name as *const _ as *const u8, ptr, size);
            let bytes = slice::from_raw_parts(ptr, size);

            str::from_utf8_unchecked(bytes)
        } else {
            ""
        };

        // Push into id map
        self.ids.insert(s, id);
        self.names.push(s);

        Ok(id)
    }
}
impl Default for AtomTable {
    fn default() -> Self {
        // Do not change the order of these atoms without updating any `From`
        // impls that may take advantage of the static order, i.e. From<bool>
        let atoms = &["true", "false", "undefined", "nil", "ok", "error"];
        AtomTable::new(atoms)
    }
}

/// This is safe to implement because the only usage is the ATOMS static, which is wrapped in an
/// `RwLock`, but it is _not_ `Sync` in general, so don't try and use it as such in other situations
unsafe impl Sync for AtomTable {}
