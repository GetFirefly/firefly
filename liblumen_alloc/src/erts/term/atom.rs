use alloc::vec::Vec;
use core::cmp;
use core::fmt::{self, Debug, Display, Write};
use core::mem;
use core::ptr;
use core::slice;
use core::str;

use hashbrown::HashMap;
use lazy_static::lazy_static;

use liblumen_arena::DroplessArena;
use liblumen_core::locks::RwLock;

use super::{AsTerm, Term};

/// The maximum number of atoms allowed
///
/// This is derived from the fact that atom values are
/// tagged in their highest 6 bits, so they are unusable.
pub const MAX_ATOMS: usize = usize::max_value() >> 6;

/// The maximum length of an atom (255)
pub const MAX_ATOM_LENGTH: usize = u16::max_value() as usize;

lazy_static! {
    /// The atom table used by the runtime system
    static ref ATOMS: RwLock<AtomTable> = Default::default();
}

/// An interned string, represented in memory as a tagged integer id.
///
/// This struct contains the untagged id
#[derive(Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct Atom(usize);
impl Atom {
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
        let id = ATOMS.write().insert(name)?;
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
impl Debug for Atom {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if let Some(name) = ATOMS.read().get_name(self.0) {
            f.write_char('\'')?;
            name.chars()
                .flat_map(char::escape_default)
                .try_for_each(|c| f.write_char(c))?;
            f.write_char('\'')
        } else {
            f.debug_tuple("Atom").field(&self.0).finish()
        }
    }
}
unsafe impl AsTerm for Atom {
    #[inline]
    unsafe fn as_term(&self) -> Term {
        Term::from_raw(self.0 | Term::FLAG_ATOM)
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

/// Produced by operations which create atoms
#[derive(Debug)]
pub struct AtomError(AtomErrorKind);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AtomErrorKind {
    TooManyAtoms,
    InvalidLength(usize),
    NonExistent,
}

impl Display for AtomError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.0 {
            AtomErrorKind::TooManyAtoms => write!(
                f,
                "exceeded system limit: maximum number of atoms ({})",
                MAX_ATOMS
            ),
            AtomErrorKind::InvalidLength(len) => write!(
                f,
                "invalid atom, length is {}, maximum length is {}",
                len, MAX_ATOM_LENGTH
            ),
            AtomErrorKind::NonExistent => {
                write!(f, "tried to convert to an atom that doesn't exist")
            }
        }
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

    fn insert(&mut self, name: &str) -> Result<usize, AtomError> {
        let name = unsafe { as_static(name) };
        let id = self.names.len();
        if id > MAX_ATOMS {
            return Err(AtomError(AtomErrorKind::TooManyAtoms));
        }
        let names = &mut self.names;
        let arena = &mut self.arena;
        Ok(*self.ids.entry(name).or_insert_with(|| {
            let size = name.len();
            // Copy string into arena
            let s = unsafe {
                let ptr = arena.alloc_raw(size, mem::align_of::<u8>());
                ptr::copy_nonoverlapping(name as *const _ as *const u8, ptr, size);
                let bytes = slice::from_raw_parts(ptr, size);
                str::from_utf8_unchecked(bytes)
            };
            // Push into id map
            names.push(s);
            id
        }))
    }
}
impl Default for AtomTable {
    fn default() -> Self {
        let atoms = &["true", "false", "undefined", "nil", "ok", "error"];
        AtomTable::new(atoms)
    }
}

/// This is safe to implement because the only usage is the ATOMS static, which is wrapped in an
/// `RwLock`, but it is _not_ `Sync` in general, so don't try and use it as such in other situations
unsafe impl Sync for AtomTable {}

#[inline]
unsafe fn as_static<'a>(s: &'a str) -> &'static str {
    mem::transmute::<&'a str, &'static str>(s)
}
