mod table;

use self::table::{self, AtomData};

use core::any::TypeId;
use core::convert::AsRef;
use core::fmt::{self, Debug, Display};
use core::hash::{Hash, Hasher};
use core::ptr::{self, NonNull};
use core::slice;
use core::str::{self, Utf8Error};

use super::{Term, Type};

/// The maximum length of an atom (255)
pub const MAX_ATOM_LENGTH: usize = u16::max_value() as usize;

/// Produced by operations which create atoms
#[derive(Debug)]
pub enum AtomError {
    InvalidLength(usize),
    NonExistent,
    InvalidString(Utf8Error),
}
#[cfg(feature = "std")]
impl std::error::Error for AtomError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::InvalidString(ref err) => Some(err),
            _ => None,
        }
    }
}
impl Display for AtomError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::InvalidLength(len) => write!(
                f,
                "invalid atom, length is {}, maximum length is {}",
                len, MAX_ATOM_LENGTH
            ),
            Self::NonExistent => f.write_str("tried to convert to an atom that doesn't exist"),
            Self::InvalidString(err) => write!(f, "invalid utf-8 bytes: {}", &err),
        }
    }
}
impl Eq for AtomError {}
impl PartialEq for AtomError {
    fn eq(&self, other: &AtomError) -> bool {
        mem::discriminant(self) == mem::discriminant(other)
    }
}

/// An atom is an interned string value with fast, constant-time equality comparison,
/// can be encoded as an immediate value, and only requires allocation once over the lifetime
/// of the program.
#[derive(Copy, Clone, PartialEq, Eq)]
pub struct Atom(NonNull<AtomData>);
impl Atom {
    pub const FALSE: Atom =
        Self(unsafe { NonNull::new_unchecked(&AtomData::FALSE as *const _ as *mut AtomData) });
    pub const TRUE: Atom =
        Self(unsafe { NonNull::new_unchecked(&AtomData::TRUE as *const _ as *mut AtomData) });

    /// Creates a new atom from a slice of bytes interpreted as Latin-1.
    ///
    /// Returns `Err` if the atom name is invalid or the table overflows
    #[inline]
    pub fn try_from_latin1_bytes(name: &[u8]) -> Result<Self, AtomError> {
        Self::try_from_str(str::from_utf8(name)?)
    }

    /// Like `try_from_latin1_bytes`, but requires that the atom already exists
    ///
    /// Returns `Err` if the atom does not exist
    #[inline]
    pub fn try_from_latin1_bytes_existing(name: &[u8]) -> Result<Self, AtomError> {
        Self::try_from_str_existing(str::from_utf8(name)?)
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
        if let Some(data) = table::get_data(name) {
            return Ok(Self(data));
        }
        Ok(Self(table::get_id_or_insert(name)?))
    }

    /// Creates a new atom from a `str`, but only if the atom already exists
    ///
    /// Returns `Err` if the atom does not exist
    #[inline]
    pub fn try_from_str_existing<S: AsRef<str>>(s: S) -> Result<Self, AtomError> {
        let name = s.as_ref();
        Self::validate(name)?;
        if let Some(data) = table::get_data(name) {
            return Ok(Self(data));
        }
        Err(AtomError::NonExistent)
    }

    /// For convenience, this function takes a `str`, creates an atom
    /// from it, and immediately encodes the resulting `Atom` as an `OpaqueTerm`
    ///
    /// Panics if the name is invalid, the table overflows, or term encoding fails
    #[inline]
    pub fn str_to_term<S: AsRef<str>>(s: S) -> OpaqueTerm {
        Self::from_str(s).into()
    }

    /// This function is intended for internal use only.
    ///
    /// # Safety
    ///
    /// You must ensure the following is true of the given pointer:
    ///
    /// * It points to a null-terminated C-string
    /// * The content of the string is valid UTF-8 data
    /// * The pointer is valid for the entire lifetime of the program
    ///
    /// If any of these constraints are violated, the behavior is undefined.
    #[inline]
    pub(crate) unsafe fn from_raw_cstr(ptr: *const core::ffi::c_char) -> Self {
        let cs = CStr::from_ptr::<'static>(ptr);
        let name = cs.to_str().unwrap_or_else(|error| {
            panic!(
                "unable to construct atom from cstr `{}` due to invalid utf-8: {:?}",
                cs.to_string_lossy(),
                error,
            )
        });
        Self(table::get_id_or_insert_static(name).unwrap())
    }

    /// Returns `true` if this atom represents a boolean
    pub fn is_boolean(self) -> bool {
        self == Self::FALSE || self == Self::TRUE
    }

    /// Converts this atom to a boolean
    ///
    /// This function will panic if the atom is not a boolean value
    pub fn as_boolean(self) -> bool {
        debug_assert!(self.is_boolean());
        self != Self::FALSE
    }

    /// Gets the string value of this atom
    pub fn as_str(&self) -> &'static str {
        // SAFETY: Atom contents are validated when creating the raw atom data, so converting back to str is safe
        unsafe { self.0.as_ref().as_str() }
    }

    #[inline(always)]
    pub(super) fn ptr(&self) -> *const AtomData {
        self.0.as_ptr()
    }

    /// Returns true if this atom requires quotes when printing as an Erlang term
    pub fn needs_quotes(&self) -> bool {
        // See https://github.com/erlang/otp/blob/ca83f680aab717fe65634247d16f18a8cbfc6d8d/erts/emulator/beam/erl_printf_term.c#L193-L212
        let mut chars = self.name().chars();

        match chars.next() {
            Some(first_char) => {
                // https://github.com/erlang/otp/blob/ca83f680aab717fe65634247d16f18a8cbfc6d8d/erts/emulator/beam/erl_printf_term.c#L198-L199
                // -> https://github.com/erlang/otp/blob/ca83f680aab717fe65634247d16f18a8cbfc6d8d/erts/emulator/beam/erl_printf_term.c#L98
                !first_char.is_ascii_lowercase() || {
                    // https://github.com/erlang/otp/blob/ca83f680aab717fe65634247d16f18a8cbfc6d8d/erts/emulator/beam/erl_printf_term.c#L201-L200
                    // -> https://github.com/erlang/otp/blob/ca83f680aab717fe65634247d16f18a8cbfc6d8d/erts/emulator/beam/erl_printf_term.c#L102
                    //    -> https://github.com/erlang/otp/blob/ca83f680aab717fe65634247d16f18a8cbfc6d8d/erts/emulator/beam/erl_printf_term.c#L91
                    //    -> https://github.com/erlang/otp/blob/ca83f680aab717fe65634247d16f18a8cbfc6d8d/erts/emulator/beam/erl_printf_term.c#L91
                    //    -> https://github.com/erlang/otp/blob/ca83f680aab717fe65634247d16f18a8cbfc6d8d/erts/emulator/beam/erl_printf_term.c#L99
                    chars.any(|c| (!c.is_ascii_alphanumeric() && c != '_'))
                }
            }
            // https://github.com/erlang/otp/blob/ca83f680aab717fe65634247d16f18a8cbfc6d8d/erts/emulator/beam/erl_printf_term.c#L187-L190
            None => true,
        }
    }

    fn validate(name: &str) -> Result<(), AtomError> {
        let len = name.len();
        if len > MAX_ATOM_LENGTH {
            return Err(AtomError::InvalidLength(len));
        }
        Ok(())
    }
}
impl From<NonNull<AtomData>> for Atom {
    #[inline]
    fn from(ptr: NonNull<AtomData>) -> Self {
        Self(ptr)
    }
}
impl From<bool> for Atom {
    fn from(b: bool) -> Atom {
        if b {
            Self::TRUE
        } else {
            Self::FALSE
        }
    }
}
impl PartialOrd for Atom {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl PartialEq<str> for Atom {
    fn eq(&self, other: &str) -> bool {
        self.as_str() == other
    }
}
impl Ord for Atom {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.as_str().cmp(other.as_str())
    }
}
impl Debug for Atom {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}
impl fmt::Pointer for Atom {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:p}", self.ptr())
    }
}
impl Display for Atom {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let needs_quotes = self.needs_quotes();

        if needs_quotes {
            f.write_char('\'')?;
        }

        for c in self.as_str().chars() {
            // https://github.com/erlang/otp/blob/dbf25321bdfdc3f4aae422b8ba2c0f31429eba61/erts/emulator/beam/erl_printf_term.c#L215-L232
            match c {
                '\'' => f.write_str("\\'")?,
                '\\' => f.write_str("\\\\")?,
                '\n' => f.write_str("\\n")?,
                '\u{C}' => f.write_str("\\f")?,
                '\t' => f.write_str("\\t")?,
                '\r' => f.write_str("\\r")?,
                '\u{8}' => f.write_str("\\b")?,
                '\u{B}' => f.write_str("\\v")?,
                _ if c.is_control() => write!(f, "\\{:o}", c as u8)?,
                _ => f.write_char(c)?,
            }
        }

        if needs_quotes {
            f.write_char('\'')?;
        }

        Ok(())
    }
}

impl Hash for Atom {
    fn hash<H: Hasher>(&self, state: &mut H) {
        ptr::hash(self.0.as_ptr(), state);
    }
}

/// This is a helper which allows compiled code to convert a pointer to a C string value into an atom directly.
#[export_name = "__lumen_builtin_atom_from_cstr"]
pub unsafe extern "C-unwind" fn atom_from_cstr(ptr: *const core::ffi::c_char) -> OpaqueTerm {
    let atom = Atom::from_raw_cstr(ptr);
    atom.into()
}
