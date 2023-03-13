#[cfg_attr(rustfmt, rustfmt_skip)]
#[allow(nonstandard_style, non_upper_case_globals)]
pub mod atoms {
    // During the build step, `build.rs` will output the generated atoms to `OUT_DIR` to avoid
    // adding it to the source directory, so we just directly include the generated code here.
    include!(concat!(env!("OUT_DIR"), "/atoms.rs"));
}

mod table;

pub use self::table::{
    with_atom_table, with_atom_table_readonly, AtomData, AtomTable, GlobalAtomTable,
};

use core::convert::AsRef;
use core::fmt::{self, Debug, Display};
use core::hash::{Hash, Hasher};
use core::ptr::NonNull;
use core::slice;
use core::str::{self, FromStr, Utf8Error};

use firefly_binary::Encoding;
use static_assertions::assert_eq_size;

use super::OpaqueTerm;

/// The maximum length of an atom (255)
pub const MAX_ATOM_LENGTH: usize = u16::max_value() as usize;

/// Produced by operations which create atoms
#[derive(Debug, Copy, Clone, PartialEq)]
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
impl From<Utf8Error> for AtomError {
    #[inline]
    fn from(err: Utf8Error) -> Self {
        Self::InvalidString(err)
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

/// An atom is an interned string value with fast, constant-time equality comparison,
/// can be encoded as an immediate value, and only requires allocation once over the lifetime
/// of the program.
#[repr(transparent)]
#[derive(Copy, Clone, PartialEq, Eq)]
pub struct Atom(NonNull<AtomData>);

assert_eq_size!(Atom, Option<Atom>);

unsafe impl Send for Atom {}
unsafe impl Sync for Atom {}
impl firefly_system::sync::Atom for Atom {
    type Repr = *mut AtomData;

    #[inline]
    fn pack(self) -> Self::Repr {
        self.0.as_ptr()
    }

    #[inline]
    fn unpack(raw: Self::Repr) -> Self {
        Self(NonNull::new(raw).unwrap())
    }
}
impl firefly_bytecode::Atom for Atom {
    type Repr = AtomData;

    #[inline]
    fn as_bytes(&self) -> &[u8] {
        self.as_str().as_bytes()
    }
    #[inline]
    fn unpack(self) -> Self::Repr {
        unsafe { *self.0.as_ptr() }
    }
    fn pack(raw: Self::Repr) -> Self {
        let name = str::from_utf8(unsafe { slice::from_raw_parts(raw.ptr, raw.size) }).unwrap();
        match name {
            "false" => atoms::False,
            "true" => atoms::True,
            name => Self(unsafe { table::get_data_or_insert(name).unwrap() }),
        }
    }
    #[inline]
    fn into_raw_parts(data: Self::Repr) -> (*const u8, usize) {
        (data.ptr, data.size)
    }
    #[inline]
    unsafe fn from_raw_parts(ptr: *const u8, size: usize) -> Self::Repr {
        AtomData { ptr, size }
    }
}
impl Atom {
    /// Creates a new atom from a slice of bytes interpreted as Latin-1.
    ///
    /// Returns `Err` if the atom does not exist
    #[inline]
    pub fn try_from_latin1_bytes_existing(name: &[u8]) -> Result<Self, AtomError> {
        Self::try_from_str_existing(str::from_utf8(name)?)
    }

    /// Creates a new atom from a `str`, but only if the atom already exists
    ///
    /// Returns `Err` if the atom does not exist
    #[inline]
    pub fn try_from_str_existing<S: AsRef<str>>(s: S) -> Result<Self, AtomError> {
        let name = s.as_ref();
        match name {
            "false" => Ok(atoms::False),
            "true" => Ok(atoms::True),
            name => {
                Self::validate(name)?;
                if let Some(data) = table::get_data(name) {
                    return Ok(Self(data));
                }
                Err(AtomError::NonExistent)
            }
        }
    }

    /// For convenience, this function takes a `str`, creates an atom
    /// from it, and immediately encodes the resulting `Atom` as an `OpaqueTerm`
    ///
    /// Panics if the name is invalid, the table overflows, or term encoding fails
    #[inline]
    pub fn str_to_term<S: AsRef<str>>(s: S) -> OpaqueTerm {
        Self::from_str(s.as_ref()).unwrap().into()
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
        use core::ffi::CStr;

        let cs = CStr::from_ptr::<'static>(ptr);
        let name = cs.to_str().unwrap_or_else(|error| {
            panic!(
                "unable to construct atom from cstr `{}` due to invalid utf-8: {:?}",
                cs.to_string_lossy(),
                error,
            )
        });
        match name {
            "false" => atoms::False,
            "true" => atoms::True,
            name => {
                let ptr = table::get_data_or_insert_static(name).unwrap();
                Self(ptr)
            }
        }
    }

    /// Returns `true` if this atom represents a boolean
    pub fn is_boolean(self) -> bool {
        self == atoms::False || self == atoms::True
    }

    /// Converts this atom to a boolean
    ///
    /// This function will return true for any atom other than `false`
    pub fn as_boolean(self) -> bool {
        self != atoms::False
    }

    /// Gets the string value of this atom
    pub fn as_str(&self) -> &'static str {
        // SAFETY: Atom contents are validated when creating the raw atom data, so converting back
        // to str is safe
        unsafe { self.0.as_ref().as_str().unwrap() }
    }

    #[inline(always)]
    pub const unsafe fn as_ptr(&self) -> *const AtomData {
        self.0.as_ptr().cast_const()
    }

    #[inline]
    pub unsafe fn from_raw(ptr: *const AtomData) -> Self {
        Self(NonNull::new(ptr.cast_mut()).unwrap())
    }

    /// Returns true if this atom requires quotes when printing as an Erlang term
    pub fn needs_quotes(&self) -> bool {
        // See https://github.com/erlang/otp/blob/ca83f680aab717fe65634247d16f18a8cbfc6d8d/erts/emulator/beam/erl_printf_term.c#L193-L212
        let mut chars = self.as_str().chars();

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

    pub fn validate(name: &str) -> Result<(), AtomError> {
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
    #[inline]
    fn from(b: bool) -> Self {
        if b {
            atoms::True
        } else {
            atoms::False
        }
    }
}
impl From<firefly_bytecode::ErrorKind> for Atom {
    fn from(kind: firefly_bytecode::ErrorKind) -> Self {
        use firefly_bytecode::ErrorKind;
        match kind {
            ErrorKind::Error => atoms::Error,
            ErrorKind::Exit => atoms::Exit,
            ErrorKind::Throw => atoms::Throw,
        }
    }
}
impl TryFrom<&str> for Atom {
    type Error = AtomError;

    fn try_from(s: &str) -> Result<Self, Self::Error> {
        match s {
            "false" => Ok(atoms::False),
            "true" => Ok(atoms::True),
            s => {
                Self::validate(s)?;
                Ok(Self(unsafe { table::get_data_or_insert(s)? }))
            }
        }
    }
}
impl TryFrom<&[u8]> for Atom {
    type Error = AtomError;

    #[inline]
    fn try_from(bytes: &[u8]) -> Result<Self, Self::Error> {
        Atom::try_from(str::from_utf8(bytes)?)
    }
}
// Support converting from atom terms to `Encoding` type
impl TryInto<Encoding> for Atom {
    type Error = anyhow::Error;

    #[inline]
    fn try_into(self) -> Result<Encoding, Self::Error> {
        self.as_str().parse()
    }
}
impl FromStr for Atom {
    type Err = AtomError;

    /// Creates a new atom from a `str`.
    ///
    /// Returns `Err` if the atom name is invalid or the table overflows
    #[inline(always)]
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Atom::try_from(s)
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
impl PartialEq<&str> for Atom {
    #[inline(always)]
    fn eq(&self, other: &&str) -> bool {
        self.as_str().eq(*other)
    }
}
impl crate::cmp::ExactEq for Atom {}
impl Ord for Atom {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.as_str().cmp(other.as_str())
    }
}
impl Debug for Atom {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Atom({} @ {:p})", self, self.0)
    }
}
impl fmt::Pointer for Atom {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:p}", self.0)
    }
}
impl Display for Atom {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use core::fmt::Write;
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
        self.0.hash(state);
    }
}

/// This is a helper which allows compiled code to convert a pointer to a C string value into an
/// atom directly.
#[export_name = "__firefly_builtin_atom_from_cstr"]
pub unsafe extern "C-unwind" fn atom_from_cstr(ptr: *const core::ffi::c_char) -> OpaqueTerm {
    let atom = Atom::from_raw_cstr(ptr);
    atom.into()
}
