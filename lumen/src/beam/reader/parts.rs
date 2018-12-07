//! A collection of the miscellaneous parts used in a BEAM file.

/// The identifier of an atom.
///
/// The corresponding atom can be retrieved from the "Atom" chunk.
/// The identifier is used as one-based index for the atom list defined in the chunk.
pub type AtomId = u32;

/// The number of input arguments of a function.
pub type Arity = u32;

/// An erlang term encoded in [External Term Fromat].
///
/// [External Term Fromat]: http://erlang.org/doc/apps/erts/erl_ext_dist.html
pub type ExternalTermFormatBinary = Vec<u8>;

/// An atom.
#[derive(Debug, PartialEq, Eq)]
pub struct Atom {
    pub name: String,
}

/// An imported function.
#[derive(Debug, PartialEq, Eq)]
pub struct Import {
    pub module: AtomId,
    pub function: AtomId,
    pub arity: Arity,
}

/// An exported function.
#[derive(Debug, PartialEq, Eq)]
pub struct Export {
    pub function: AtomId,
    pub arity: Arity,
    pub label: u32,
}

/// A local function.
#[derive(Debug, PartialEq, Eq)]
pub struct Local {
    pub function: AtomId,
    pub arity: Arity,
    pub label: u32,
}

/// An anonymous function.
#[derive(Debug, PartialEq, Eq)]
pub struct Function {
    pub function: AtomId,
    pub arity: Arity,
    pub label: u32,
    pub index: u32,
    pub num_free: u32,
    pub old_uniq: u32,
}
