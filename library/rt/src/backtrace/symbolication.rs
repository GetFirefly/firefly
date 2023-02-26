use alloc::alloc::{AllocError, Layout};
use alloc::string::{String, ToString};

use firefly_alloc::heap::Heap;

use crate::function::ModuleFunctionArity;
use crate::term::*;

#[derive(Debug, Clone)]
pub enum Symbol {
    Erlang(ModuleFunctionArity),
    Native(String),
}
impl From<ModuleFunctionArity> for Symbol {
    fn from(mfa: ModuleFunctionArity) -> Self {
        Self::Erlang(mfa)
    }
}
impl From<&str> for Symbol {
    fn from(sym: &str) -> Self {
        match rustc_demangle::try_demangle(sym) {
            Ok(demangled) => Self::Native(alloc::format!("{:#}", demangled)),
            Err(_) => Self::Native(sym.to_string()),
        }
    }
}
impl Symbol {
    pub fn module(&self) -> Option<Atom> {
        match self {
            Self::Erlang(mfa) => Some(mfa.module),
            Self::Native(_) => None,
        }
    }

    pub fn function(&self) -> Option<Atom> {
        match self {
            Self::Erlang(mfa) => Some(mfa.function),
            Self::Native(_) => None,
        }
    }

    pub fn arity(&self) -> Option<u8> {
        match self {
            Self::Erlang(mfa) => Some(mfa.arity),
            Self::Native(_) => None,
        }
    }

    pub fn mfa(&self) -> Option<ModuleFunctionArity> {
        match self {
            Self::Erlang(mfa) => Some(*mfa),
            Self::Native(_) => None,
        }
    }
}

#[derive(Default, Debug, Clone)]
pub struct Symbolication {
    pub(super) symbol: Option<Symbol>,
    pub(super) filename: Option<String>,
    pub(super) line: Option<u32>,
    pub(super) column: Option<u32>,
}
impl From<ModuleFunctionArity> for Symbolication {
    fn from(mfa: ModuleFunctionArity) -> Self {
        Self {
            symbol: Some(mfa.into()),
            filename: None,
            line: None,
            column: None,
        }
    }
}
impl Symbolication {
    pub fn new(
        symbol: Symbol,
        filename: Option<String>,
        line: Option<u32>,
        column: Option<u32>,
    ) -> Self {
        Self {
            symbol: Some(symbol),
            filename,
            line,
            column,
        }
    }

    #[inline]
    pub fn module(&self) -> Option<Atom> {
        self.symbol.as_ref().and_then(|mfa| mfa.module())
    }

    #[inline]
    pub fn function(&self) -> Option<Atom> {
        self.symbol.as_ref().and_then(|mfa| mfa.function())
    }

    #[inline]
    pub fn arity(&self) -> Option<u8> {
        self.symbol.as_ref().and_then(|mfa| mfa.arity())
    }

    #[inline]
    pub fn mfa(&self) -> Option<ModuleFunctionArity> {
        self.symbol.as_ref().and_then(|mfa| mfa.mfa())
    }

    #[inline]
    pub fn symbol(&self) -> Option<&Symbol> {
        self.symbol.as_ref()
    }

    #[inline]
    pub fn filename(&self) -> Option<&str> {
        self.filename.as_deref()
    }

    #[inline]
    pub fn line(&self) -> Option<u32> {
        self.line
    }

    #[inline]
    pub fn column(&self) -> Option<u32> {
        self.column
    }
}
impl TryFrom<Term> for Symbolication {
    type Error = ();

    fn try_from(term: Term) -> Result<Self, Self::Error> {
        let Term::Tuple(tuple) = term else { return Err(()); };

        // If this tuple doesn't have 4 elements, then it is not a symbolicated frame
        if tuple.len() != 4 {
            return Err(());
        }

        let Term::Atom(module) = tuple[0].into() else { return Err(()); };
        let Term::Atom(function) = tuple[1].into() else { return Err(()); };

        // The third element is either the function arity or its argument list
        let mfa = match tuple[2].into() {
            Term::Int(arity) => ModuleFunctionArity {
                module,
                function,
                arity: arity.try_into().unwrap(),
            },
            Term::Nil => ModuleFunctionArity {
                module,
                function,
                arity: 0,
            },
            Term::Cons(cons) => ModuleFunctionArity {
                module,
                function,
                arity: cons.iter().count().try_into().unwrap(),
            },
            _ => return Err(()),
        };

        // The fourth element is optional metadata, including location information
        match tuple[3].into() {
            // No location metadata
            Term::Nil => Ok(Self {
                symbol: Some(Symbol::Erlang(mfa)),
                filename: None,
                line: None,
                column: None,
            }),
            Term::Cons(list) => {
                let filename = if let Some(term) = list.keyfind(0, atoms::File).ok().unwrap_or(None)
                {
                    match term.into() {
                        Term::Cons(list) => Some(list.to_string()),
                        _ => None,
                    }
                } else {
                    None
                };
                let line = if let Some(Term::Int(line)) =
                    list.keyfind(0, atoms::Line).ok().unwrap_or(None)
                {
                    Some(line.try_into().unwrap())
                } else {
                    None
                };
                Ok(Self {
                    symbol: Some(Symbol::Erlang(mfa)),
                    filename,
                    line,
                    column: None,
                })
            }
            // Technically a bug, but it is optional info, so we ignore it in the
            // interest of getting usable traces
            _ => Ok(Self {
                symbol: Some(Symbol::Erlang(mfa)),
                filename: None,
                line: None,
                column: None,
            }),
        }
    }
}
impl TryFrom<OpaqueTerm> for Symbolication {
    type Error = ();

    fn try_from(term: OpaqueTerm) -> Result<Self, Self::Error> {
        let term: Term = term.into();
        term.try_into()
    }
}

pub const MAX_FILENAME_LEN: usize = 120;

/// Calculates the Layout which should contain enough memory to hold the entire
/// stacktrace as an Erlang term. This allows the trace to live separately
/// from the process which raised it
///
/// Each frame looks like:
///
/// ```erlang,ignore
///     {module, function, arity, [{file, "path/to/file"}, {line, 1}]}
/// ```
///
/// However, the first frame may optionally contain the arguments, used for certain
/// classes of errors like function_clause errors. In this case, one frame will look
/// like so:
///
/// ```erlang,ignore
///     {module, function, [..args], [{file, "path/to/file"}, {line, 1}]}
/// ```
///
/// The frames themselves are contained in a list.
///
/// To calculate the total heap needed to allocate all of the terms, we define a heap
/// layout something like the following:
///
/// ```text,ignore
/// struct Frames {
///   list: [Cons<Frame>; NUM_FRAMES], // the cons cells for each frame, references `data`
///   data: [Frame; NUM_FRAMES], // the data for each frame
/// }
///
/// struct Frame {
///   frame: Tuple<4>, // references `meta`
///   arity: Int, // elided for the first frame if the argument list was provided
///   args: Nil | [Cons<Term>; ARITY], // elided for all but the first frame if the argument list was provided
///   meta: [Cons<Tuple>; 2], // cons cells for `file` and `line`
///   file: Tuple<2>, // references data in `filename`
///   line: Tuple<2>, // both elements are immediate
///   filename: [Cons<Int>; MAX_FILENAME_LEN],
/// }
///
/// struct Int(i64);
///
/// struct Nil(i64);
///
/// struct Cons<T>(*const T, *const Cons<T>);
///
/// struct Tuple<const N: usize>([OpaqueTerm; N]);
/// ```
///
/// The layout is then effectively equivalent to `mem::size_of::<Frames>()`. All types
/// should have the same alignment, so even if things are organized on the heap a little
/// differently, the total size of the allocation should remain the same.
///
/// The goal here is to allow plenty of room to hold the trace, without the risk of
/// running out of memory while constructing the trace for display and potentially
/// losing the whole trace. Especially on 64-bit systems, the address space is relatively
/// plentiful, so we're better off erring on the side of too much.
pub fn calculate_fragment_layout(
    num_frames: usize,
    arguments: Option<&[OpaqueTerm]>,
) -> Option<Layout> {
    if num_frames == 0 {
        return None;
    }

    let mut arguments_layout = LayoutBuilder::new();
    if let Some(terms) = arguments {
        for term in terms.iter().copied() {
            if !term.is_gcbox() {
                arguments_layout.build_cons();
                continue;
            }
            let term: Term = term.into();
            arguments_layout += term.layout();
            arguments_layout.build_cons();
        }
    }
    let arguments_layout = arguments_layout.finish();

    let mut metadata_layout = LayoutBuilder::new();
    metadata_layout
        .build_list(MAX_FILENAME_LEN) // filename charlist
        .build_tuple(2) // file tuple
        .build_tuple(2) // line tuple
        .build_list(2); // metadata list
    let metadata_layout = metadata_layout.finish();

    // The first frame is a bit larger because it may contain arguments
    let mut first_frame_base_layout = LayoutBuilder::new();
    first_frame_base_layout += arguments_layout;
    first_frame_base_layout += metadata_layout;
    first_frame_base_layout.build_tuple(4); // {module, function, arity_or_args, meta}
    let first_frame_base_layout = first_frame_base_layout.finish();

    // Most frames just contain the arity however
    let mut frame_base_layout = LayoutBuilder::new();
    frame_base_layout += metadata_layout;
    frame_base_layout.build_tuple(4); // {module, function, arity, meta}
    let frame_base_layout = frame_base_layout.finish();

    // The list of frames is either an empty list (nil) or a list of frames,
    // where the first frame is potentially larger than the rest
    let mut frame_list_layout = LayoutBuilder::new();
    for i in 0..num_frames {
        if i > 0 {
            frame_list_layout += frame_base_layout;
        } else {
            frame_list_layout += first_frame_base_layout;
        }
        frame_list_layout.build_cons();
    }
    Some(frame_list_layout.finish())
}

pub fn format_mfa<H>(
    mfa: &ModuleFunctionArity,
    argv: Option<&[OpaqueTerm]>,
    filename: Option<&str>,
    line: Option<u32>,
    alloc: &H,
) -> Result<Term, AllocError>
where
    H: Heap,
{
    let module: OpaqueTerm = mfa.module.into();
    let function: OpaqueTerm = mfa.function.into();

    let locs = format_locations(filename, line, alloc)
        .unwrap_or(OpaqueTerm::NIL)
        .into();

    let frame = if let Some(args) = argv {
        let mut builder = ListBuilder::new(alloc);
        for arg in args.iter().rev().copied() {
            builder.push(arg.into())?;
        }
        let arglist: OpaqueTerm = builder.finish().unwrap().into();
        Tuple::from_slice(&[module, function, arglist, locs], alloc).map(Term::Tuple)?
    } else {
        let arity: OpaqueTerm = Term::Int(mfa.arity as i64).into();
        Tuple::from_slice(&[module, function, arity, locs], alloc).map(Term::Tuple)?
    };

    Ok(frame)
}

pub fn format_locations<H>(
    filename: Option<&str>,
    line: Option<u32>,
    alloc: &H,
) -> Result<OpaqueTerm, AllocError>
where
    H: Heap,
{
    // Each location is a pair of: {file, "<path>"}, {line, <line>}
    let file_key = atoms::File.into();
    let line_key = atoms::Line.into();
    let file = if let Some(f) = filename {
        let filename = to_trimmed_charlist(f, alloc).unwrap_or(OpaqueTerm::NIL);
        Tuple::from_slice(&[file_key, filename], alloc).map(Term::Tuple)?
    } else {
        Tuple::from_slice(&[file_key, OpaqueTerm::NIL], alloc).map(Term::Tuple)?
    };
    let line = Term::Int(line.unwrap_or_default().try_into().unwrap());
    let line = Tuple::from_slice(&[line_key, line.into()], alloc).map(Term::Tuple)?;

    let mut builder = ListBuilder::new(alloc);
    builder.push(line)?;
    builder.push(file)?;

    Ok(builder.finish().unwrap().into())
}

pub fn to_trimmed_charlist<H, S>(filename: S, alloc: &H) -> Result<OpaqueTerm, AllocError>
where
    S: AsRef<str>,
    H: Heap,
{
    let f = filename.as_ref();
    let len = f.len();
    if len > MAX_FILENAME_LEN {
        let begin = len - MAX_FILENAME_LEN;
        Ok(Cons::charlist_from_str(&f[begin..], alloc)?
            .map(Term::Cons)
            .unwrap_or(Term::Nil)
            .into())
    } else {
        Ok(Cons::charlist_from_str(f, alloc)?
            .map(Term::Cons)
            .unwrap_or(Term::Nil)
            .into())
    }
}
