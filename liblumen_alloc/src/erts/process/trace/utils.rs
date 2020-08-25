use std::alloc::Layout;
use std::borrow::Cow;
use std::convert::{TryFrom, TryInto};
use std::mem;
use std::path::{Path, PathBuf};

use crate::borrow::CloneToProcess;
use crate::erts::process::{AllocResult, TermAlloc};
use crate::erts::term::prelude::*;
use crate::erts::ModuleFunctionArity;

use super::Trace;

#[derive(Debug, Clone)]
pub struct Symbolication {
    pub(super) mfa: Option<ModuleFunctionArity>,
    pub(super) filename: Option<PathBuf>,
    pub(super) line: Option<u32>,
}
impl Symbolication {
    #[inline]
    pub fn module(&self) -> Option<Atom> {
        self.module_function_arity().map(|mfa| mfa.module)
    }

    #[inline]
    pub fn function(&self) -> Option<Atom> {
        self.module_function_arity().map(|mfa| mfa.function)
    }

    #[inline]
    pub fn arity(&self) -> Option<u8> {
        self.module_function_arity().map(|mfa| mfa.arity)
    }

    #[inline]
    pub fn module_function_arity(&self) -> Option<&ModuleFunctionArity> {
        self.mfa.as_ref()
    }

    #[inline]
    pub fn filename(&self) -> Option<&Path> {
        self.filename.as_ref().map(|p| p.as_path())
    }

    #[inline]
    pub fn line(&self) -> Option<u32> {
        self.line.clone()
    }
}
impl Default for Symbolication {
    fn default() -> Self {
        Self {
            mfa: None,
            filename: None,
            line: None,
        }
    }
}

impl TryFrom<Term> for Symbolication {
    type Error = ();

    fn try_from(term: Term) -> Result<Self, Self::Error> {
        let term = term.decode().map_err(|_| ())?;
        if let Ok(TypedTerm::Tuple(tuple)) = term.try_into() {
            // If this isn't a 4-tuple, then its not a symbolicated frame
            if tuple.len() != 4 {
                return Err(());
            }

            let elements = tuple.elements();
            let module_tt = elements[0].decode().map_err(|_| ())?;
            let function_tt = elements[1].decode().map_err(|_| ())?;
            let arity_or_args = elements[2].decode().map_err(|_| ())?;
            let meta_tt = elements[3].decode().map_err(|_| ())?;

            let module: Atom = module_tt.try_into().map_err(|_| ())?;
            let function: Atom = function_tt.try_into().map_err(|_| ())?;
            let mut symbol = match arity_or_args {
                TypedTerm::SmallInteger(i) => Symbolication {
                    mfa: Some(ModuleFunctionArity {
                        module,
                        function,
                        arity: i.try_into().unwrap(),
                    }),
                    filename: None,
                    line: None,
                },
                TypedTerm::Nil => Symbolication {
                    mfa: Some(ModuleFunctionArity {
                        module,
                        function,
                        arity: 0,
                    }),
                    filename: None,
                    line: None,
                },
                TypedTerm::List(list) => Symbolication {
                    mfa: Some(ModuleFunctionArity {
                        module,
                        function,
                        arity: list.iter().count() as u8,
                    }),
                    filename: None,
                    line: None,
                },
                _ => return Err(()),
            };

            match meta_tt {
                // No location metadata
                TypedTerm::Nil => Ok(symbol),
                TypedTerm::List(list) => {
                    // Get file metadata and convert to path
                    let file = list
                        .keyfind(0, Atom::str_to_term("file"))
                        .ok()
                        .unwrap_or(None);
                    if let Some(TypedTerm::List(chars)) = file.map(|t| t.decode().unwrap()) {
                        symbol.filename = Some(PathBuf::from(format!("{}", chars.as_ref())));
                    }

                    // Get line metadata and convert to u32
                    let line = list
                        .keyfind(0, Atom::str_to_term("line"))
                        .ok()
                        .unwrap_or(None);
                    if let Some(TypedTerm::SmallInteger(i)) = line.map(|t| t.decode().unwrap()) {
                        let i: u32 = i.try_into().unwrap();
                        symbol.line = Some(i);
                    }

                    Ok(symbol)
                }
                // This technically is a bug, but its non-essential info, so ignore it
                _ => Ok(symbol),
            }
        } else {
            Err(())
        }
    }
}

// Minimum frame size is a 4 element tuple with all immediates (i.e. nil for the location info)
pub const MIN_FRAME_SIZE: usize = 5 * mem::size_of::<Term>();
// Base frame size is a 4 element tuple, 2 cons cells, and 2 2-element tuples,
// and 1 character list of up to 120 chars of which each char requires a cons cell
pub const MAX_FILENAME_LEN: usize = 120;
pub const FILENAME_SIZE: usize = MAX_FILENAME_LEN * mem::size_of::<Cons>();
pub const TWO_TUPLE_SIZE: usize = 3 * mem::size_of::<Term>();
pub const BASE_FRAME_SIZE: usize =
    MIN_FRAME_SIZE + (2 * mem::size_of::<Cons>()) + (2 * TWO_TUPLE_SIZE) + FILENAME_SIZE;

/// Calculates the Layout which should contain enough memory to hold the entire
/// stacktrace as an Erlang term. This allows the trace to live separately
/// from the process which raised it
///
/// Each (full) frame looks like so:
///
///     [{module, function, arity, [{file, "path/to/file"}, {line, 1}]}]
///
/// The amount allocated for a trace is calculated by taking the number of frames
/// in the trace, multiplying that by `BASE_FRAME_SIZE`, and then adding on an extra
/// 25% padding to that. If `extra` is specified, that is added as an additional amount
/// above and beyond the padded number.
///
/// The goal here is to allow plenty of room to hold the trace, without the risk of
/// running out of memory while constructing the trace for display and potentially
/// losing the whole trace. Especially on 64-bit systems, the address space is relatively
/// plentiful, so we're better off erring on the side of too much.
pub fn calculate_fragment_layout(num_frames: usize, extra: usize) -> Option<Layout> {
    if num_frames == 0 {
        return None;
    }

    let base_trace_size = num_frames * BASE_FRAME_SIZE;
    let padded_trace_size = base_trace_size + (base_trace_size / 4);
    let total_size = extra + padded_trace_size;
    Layout::from_size_align(total_size, mem::align_of::<Term>()).ok()
}

pub fn format_mfa<A>(
    heap: &mut A,
    mfa: &ModuleFunctionArity,
    argv: Option<&[Term]>,
    filename: Option<&Path>,
    line: Option<u32>,
) -> AllocResult<Term>
where
    A: TermAlloc,
{
    let module = mfa.module.as_term();
    let fun = mfa.function.as_term();
    // If we have arguments for this frame, use them, otherwise use the arity
    let mut frame = if let Some(args) = argv {
        // First allocate space for the arguments
        let mut cells = Vec::with_capacity(args.len());
        for arg in args {
            cells.push(arg.clone_to_heap(heap)?);
        }
        let arglist = heap.list_from_slice(cells.as_slice())?;
        // Then allocate space for the frame
        let mut frame = heap.mut_tuple(4)?;
        frame.set_element(0, module).unwrap();
        frame.set_element(1, fun).unwrap();
        frame.set_element(2, arglist.into()).unwrap();
        frame
    } else {
        let arity = mfa.arity;
        let mut frame = heap.mut_tuple(4)?;
        frame.set_element(0, module).unwrap();
        frame.set_element(1, fun).unwrap();
        frame.set_element(2, arity.into()).unwrap();
        frame
    };

    let locs = format_locations(heap, filename, line).unwrap_or(Term::NIL);
    frame.set_element(3, locs).unwrap();
    Ok(frame.into())
}

pub fn format_locations<A>(
    heap: &mut A,
    filename: Option<&Path>,
    line: Option<u32>,
) -> AllocResult<Term>
where
    A: TermAlloc,
{
    // Each location is a pair of: {file, "<path>"}, {line, <line>}
    let file_atom = Atom::str_to_term("file");
    let line_atom = Atom::str_to_term("line");
    let file = if let Some(f) = filename {
        let filename_list = match f.to_string_lossy() {
            Cow::Borrowed(s) => to_trimmed_charlist(heap, s),
            Cow::Owned(ref s) => to_trimmed_charlist(heap, s),
        };
        heap.tuple_from_slice(&[file_atom, filename_list.unwrap_or(Term::NIL)])?
    } else {
        heap.tuple_from_slice(&[file_atom, Term::NIL])?
    };
    let file = file.into();
    let line_int: SmallInteger = line.unwrap_or_default().into();
    let line = heap.tuple_from_slice(&[line_atom, line_int.into()])?.into();

    Ok(heap.list_from_slice(&[file, line])?.into())
}

pub fn to_trimmed_charlist<A, S>(heap: &mut A, filename: S) -> AllocResult<Term>
where
    A: TermAlloc,
    S: AsRef<str>,
{
    let f = filename.as_ref();
    let len = f.len();
    if len > MAX_FILENAME_LEN {
        let begin = len - MAX_FILENAME_LEN;
        let trimmed = &f[begin..];
        heap.charlist_from_str(&f[begin..]).map(|t| t.into())
    } else {
        heap.charlist_from_str(f).map(|t| t.into())
    }
}
