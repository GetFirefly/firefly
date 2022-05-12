use alloc::alloc::{Allocator, AllocErr, Layout};
use alloc::borrow::Cow;

use core::mem;

use crate::term::*;
use crate::function::ModuleFunctionArity;

#[derive(Default, Debug, Clone)]
pub struct Symbolication {
    pub(super) mfa: Option<ModuleFunctionArity>,
    pub(super) filename: Option<String>,
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
    pub fn filename(&self) -> Option<&str> {
        self.filename.as_deref()
    }

    #[inline]
    pub fn line(&self) -> Option<u32> {
        self.line.copied()
    }
}
impl TryFrom<OpaqueTerm> for Symbolication {
    type Error = ();

    fn try_from(term: OpaqueTerm) -> Result<Self, Self::Error> {
        let Ok(Term::Tuple(ptr)) = term.into() else { return Err(()); };

        let tuple = ptr.as_ref();

        // If this tuple doesn't have 4 elements, then it is not a symbolicated frame
        if tuple.len() != 4 {
            return Err(());
        }

        let Term::Atom(module) = tuple[0] else { return Err(()); };
        let Term::Atom(function) = tuple[1] else { return Err(()); };

        // The third element is either the function arity or its argument list
        let mfa = match tuple[2] {
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
            Term::List(ptr) => ModuleFunctionArity {
                    module,
                    function,
                    arity: ptr.as_ref().iter().count().try_into().unwrap(),
                },
            _ => return Err(()),
        };

        // The fourth element is optional metadata, including location information
        match tuple[3] {
            // No location metadata
            Term::Nil => Ok(Self {
                mfa: Some(mfa),
                filename: None,
                line: None,
            }),
            Term::List(ptr) => {
                let list = ptr.as_ref();
                let filename = if let Some(term) = list.keyfind(0, Atom::from("file")).ok().unwrap_or(None) {
                    term.try_into().ok()
                } else {
                    None
                };
                let line = if let Some(Term::Int(line)) = list.keyfind(0, Atom::from("line")).ok().unwrap_or(None) {
                    Some(line.try_into().unwrap())
                } else {
                    None
                };
                Ok(Self {
                    mfa: Some(mfa),
                    filename,
                    line,
                })
            }
            // Technically a bug, but it is optional info, so we ignore it in the
            // interest of getting usable traces
            _ => Ok(Self {
                mfa: Some(mfa),
                filename: None,
                line: None,
            })
        }
    }
}

pub const MAX_FILENAME_LEN: usize = 120;

/// Calculates the Layout which should contain enough memory to hold the entire
/// stacktrace as an Erlang term. This allows the trace to live separately
/// from the process which raised it
///
/// Each frame looks like so:
///
/// ```ignore
///     {module, function, arity, [{file, "path/to/file"}, {line, 1}]}
/// ```
///
/// The frames themselves are contained in a list.
///
/// To calculate the total heap needed to allocate all of the terms, we define a heap
/// layout something like the following:
///
/// ```ignore
/// struct Frames {
///   list: [Cons<Frame>; NUM_FRAMES], // the cons cells for each frame, references `data`
///   data: [Frame; NUM_FRAMES], // the data for each frame
/// }
///
/// struct Frame {
///   frame: Tuple<4>, // references `meta`
///   meta: [Cons<Tuple>; 2], // cons cells for `file` and `line`
///   file: Tuple<2>, // references data in `filename`
///   line: Tuple<2>, // both elements are immediate
///   filename: [Cons<Int>; MAX_FILENAME_LEN],
/// }
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
pub fn calculate_fragment_layout(num_frames: usize) -> Option<Layout> {
    const MIN_FRAME_SIZE: usize = 4;

    if num_frames == 0 {
        return None;
    }

    let frame_layout = min_tuple_layout(4)
        .extend(Layout::array::<Cons>(2))
        .unwrap().0.extend(min_tuple_layout(2))
        .unwrap().0.extend(min_tuple_layout(2))
        .unwrap().0.extend(Layout::array::<Cons>(MAX_FILENAME_LEN))
        .unwrap().0
        .pad_to_align();

    let frame_list_layout = Layout::array::<Cons>(num_frames)
        .extend(frame_layout.repeat(num_frames).unwrap().0)
        .unwrap().0
        .pad_to_align();

    Some(frame_list_layout)
}

#[inline]
fn min_tuple_layout(capacity: usize) -> Layout {
    let ptr: *const Tuple = ptr::from_raw_parts(ptr::null(), capacity);
    unsafe { Layout::for_value_raw(ptr) }
}

pub fn format_mfa<A: Allocator>(
    mfa: &ModuleFunctionArity,
    argv: Option<&[Term]>,
    filename: Option<&str>,
    line: Option<u32>,
    alloc: &mut A
) -> Result<Term, AllocError> {
    let module: OpaqueTerm = mfa.module.into();
    let function: OpaqueTerm = mfa.function.into();

    let locs = format_locations(filename, line, alloc).unwrap_or(OpaqueTerm::NIL);

    let mut frame = if let Some(args) = argv {
        let mut builder = ListBuilder::new(alloc);
        for arg in args.iter().rev().cloned() {
            builder.push(arg)?;
        }
        let arglist: OpaqueTerm = builder.finish().unwrap().into();
        Tuple::from_slice(&[module, function, arglist, locs], alloc)?
    } else {
        let arity: OpaqueTerm = mfa.arity.into();
        Tuple::from_slice(&[module, function, arity, locs], alloc)?
    }

    Ok(frame.into())
}

pub fn format_locations<A>(
    filename: Option<&str>,
    line: Option<u32>,
    alloc: &mut A,
) -> Result<Term, AllocError>
where
    A: Allocator,
{
    // Each location is a pair of: {file, "<path>"}, {line, <line>}
    let file_key = Atom::str_to_term("file");
    let line_key = Atom::str_to_term("line");
    let file = if let Some(f) = filename {
        let filename = to_trimmed_charlist(f, alloc).unwrap_or(OpaqueTerm::NIL);
        Tuple::from_slice(&[file_key, filename], alloc)?
    } else {
        Tuple::from_slice(&[file_key, OpaqueTerm::NIL], alloc)?
    };
    let line: Term = line.unwrap_or_default().try_into().unwrap();
    let line = Tuple::from_slice(&[line_key, line.into()], alloc)?;

    let mut builder = ListBuilder::new(alloc);
    builder.push(line);
    builder.push(file);

    Ok(builder.finish().unwrap().into())
}

pub fn to_trimmed_charlist<A, S>(filename: S, alloc: &mut A) -> Result<OpaqueTerm, AllocError>
where
    S: AsRef<str>,
    A: Allocator,
{
    let f = filename.as_ref();
    let len = f.len();
    if len > MAX_FILENAME_LEN {
        let begin = len - MAX_FILENAME_LEN;
        let trimmed = &f[begin..];
        Ok(Cons::charlist_from_str(&f[begin..], alloc)?.into())
    } else {
        Ok(Cons::charlist_from_str(f)?.into())
    }
}
