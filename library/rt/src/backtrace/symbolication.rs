use alloc::alloc::{AllocError, Layout};
use alloc::string::String;
use core::ptr;

use liblumen_alloc::heap::Heap;

use crate::function::ModuleFunctionArity;
use crate::term::*;

#[derive(Default, Debug, Clone)]
pub struct Symbolication {
    pub(super) mfa: Option<ModuleFunctionArity>,
    pub(super) filename: Option<String>,
    pub(super) line: Option<u32>,
}
impl Symbolication {
    #[inline]
    pub fn module(&self) -> Option<Atom> {
        self.mfa.map(|mfa| mfa.module)
    }

    #[inline]
    pub fn function(&self) -> Option<Atom> {
        self.mfa.map(|mfa| mfa.function)
    }

    #[inline]
    pub fn arity(&self) -> Option<u8> {
        self.mfa.map(|mfa| mfa.arity)
    }

    #[inline]
    pub fn filename(&self) -> Option<&str> {
        self.filename.as_deref()
    }

    #[inline]
    pub fn line(&self) -> Option<u32> {
        self.line
    }
}
impl TryFrom<Term> for Symbolication {
    type Error = ();

    fn try_from(term: Term) -> Result<Self, Self::Error> {
        let Term::Tuple(ptr) = term else { return Err(()); };

        let tuple = unsafe { ptr.as_ref() };

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
            Term::Cons(ptr) => unsafe {
                ModuleFunctionArity {
                    module,
                    function,
                    arity: ptr.as_ref().iter().count().try_into().unwrap(),
                }
            },
            _ => return Err(()),
        };

        // The fourth element is optional metadata, including location information
        match tuple[3].into() {
            // No location metadata
            Term::Nil => Ok(Self {
                mfa: Some(mfa),
                filename: None,
                line: None,
            }),
            Term::Cons(ptr) => {
                let list = unsafe { ptr.as_ref() };
                let filename = if let Some(term) = list.keyfind(0, atoms::File).ok().unwrap_or(None)
                {
                    match term.into() {
                        Term::Cons(list) => unsafe { list.as_ref().to_string() },
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
/// ```ignore
///     {module, function, arity, [{file, "path/to/file"}, {line, 1}]}
/// ```
///
/// However, the first frame may optionally contain the arguments, used for certain
/// classes of errors like function_clause errors. In this case, one frame will look
/// like so:
///
/// ```ignore
///     {module, function, [..args], [{file, "path/to/file"}, {line, 1}]}
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
pub fn calculate_fragment_layout(num_frames: usize, arguments: Option<&[Term]>) -> Option<Layout> {
    if num_frames == 0 {
        return None;
    }

    let arguments_layout = match arguments {
        None => Layout::new::<OpaqueTerm>(),
        Some(terms) if terms.is_empty() => Layout::new::<OpaqueTerm>(),
        Some(terms) => {
            let base = Layout::array::<Cons>(terms.len()).unwrap();
            terms.iter().fold(base, |layout, term| {
                let (extended, _) = layout.extend(term.layout()).unwrap();
                extended.pad_to_align()
            })
        }
    };

    let base = min_tuple_layout(4);
    let first_frame_arity_or_args = arguments_layout;
    let arity_or_args = Layout::new::<OpaqueTerm>();
    let meta = Layout::array::<Cons>(2).unwrap();
    let file = min_tuple_layout(2);
    let line = min_tuple_layout(2);
    let filename = Layout::array::<Cons>(MAX_FILENAME_LEN).unwrap();

    let frame_tail_layout = meta
        .extend(file)
        .unwrap()
        .0
        .extend(line)
        .unwrap()
        .0
        .extend(filename)
        .unwrap()
        .0
        .pad_to_align();
    let first_frame_base_layout = base
        .extend(first_frame_arity_or_args)
        .unwrap()
        .0
        .extend(frame_tail_layout)
        .unwrap()
        .0;
    let frame_base_layout = base
        .extend(arity_or_args)
        .unwrap()
        .0
        .extend(frame_tail_layout)
        .unwrap()
        .0;

    let frame_list_layout = if num_frames > 1 {
        Layout::array::<Cons>(num_frames)
            .unwrap()
            .extend(first_frame_base_layout)
            .unwrap()
            .0
            .pad_to_align()
            .extend(frame_base_layout.repeat(num_frames).unwrap().0)
            .unwrap()
            .0
            .pad_to_align()
    } else {
        Layout::array::<Cons>(1)
            .unwrap()
            .extend(first_frame_base_layout)
            .unwrap()
            .0
            .pad_to_align()
    };

    Some(frame_list_layout)
}

#[inline]
fn min_tuple_layout(capacity: usize) -> Layout {
    let ptr: *const Tuple = ptr::from_raw_parts(ptr::null(), capacity);
    unsafe { Layout::for_value_raw(ptr) }
}

pub fn format_mfa<H>(
    mfa: &ModuleFunctionArity,
    argv: Option<&[Term]>,
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
        .unwrap_or(Term::Nil)
        .into();

    let frame = if let Some(args) = argv {
        let mut builder = ListBuilder::new(alloc);
        for arg in args.iter().rev().cloned() {
            builder.push(arg)?;
        }
        let arglist: OpaqueTerm = builder.finish().unwrap().into();
        Tuple::from_slice(&[module, function, arglist, locs], alloc)?
    } else {
        let arity: OpaqueTerm = Term::Int(mfa.arity as i64).into();
        Tuple::from_slice(&[module, function, arity, locs], alloc)?
    };

    Ok(frame.into())
}

pub fn format_locations<H>(
    filename: Option<&str>,
    line: Option<u32>,
    alloc: &H,
) -> Result<Term, AllocError>
where
    H: Heap,
{
    // Each location is a pair of: {file, "<path>"}, {line, <line>}
    let file_key = atoms::File.into();
    let line_key = atoms::Line.into();
    let file = if let Some(f) = filename {
        let filename = to_trimmed_charlist(f, alloc).unwrap_or(OpaqueTerm::NIL);
        Tuple::from_slice(&[file_key, filename], alloc)?
    } else {
        Tuple::from_slice(&[file_key, OpaqueTerm::NIL], alloc)?
    };
    let line = Term::Int(line.unwrap_or_default().try_into().unwrap());
    let line = Tuple::from_slice(&[line_key, line.into()], alloc)?;

    let mut builder = ListBuilder::new(alloc);
    builder.push(line.into())?;
    builder.push(file.into())?;

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
