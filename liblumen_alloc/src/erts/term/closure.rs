use core::alloc::Layout;
use core::cmp::Ordering;
use core::convert::TryFrom;
use core::fmt::{self, Debug, Display, Write};
use core::hash::{Hash, Hasher};
use core::ptr;
use core::slice;

use alloc::sync::Arc;

use crate::borrow::CloneToProcess;
use crate::erts::exception::system::Alloc;
use crate::erts::process::code::stack::frame::{Frame, Placement};
use crate::erts::process::code::Code;
use crate::erts::process::Process;
use crate::erts::term::prelude::{Header, Encoded};
use crate::erts::{self, HeapAlloc, ModuleFunctionArity};

use super::prelude::{TypeError, Term, TypedTerm, Boxed};

#[repr(C)]
pub struct Closure {
    header: Header<Closure>,
    creator: Term, // pid of creator process, possible to be either Pid or ExternalPid
    module_function_arity: Arc<ModuleFunctionArity>,
    /// Pointer to function entry
    pub code: Code,
    pub env_len: usize,
    env: [Term]
}

#[derive(Clone, Copy)]
struct ClosureLayout {
    layout: Layout,
    creator_offset: usize,
    mfa_offset: usize,
    code_offset: usize,
    env_offset: usize,
}
impl ClosureLayout {
    fn base_size() -> usize {
        let (layout, _creator_offset) = Layout::new::<Header<Closure>>()
            .extend(Layout::new::<Term>())
            .unwrap();
        let (layout, _mfa_offset) = layout
            .extend(Layout::new::<Arc<ModuleFunctionArity>>())
            .unwrap();
        let (layout, _code_offset) = layout
            .extend(Layout::new::<Code>())
            .unwrap();
        layout.size()
    }

    fn for_code_and_env(code: &Code, env: &[Term]) -> Self {
        let (layout, creator_offset) = Layout::new::<Header<Closure>>()
            .extend(Layout::new::<Term>())
            .unwrap();
        let (layout, mfa_offset) = layout
            .extend(Layout::new::<Arc<ModuleFunctionArity>>())
            .unwrap();
        let (layout, code_offset) = layout
            .extend(Layout::for_value(code))
            .unwrap();
        let (layout, env_offset) = layout
            .extend(Layout::for_value(env))
            .unwrap();

        let layout = layout.pad_to_align().unwrap();

        Self {
            layout,
            creator_offset,
            mfa_offset,
            code_offset,
            env_offset,
        }
    }

    fn for_code_and_env_len(code: &Code, env_len: usize) -> Self {
        unsafe {
            let ptr = ptr::null_mut() as *mut Term;
            let arr = core::slice::from_raw_parts(ptr as *const (), env_len);
            let env = &*(arr as *const [()] as *mut [Term]);
            Self::for_code_and_env(code, env)
        }
    }
}

impl Closure {
    /// Constructs a new `Closure` with an env of size `len` using `heap`
    ///
    /// The constructed closure will contain an environment of invalid words until
    /// individual elements are written, this is intended for cases where we don't
    /// already have a slice of elements to construct a tuple from
    pub fn new<A>(heap: &mut A, mfa: Arc<ModuleFunctionArity>, code: Code, creator: Term, env_len: usize) -> Result<Boxed<Self>, Alloc> 
    where
        A: ?Sized + HeapAlloc,
    {
        let closure_layout = ClosureLayout::for_code_and_env_len(&code, env_len);
        let layout = closure_layout.layout.clone();

        let header = Header::from_arity(env_len);
        unsafe {
            // Allocate space for closure
            let ptr = heap.alloc_layout(layout)?.as_ptr() as *mut u8;
            let header_ptr = ptr as *mut Header<Self>;
            // Write header
            header_ptr.write(header);
            // Construct pointer to each field and write the corresponding value
            let creator_ptr = ptr.offset(closure_layout.creator_offset as isize) as *mut Term;
            creator_ptr.write(creator);
            let mfa_ptr = ptr.offset(closure_layout.mfa_offset as isize) as *mut Arc<ModuleFunctionArity>;
            mfa_ptr.write(mfa);
            let code_ptr = ptr.offset(closure_layout.code_offset as isize) as *mut Code;
            code_ptr.write(code);
            // Construct actual Tuple reference
            Ok(Self::from_raw_parts(ptr, env_len))
        }
    }

    pub fn from_slice<A>(heap: &mut A, mfa: Arc<ModuleFunctionArity>, code: Code, creator: Term, env: &[Term]) -> Result<Boxed<Self>, Alloc> 
    where
        A: ?Sized + HeapAlloc,
    {
        let closure_layout = ClosureLayout::for_code_and_env(&code, env);
        let layout = closure_layout.layout.clone();

        // The result of calling this will be a Tuple with everything located
        // contiguously in memory
        let arity = env.len();
        let header = Header::from_arity(arity);
        unsafe {
            // Allocate space for tuple and immediate elements
            let ptr = heap.alloc_layout(layout)?.as_ptr() as *mut u8;
            // Write tuple header
            let header_ptr = ptr as *mut Header<Self>;
            header_ptr.write(header);
            // Construct pointer to each field and write the corresponding value
            let creator_ptr = ptr.offset(closure_layout.creator_offset as isize) as *mut Term;
            creator_ptr.write(creator);
            let mfa_ptr = ptr.offset(closure_layout.mfa_offset as isize) as *mut Arc<ModuleFunctionArity>;
            mfa_ptr.write(mfa);
            let code_ptr = ptr.offset(closure_layout.code_offset as isize) as *mut Code;
            code_ptr.write(code);
            // Construct pointer to first env element
            let mut env_ptr = (ptr as *mut u8).offset(closure_layout.env_offset as isize) as *mut Term;
            // Walk original slice of terms and copy them into new memory region,
            // copying boxed terms recursively as necessary
            for element in env {
                if element.is_immediate() {
                    env_ptr.write(*element);
                } else {
                    // Recursively call clone_to_heap, and then write the box header here
                    let boxed = element.clone_to_heap(heap)?;
                    env_ptr.write(boxed);
                }

                env_ptr = env_ptr.offset(1);
            }
            // Construct actual Tuple reference
            Ok(Self::from_raw_parts(ptr as *mut u8, arity))
        }
    }

    #[inline]
    pub fn base_size_words() -> usize {
        erts::to_word_size(ClosureLayout::base_size())
    }

    pub unsafe fn from_raw_term(term: *mut Term) -> Boxed<Self> {
        let header = &*(term as *mut Header<Closure>);
        let arity = header.arity();

        Self::from_raw_parts(term as *const u8, arity)
    }

    #[inline]
    pub(in super) unsafe fn from_raw_parts(ptr: *const u8, len: usize) -> Boxed<Self> {
        // Invariants of slice::from_raw_parts.
        assert!(!ptr.is_null());
        assert!(len <= isize::max_value() as usize);

        let slice = core::slice::from_raw_parts(ptr as *const (), len);
        Boxed::new_unchecked(slice as *const [()] as *mut Self)
    }

    pub fn arity(&self) -> u8 {
        self.module_function_arity.arity
    }

    pub fn frame(&self) -> Frame {
        Frame::new(Arc::clone(&self.module_function_arity), self.code)
    }

    pub fn module_function_arity(&self) -> Arc<ModuleFunctionArity> {
        Arc::clone(&self.module_function_arity)
    }

    pub fn place_frame_with_arguments(
        &self,
        process: &Process,
        placement: Placement,
        arguments: Vec<Term>,
    ) -> Result<(), Alloc> {
        assert_eq!(arguments.len(), self.arity() as usize);
        for argument in arguments.iter().rev() {
            process.stack_push(*argument)?;
        }

        self.push_env_to_stack(process)?;

        process.place_frame(self.frame(), placement);

        Ok(())
    }

    fn push_env_to_stack(&self, process: &Process) -> Result<(), Alloc> {
        for term in self.env.iter().rev() {
            process.stack_push(*term)?;
        }

        Ok(())
    }

    /// Returns the length of the closure environment in terms.
    #[inline]
    pub fn env_len(&self) -> usize {
        self.env.len()
    }

    /// Returns a slice containing the closure environment.
    #[inline]
    pub fn env_slice(&self) -> &[Term] {
        &self.env
    }

    /// Returns a mutable slice containing the closure environment.
    #[inline]
    pub fn env_slice_mut(&mut self) -> &mut [Term] {
        &mut self.env
    }

    /// Iterator over the terms in the closure environment.
    #[inline]
    pub fn env_iter<'a>(&'a self) -> slice::Iter<'a, Term> {
        self.env.iter()
    }

    /// Gets an element from the environment directly by index.
    /// Panics if outside of bounds.
    #[inline]
    pub fn get_env_element(&self, idx: usize) -> Term {
        self.env[idx]
    }
}

impl CloneToProcess for Closure {
    fn clone_to_heap<A>(&self, heap: &mut A) -> Result<Term, Alloc> 
    where
        A: ?Sized + HeapAlloc,
    {
        let mfa = self.module_function_arity.clone();
        let code = self.code;
        let creator = self.creator;
        let ptr = Self::from_slice(heap, mfa, code, creator, &self.env)?;

        Ok(ptr.into())
    }

    fn size_in_words(&self) -> usize {
        erts::to_word_size(Layout::for_value(self).size())
    }
}

impl Debug for Closure {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Closure")
            .field("header", &self.header)
            .field("code", &(self.code as usize))
            .field("module_function_arity", &self.module_function_arity)
            .field("creator", &self.creator)
            .field("env_len", &self.env_len)
            .finish()
    }
}

impl Display for Closure {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let module_function_arity = &self.module_function_arity;

        write!(f, "&{}.", module_function_arity.module)?;
        f.write_char('\"')?;
        module_function_arity
            .function
            .name()
            .chars()
            .flat_map(char::escape_default)
            .try_for_each(|c| f.write_char(c))?;
        f.write_char('\"')?;
        write!(f, "/{}", module_function_arity.arity)
    }
}

impl Eq for Closure {}

impl Hash for Closure {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.module_function_arity.hash(state);
        state.write_usize(self.code as usize);
    }
}

impl Ord for Closure {
    fn cmp(&self, other: &Self) -> Ordering {
        (self.module_function_arity.cmp(&other.module_function_arity))
            .then_with(|| (self.code as usize).cmp(&(other.code as usize)))
    }
}

impl PartialEq for Closure {
    fn eq(&self, other: &Self) -> bool {
        (self.module_function_arity == other.module_function_arity)
            && ((self.code as usize) == (other.code as usize))
    }
}
impl<T> PartialEq<Boxed<T>> for Closure
where
    T: ?Sized + PartialEq<Closure>,
{
    #[inline]
    fn eq(&self, other: &Boxed<T>) -> bool {
        other.as_ref().eq(self)
    }
}

impl PartialOrd for Closure {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl<T> PartialOrd<Boxed<T>> for Closure
where
    T: ?Sized + PartialOrd<Closure>,
{
    #[inline]
    fn partial_cmp(&self, other: &Boxed<T>) -> Option<Ordering> {
        other.as_ref().partial_cmp(self).map(|o| o.reverse())
    }
}

impl TryFrom<TypedTerm> for Boxed<Closure> {
    type Error = TypeError;

    fn try_from(typed_term: TypedTerm) -> Result<Self, Self::Error> {
        match typed_term {
            TypedTerm::Closure(closure) => Ok(closure),
            _ => Err(TypeError),
        }
    }
}
