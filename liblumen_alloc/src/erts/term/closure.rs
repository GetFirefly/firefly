use core::alloc::Layout;
use core::cmp::Ordering;
use core::convert::{TryFrom, TryInto};
use core::fmt::{self, Debug, Display, Write};
use core::hash::{Hash, Hasher};
use core::mem;

use alloc::sync::Arc;

use super::{to_word_size, AsTerm, Term};

use crate::borrow::CloneToProcess;
use crate::erts::exception::system::Alloc;
use crate::erts::process::code::stack::frame::{Frame, Placement};
use crate::erts::process::code::Code;
use crate::erts::process::ProcessControlBlock;
use crate::erts::term::{arity_of, Boxed, TypeError, TypedTerm};
use crate::erts::{HeapAlloc, ModuleFunctionArity};

#[repr(C)]
pub struct Closure {
    header: Term,
    creator: Term, // pid of creator process, possible to be either Pid or ExternalPid
    module_function_arity: Arc<ModuleFunctionArity>,
    code: Code, // pointer to function entry
    pub env_len: usize,
}

impl Closure {
    pub fn new(
        module_function_arity: Arc<ModuleFunctionArity>,
        code: Code,
        creator: Term,
        env_len: usize,
    ) -> Self {
        Self {
            header: Term::make_header(arity_of::<Self>() + env_len, Term::FLAG_CLOSURE),
            creator,
            module_function_arity,
            code,
            env_len,
        }
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
        process: &ProcessControlBlock,
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

    fn push_env_to_stack(&self, process: &ProcessControlBlock) -> Result<(), Alloc> {
        for term in self.env_slice().iter().rev() {
            process.stack_push(*term)?;
        }

        Ok(())
    }

    /// Returns the pointer to the head of the closure environment
    pub fn env_head(&self) -> *const Term {
        unsafe { ((self as *const Closure) as *const Term).add(Self::base_size_words()) }
    }

    /// Returns the length of the closure environment in terms.
    pub fn env_len(&self) -> usize {
        self.env_len
    }

    /// Returns a slice containing the closure environment.
    pub fn env_slice(&self) -> &[Term] {
        unsafe { core::slice::from_raw_parts(self.env_head(), self.env_len()) }
    }

    /// Iterator over the terms in the closure environment.
    pub fn env_iter(&self) -> EnvIter {
        EnvIter::new(self)
    }

    /// Gets an element from the environment directly by index.
    /// Panics if outside of bounds.
    pub fn get_env_element(&self, idx: usize) -> Term {
        self.env_slice()[idx]
    }

    #[inline]
    pub fn layout(env_len: usize) -> Layout {
        let size = Self::need_in_bytes_from_env_len(env_len);
        unsafe { Layout::from_size_align_unchecked(size, mem::align_of::<Term>()) }
    }

    /// The number of bytes for the header and immediate terms or box term pointer to elements
    /// allocated elsewhere.
    pub fn need_in_bytes_from_env_len(env_len: usize) -> usize {
        Closure::base_size() + (env_len * mem::size_of::<Term>())
    }

    /// The number of words for the header and immediate terms or box term pointer to elements
    /// allocated elsewhere.
    pub fn need_in_words_from_env_len(env_len: usize) -> usize {
        to_word_size(Self::need_in_bytes_from_env_len(env_len))
    }

    /// Since we are storing terms directly following the closure header, we need the pointer
    /// following the base to meet the alignment requirements for Term.
    pub fn base_size() -> usize {
        use liblumen_core::alloc::alloc_utils::round_up_to_multiple_of;

        round_up_to_multiple_of(mem::size_of::<Self>(), mem::align_of::<Term>())
    }

    pub fn base_size_words() -> usize {
        to_word_size(Self::base_size())
    }
}

unsafe impl AsTerm for Closure {
    #[inline]
    unsafe fn as_term(&self) -> Term {
        Term::make_boxed(self as *const Self)
    }
}

impl CloneToProcess for Closure {
    fn clone_to_heap<A: HeapAlloc>(&self, heap: &mut A) -> Result<Term, Alloc> {
        unsafe {
            let len = self.env_len();
            let words = Self::need_in_words_from_env_len(len);

            let base_ptr = heap.alloc(words)?.as_ptr() as *mut Term;
            let closure_ptr = base_ptr as *mut Self;
            // Write header
            closure_ptr.write(Closure::new(
                self.module_function_arity.clone(),
                self.code,
                self.creator,
                len,
            ));

            // Write the elements
            let mut element_ptr = base_ptr.offset(Self::base_size_words() as isize);
            for element in self.env_iter() {
                if element.is_immediate() {
                    element_ptr.write(element);
                } else {
                    let boxed = element.clone_to_heap(heap)?;
                    element_ptr.write(boxed);
                }

                element_ptr = element_ptr.offset(1);
            }

            Ok(Term::make_boxed(closure_ptr))
        }
    }
    fn size_in_words(&self) -> usize {
        Self::need_in_words_from_env_len(self.env_len())
    }
}

impl Debug for Closure {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Closure")
            .field("header", &format_args!("{:#b}", &self.header.as_usize()))
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

impl PartialOrd for Closure {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl TryFrom<Term> for Boxed<Closure> {
    type Error = TypeError;

    fn try_from(term: Term) -> Result<Self, Self::Error> {
        term.to_typed_term().unwrap().try_into()
    }
}

impl TryFrom<TypedTerm> for Boxed<Closure> {
    type Error = TypeError;

    fn try_from(typed_term: TypedTerm) -> Result<Self, Self::Error> {
        match typed_term {
            TypedTerm::Boxed(boxed_closure) => boxed_closure.to_typed_term().unwrap().try_into(),
            TypedTerm::Closure(closure) => Ok(closure),
            _ => Err(TypeError),
        }
    }
}

pub struct EnvIter {
    pointer: *const Term,
    limit: *const Term,
}

impl EnvIter {
    pub fn new(closure: &Closure) -> Self {
        let pointer = closure.env_head();
        let limit = unsafe { pointer.add(closure.env_len()) };

        Self { pointer, limit }
    }
}

impl Iterator for EnvIter {
    type Item = Term;

    fn next(&mut self) -> Option<Term> {
        if self.pointer == self.limit {
            None
        } else {
            let old_pointer = self.pointer;

            unsafe {
                self.pointer = self.pointer.add(1);
                old_pointer.as_ref().map(|r| *r)
            }
        }
    }
}

impl DoubleEndedIterator for EnvIter {
    fn next_back(&mut self) -> Option<Term> {
        if self.pointer == self.limit {
            None
        } else {
            unsafe {
                // limit is +1 past he actual elements, so pre-decrement unlike `next`, which
                // post-decrements
                self.limit = self.limit.offset(-1);
                self.limit.as_ref().map(|r| *r)
            }
        }
    }
}
