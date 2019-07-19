use core::alloc::AllocErr;
use core::cmp::Ordering;
use core::fmt;
use core::hash::{Hash, Hasher};
use core::mem;
use core::ptr;

use alloc::sync::Arc;

use super::{AsTerm, Term};

use crate::borrow::CloneToProcess;
use crate::erts::process::code::stack::frame::Frame;
use crate::erts::process::code::Code;
use crate::erts::{to_word_size, HeapAlloc, ModuleFunctionArity};

pub struct Closure {
    header: Term,
    creator: Term, // pid of creator process, possible to be either Pid or ExternalPid
    module_function_arity: Arc<ModuleFunctionArity>,
    code: Code,     // pointer to function entry
    next: *mut u8,  // off heap header
    env_len: usize, // the number of free variables
    env: *mut Term, // pointer to first element of free variable array
}

impl Closure {
    // The size of the non-header fields in bytes
    const ARITYVAL: usize = mem::size_of::<Self>() - mem::size_of::<Term>();

    pub fn new(module_function_arity: Arc<ModuleFunctionArity>, code: Code, creator: Term) -> Self {
        Self {
            header: Term::make_header(Self::ARITYVAL, Term::FLAG_CLOSURE),
            creator,
            module_function_arity,
            code,
            next: ptr::null_mut(),
            env_len: 0,
            env: ptr::null_mut(),
        }
    }

    pub fn module_function_arity(&self) -> Arc<ModuleFunctionArity> {
        Arc::clone(&self.module_function_arity)
    }

    pub fn frame(&self) -> Frame {
        Frame::new(Arc::clone(&self.module_function_arity), self.code)
    }
}

unsafe impl AsTerm for Closure {
    #[inline]
    unsafe fn as_term(&self) -> Term {
        Term::make_boxed(self as *const Self)
    }
}

impl CloneToProcess for Closure {
    fn clone_to_heap<A: HeapAlloc>(&self, heap: &mut A) -> Result<Term, AllocErr> {
        // Allocate space on process heap
        let words = self.size_in_words();
        let bytes = words * mem::size_of::<usize>();
        unsafe {
            let ptr = heap.alloc(words)?.as_ptr();
            // Copy to newly allocated region
            ptr::copy_nonoverlapping(self as *const _ as *const u8, ptr as *mut u8, bytes);
            // Return term
            Ok(Term::make_boxed(ptr))
        }
    }

    fn size_in_words(&self) -> usize {
        let mut size = to_word_size(mem::size_of::<Self>());
        for offset in 0..self.env_len {
            let ptr = unsafe { self.env.offset(offset as isize) };
            let term = unsafe { &*ptr };
            size += term.size_in_words()
        }
        size
    }
}

impl fmt::Debug for Closure {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Closure")
            .field("header", &format_args!("{:#b}", &self.header.as_usize()))
            .field("code", &(self.code as usize))
            .field("next", &self.next)
            .field("module_function_arity", &self.module_function_arity)
            .field("creator", &self.creator)
            .field("env_len", &self.env_len)
            .field("env", &self.env)
            .finish()
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
