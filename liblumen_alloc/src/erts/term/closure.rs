use core::cmp::Ordering;
use core::convert::{TryFrom, TryInto};
use core::fmt::{self, Debug, Display, Write};
use core::hash::{Hash, Hasher};
use core::mem;
use core::ptr;

use alloc::sync::Arc;
use alloc::vec::Vec;

use super::{AsTerm, Term};

use crate::borrow::CloneToProcess;
use crate::erts::exception::system::Alloc;
use crate::erts::process::code::stack::frame::{Frame, Placement};
use crate::erts::process::code::Code;
use crate::erts::process::ProcessControlBlock;
use crate::erts::term::{arity_of, Boxed, TypeError, TypedTerm};
use crate::erts::{HeapAlloc, ModuleFunctionArity};

pub struct Closure {
    header: Term,
    creator: Term, // pid of creator process, possible to be either Pid or ExternalPid
    module_function_arity: Arc<ModuleFunctionArity>,
    code: Code, // pointer to function entry
    env: Vec<Term>,
}

impl Closure {
    pub fn new(
        module_function_arity: Arc<ModuleFunctionArity>,
        code: Code,
        creator: Term,
        env: Vec<Term>,
    ) -> Self {
        Self {
            header: Term::make_header(arity_of::<Self>(), Term::FLAG_CLOSURE),
            creator,
            module_function_arity,
            code,
            env,
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
        for term in self.env.iter().rev() {
            process.stack_push(*term)?;
        }

        Ok(())
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
}

impl Debug for Closure {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Closure")
            .field("header", &format_args!("{:#b}", &self.header.as_usize()))
            .field("code", &(self.code as usize))
            .field("module_function_arity", &self.module_function_arity)
            .field("creator", &self.creator)
            .field("env", &self.env)
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
