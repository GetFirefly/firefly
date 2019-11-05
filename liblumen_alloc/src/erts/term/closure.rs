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
pub use crate::erts::module_function_arity::Arity;
use crate::erts::process::code::stack::frame::{Frame, Placement};
use crate::erts::process::code::Code;
use crate::erts::process::Process;
use crate::erts::term::{arity_of, Atom, Boxed, ExternalPid, Pid, TypeError, TypedTerm};
use crate::erts::{HeapAlloc, ModuleFunctionArity};

#[derive(Clone)]
#[repr(C)]
pub struct Closure {
    header: Term,
    pub module: Atom,
    pub definition: Definition,
    pub arity: u8,
    /// Pointer to function entry.  When a closure is received over ETF, `code` may be `None`.
    option_code: Option<Code>,
}

impl Closure {
    pub fn anonymous(
        module: Atom,
        index: Index,
        old_unique: OldUnique,
        unique: Unique,
        arity: Arity,
        env_len: usize,
        option_code: Option<Code>,
        creator: Creator,
    ) -> Self {
        Self {
            header: Self::anonymous_header(env_len),
            module,
            definition: Definition::Anonymous {
                index,
                unique,
                old_unique,
                creator,
                env_len,
            },
            arity,
            option_code,
        }
    }

    pub fn export(module: Atom, function: Atom, arity: Arity, option_code: Option<Code>) -> Self {
        Self {
            header: Self::export_header(),
            module,
            definition: Definition::Export { function },
            arity,
            option_code,
        }
    }

    pub fn code(&self) -> Code {
        self.option_code.unwrap_or_else(|| {
            panic!(
                "{} does not have code associated with it",
                self.module_function_arity()
            )
        })
    }

    pub fn frame(&self) -> Frame {
        let mfa = self.module_function_arity();
        Frame::from_definition(mfa.module, self.definition.clone(), mfa.arity, self.code())
    }

    pub fn module_function_arity(&self) -> Arc<ModuleFunctionArity> {
        Arc::new(ModuleFunctionArity {
            module: self.module,
            function: self.function(),
            arity: self.arity,
        })
    }

    pub fn place_frame_with_arguments(
        &self,
        process: &Process,
        placement: Placement,
        arguments: Vec<Term>,
    ) -> Result<(), Alloc> {
        assert_eq!(arguments.len(), self.arity as usize);
        for argument in arguments.iter().rev() {
            process.stack_push(*argument)?;
        }

        self.push_env_to_stack(process)?;

        process.place_frame(self.frame(), placement);

        Ok(())
    }

    fn push_env_to_stack(&self, process: &Process) -> Result<(), Alloc> {
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
        match self.definition {
            Definition::Export { .. } => 0,
            Definition::Anonymous { env_len, .. } => env_len,
        }
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

    // Private

    fn anonymous_header(env_len: usize) -> Term {
        Term::make_header(arity_of::<Self>() + env_len, Term::FLAG_CLOSURE)
    }

    fn export_header() -> Term {
        Self::anonymous_header(0)
    }

    fn function(&self) -> Atom {
        self.definition.function_name()
    }

    fn option_code_address(&self) -> Option<usize> {
        self.option_code.map(|code| code as usize)
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
            closure_ptr.write(self.clone());

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
            .field("module", &self.module)
            .field("definition", &self.definition)
            .field("arity", &self.arity)
            .field("option_code", &self.option_code_address())
            .finish()
    }
}

impl Display for Closure {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let module_function_arity = &self.module_function_arity();

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
        self.module.hash(state);
        self.definition.hash(state);
        self.arity.hash(state);
        self.option_code_address().hash(state);
        self.env_slice().hash(state);
    }
}

impl Ord for Closure {
    fn cmp(&self, other: &Self) -> Ordering {
        self.module
            .cmp(&other.module)
            .then_with(|| self.definition.cmp(&other.definition))
            .then_with(|| self.arity.cmp(&other.arity))
            .then_with(|| self.option_code_address().cmp(&other.option_code_address()))
            .then_with(|| self.env_slice().cmp(other.env_slice()))
    }
}

impl PartialEq for Closure {
    fn eq(&self, other: &Self) -> bool {
        (self.module == other.module)
            && (self.definition == other.definition)
            && (self.arity == other.arity)
            && (self.option_code_address() == other.option_code_address())
            && (self.env_slice() == other.env_slice())
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

#[derive(Clone)]
pub enum Creator {
    Local(Pid),
    External(ExternalPid),
}

impl Debug for Creator {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Local(pid) => write!(f, "{:?}", pid),
            Self::External(external_pid) => write!(f, "{:?}", external_pid),
        }
    }
}

impl From<Pid> for Creator {
    fn from(pid: Pid) -> Self {
        Self::Local(pid)
    }
}

impl From<ExternalPid> for Creator {
    fn from(external_pid: ExternalPid) -> Self {
        Self::External(external_pid)
    }
}

#[derive(Clone, Debug)]
pub enum Definition {
    /// External functions captured with `fun M:F/A` in Erlang or `&M.f/a` in Elixir.
    Export { function: Atom },
    /// Anonymous functions declared with `fun` in Erlang or `fn` in Elixir.
    Anonymous {
        /// Each anonymous function within a module has an unique index.
        index: u32,
        /// The 16 bytes MD5 of the significant parts of the Beam file.
        unique: [u8; 16],
        /// The hash value of the parse tree for the fun, but must fit in i32, so not the same as
        /// `unique`.
        old_unique: u32,
        creator: Creator,
        env_len: usize,
    },
}

impl Definition {
    pub fn function_name(&self) -> Atom {
        match self {
            Definition::Export { function } => *function,
            Definition::Anonymous {
                index,
                old_unique,
                unique,
                ..
            } => Atom::try_from_str(format!(
                "{}-{}-{}",
                index,
                old_unique,
                Self::format_unique(&unique)
            ))
            .unwrap(),
        }
    }

    fn format_unique(unique: &[u8; 16]) -> String {
        let mut string = String::with_capacity(unique.len() * 2);

        for byte in unique {
            string.push_str(&format!("{:02x}", byte));
        }

        string
    }
}

impl Eq for Definition {}

impl Hash for Definition {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            Definition::Export { function } => function.hash(state),
            Definition::Anonymous {
                index,
                unique,
                old_unique,
                ..
            } => {
                index.hash(state);
                unique.hash(state);
                old_unique.hash(state);
            }
        }
    }
}

impl PartialEq for Definition {
    fn eq(&self, other: &Definition) -> bool {
        match (self, other) {
            (
                Definition::Export {
                    function: self_function,
                },
                Definition::Export {
                    function: other_function,
                },
            ) => self_function == other_function,
            (
                Definition::Anonymous {
                    index: self_index,
                    unique: self_unique,
                    old_unique: self_old_unique,
                    ..
                },
                Definition::Anonymous {
                    index: other_index,
                    unique: other_unique,
                    old_unique: other_old_unique,
                    ..
                },
            ) => {
                (self_index == other_index)
                    && (self_unique == other_unique)
                    && (self_old_unique == other_old_unique)
            }
            _ => false,
        }
    }
}

impl PartialOrd for Definition {
    fn partial_cmp(&self, other: &Definition) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Definition {
    fn cmp(&self, other: &Definition) -> Ordering {
        match (self, other) {
            (
                Definition::Export {
                    function: self_function,
                },
                Definition::Export {
                    function: other_function,
                },
            ) => self_function.cmp(other_function),
            (Definition::Export { .. }, Definition::Anonymous { .. }) => Ordering::Greater,
            (Definition::Anonymous { .. }, Definition::Export { .. }) => Ordering::Less,
            (
                Definition::Anonymous {
                    index: self_index,
                    unique: self_unique,
                    old_unique: self_old_unique,
                    ..
                },
                Definition::Anonymous {
                    index: other_index,
                    unique: other_unique,
                    old_unique: other_old_unique,
                    ..
                },
            ) => self_index
                .cmp(other_index)
                .then_with(|| self_unique.cmp(other_unique))
                .then_with(|| self_old_unique.cmp(other_old_unique)),
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
                // limit is +1 past the actual elements, so pre-decrement unlike `next`, which
                // post-decrements
                self.limit = self.limit.offset(-1);
                self.limit.as_ref().map(|r| *r)
            }
        }
    }
}

/// Index of anonymous function in module's function table
pub type Index = u32;

/// Hash of the parse of the function.  Replaced by `Unique`
pub type OldUnique = u32;

/// 16 byte MD5 of the significant parts of the BEAM file.
pub type Unique = [u8; 16];
