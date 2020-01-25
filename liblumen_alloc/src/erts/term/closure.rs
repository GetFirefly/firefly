use core::alloc::Layout;
use core::cmp::Ordering;
use core::convert::TryFrom;
use core::fmt::{self, Debug, Display};
use core::hash::{Hash, Hasher};
use core::ptr;
use core::slice;

use crate::borrow::CloneToProcess;
use crate::erts::exception::AllocResult;
use crate::erts::process::alloc::{Heap, TermAlloc};
use crate::erts::process::code::stack::frame::{Frame, Placement};
use crate::erts::process::code::LocatedCode;
use crate::erts::process::Process;
use crate::erts::{self, Arity};

use super::prelude::*;

#[repr(C)]
pub struct Closure {
    header: Header<Closure>,
    module: Atom,
    definition: Definition,
    arity: Arity,
    /// Pointer to function entry with its location for stacktraces.  When a closure is received
    /// over ETF, `located_code` may be `None`.
    located_code: Option<LocatedCode>,
    env: [Term],
}
impl_dynamic_header!(Closure, Term::HEADER_CLOSURE);

#[derive(Debug, Clone, Copy)]
struct ClosureLayout {
    layout: Layout,
    module_offset: usize,
    definition_offset: usize,
    arity_offset: usize,
    code_offset: usize,
    env_offset: usize,
}
impl ClosureLayout {
    fn base_size() -> usize {
        let (layout, _module_offset) = Layout::new::<Header<Closure>>()
            .extend(Layout::new::<Atom>())
            .unwrap();
        let (layout, _definition_offset) = layout.extend(Layout::new::<Definition>()).unwrap();
        let (layout, _arity_offset) = layout.extend(Layout::new::<usize>()).unwrap();
        let (layout, _code_offset) = layout.extend(Layout::new::<Option<LocatedCode>>()).unwrap();
        layout.size()
    }

    fn for_env(env: &[Term]) -> Self {
        let (layout, module_offset) = Layout::new::<Header<Closure>>()
            .extend(Layout::new::<Atom>())
            .unwrap();
        let (layout, definition_offset) = layout.extend(Layout::new::<Definition>()).unwrap();
        let (layout, arity_offset) = layout.extend(Layout::new::<usize>()).unwrap();
        let (layout, code_offset) = layout.extend(Layout::new::<Option<LocatedCode>>()).unwrap();
        let (layout, env_offset) = layout.extend(Layout::for_value(env)).unwrap();

        let layout = layout.pad_to_align();

        Self {
            layout,
            module_offset,
            definition_offset,
            arity_offset,
            code_offset,
            env_offset,
        }
    }

    fn for_env_len(env_len: usize) -> Self {
        unsafe {
            let ptr = ptr::null_mut() as *mut Term;
            let arr = core::slice::from_raw_parts(ptr as *const (), env_len);
            let env = &*(arr as *const [()] as *mut [Term]);
            Self::for_env(env)
        }
    }
}
impl Closure {
    /// Constructs a new `Closure` with a definition, with an env of size `len` using
    /// `heap`.
    ///
    /// The constructed closure will contain an environment of invalid words until
    /// individual elements are written, this is intended for cases where we don't
    /// already have a slice of elements to construct a tuple from
    pub fn new<A>(
        heap: &mut A,
        module: Atom,
        definition: Definition,
        arity: Arity,
        located_code: Option<LocatedCode>,
        env_len: usize,
    ) -> AllocResult<Boxed<Self>>
    where
        A: ?Sized + Heap,
    {
        let closure_layout = ClosureLayout::for_env_len(env_len);
        let layout = closure_layout.layout.clone();

        let header = Header::from_arity(env_len);
        unsafe {
            // Allocate space for closure
            let ptr = heap.alloc_layout(layout)?.as_ptr() as *mut u8;
            let header_ptr = ptr as *mut Header<Self>;
            // Write header
            header_ptr.write(header);
            // Construct pointer to each field and write the corresponding value
            let module_ptr = ptr.offset(closure_layout.module_offset as isize) as *mut Atom;
            module_ptr.write(module);
            let definition_ptr =
                ptr.offset(closure_layout.definition_offset as isize) as *mut Definition;
            definition_ptr.write(definition);
            let arity_ptr = ptr.offset(closure_layout.arity_offset as isize) as *mut Arity;
            arity_ptr.write(arity);
            let located_code_ptr =
                ptr.offset(closure_layout.code_offset as isize) as *mut Option<LocatedCode>;
            located_code_ptr.write(located_code);
            // Construct actual Closure reference
            Ok(Self::from_raw_parts::<Term>(ptr as *mut Term, env_len))
        }
    }

    pub fn from_slice<A>(
        heap: &mut A,
        module: Atom,
        index: Index,
        old_unique: OldUnique,
        unique: Unique,
        arity: Arity,
        located_code: Option<LocatedCode>,
        creator: Creator,
        env: &[Term],
    ) -> AllocResult<Boxed<Self>>
    where
        A: ?Sized + TermAlloc,
    {
        let definition = Definition::Anonymous {
            index,
            unique,
            old_unique,
            creator,
        };

        Self::new_from_slice(heap, module, definition, arity, located_code, env)
    }

    fn new_from_slice<A>(
        heap: &mut A,
        module: Atom,
        definition: Definition,
        arity: Arity,
        located_code: Option<LocatedCode>,
        env: &[Term],
    ) -> AllocResult<Boxed<Self>>
    where
        A: ?Sized + TermAlloc,
    {
        let closure_layout = ClosureLayout::for_env(env);
        let layout = closure_layout.layout.clone();

        // The result of calling this will be a Closure with everything located
        // contiguously in memory
        let env_arity = env.len();
        let header = Header::from_arity(env_arity);
        unsafe {
            // Allocate space for tuple and immediate elements
            let ptr = heap.alloc_layout(layout)?.as_ptr() as *mut u8;
            // Write tuple header
            let header_ptr = ptr as *mut Header<Self>;
            header_ptr.write(header);
            // Construct pointer to each field and write the corresponding value
            let module_ptr = ptr.offset(closure_layout.module_offset as isize) as *mut Atom;
            module_ptr.write(module);
            let definition_ptr =
                ptr.offset(closure_layout.definition_offset as isize) as *mut Definition;
            definition_ptr.write(definition);
            let arity_ptr = ptr.offset(closure_layout.arity_offset as isize) as *mut Arity;
            arity_ptr.write(arity);
            let located_code_ptr =
                ptr.offset(closure_layout.code_offset as isize) as *mut Option<LocatedCode>;
            located_code_ptr.write(located_code);
            // Construct pointer to first env element
            let mut env_ptr = ptr.offset(closure_layout.env_offset as isize) as *mut Term;
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
            // Construct actual Closure reference
            Ok(Self::from_raw_parts::<Term>(ptr as *mut Term, env_arity))
        }
    }

    #[inline]
    pub fn base_size_words() -> usize {
        erts::to_word_size(ClosureLayout::base_size())
    }

    #[inline]
    pub fn definition(&self) -> &Definition {
        &self.definition
    }

    #[inline]
    pub fn arity(&self) -> Arity {
        self.arity
    }

    pub fn located_code(&self) -> LocatedCode {
        self.located_code
            .unwrap_or_else(|| panic!("{} does not have code associated with it", self))
    }

    pub fn frame(&self) -> Frame {
        let LocatedCode { code, location } = self.located_code();

        Frame::from_definition(
            self.module,
            self.definition.clone(),
            self.arity,
            location,
            code,
        )
    }

    pub fn place_frame_with_arguments(
        &self,
        process: &Process,
        placement: Placement,
        arguments: Vec<Term>,
    ) -> AllocResult<()> {
        assert_eq!(arguments.len(), self.arity() as usize);
        for argument in arguments.iter().rev() {
            process.stack_push(*argument)?;
        }

        self.push_env_to_stack(process)?;

        process.place_frame(self.frame(), placement);

        Ok(())
    }

    fn push_env_to_stack(&self, process: &Process) -> AllocResult<()> {
        for term in self.env.iter().rev() {
            process.stack_push(*term)?;
        }

        Ok(())
    }

    #[inline]
    pub fn module(&self) -> Atom {
        self.module
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

    /// Given a raw pointer to some memory, and a length in units of `Self::Element`,
    /// this function produces a fat pointer to `Self` which represents a value
    /// containing `len` elements in its variable-length field
    ///
    /// For example, given a pointer to the memory containing a `Closure`, and the
    /// number of elements it contains, this function produces a valid pointer of
    /// type `Closure`
    unsafe fn from_raw_parts<E: super::arch::Repr>(ptr: *const E, len: usize) -> Boxed<Closure> {
        // Invariants of slice::from_raw_parts.
        assert!(!ptr.is_null());
        assert!(len <= isize::max_value() as usize);

        let slice = core::slice::from_raw_parts_mut(ptr as *mut E, len);
        let ptr = slice as *mut [E] as *mut Closure;
        Boxed::new_unchecked(ptr)
    }
}

impl<E: crate::erts::term::arch::Repr> Boxable<E> for Closure {}

impl<E: super::arch::Repr> UnsizedBoxable<E> for Closure {
    unsafe fn from_raw_term(ptr: *mut E) -> Boxed<Closure> {
        let header = &*(ptr as *mut Header<Closure>);
        let arity = header.arity();

        Self::from_raw_parts::<E>(ptr, arity)
    }
}

impl CloneToProcess for Closure {
    fn clone_to_heap<A>(&self, heap: &mut A) -> AllocResult<Term>
    where
        A: ?Sized + TermAlloc,
    {
        let module = self.module.clone();
        let definition = self.definition.clone();
        let arity = self.arity;
        let located_code = self.located_code.clone();
        let ptr = Self::new_from_slice(
            heap,
            module,
            definition,
            arity as u8,
            located_code,
            &self.env,
        )?;

        Ok(ptr.into())
    }

    fn size_in_words(&self) -> usize {
        let mut size = erts::to_word_size(Layout::for_value(self).size());
        for element in &self.env {
            size += element.size_in_words()
        }
        size
    }
}

impl Debug for Closure {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Closure")
            .field("header", &self.header)
            .field("module", &self.module)
            .field("definition", &self.definition)
            .field("arity", &self.arity)
            .field("located_code", &self.located_code)
            .field("env_len", &self.env.len())
            .field("env", &self.env.iter().copied().collect::<Vec<Term>>())
            .finish()
    }
}

impl Display for Closure {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "&{}.{}/{}", self.module, self.definition, self.arity)
    }
}

impl Eq for Closure {}

impl Hash for Closure {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.module.hash(state);
        self.definition.hash(state);
        self.arity.hash(state);
        self.located_code.hash(state);
        self.env_slice().hash(state);
    }
}

impl Ord for Closure {
    fn cmp(&self, other: &Self) -> Ordering {
        self.module
            .cmp(&other.module)
            .then_with(|| self.definition.cmp(&other.definition))
            .then_with(|| self.arity.cmp(&other.arity))
            .then_with(|| self.located_code.cmp(&other.located_code))
            .then_with(|| self.env_slice().cmp(other.env_slice()))
    }
}

impl PartialEq for Closure {
    fn eq(&self, other: &Self) -> bool {
        (self.module == other.module)
            && (self.definition == other.definition)
            && (self.arity == other.arity)
            && (self.located_code == other.located_code)
            && (self.env_slice() == other.env_slice())
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
    },
}

impl Definition {
    fn format_unique(unique: &[u8; 16]) -> String {
        let mut string = String::with_capacity(unique.len() * 2);

        for byte in unique {
            string.push_str(&format!("{:02x}", byte));
        }

        string
    }
}

impl Display for Definition {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Export { function } => write!(f, "{}", function),
            Self::Anonymous {
                index,
                unique,
                old_unique,
                ..
            } => write!(
                f,
                "{}-{}-{}",
                index,
                Self::format_unique(unique),
                old_unique
            ),
        }
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

/// Index of anonymous function in module's function table
pub type Index = u32;

/// Hash of the parse of the function.  Replaced by `Unique`
pub type OldUnique = u32;

/// 16 byte MD5 of the significant parts of the BEAM file.
pub type Unique = [u8; 16];
