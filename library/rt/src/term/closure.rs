use alloc::alloc::{AllocError, Allocator, Layout};
use core::fmt;
use core::hash::{Hash, Hasher};
use core::ops::Deref;
use core::ptr;

use firefly_alloc::heap::Heap;
use firefly_macros_seq::seq;

use crate::function::{ErlangResult, ModuleFunctionArity};
use crate::gc::Gc;
use crate::process::ProcessLock;

use super::{Atom, Boxable, Header, LayoutBuilder, OpaqueTerm, Tag, Term};

bitflags::bitflags! {
    pub struct ClosureFlags: u8 {
        /// Set if the closure points to a bytecoded function
        ///
        /// When this is true, the callee pointer is an instruction offset in the
        /// loaded bytecode, rather than a pointer to a function.
        const BYTECODE = 1;
        /// Set if this closure is a thin closure
        const THIN = 1 << 1;
    }
}
impl Default for ClosureFlags {
    fn default() -> Self {
        Self::empty()
    }
}

/// This struct unifies function captures and closures under a single type.
///
/// Closure contains all the metadata about the callee required to answer questions like
/// what is the arity, what module was it defined in, etc.
///
/// Closures (as opposed to function captures) have an implicit extra argument that comes first
/// in the argument list of the callee, which is a fat pointer to the Closure struct. This enables
/// the callee to access the closed-over values from its environment.
///
/// Function captures do not have the extra self argument, and always have an implicitly empty environment.
#[repr(C)]
pub struct Closure {
    pub header: Header,
    pub module: Atom,
    pub name: Atom,
    pub arity: u8,
    pub flags: ClosureFlags,
    pub callee: *const (),
    env: [OpaqueTerm],
}
impl fmt::Debug for Closure {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Closure")
            .field("header", &self.header)
            .field("module", &self.module.as_str())
            .field("function", &self.name.as_str())
            .field("arity", &self.arity)
            .field("flags", &self.flags)
            .field("callee", &self.callee)
            .field("env", &&self.env)
            .finish()
    }
}
impl fmt::Display for Closure {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "#Fun<{}:{}/{}>", self.module, self.name, self.arity)
    }
}
impl Closure {
    /// Allocates a new Gc'd closure with the given name, callee, and environment, using the provided allocator
    ///
    /// # Safety
    ///
    /// This is a risky low-level operation, and is only safe if the following guarantees are upheld by the caller:
    ///
    /// * The callee pointer must point to an actual function
    /// * The callee must be guaranteed to outlive the closure itself
    /// * The callee must expect to receive `arity` arguments in addition to the closure self argument
    pub fn new_in<A: ?Sized + Allocator>(
        module: Atom,
        name: Atom,
        arity: u8,
        callee: *const (),
        env: &[OpaqueTerm],
        alloc: &A,
    ) -> Result<Gc<Self>, AllocError> {
        let mut flags = ClosureFlags::empty();
        if env.is_empty() {
            flags |= ClosureFlags::THIN;
        }
        let mut this = Gc::<Self>::with_capacity_in(env.len(), alloc)?;
        this.header = Header::new(Tag::Closure, env.len());
        this.module = module;
        this.name = name;
        this.arity = arity;
        this.flags = flags;
        this.callee = callee;
        this.env.copy_from_slice(env);
        Ok(this)
    }

    pub fn new_with_flags_in<A: ?Sized + Allocator>(
        module: Atom,
        name: Atom,
        arity: u8,
        mut flags: ClosureFlags,
        callee: *const (),
        env: &[OpaqueTerm],
        alloc: &A,
    ) -> Result<Gc<Self>, AllocError> {
        if env.is_empty() {
            flags |= ClosureFlags::THIN;
        }
        let mut this = Gc::<Self>::with_capacity_in(env.len(), alloc)?;
        this.header = Header::new(Tag::Closure, env.len());
        this.module = module;
        this.name = name;
        this.arity = arity;
        this.flags = flags;
        this.callee = callee;
        this.env.copy_from_slice(env);
        Ok(this)
    }

    pub unsafe fn with_capacity_in<A: ?Sized + Allocator>(
        capacity: usize,
        alloc: &A,
    ) -> Result<Gc<Self>, AllocError> {
        let mut this = Gc::<Self>::with_capacity_in(capacity, alloc)?;
        if capacity == 0 {
            this.flags |= ClosureFlags::THIN;
        }
        Ok(this)
    }

    #[inline]
    pub fn clone_from<A: ?Sized + Allocator>(
        other: &Self,
        alloc: &A,
    ) -> Result<Gc<Self>, AllocError> {
        Self::new_with_flags_in(
            other.module,
            other.name,
            other.arity,
            other.flags,
            other.callee,
            &other.env,
            alloc,
        )
    }

    /// Returns true if this closure is a function capture, i.e. it has no free variables.
    #[inline]
    pub fn is_thin(&self) -> bool {
        self.flags.contains(ClosureFlags::THIN)
    }

    /// Returns `true` if this closure points to a function in memory
    ///
    /// If the function is bytecoded, it returns `false`
    #[inline]
    pub fn is_native(&self) -> bool {
        !self.flags.contains(ClosureFlags::BYTECODE)
    }

    /// Returns the size of the environment (in units of `OpaqueTerm`) bound to this closure
    #[inline]
    pub fn env_size(&self) -> usize {
        self.header.arity()
    }

    #[inline]
    pub const fn env(&self) -> &[OpaqueTerm] {
        &self.env
    }

    #[inline]
    pub const fn env_mut(&mut self) -> &mut [OpaqueTerm] {
        &mut self.env
    }

    #[inline]
    pub fn mfa(&self) -> ModuleFunctionArity {
        ModuleFunctionArity::new(self.module, self.name, self.arity as usize)
    }

    /// Copies the env from `other` into this closure's environment
    ///
    /// This function will panic if the env arities are different
    pub fn copy_from(&mut self, other: &Self) {
        assert_eq!(self.env.len(), other.env.len());
        self.env.copy_from_slice(&other.env);
    }

    /// Applies the given slice of arguments to this closure.
    ///
    /// This function will panic if the number of arguments given does not match
    /// the arity of the closure.
    ///
    /// NOTE: Currently, a max arity of 10 is supported for dynamic apply via this function.
    /// If the number of arguments exceeds this number, this function will panic.
    #[inline]
    pub fn apply(&self, process: &mut ProcessLock, args: &[OpaqueTerm]) -> ErlangResult {
        seq!(N in 0..10 {
            match args.len() {
                #(
                    N => apply~N(self, process, args),
                )*
                n => panic!("apply failed: too many arguments, got {}, expected no more than 10", n),
            }
        })
    }
}
impl Boxable for Closure {
    type Metadata = usize;

    const TAG: Tag = Tag::Closure;

    #[inline]
    fn header(&self) -> &Header {
        &self.header
    }

    #[inline]
    fn header_mut(&mut self) -> &mut Header {
        &mut self.header
    }

    fn layout_excluding_heap<H: ?Sized + Heap>(&self, heap: &H) -> Layout {
        if heap.contains((self as *const Self).cast()) {
            return Layout::new::<()>();
        }

        let mut builder = LayoutBuilder::new();
        for element in self.env.iter().copied() {
            if !element.is_gcbox() {
                continue;
            }
            let element: Term = element.into();
            builder.extend(&element);
        }
        builder += Layout::for_value(self);
        builder.finish()
    }

    fn unsafe_clone_to_heap<H: ?Sized + Heap>(&self, heap: &H) -> Gc<Self> {
        let ptr = self as *const Self;
        if heap.contains(ptr.cast()) {
            unsafe { Gc::from_raw(ptr.cast_mut()) }
        } else {
            let mut cloned = Self::clone_from(self, heap).unwrap();
            let target_env = cloned.env_mut();
            for (i, term) in self.env().iter().copied().enumerate() {
                if !term.is_gcbox() {
                    term.maybe_increment_refcount();
                    unsafe {
                        *target_env.get_unchecked_mut(i) = term;
                    }
                    continue;
                }
                let term: Term = term.into();
                unsafe {
                    *target_env.get_unchecked_mut(i) = term.unsafe_clone_to_heap(heap).into();
                }
            }
            cloned
        }
    }
}
impl Closure {
    pub unsafe fn unsafe_move_to_heap<H: ?Sized + Heap>(&self, heap: &H) -> Gc<Self> {
        use crate::term::Cons;

        let ptr = self as *const Self;
        if heap.contains(ptr.cast()) {
            unsafe { Gc::from_raw(ptr.cast_mut()) }
        } else {
            let mut cloned = Self::clone_from(self, heap).unwrap();
            let target_env = cloned.env_mut();
            for (i, term) in self.env().iter().copied().enumerate() {
                if term.is_rc() {
                    *target_env.get_unchecked_mut(i) = term;
                } else if term.is_nonempty_list() {
                    let mut cons = Gc::from_raw(term.as_ptr() as *mut Cons);
                    let moved = cons.unsafe_move_to_heap(heap);
                    *target_env.get_unchecked_mut(i) = moved.into();
                } else if term.is_gcbox() || term.is_tuple() {
                    let term: Term = term.into();
                    let moved = term.unsafe_move_to_heap(heap);
                    *target_env.get_unchecked_mut(i) = moved.into();
                } else {
                    *target_env.get_unchecked_mut(i) = term;
                }
            }
            cloned
        }
    }
}

seq!(A in 0..10 {
    #(
        seq!(N in 0..A {
            /// This type represents a function which implements a closure of arity A
            ///
            /// See the `Closure` docs for more information on how closures are implemented.
            pub type Closure~A<'a, 'b> = extern "C-unwind" fn (&'a mut ProcessLock<'b>, #(
                                                    OpaqueTerm,
                                                )*
                                                OpaqueTerm
                                                ) -> ErlangResult;

            /// This type represents a function capture of arity A
            ///
            /// This differs from `ClosureA` in that a function capture has no implicit self argument.
            pub type Fun~A<'a, 'b> = extern "C-unwind" fn (&'a mut ProcessLock<'b>, #(OpaqueTerm,)*) -> ErlangResult;

            /// This type represents a tuple of A arguments
            pub type Args~A<'a, 'b> = (&'a mut ProcessLock<'b>, #(OpaqueTerm,)*);
        });

        seq!(N in 0..(A + 1) {
            impl<'a, 'b> FnOnce<Args~A<'a, 'b>> for &Closure {
                type Output = ErlangResult;

                #[inline]
                extern "rust-call" fn call_once(self, _args: Args~A<'a, 'b>) -> Self::Output {
                    assert!(self.is_native());
                    if self.is_thin() {
                        assert_eq!(self.arity, A, "mismatched arity");
                        let fun = unsafe { core::mem::transmute::<_, Fun~A<'a, 'b>>(self.callee) };
                        fun(#(_args.N,)*)
                    } else {
                        assert_eq!(self.arity, A + 1, "mismatched arity");
                        let fun = unsafe { core::mem::transmute::<_, Closure~A<'a, 'b>>(self.callee) };
                        let this = unsafe { OpaqueTerm::from_gcbox_closure(self) };
                        fun(#(_args.N,)* this)
                    }
                }
            }
            impl<'a, 'b> FnMut<Args~A<'a, 'b>> for &Closure {
                #[inline]
                extern "rust-call" fn call_mut(&mut self, _args: Args~A<'a, 'b>) -> Self::Output {
                    assert!(self.is_native());
                    if self.is_thin() {
                        assert_eq!(self.arity, A, "mismatched arity");
                        let fun = unsafe { core::mem::transmute::<_, Fun~A<'a, 'b>>(self.callee) };
                        fun(#(_args.N,)*)
                    } else {
                        assert_eq!(self.arity, A + 1, "mismatched arity");
                        let fun = unsafe { core::mem::transmute::<_, Closure~A<'a, 'b>>(self.callee) };
                        let this = unsafe { OpaqueTerm::from_gcbox_closure(*self) };
                        fun(#(_args.N,)* this)
                    }
                }
            }
            impl<'a, 'b> Fn<Args~A<'a, 'b>> for &Closure {
                #[inline]
                extern "rust-call" fn call(&self, _args: Args~A<'a, 'b>) -> Self::Output {
                    assert!(self.is_native());
                    if self.is_thin() {
                        assert_eq!(self.arity, A, "mismatched arity");
                        let fun = unsafe { core::mem::transmute::<_, Fun~A<'a, 'b>>(self.callee) };
                        fun(#(_args.N,)*)
                    } else {
                        assert_eq!(self.arity, A + 1, "mismatched arity");
                        let fun = unsafe { core::mem::transmute::<_, Closure~A<'a, 'b>>(self.callee) };
                        let this = unsafe { OpaqueTerm::from_gcbox_closure(*self) };
                        fun(#(_args.N,)* this)
                    }
                }
            }
        });

        seq!(M in 0..A {
            /// Applies the given slice of arguments to a function of arity A
            ///
            /// NOTE: This function asserts that the length of `args` matches the arity of `fun`,
            /// if they do not match the function panics.
            #[inline]
            pub fn apply~A<F>(fun: F, process: &mut ProcessLock, _args: &[OpaqueTerm]) -> ErlangResult
            where
                F: Fn(&mut ProcessLock, #(OpaqueTerm,)*) -> ErlangResult,
            {
                assert_eq!(_args.len(), A, "mismatched arity");

                fun(process, #(_args[M],)*)
            }
        });
    )*
});

impl Eq for Closure {}
impl crate::cmp::ExactEq for Closure {}
impl PartialEq for Closure {
    fn eq(&self, other: &Self) -> bool {
        self.module == other.module
            && self.name == other.name
            && self.arity == other.arity
            && self.flags == other.flags
            && core::ptr::eq(self.callee, other.callee)
    }
}
impl PartialEq<Gc<Closure>> for Closure {
    fn eq(&self, other: &Gc<Closure>) -> bool {
        self.eq(other.deref())
    }
}
impl PartialOrd for Closure {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Closure {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        use core::cmp::Ordering;

        match self.module.cmp(&other.module) {
            Ordering::Equal => match self.name.cmp(&other.name) {
                Ordering::Equal => self.arity.cmp(&other.arity),
                other => other,
            },
            other => other,
        }
    }
}
impl Hash for Closure {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.module.hash(state);
        self.name.hash(state);
        self.arity.hash(state);
        self.flags.hash(state);
        ptr::hash(self.callee, state);
    }
}
