mod ffi;

use alloc::boxed::Box;
use alloc::format;
use core::convert::Infallible;
use core::fmt;
use core::future::Future;
use core::ops::{self, ControlFlow};
use core::pin::Pin;
use core::ptr::NonNull;
use core::task::{Context, Poll};

use intrusive_collections::{intrusive_adapter, LinkedListLink, UnsafeRef};

use crate::error::ErlangException;
use crate::function::ErlangResult;
use crate::term::OpaqueTerm;

pub use self::ffi::{ContextExt, FfiContext};

intrusive_adapter!(pub(crate) ContinuationAdapter<T, E> = UnsafeRef<Continuation<T, E>>: Continuation<T, E> { link: LinkedListLink });

/// This type represents the poll function of a continuation
///
/// The `cont` argument is a reference to the Continuation being polled, and can be used to get access to the underlying
/// future for the "real" poll implementation. The reason for passing the continuation and not the future directly,
/// is because the continuation also contains a pointer to the next continuation in the call stack, if one exists,
/// which is used to replay the call stack when resuming execution. The structure of the underlying future is only
/// known to the implementation of the poll function.
///
/// The `context` argument is an FFI-safe wrapper around `core::task::Context`, and can be used to call `Context`
/// functions from outside Rust.
///
/// Unlike `Future:poll`, this function must return an `ErlangFuture<T, E>`, which is similar to `Poll<Result<T, E>>>`,
/// but is both FFI-safe, and has a more efficient layout in memory so that it can be returned in registers when both
/// `T` and `E` are pointer-sized (or smaller).
pub type PollFn<T = OpaqueTerm, E = NonNull<ErlangException>> =
    unsafe extern "C-unwind" fn(
        cont: Pin<&mut Continuation<T, E>>,
        context: *mut FfiContext,
    ) -> ErlangFuture<T, E>;

/// This type represents the drop function for a continuation's future
///
/// This function is called in one of two scenarios:
///
/// 1. The continuation is being dropped, and thus the underlying future is as well
/// 2. The continuation is being set aside for reuse, but the future is no longer needed, so it is being dropped
pub type DropFn = unsafe extern "C-unwind" fn(*mut ());

/// A continuation is an FFI-safe future that represents the rest of the program following a suspension point.
///
/// Continuations are created either from a Rust future, or by the compiler in the prologue of an async function.
/// It is much like a Rust future, except it is safe to use in our calling convention, and has a layout which
/// allows us to manipulate them in compiler-generated code.
///
/// The compiler will allocate continuations in any async function it compiles, when the function is entered for
/// the first time. Each suspension point in that function will then re-use the continuation to manage the state
/// machine for that function, as well as store stack values which must be live across a yield point. The compiler
/// ensures that continuations are large enough to store those values without reallocation.
///
/// Compiler-generated async functions can be broken up two different ways, depending on compilation-strategy, and
/// both are supported by this type:
///
/// * switched-return lowering, where there is a ramp function and one split where resumption occurs. The split
/// function contains effectively a big switch statement for each yield point in the function, and the current
/// switch index is stored in the continuation structure.
/// * returned-continuation lowering, where there is a ramp function and one or more splits depending on the number
/// of yield points in the function.
///
/// In both cases, the continuation is allocated in the ramp function, and the pointer to the first split function
/// is stored in this structure as `poll_fn`. In the returned-continuation lowering, each split is responsible for
/// updating `poll_fn` before yielding. Splits are responsible for unpacking any values that were live across the
/// last suspension point using the future reference passed to `poll_fn`. The layout of values is context-sensitive,
/// and is expected to be known only to the code just before and after a yield point.
///
/// Continuations are intended to be allocated on the heap, ideally in a special region managed by a bump allocator
/// for efficiency, but they can be allocated via `Box` as well.
#[repr(C)]
pub struct Continuation<T = OpaqueTerm, E = NonNull<ErlangException>> {
    /// This link is used to store this continuation in a doubly-linked list of
    /// continuations; which can be used for multiple purposes, such as tracking a
    /// call stack, storing continuations in a free list for reuse, etc.
    pub link: LinkedListLink,
    /// This is a pointer to the future object, known only to the creator of the continuation
    future: *mut (),
    /// This is a pointer to the function used to poll the future, i.e. resume the continuation
    poll_fn: PollFn<T, E>,
    /// This is a pointer to the `drop` function for `future`
    drop_fn: DropFn,
}
impl<T, E> Continuation<T, E> {
    /// Creates a new continuation from its raw components:
    ///
    /// * A pointer to the underlying future's data structure
    /// * A function pointer to be used as the poll function for the underlying future
    /// * A function pointer to be used as the drop function for the underlying future
    ///
    /// # Safety
    ///
    /// There are a few safety invariants that must be enforced by callers of this function:
    ///
    /// * The pointee type of `future` _must_ match the future type expected by the implementations of
    /// `poll_fn` and `drop_fn`.
    ///
    /// * Ownership of the value pointed to by `future` implicitly belongs to the Continuation created.
    /// However, if you wish to manage the lifetime of the future separately, then you must use reference
    /// counting, and leverage `drop_fn` to decrement the reference count when the Continuation releases
    /// ownership of the future. This ensures that the future is never freed while the Continuation is
    /// still considered live. If you do not maintain a reference count, then you must use `drop_fn` to
    /// signal that the future is no longer needed, otherwise the future will be leaked.
    ///
    /// * Both `poll_fn` and `drop_fn` must be unwind safe, or use the "C-unwind" ABI, it is not permitted
    /// to panic from an extern "C" function otherwise. This should be enforced by the Rust type system,
    /// but it is _not_ safe to cast function pointers of extern "C" type if they are not guaranteed to
    /// be unwind-safe.
    ///
    #[inline(always)]
    pub unsafe fn from_raw_parts(future: *mut (), poll_fn: PollFn<T, E>, drop_fn: DropFn) -> Self {
        Self {
            link: LinkedListLink::new(),
            future,
            poll_fn,
            drop_fn,
        }
    }

    /// This is only intended for use by the ContinuationTracker when reusing continuations
    #[allow(unused)]
    #[inline(always)]
    pub(crate) fn set_poll(&mut self, poll_fn: PollFn<T, E>) {
        self.poll_fn = poll_fn;
    }
}
impl<T, E> Drop for Continuation<T, E> {
    fn drop(&mut self) {
        unsafe { (self.drop_fn)(self.future) }
    }
}
impl<T, E> Future for Continuation<T, E> {
    type Output = Result<T, E>;

    fn poll(self: Pin<&mut Self>, ctx: &mut Context<'_>) -> Poll<Self::Output> {
        ctx.with_ffi_context(|ctx| unsafe { (self.poll_fn)(self, ctx).into() })
    }
}

pub trait FutureExt: Future + Sized {
    /// Convert a Rust `Future` into a `Continuation`
    fn into_continuation<T, E>(self) -> Continuation<T, E>
    where
        Self: Future<Output = Result<T, E>>,
    {
        BoxedFuture::new(self).into()
    }
}
impl<F> FutureExt for F where F: Future + Sized {}

/// Represents a Future which has been allocated on the global heap for use with `Continuation`
struct BoxedFuture<F: Future> {
    future: Box<F>,
}
impl<F: Future> BoxedFuture<F> {
    #[inline(always)]
    fn new(fut: F) -> Self {
        Self {
            future: Box::new(fut),
        }
    }
}
impl<T, E, F> Into<Continuation<T, E>> for BoxedFuture<F>
where
    F: Future<Output = Result<T, E>>,
{
    fn into(self) -> Continuation<T, E> {
        // Polls the inner future
        unsafe extern "C-unwind" fn poll_fn<T, E, F: Future<Output = Result<T, E>>>(
            cont: Pin<&mut Continuation<T, E>>,
            context_ptr: *mut FfiContext,
        ) -> ErlangFuture<T, E> {
            let fut_pin = Pin::new_unchecked(&mut *cont.future.cast::<F>());
            (*context_ptr)
                .with_context(|ctx| F::poll(fut_pin, ctx))
                .into()
        }

        // Drops the inner future
        unsafe extern "C-unwind" fn drop_fn<T>(ptr: *mut ()) {
            drop(Box::from_raw(ptr.cast::<T>()));
        }

        let ptr = Box::into_raw(self.future);

        Continuation {
            link: LinkedListLink::new(),
            future: ptr.cast(),
            poll_fn: poll_fn::<_, _, F>,
            drop_fn: drop_fn::<F>,
        }
    }
}

/// This enum is intended to have the same layout in memory as `ErlangResult`,
/// so that `ErlangResult` can be safely transmuted to `ErlangFuture`. In such
/// cases, the result would appear to be a future that is ready immediately.
///
/// Similarly, `ErlangFuture<T, E>` is intended to mirror `Poll<Result<T, E>>`,
/// but with a layout that can be passed in registers like `ErlangResult` when both
/// `T` and `E` are pointer-sized (or less). This type is used as an FFI-safe version
/// of `Poll`, and is used as the result type of `PollFn` for that reason.
///
/// # Safety
///
/// It is safe to transmute `ErlangResult<T, E>` to `ErlangFuture<T, E>`, but not
/// the reverse. Even though they share the same layout. The `Suspended` variant
/// cannot be meaningfully translated to `ErlangResult`, so the future which was polled
/// to produce the `ErlangFuture` must be run to completion, or a panic must be raised.
///
/// If for some reason an `ErlangFuture::Suspended` is transmuted to an `ErlangResult`,
/// the best case scenario is that it behaves like `ErlangResult::Ok` with zeroed bits
/// for `T`, but it is extremely likely that undefined behavior will result instead, because
/// there is no guarantee that rustc won't generate code that checks discriminant values exactly,
/// or that the bits of `T` would be zero-initialized - more than likely they are not. In
/// other words, just don't do it, and don't allow it to happen.
#[derive(Debug, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(u8)]
pub enum ErlangFuture<T = OpaqueTerm, E = NonNull<ErlangException>> {
    /// Occurs when the future is complete and the result is ready
    Ready(T) = 0,
    /// Occurs when the future is complete, but failed due to an error
    Failed(E) = 1,
    /// Occurs when a future is first created and is started suspended
    Spawned(NonNull<Continuation>) = 2,
    /// Occurs when a future is polled but the result is not ready yet for some reason
    Suspended = 3,
}
impl<T, E> const Clone for ErlangFuture<T, E>
where
    T: ~const Clone + ~const core::marker::Destruct,
    E: ~const Clone + ~const core::marker::Destruct,
{
    #[inline]
    fn clone(&self) -> Self {
        match self {
            Self::Ready(x) => Self::Ready(x.clone()),
            Self::Failed(x) => Self::Failed(x.clone()),
            Self::Spawned(c) => Self::Spawned(*c),
            Self::Suspended => Self::Suspended,
        }
    }

    #[inline]
    fn clone_from(&mut self, source: &Self) {
        match (self, source) {
            (Self::Ready(to), Self::Ready(from)) => to.clone_from(from),
            (Self::Failed(to), Self::Failed(from)) => to.clone_from(from),
            (to, from) => *to = from.clone(),
        }
    }
}
impl<T, E> ErlangFuture<T, E> {
    #[inline]
    pub fn is_completed(&self) -> bool {
        match self {
            Self::Ready(_) | Self::Failed(_) => true,
            _ => false,
        }
    }

    #[inline]
    pub fn is_suspended(&self) -> bool {
        match self {
            Self::Spawned(_) | Self::Suspended => true,
            _ => false,
        }
    }

    #[inline]
    pub fn is_ready(&self) -> bool {
        match self {
            Self::Ready(_) => true,
            _ => false,
        }
    }

    #[inline]
    pub fn is_failed(&self) -> bool {
        match self {
            Self::Failed(_) => true,
            _ => false,
        }
    }

    /// Attempts to convert this future to `Option<T>`
    ///
    /// Returns `Ok(Option<T>)` if the conversion is allowed,
    /// otherwise returns `Err(Self)` if the future is in a
    /// suspended state.
    #[inline]
    pub fn ok(self) -> Result<Option<T>, Self> {
        match self {
            Self::Ready(v) => Ok(Some(v)),
            Self::Failed(_) => Ok(None),
            other => Err(other),
        }
    }

    #[inline]
    pub fn map<U, F>(self, op: F) -> ErlangFuture<U, E>
    where
        F: FnOnce(T) -> U,
    {
        match self {
            Self::Ready(v) => ErlangFuture::Ready(op(v)),
            Self::Failed(e) => ErlangFuture::Failed(e),
            Self::Spawned(c) => ErlangFuture::Spawned(c),
            Self::Suspended => ErlangFuture::Suspended,
        }
    }

    #[inline]
    pub fn map_failed<F, O>(self, op: O) -> ErlangFuture<T, F>
    where
        O: FnOnce(E) -> F,
    {
        match self {
            Self::Ready(v) => ErlangFuture::Ready(v),
            Self::Failed(e) => ErlangFuture::Failed(op(e)),
            Self::Spawned(c) => ErlangFuture::Spawned(c),
            Self::Suspended => ErlangFuture::Suspended,
        }
    }

    #[inline]
    pub fn expect(self, msg: &str) -> T
    where
        E: fmt::Debug,
    {
        match self {
            Self::Ready(v) => v,
            Self::Failed(ref e) => unwrap_failed(msg, e),
            Self::Spawned(_) => unwrap_suspended(&format!(
                "called `ErlangFuture::expect` on a `Spawned` value: {}",
                msg
            )),
            Self::Suspended => unwrap_suspended(&format!(
                "called `ErlangFuture::expect` on a `Suspended` value: {}",
                msg
            )),
        }
    }

    #[inline]
    pub fn unwrap(self) -> T
    where
        E: fmt::Debug,
    {
        match self {
            Self::Ready(v) => v,
            Self::Failed(ref e) => {
                unwrap_failed("called `ErlangFuture::unwrap` on a `Failed` value", e)
            }
            Self::Spawned(_) => {
                unwrap_suspended("called `ErlangFuture::unwrap` on a `Spawned` value")
            }
            Self::Suspended => {
                unwrap_suspended("called `ErlangFuture::unwrap` on a `Suspended` value")
            }
        }
    }

    #[inline]
    pub fn unwrap_failed(self) -> E
    where
        T: fmt::Debug,
    {
        match self {
            Self::Ready(ref v) => {
                unwrap_failed("called `ErlangFuture::unwrap_failed` on a `Ready` value", v)
            }
            Self::Spawned(_) => {
                unwrap_suspended("called `ErlangFuture::unwrap_failed` on a `Spawned` value")
            }
            Self::Suspended => {
                unwrap_suspended("called `ErlangFuture::unwrap_failed` on a `Suspended` value")
            }
            Self::Failed(e) => e,
        }
    }

    #[inline]
    pub unsafe fn unwrap_failed_unchecked(self) -> E {
        debug_assert!(self.is_failed());
        match self {
            // SAFETY: the safety contract must be upheld by the caller.
            Self::Ready(_) | Self::Spawned(_) | Self::Suspended => unsafe {
                core::hint::unreachable_unchecked()
            },
            Self::Failed(e) => e,
        }
    }
}

#[inline(never)]
#[cold]
#[track_caller]
fn unwrap_failed(msg: &str, error: &dyn fmt::Debug) -> ! {
    panic!("{msg}: {error:?}")
}

#[inline(never)]
#[cold]
#[track_caller]
fn unwrap_suspended(msg: &str) -> ! {
    panic!("{msg}")
}

#[inline(never)]
#[cold]
#[track_caller]
fn invalid_poll(msg: &str) -> ! {
    panic!("{msg}")
}

impl<T, E> From<Poll<Result<T, E>>> for ErlangFuture<T, E> {
    #[inline]
    fn from(poll: Poll<Result<T, E>>) -> Self {
        match poll {
            Poll::Ready(Ok(v)) => Self::Ready(v),
            Poll::Ready(Err(e)) => Self::Failed(e),
            Poll::Pending => Self::Suspended,
        }
    }
}
impl<T, E> Into<Poll<Result<T, E>>> for ErlangFuture<T, E> {
    #[inline]
    fn into(self) -> Poll<Result<T, E>> {
        match self {
            Self::Ready(v) => Poll::Ready(Ok(v)),
            Self::Failed(e) => Poll::Ready(Err(e)),
            Self::Suspended => Poll::Pending,
            Self::Spawned(_) => invalid_poll("called Into<Poll> on a `Spawned` value"),
        }
    }
}
impl<T, E> From<Result<T, E>> for ErlangFuture<T, E> {
    fn from(result: Result<T, E>) -> Self {
        match result {
            Ok(v) => Self::Ready(v),
            Err(v) => Self::Failed(v),
        }
    }
}
impl<T, E> From<ErlangResult<T, E>> for ErlangFuture<T, E> {
    fn from(result: ErlangResult<T, E>) -> Self {
        match result {
            ErlangResult::Ok(v) => Self::Ready(v),
            ErlangResult::Err(v) => Self::Failed(v),
        }
    }
}
impl<T, E> TryInto<Result<T, E>> for ErlangFuture<T, E> {
    type Error = ErlangFuture<T, E>;

    fn try_into(self) -> Result<Result<T, E>, Self::Error> {
        match self {
            Self::Ready(v) => Ok(Ok(v)),
            Self::Failed(v) => Ok(Err(v)),
            other => Err(other),
        }
    }
}
impl<T, E> ops::Try for ErlangFuture<T, E> {
    type Output = T;
    type Residual = ErlangFuture<Infallible, E>;

    #[inline]
    fn from_output(output: Self::Output) -> Self {
        Self::Ready(output)
    }

    #[inline]
    fn branch(self) -> ControlFlow<Self::Residual, Self::Output> {
        match self {
            Self::Ready(v) => ControlFlow::Continue(v),
            Self::Failed(e) => ControlFlow::Break(ErlangFuture::Failed(e)),
            Self::Spawned(c) => ControlFlow::Break(ErlangFuture::Spawned(c)),
            Self::Suspended => ControlFlow::Break(ErlangFuture::Suspended),
        }
    }
}
impl<T, E, F: From<E>> ops::FromResidual<ErlangFuture<Infallible, E>> for ErlangFuture<T, F> {
    #[inline]
    #[track_caller]
    fn from_residual(residual: ErlangFuture<Infallible, E>) -> Self {
        match residual {
            ErlangFuture::Failed(e) => Self::Failed(From::from(e)),
            ErlangFuture::Spawned(c) => Self::Spawned(c),
            ErlangFuture::Suspended => Self::Suspended,
            _ => unreachable!(),
        }
    }
}
impl<T, E, F: From<E>> ops::FromResidual<ErlangResult<Infallible, E>> for ErlangFuture<T, F> {
    #[inline]
    #[track_caller]
    fn from_residual(residual: ErlangResult<Infallible, E>) -> Self {
        match residual {
            ErlangResult::Err(e) => Self::Failed(From::from(e)),
            _ => unreachable!(),
        }
    }
}
impl<T, E, F: From<E>> ops::FromResidual<Result<Infallible, E>> for ErlangFuture<T, F> {
    #[inline]
    #[track_caller]
    fn from_residual(residual: Result<Infallible, E>) -> Self {
        match residual {
            Err(e) => Self::Failed(From::from(e)),
            _ => unreachable!(),
        }
    }
}
impl<T, E> ops::Residual<T> for ErlangFuture<Infallible, E> {
    type TryType = ErlangFuture<T, E>;
}
