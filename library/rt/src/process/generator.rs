use core::fmt;

use crate::term::OpaqueTerm;

use super::ProcessLock;

/// A `Continuation` is a function which when called, may yield to the caller in the middle of
/// execution, or run all the way to completion. In our case, a continuation is specifically
/// designed to operate in the context of a scheduled process, with some opaque state allocated by
/// the function which created the continuation. When resumed, the continuation function receives
/// the current process lock, as well as an opaque pointer to the state on which it may need to
/// operate.
///
/// When a continuation chooses to suspend prior to completion, it must return a new continuation to
/// use on the next resumption, along with an optional value which the caller might use at each
/// suspension point.
///
/// On completion, a continuation function behaves like a normal function, i.e. it returns whatever
/// the return value of the function is as the completion value.
pub type Continuation<T = (), U = Result<OpaqueTerm, ()>> =
    extern "C-unwind" fn(&mut ProcessLock, *mut ()) -> ContinuationResult<T, U>;

/// Continuation functions may either suspend or execute to completion,
/// this result type is used to indicate which of the two states the continuation
/// reached, along with values corresponding to the outputs of those states.
pub enum ContinuationResult<T = (), U = Result<OpaqueTerm, ()>> {
    /// Yield during execution of a continuation, producing a value, and returning a new
    /// continuation to call when resuming execution again in the future.
    Yield(Continuation<T, U>, T),
    /// The continuation executed to completion, returning the given value.
    Complete(U),
}

/// Represents the current state of a `Generator` after a call to `resume` returns.
pub enum GeneratorState<T, U> {
    /// The generator yielded during execution, returning the given value.
    Yielded(T),
    /// The generator executing, returning the given value.
    Completed(U),
}

/// A `Generator` is used to run a `Continuation` to completion, i.e. it tracks the current
/// continuation and generator state, and on each resumption, updates itself with the next
/// continuation, returning any intermediate values along the way.
#[derive(Clone)]
pub struct Generator<T = (), U = Result<OpaqueTerm, ()>> {
    next: Continuation<T, U>,
    state: *mut (),
}
impl<T, U> fmt::Debug for Generator<T, U> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Generator")
            .field("next", &format_args!("{:p}", &(self.next as *const ())))
            .field("state", &format_args!("{:p}", &self.state))
            .finish()
    }
}
impl<T, U> Generator<T, U> {
    pub const fn new(start: Continuation<T, U>, state: *mut ()) -> Self {
        Self { next: start, state }
    }

    /// Resumes this generator until it either yields, or completes.
    pub fn resume(&mut self, process: &mut ProcessLock) -> GeneratorState<T, U> {
        match (self.next)(process, self.state) {
            ContinuationResult::Yield(next, value) => {
                self.next = next;
                GeneratorState::Yielded(value)
            }
            ContinuationResult::Complete(result) => GeneratorState::Completed(result),
        }
    }
}
