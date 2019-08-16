use std::convert::TryInto;
use std::sync::Arc;

use wasm_bindgen::JsValue;

use js_sys::{Function, Promise};

use liblumen_core::locks::Mutex;

use liblumen_alloc::erts::exception::system::Alloc;
use liblumen_alloc::erts::process::{code, ProcessControlBlock};
use liblumen_alloc::erts::term::{resource, Atom, Term, TypedTerm};

use lumen_runtime::scheduler::Scheduled;
use lumen_runtime::{process, registry};

/// Spawns process with this as the first frame, so that the next frame added in `call` can fulfill
/// the promise.
pub fn spawn<F>(
    parent_process: &ProcessControlBlock,
    heap: *mut Term,
    heap_size: usize,
    place_frame_with_arguments: F,
) -> Result<Promise, Alloc>
where
    F: Fn(&ProcessControlBlock) -> Result<(), Alloc>,
{
    let (process, promise) = spawn_unscheduled(parent_process, heap, heap_size)?;

    place_frame_with_arguments(&process)?;

    let parent_scheduler = parent_process.scheduler().unwrap();
    let arc_process = parent_scheduler.schedule(process);
    registry::put_pid_to_process(&arc_process);

    Ok(promise)
}

// Private

fn code(arc_process: &Arc<ProcessControlBlock>) -> code::Result {
    let return_term = arc_process.stack_pop().unwrap();
    let executor_term = arc_process.stack_pop().unwrap();
    assert!(executor_term.is_resource_reference());

    let executor_resource_reference: resource::Reference = executor_term.try_into().unwrap();
    let executor_mutex: &Mutex<Executor> = executor_resource_reference.downcast_ref().unwrap();
    executor_mutex.lock().resolve(return_term);

    arc_process.remove_last_frame();

    ProcessControlBlock::call_code(arc_process)
}

fn function() -> Atom {
    Atom::try_from_str("with_return").unwrap()
}

/// Spawns process with this as the first frame, so that any later `Frame`s can return to it.
///
/// The returns `ProcessControlBlock` is **NOT** scheduled with the scheduler yet, so that
/// the frame that will return to this frame can be added prior to running the process to
/// prevent a race condition on the `parent_process`'s scheduler running the new child process
/// when only the `with_return/0` frame is there.
///
/// ```
/// use liblumen_alloc::default_heap;
/// use liblumen_alloc::erts::process::code::stack::frame::Placement;
///
/// use lumen_runtime::otp::erlang::self_0;
/// use lumen_runtime::registry;
/// use lumen_runtime::scheduler::{Scheduler, Scheduler};
///
/// # let arc_scheduler = Scheduler::current();
/// # let parent_arc_process = arc_scheduler.spawn_init(0).unwrap();
///
/// # let (heap, heap_size) = default_heap();
/// let (process, promise) = lumen_web::wait::with_return_0::spawn_unscheduled(&parent_arc_process, heap, heap_size);
/// self_0::place_frame(promise, Placement::Push);
///
/// let parent_scheduler = parent_process.scheduler().unwrap();
/// let arc_process = parent_scheduler.schedule(process);
/// registry::put_pid_to_process(arc_process);
///
/// promise
/// ```
fn spawn_unscheduled(
    parent_process: &ProcessControlBlock,
    heap: *mut Term,
    heap_size: usize,
) -> Result<(ProcessControlBlock, Promise), Alloc> {
    let process = process::spawn(
        parent_process,
        super::module(),
        function(),
        vec![],
        code,
        heap,
        heap_size,
    )?;

    let mut executor = Executor::new();
    let promise = executor.promise();

    let executor_resource_reference = process.resource(Box::new(Mutex::new(executor)))?;
    process.stack_push(executor_resource_reference)?;

    Ok((process, promise))
}

fn term_to_js_value(term: Term) -> JsValue {
    match term.to_typed_term().unwrap() {
        TypedTerm::SmallInteger(small_integer) => {
            let i: isize = small_integer.into();

            if (std::i32::MIN as isize) <= i && i <= (std::i32::MAX as isize) {
                (i as i32).into()
            } else {
                (i as f64).into()
            }
        }
        _ => unimplemented!("Convert {:?} to JsValue", term),
    }
}

/// The executor for a `js_sys::Promise` that will be resolved by `code` or rejected when the owning
/// promise exits and the executor is dropped.
struct Executor {
    state: State,
}

impl Executor {
    pub fn new() -> Self {
        Self {
            state: State::Uninitialized,
        }
    }

    pub fn promise(&mut self) -> Promise {
        match self.state {
            State::Uninitialized => {
                let executor = self;

                Promise::new(&mut |resolve, reject| {
                    executor.state = State::Pending { resolve, reject };
                })
            }
            _ => panic!("Can only create promise once"),
        }
    }

    pub fn reject(&mut self) {
        match &self.state {
            State::Pending { reject, .. } => {
                drop(reject.call1(&JsValue::undefined(), &JsValue::undefined()));
                self.state = State::Rejected;
            }
            _ => panic!("Can only reject executor when pending"),
        }
    }

    pub fn resolve(&mut self, term: Term) {
        match &self.state {
            State::Pending { resolve, .. } => {
                drop(resolve.call1(&JsValue::undefined(), &term_to_js_value(term)));
                self.state = State::Resolved;
            }
            _ => panic!("Can only resolve executor when pending"),
        }
    }
}

impl Drop for Executor {
    fn drop(&mut self) {
        match self.state {
            State::Pending { .. } => self.reject(),
            _ => (),
        };
    }
}

enum State {
    Uninitialized,
    Pending { resolve: Function, reject: Function },
    Resolved,
    Rejected,
}
