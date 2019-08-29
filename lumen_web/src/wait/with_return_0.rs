use std::any::TypeId;
use std::convert::TryInto;
use std::sync::Arc;

use wasm_bindgen::JsValue;

use js_sys::{Function, Promise};

use web_sys::{Document, Element, HtmlBodyElement, HtmlElement, HtmlTableElement, Node, Text};

use liblumen_core::locks::Mutex;

use liblumen_alloc::erts::exception::system::Alloc;
use liblumen_alloc::erts::process::{code, ProcessControlBlock};
use liblumen_alloc::erts::term::{resource, Atom, Pid, SmallInteger, Term, Tuple, TypedTerm};

use lumen_runtime::process::spawn::options::Options;
use lumen_runtime::scheduler::Scheduler;
use lumen_runtime::{process, registry};

/// Spawns process with this as the first frame, so that the next frame added in `call` can fulfill
/// the promise.
pub fn spawn<F>(options: Options, place_frame_with_arguments: F) -> Result<Promise, Alloc>
where
    F: Fn(&ProcessControlBlock) -> Result<(), Alloc>,
{
    let (process, promise) = spawn_unscheduled(options)?;

    place_frame_with_arguments(&process)?;

    let arc_process = Scheduler::current().schedule(process);
    registry::put_pid_to_process(&arc_process);

    Ok(promise)
}

// Private

fn atom_to_js_value(atom: Atom) -> JsValue {
    js_sys::Symbol::for_(atom.name()).into()
}

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

fn pid_to_js_value(pid: Pid) -> JsValue {
    let array = js_sys::Array::new();

    array.push(&(pid.number() as i32).into());
    array.push(&(pid.serial() as i32).into());

    array.into()
}

fn resource_reference_to_js_value(resource_reference: resource::Reference) -> JsValue {
    let resource_type_id = resource_reference.type_id();

    if resource_type_id == TypeId::of::<Document>() {
        let document: &Document = resource_reference.downcast_ref().unwrap();

        document.into()
    } else if resource_type_id == TypeId::of::<Element>() {
        let element: &Element = resource_reference.downcast_ref().unwrap();

        element.into()
    } else if resource_type_id == TypeId::of::<HtmlBodyElement>() {
        let html_body_element: &HtmlBodyElement = resource_reference.downcast_ref().unwrap();

        html_body_element.into()
    } else if resource_type_id == TypeId::of::<HtmlElement>() {
        let html_element: &HtmlElement = resource_reference.downcast_ref().unwrap();

        html_element.into()
    } else if resource_type_id == TypeId::of::<HtmlTableElement>() {
        let html_table_element: &HtmlTableElement = resource_reference.downcast_ref().unwrap();

        html_table_element.into()
    } else if resource_type_id == TypeId::of::<Node>() {
        let node: &Node = resource_reference.downcast_ref().unwrap();

        node.into()
    } else if resource_type_id == TypeId::of::<Text>() {
        let text: &Text = resource_reference.downcast_ref().unwrap();

        text.into()
    } else {
        unimplemented!("Convert {:?} to JsValue", resource_reference);
    }
}

fn small_integer_to_js_value(small_integer: SmallInteger) -> JsValue {
    let i: isize = small_integer.into();

    if (std::i32::MIN as isize) <= i && i <= (std::i32::MAX as isize) {
        (i as i32).into()
    } else {
        (i as f64).into()
    }
}

/// Spawns process with this as the first frame, so that any later `Frame`s can return to it.
///
/// The returns `ProcessControlBlock` is **NOT** scheduled with the scheduler yet, so that
/// the frame that will return to this frame can be added prior to running the process to
/// prevent a race condition on the `parent_process`'s scheduler running the new child process
/// when only the `with_return/0` frame is there.
///
/// ```
/// use liblumen_alloc::erts::process::code::stack::frame::Placement;
///
/// use lumen_runtime::otp::erlang::self_0;
/// # use lumen_runtime::process::spawn::options::Options;
/// use lumen_runtime::registry;
/// use lumen_runtime::scheduler::{Scheduled, Scheduler};
///
/// # let options: Options = Default::default();
/// let (process, promise) = lumen_web::wait::with_return_0::spawn_unscheduled(options);
/// self_0::place_frame(promise, Placement::Push);
///
/// let arc_process = Scheduler::current().schedule(process);
/// registry::put_pid_to_process(arc_process);
///
/// promise
/// ```
fn spawn_unscheduled(options: Options) -> Result<(ProcessControlBlock, Promise), Alloc> {
    let parent_process = None;
    let process = process::spawn::code(
        parent_process,
        options,
        super::module(),
        function(),
        vec![],
        code,
    )?;

    let mut executor = Executor::new();
    let promise = executor.promise();

    let executor_resource_reference = process.resource(Box::new(Mutex::new(executor)))?;
    process.stack_push(executor_resource_reference)?;

    Ok((process, promise))
}

fn term_to_js_value(term: Term) -> JsValue {
    match term.to_typed_term().unwrap() {
        TypedTerm::Atom(atom) => atom_to_js_value(atom),
        TypedTerm::Boxed(boxed) => match boxed.to_typed_term().unwrap() {
            TypedTerm::ResourceReference(resource_reference) => {
                resource_reference_to_js_value(resource_reference)
            }
            TypedTerm::Tuple(tuple) => tuple_to_js_value(&tuple),
            _ => unimplemented!("Convert {:?} to JsValue", term),
        },
        TypedTerm::Pid(pid) => pid_to_js_value(pid),
        TypedTerm::SmallInteger(small_integer) => small_integer_to_js_value(small_integer),
        _ => unimplemented!("Convert {:?} to JsValue", term),
    }
}

fn tuple_to_js_value(tuple: &Tuple) -> JsValue {
    let array = js_sys::Array::new();

    for element_term in tuple.iter() {
        let element_js_value = term_to_js_value(element_term);
        array.push(&element_js_value);
    }

    array.into()
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
