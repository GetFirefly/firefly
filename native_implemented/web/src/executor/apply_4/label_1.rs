use std::convert::TryInto;
use std::sync::Arc;

use liblumen_core::locks::Mutex;

use liblumen_alloc::erts::term::prelude::*;

use crate::executor::Executor;

#[native_implemented::label]
pub fn result(apply_returned: Term, executor: Term) -> Term {
    let executor_boxed_resource: Boxed<Resource> = executor.try_into().unwrap();
    let executor_resource: Resource = executor_boxed_resource.into();
    let executor_mutex: &Arc<Mutex<Executor>> = executor_resource.downcast_ref().unwrap();
    executor_mutex.lock().resolve(apply_returned);

    apply_returned
}
