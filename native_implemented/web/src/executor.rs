//! An executor has the ability to resolve or reject a `Promise`

pub mod apply_4;

use wasm_bindgen::JsValue;

use js_sys::{Function, Promise};

use liblumen_alloc::erts::term::prelude::*;

use super::js_value;

pub fn module() -> Atom {
    Atom::from_str("Elixir.Lumen.Web.Executor")
}

/// The executor for a `js_sys::Promise` that will be resolved by `code` or rejected when the owning
/// process exits and the executor is dropped.
pub struct Executor {
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
                drop(resolve.call1(&JsValue::undefined(), &js_value::from_term(term)));
                self.state = State::Resolved;
            }
            _ => panic!("Can only resolve executor when pending"),
        }
    }
}

impl Drop for Executor {
    fn drop(&mut self) {
        if let State::Pending { .. } = self.state {
            self.reject()
        };
    }
}

enum State {
    Uninitialized,
    Pending { resolve: Function, reject: Function },
    Resolved,
    Rejected,
}
