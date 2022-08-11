use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use liblumen_intern::{symbols, Symbol};

use super::*;

// Represents the environment in which names are bound
// Functions and variables can share the same name as their usage is unambiguous
#[derive(Default, Debug, Clone)]
pub struct Scope {
    parent: Option<Rc<RefCell<Scope>>>,
    funs: HashMap<Symbol, FuncRef>,
    vars: HashMap<Symbol, Value>,
}
impl Scope {
    /// Create an empty root environment
    pub fn new() -> Self {
        Self::default()
    }

    /// Create an empty environment as a child of the given parent scope
    pub fn with_parent(parent: Rc<RefCell<Scope>>) -> Self {
        Self {
            parent: Some(parent),
            funs: HashMap::new(),
            vars: HashMap::new(),
        }
    }

    /// Associates `name` to the given function ref, returning the previous value
    /// if one was present.
    pub fn define_function(&mut self, name: Symbol, f: FuncRef) -> Option<FuncRef> {
        use std::collections::hash_map::Entry;
        debug_assert_ne!(
            name,
            symbols::Underscore,
            "the wildcard symbol '_' cannot be defined as a functino"
        );

        match self.funs.entry(name) {
            Entry::Vacant(e) => {
                e.insert(f);
                None
            }
            Entry::Occupied(mut e) => Some(e.insert(f)),
        }
    }

    /// Returns the function ref bound to the given name, if in scope
    pub fn function(&self, name: Symbol) -> Option<FuncRef> {
        if name == symbols::Underscore {
            return None;
        }
        if let Some(f) = self.funs.get(&name) {
            return Some(*f);
        }
        if let Some(parent) = self.parent.as_ref() {
            let cell = parent.as_ref();
            let scope = cell.borrow();
            return scope.function(name);
        }
        None
    }

    /// Associates `name` to the given value, returning the previous value
    /// if one was present.
    pub fn define_var(&mut self, name: Symbol, v: Value) -> Option<Value> {
        use std::collections::hash_map::Entry;
        debug_assert_ne!(
            name,
            symbols::Underscore,
            "the wildcard symbol '_' cannot be defined as a variable"
        );

        match self.vars.entry(name) {
            Entry::Vacant(e) => {
                e.insert(v);
                None
            }
            Entry::Occupied(mut e) => Some(e.insert(v)),
        }
    }

    /// Returns the value bound to the given name, if in scope
    pub fn var(&self, name: Symbol) -> Option<Value> {
        if name == symbols::Underscore {
            return None;
        }
        if let Some(v) = self.vars.get(&name) {
            return Some(*v);
        }
        if let Some(parent) = self.parent.as_ref() {
            let cell = parent.as_ref();
            let scope = cell.borrow();
            return scope.var(name);
        }
        None
    }

    /// Returns the parent of this scope, if there is one
    ///
    /// If this returns None, the current scope is the root scope
    pub fn parent(&self) -> Option<Rc<RefCell<Scope>>> {
        self.parent.as_ref().map(|p| Rc::clone(p))
    }
}
