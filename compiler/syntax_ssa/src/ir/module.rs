use std::cell::{Ref, RefCell};
use std::collections::{BTreeMap, BTreeSet};
use std::rc::Rc;

use cranelift_entity::PrimaryMap;

use firefly_diagnostics::SourceSpan;
use firefly_intern::{symbols, Ident, Symbol};
use firefly_syntax_base::*;
use firefly_util::emit::Emit;

use super::*;

/// Represents a SSA IR module
///
/// This module is largely a container for functions, but it also acts
/// as the owner for pooled resources available to functions:
///
/// * Mapping from Signature to FuncRef
/// * Mapping from FunctionName to FuncRef
/// * Constant pool
#[derive(Debug, Clone)]
pub struct Module {
    /// The name and source span of this module
    pub name: Ident,
    /// This is the list of functions defined in this module
    pub functions: Vec<Function>,
    /// This map associates known functions (locals, builtins, imports, or externals) to a reference value
    pub signatures: Rc<RefCell<PrimaryMap<FuncRef, Signature>>>,
    /// This map provides a quick way to look up function references given an MFA, useful when lowering
    pub callees: Rc<RefCell<BTreeMap<FunctionName, FuncRef>>>,
    /// This map provides a quick way to look up function references for native functions which do not
    /// use our naming convention
    pub natives: Rc<RefCell<BTreeMap<Symbol, FuncRef>>>,
    /// This pool contains de-duplicated constant values/data used in the current module
    pub constants: Rc<RefCell<ConstantPool>>,
    /// This set contains the local functions which are lifted closures (i.e. expect a closure env)
    pub closures: BTreeSet<FunctionName>,
}
// UNSAFE: These is only safe because we make sure we don't actually use
// a module in more than one thread
unsafe impl Send for Module {}
unsafe impl Sync for Module {}
impl Eq for Module {}
impl PartialEq for Module {
    fn eq(&self, other: &Self) -> bool {
        self.name == other.name && self.functions.len() == other.functions.len()
    }
}
impl Emit for Module {
    fn file_type(&self) -> Option<&'static str> {
        Some("ssa")
    }

    fn emit(&self, f: &mut std::fs::File) -> anyhow::Result<()> {
        use std::io::Write;
        write!(f, "module {}\n", &self.name)?;
        for function in self.functions.iter() {
            writeln!(f)?;
            crate::write::write_function(f, function)?;
        }

        Ok(())
    }
}
impl Module {
    pub fn new(name: Ident) -> Self {
        Self {
            name,
            functions: vec![],
            signatures: Rc::new(RefCell::new(PrimaryMap::new())),
            callees: Rc::new(RefCell::new(BTreeMap::new())),
            natives: Rc::new(RefCell::new(BTreeMap::new())),
            constants: Rc::new(RefCell::new(ConstantPool::new())),
            closures: BTreeSet::new(),
        }
    }

    /// Returns the source span associated with this module's declaration
    pub fn span(&self) -> SourceSpan {
        self.name.span
    }

    /// Returns the name of the current module as a symbol
    pub fn name(&self) -> Symbol {
        self.name.name
    }

    /// If the given function is defined in this module, return true
    pub fn is_local(&self, name: &FunctionName) -> bool {
        if let Some(id) = self.get_callee(*name) {
            let sigs = self.signatures.borrow();
            sigs.get(id)
                .map(|sig| sig.visibility.is_locally_defined())
                .is_some()
        } else {
            false
        }
    }

    pub fn is_closure(&self, name: &FunctionName) -> bool {
        self.closures.contains(name)
    }

    /// If the given function was imported, return the fully resolved name
    pub fn get_import(&self, name: &FunctionName) -> Option<FunctionName> {
        let callees = self.callees.borrow();
        if let Some(func_ref) = callees.get(name).copied() {
            let signatures = self.signatures.borrow();
            signatures.get(func_ref).map(|sig| sig.mfa())
        } else {
            None
        }
    }

    /// Imports a function from another module into scope as if defined in the current module
    ///
    /// The returned reference may be modified to refer to another function if a shadowing declaration is created later
    pub fn import_function(&mut self, mut signature: Signature) -> FuncRef {
        assert!(
            signature.module != self.name.name,
            "cannot import a local function"
        );

        // Ensure the visibility flags for this signature are properly set
        signature
            .visibility
            .insert(Visibility::PUBLIC | Visibility::IMPORTED | Visibility::EXTERNAL);

        let mfa = signature.mfa();
        // If already imported, we're done
        {
            let callees = self.callees.borrow();
            if let Some(f) = callees.get(&mfa) {
                return *f;
            }
        }
        // Create the function reference to the fully-qualified name
        let mut signatures = self.signatures.borrow_mut();
        let mut callees = self.callees.borrow_mut();
        let f = signatures.push(signature);
        // Register the fully-qualified name as a callee
        callees.insert(mfa, f);
        // Register the local name as a callee
        callees.insert(mfa.to_local(), f);
        // Return the reference
        f
    }

    pub fn get_native(&self, op: Symbol) -> Option<FuncRef> {
        let natives = self.natives.borrow();
        natives.get(&op).copied()
    }

    /// Registers a known primop/bif as a callee of this module.
    ///
    /// Rather than always insert every builtin in every module, we lazily do so
    /// on demand.
    ///
    /// The given MFA must have a signature defined in syntax_base::bifs
    pub fn get_or_register_builtin(&mut self, op: FunctionName) -> FuncRef {
        assert_eq!(op.module, Some(symbols::Erlang));
        assert!(op.is_primop() || op.is_bif());

        {
            let callees = self.callees.borrow();
            if let Some(f) = callees.get(&op).copied() {
                return f;
            }
        }

        let signature = bifs::fetch(&op).clone();
        // Create the function reference to the fully-qualified name
        let mut signatures = self.signatures.borrow_mut();
        let mut callees = self.callees.borrow_mut();
        let f = signatures.push(signature);
        // Register the fully-qualified name as a callee
        callees.insert(op, f);
        // Return the reference
        f
    }

    /// Registers a known native function as a callee of this module.
    ///
    /// The given name must have a signature defined in syntax_base::nifs to ensure
    /// we have a canonical source of known native functions
    ///
    /// NOTE: Signatures of native functions must have a module symbol, but it is
    /// ignored when lowered, you should prefer to use symbols::Empty when defining
    /// signatures for these functions
    pub fn get_or_register_native(&mut self, op: Symbol) -> FuncRef {
        {
            let natives = self.natives.borrow();
            if let Some(f) = natives.get(&op).copied() {
                return f;
            }
        }

        let signature = nifs::fetch(&op).clone();
        // Create the function reference to the fully-qualified name
        let mut signatures = self.signatures.borrow_mut();
        let mut natives = self.natives.borrow_mut();
        let f = signatures.push(signature);
        // Register the fully-qualified name as a callee
        natives.insert(op, f);
        // Return the reference
        f
    }

    /// Same as `declare_function`, but marks the function as a known closure
    pub fn declare_closure(&mut self, signature: Signature) -> FuncRef {
        let local_mfa = signature.mfa().to_local();
        self.closures.insert(local_mfa);
        self.declare_function(signature)
    }

    /// Declares a function in the current module with the given signature, and creates the empty
    /// definition for it. Use the returned funcref to obtain a reference to that definition using
    /// `get_function` or `get_function_mut`.
    ///
    /// If the declared function has the same name/arity as a builtin or imported function, the
    /// declaration shadows the previous signature. If there are any existing references to the
    /// shadowed function, they are transparently updated to refer to the new declaration
    pub fn declare_function(&mut self, mut signature: Signature) -> FuncRef {
        // Ensure the visibility flags for this signature are properly set
        signature
            .visibility
            .remove(Visibility::IMPORTED | Visibility::EXTERNAL);

        // When declaring a function that was already imported, the import is shadowed,
        // so we need to update existing references to point to the new signature
        let mfa = signature.mfa();
        let local_mfa = mfa.to_local();
        let shadowed = {
            let callees = self.callees.borrow();
            callees.get(&local_mfa).copied()
        };
        let mut signatures = self.signatures.borrow_mut();
        let mut callees = self.callees.borrow_mut();
        if let Some(f) = shadowed {
            // Rewrite the signature of the shadowed import
            let sig = signatures.get_mut(f).unwrap();
            *sig = signature.clone();
            f
        } else {
            // Register the signature
            let f = signatures.push(signature.clone());
            // Register both the fully-qualified and local names
            callees.insert(mfa, f);
            callees.insert(local_mfa, f);
            f
        }
    }

    /// Adds the definition of a function which was previously declared
    ///
    /// This function will panic if the function provided is not declared in this module
    pub fn define_function(&mut self, function: Function) {
        let signatures = self.signatures.borrow();
        assert!(
            signatures.get(function.id).is_some(),
            "cannot define a function which was not declared in this module"
        );

        self.functions.push(function);
    }

    /// Returns the signature of the given function reference
    pub fn call_signature(&self, callee: FuncRef) -> Ref<'_, Signature> {
        Ref::map(self.signatures.borrow(), |sigs| sigs.get(callee).unwrap())
    }

    /// Looks up the concrete function for the given MFA (module of None indicates that it is a local or imported function)
    pub fn get_callee(&self, mfa: FunctionName) -> Option<FuncRef> {
        self.callees.borrow().get(&mfa).copied()
    }

    /// Returns a reference to the definition of the given funcref, if it refers to a local definition
    pub fn get_function(&self, id: FuncRef) -> Option<&Function> {
        self.functions.iter().find(|f| f.id == id)
    }

    /// Returns a mutable reference to the definition of the given funcref, if it refers to a local definition
    pub fn get_function_mut(&mut self, id: FuncRef) -> Option<&mut Function> {
        self.functions.iter_mut().find(|f| f.id == id)
    }
}
