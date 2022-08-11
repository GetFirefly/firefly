use std::cell::{Ref, RefCell};
use std::collections::BTreeMap;
use std::rc::Rc;

use cranelift_entity::PrimaryMap;

use liblumen_diagnostics::SourceSpan;
use liblumen_intern::{Ident, Symbol};
use liblumen_util::emit::Emit;

use super::*;

/// Represents a SSA IR module
///
/// This module is largely a container for functions, but it also acts
/// as the owner for pooled resources available to functions:
///
/// * Mapping from Signature to FuncRef
/// * Mapping from FunctionName to FuncRef
/// * Mapping from Annotation to AnnotationData
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
    /// This map stores annotation data and provides a handle for associating to instructions
    pub annotations: Rc<RefCell<PrimaryMap<Annotation, AnnotationData>>>,
    /// This pool contains de-duplicated constant values/data used in the current module
    pub constants: Rc<RefCell<ConstantPool>>,
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
        Some("core")
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
        let mut annotations = PrimaryMap::new();
        // This should always be the first annotation so we can construct without interacting with the annotation map
        annotations.push(AnnotationData::CompilerGenerated);
        Self {
            name,
            functions: vec![],
            signatures: Rc::new(RefCell::new(PrimaryMap::new())),
            callees: Rc::new(RefCell::new(BTreeMap::new())),
            annotations: Rc::new(RefCell::new(annotations)),
            constants: Rc::new(RefCell::new(ConstantPool::new())),
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

    /// Registers a builtin function in the current module with the given signature.
    ///
    /// Calling this function twice for the same MFA will return the signature that was
    /// registered first.
    pub fn register_builtin(&mut self, signature: Signature) -> FuncRef {
        let mfa = signature.mfa();
        assert!(mfa.module.is_some());

        {
            let callees = self.callees.borrow();
            if let Some(f) = callees.get(&mfa).copied() {
                return f;
            }
        }
        // Create the function reference to the fully-qualified name
        let mut signatures = self.signatures.borrow_mut();
        let mut callees = self.callees.borrow_mut();
        let f = signatures.push(signature);
        // Register the fully-qualified name as a callee
        callees.insert(mfa, f);
        // Return the reference
        f
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
