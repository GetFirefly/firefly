use std::fmt::Display;
use std::collections::HashMap;
use std::boxed::Box;

use syntax::ast::ast::ModuleDecl;
use syntax::ast::ast::form::Form;
use syntax::ast::ast::clause::Clause;

use super::*;

pub struct Parameter {
    name: String,
    ty: Type,
    attrs: Vec<FunctionAttribute>,
}
impl Parameter {
    pub fn new(name: &str, ty: Type) -> Parameter {
        Parameter { name: name.to_string(), ty, attrs: Vec::new() }
    }

    pub fn set_attr(&mut self, attr: FunctionAttribute) {
        self.attrs.push(attr);
    }
}

pub struct Module {
    pub name: String,
    globals: HashMap<String, Global>,
    declared: HashMap<String, Function>,
    defined: HashMap<String, Function>,
    next_id: u32,
}
impl Module {
    pub fn new(name: &str) -> Module {
        Module {
            name: name.to_string(),
            globals: HashMap::new(),
            declared: HashMap::new(),
            defined: HashMap::new(),
            next_id: 0,
        }
    }

    pub fn from_ast(name: &str, ast: ModuleDecl) -> Result<Module, CodeGenError> {
        let module = Module::new(name);
        // Declare prelude
        let puts = Function::new("__lumen_system_io_puts");
        puts.set_return_type(Type::Void);
        puts.push_arg(Parameter::new("item", pointer_type(Type::Integer(8))));
        module.declare_function(puts);

        for form in ast.forms.iter() {
            form.generate(&module)?;
        }
        Ok(module)
    }

    pub fn set_current_function(&mut self, fun: Function) -> &Function {
        self.current_function = Some(fun);
        &fun
    }

    pub fn set_current_block(&mut self, block: Block) -> &Block {
        self.current_block = Some(block);
        &block;
    }

    pub fn end_current_block(&mut self) {
        let block = self.current_block.expect("expected current block");
        let fun = self.current_function.expect("expected current function");
        fun.push_block(block);
        self.current_function(fun);
    }

    pub fn end_current_function(&mut self) {
        match self.block {
            None => {
                let fun = self.current_function.expect("expected current function");
                self.define_function(fun);
                self.current_function = None;
            },
            Some(block) => {
                let fun = self.current_function.expect("expected current function");
                fun.push_block(block);
                self.define_function(fun);
                self.current_block = None;
                self.current_function = None;
            }
        }
    }

    pub fn declare_function(&mut self, fun: Function) {
        self.declared.insert(fun.name, fun);
    }

    pub fn define_function(&mut self, fun: Function) {
        self.defined.insert(fun.name, fun);
    }

    pub fn declare_global(&mut self, global: Global) {
        self.globals.insert(global.name, global);

    }

    pub fn to_string(&self) -> String {
        let mut ir = format!("; ModuleID = '{}'\n\n", self.name);
        for (_name, global) in self.globals.iter() {
            ir.push_str(global.to_string())
        }
        ir.push_str("\n");
        for (_name, fun) in self.declared.iter() {
            ir.push_str(fun.to_string())
        }
        ir.push_str("\n");
        for (_name, fun) in self.defined.iter() {
            ir.push_str(fun.to_string())
        }
        ir.push_str("\n");
        return ir.to_string();
    }

    pub fn is_global(&self, name: &str) -> bool {
        self.globals.contains_key(name.to_string())
    }

    pub fn is_function(&self, name: &str) -> bool {
        let name = name.to_string();
        self.declared.contains_key(name) || self.defined.contains_key(name)
    }

    pub fn next_id(&mut self) -> u32 {
        let next = self.next_id;
        self.next_id = next + 1;
        next
    }
}

pub struct Global {
    pub name: String,
    constant: bool,
    linkage: Option<Linkage>,
    visibility: Visibility,
    unnamed_addr: Option<UnnamedAddr>,
    ty: Type,
    value: Option<String>,
}
impl Global {
    pub fn new(name: &str, constant: bool, ty: Type, value: Option<String>) -> Global {
        Global {
            name: name.to_string(),
            constant,
            linkage: None,
            visibility: Visibility::Default,
            ty,
            value,
            attrs: Vec::new(),
        }
    }

    pub fn declare(name: &str, ty: Type) -> Global {
        Global::new(name, false, ty, None)
    }

    pub fn declare_constant(name: &str, ty: Type) -> Global {
        Global::new(name, true, ty, None)
    }

    pub fn define(name: &str, ty: Type, value: &str) -> Global {
        Global::new(name, false, ty, Some(value.to_string()))
    }

    pub fn define_constant(name: &str, ty: Type, value: Value) -> Global {
        Global::new(name, true, ty, Some(value.to_string()))
    }

    pub fn set_linkage(&mut self, linkage: Linkage) {
        self.linkage = Some(linkage);
    }

    pub fn set_visibility(&mut self, visibility: Visibility) {
        self.visibility = visibility;
    }

    pub fn set_address_is_insignificant_globally(&mut self) {
        self.unnamed_addr = Some(UnnamedAddr::Global);
    }

    pub fn set_address_is_insignificant_locally(&mut self) {
        self.unnamed_addr = Some(UnnamedAddr::Local);
    }

    pub fn to_string(&self) -> &str {
        let linkage = self.linkage.unwrap_or("");
        let visibility = self.visibility;
        let unnamed_addr = self.unnamed_addr.unwrap_or("");
        let var_type = if self.constant {
            "constant"
        } else {
            "global"
        };
        let val_type = self.ty;
        let value = self.value.unwrap_or("");
        format!("@{} = {} {} {} {} {}", linkage, visibility, unnamed_addr, var_type, val_type, value)
    }
}
impl Display for Global {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}", self.to_string())
    }
}

pub enum Linkage {
    Private,
    Internal, // like 'static' in C
    AvailableExternally, // expected to be imported from another module
    External, // exporting from the current module
    Appending, // Used to merge globals of array type into one global array
    // There are others, but we don't use them
}
impl Display for Linkage {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let s = match *self {
            Linkage::Private => "private",
            Linkage::Internal => "internal",
            Linkage::AvailableExternally => "available_externally",
            Linkage::External => "external",
            Linkage::Appending => "appending"
        };
        write!(f, "{}", s)
    }
}

pub enum CallingConvention {
    // Necessary to support FFI, and LLVMs default
    C,
    // Lets LLVM optimize for fast code, supports tail calls, should not be used for FFI
    Fast,
    // Lets LLVM optimize for callers, by preserving registers when calling cold funs, should not be used for FFI
    Cold,
    // Target-specific or custom calling conventions (by numeric id, target-specific ids start at 64)
    Custom(u32)
}
impl Display for CallingConvention {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let s = match *self {
            CallingConvention::C => "ccc",
            CallingConvention::Fast => "fastcc",
            CallingConvention::Cold => "coldcc",
            CallingConvention::Custom(id) => format!("cc {}", id)
        };
        write!(f, "{}", s)
    }
}

// All global variables and functions have a visibility
// When using internal/private linkage, use default visibility
pub enum Visibility {
    // visible to other modules, can be overridden (i.e. externally linked)
    Default,
    // not visible to other modules
    Hidden,
    // visible to other modules, can not be overridden
    Protected
}
impl Display for Visibility {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let s = match *self {
            Visibility::Default => "default",
            Visibility::Hidden => "hidden",
            Visibility::Protected => "protected"
        };
        write!(f, "{}", s)
    }
}

// Indicates that the address of a function is insignificant, only the content
pub enum UnnamedAddr {
    // Applies only to the current module
    Local,
    // Applies globally
    Global
}
impl Display for UnnamedAddr {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let s = match *self {
            UnnamedAddr::Local => "local_unnamed_addr",
            UnnamedAddr::Global => "unnamed_addr",
        };
        write!(f, "{}", s)
    }
}

pub enum FunctionAttribute {
    Inline, // hint LLVM that it might be good to inline this fun
    AlwaysInline, // tell LLVM to always inline this fun
    NoInline, // tell LLVM to never inline this fun
    NoReturn, // this fun never returns (e.g., exit/1)
    NoUnwind, // this fun never returns with an unwind or exceptional control flow
    BuiltIn, // tell LLVM to treat calls to this as if it is builtin
    NoBuiltIn, // tell LLVM to not treat calls as builtins
    Cold, // tell LLVM that this function is rarely called
    // This attribute indicates that the function computes its result (or decides to unwind an exception)
    // based strictly on its arguments, without dereferencing any pointer arguments or otherwise accessing
    // any mutable state (e.g. memory, control registers, etc) visible to caller functions. It does not write
    // through any pointer arguments (including byval arguments) and never changes any state visible to callers.
    // This means that it cannot unwind exceptions by calling the C++ exception throwing methods, but could
    // use the unwind instruction.
    ReadNone,
    // This attribute indicates that the function does not write through any pointer arguments (including byval arguments)
    // or otherwise modify any state (e.g. memory, control registers, etc) visible to caller functions.
    // It may dereference pointer arguments and read state that may be set in the caller. A readonly function always
    // returns the same value (or unwinds an exception identically) when called with the same set of arguments and
    // global state. It cannot unwind an exception by calling the C++ exception throwing methods, but may use the
    // unwind instruction.
    ReadOnly,
    WriteOnly, // like readonly, but for writes
    ArgMemoryOnly, // loosens readonly to allow derefrencing argument pointers
    InaccessibleMemoryOnly, // loosens readonly to allow accessing memory unknown to the program
    InaccessibleMemoryOrArgMemoryOnly, // combines the previous two
    JumpTable, // convert pointers to this function to lookups in a jump table
    MinSize, // tell LLVM to reduce size over all other priorities
    OptSize, // tell LLVM to reduce size, but do not sacrifice performance completely
    Naked, // tell LLVM to omit the prologue/epilogue of the function (which manages the stack)
    ReturnsTwice, // if a function can return twice, e.g. due to setjmp
    StrictFloatingPoint, // tell LLVM to never perform floating point optimizations in this function
    Thunk, // the current function delegates to another function with a tail call
}
impl Display for FunctionAttribute {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        use self::FunctionAttribute::*;
        let s = match *self {
            Inline => "inlinehint",
            AlwaysInline => "alwaysinline",
            NoInline => "noinline",
            NoReturn => "noreturn",
            NoUnwind => "nounwind",
            BuiltIn => "builtin",
            NoBuiltIn => "nobuiltin",
            Cold => "cold",
            ReadNone => "readnone",
            ReadOnly => "readonly",
            WriteOnly => "writeonly",
            ArgMemoryOnly => "argmemonly",
            InaccessibleMemoryOnly => "inaccessiblememonly",
            InaccessibleMemoryOrArgMemoryOnly => "inaccessiblemem_or_argmemonly",
            JumpTable => "jumptable",
            MinSize => "minsize",
            OptSize => "optsize",
            Naked => "naked",
            ReturnsTwice => "returns_twice",
            StrictFloatingPoint => "strictfp",
            Thunk => "thunk"
        };
        write!(f, "{}", s)
    }
}

pub struct Function {
    pub name: String,
    linkage: Option<Linkage>,
    visibility: Visibility,
    unnamed_addr: Option<UnnamedAddr>,
    convention: Option<CallingConvention>,
    return_type: Option<Type>,
    return_type_attr: Option<String>,
    args: Vec<(String, Parameter)>,
    attrs: Vec<FunctionAttribute>,
    gc: Option<String>,
    blocks: Vec<Block>,
    next_id: u32,
}
impl Function {
    pub fn new(name: &str) -> Function {
        Function {
            name: name.to_string(),
            linkage: None,
            visibility: Visibility::Default,
            return_type: None,
            args: HashMap::new(),
            attrs: Vec::new(),
            blocks: Vec::new(),
            next_id: 0,
        }
    }

    pub fn set_linkage(&mut self, linkage: Linkage) {
        self.linkage = Some(linkage);
    }

    pub fn set_return_type(&mut self, ty: Type) {
        self.return_type = Some(ty);
    }

    pub fn set_attr(&mut self, attr: FunctionAttribute) {
        self.attrs.push(attr);
    }

    pub fn push_arg(&mut self, name: &str, arg: Parameter) {
        self.args.push((name.to_string(), arg));
    }

    pub fn push_block(&mut self, block: Block) {
        self.blocks.push(block);
    }

    pub fn next_id(&mut self) -> u32 {
        let next = self.next_id;
        self.next_id = next + 1;
        next
    }
}
impl Display for Function {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        // TODO
        unimplemented!()
    }
}

pub struct Block {
    pub name: String,
    instructions: Vec<String>,
    next_id: u32,
}
impl Block {
    pub fn new(name: &str) -> Block {
        Block { name: name.to_string(), instructions: Vec::new(), next_id: 0 }
    }

    pub fn next_id(&mut self) -> u32 {
        let next = self.next_id;
        self.next_id = next + 1;
        next
    }

    pub fn call(&mut self, module: &Module, name: &str, return_type: Type, args: Vec<Value>) {
        let args = args.iter().map(|v| {
            match arg {
                Value::GlobalRef(name) => {
                    let g = module.globals.get(name).unwrap();
                    format!("i8* getelementptr inbounds ({}* @{}, i32 0, i32 0)", global.name, global.ty)
                },
                Value::IntegerLiteral(size, val) => {
                    format!("i{} {}", size, val)
                },
                Value::StringLiteral(s) => {
                    panic!("invalid usage of Value::StringLiteral");
                }
            }
        }).collect();
        let id = self.next_id();
        let ix = format!("%{} = call {} @{}({})", id, name, return_type, args);
        self.instructions.push(ix);
    }

    pub fn ret(&mut self, val: Value) {
        self.instructions.push(format!("ret {}", val))
    }
}

pub enum Type {
    Void,
    // First class
    Integer(u32),
    Float,
    Double,
    DoubleDouble,
    Pointer(Box<Type>),
    Vector(u32, Box<Type>),
    Function(Box<Type>, Vec<Box<Type>>),
    // Aggregates
    Array(u32, Box<Type>),
    Struct { name: String, packed: bool, fields: Vec<Box<Type>> },
    Opaque
}
impl Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let s = match *self {
            Type::Void => "void",
            Type::Integer(size) => &format!("i{}", size),
            Type::Float => "float",
            Type::Double => "double",
            Type::DoubleDouble => "fp128",
            Type::Pointer(ty) => &format!("{}*", ty),
            Type::Vector(size, ty) => &format!("<{} x {}>", size, ty),
            Type::Function(return_ty, params) => {
                let params: Vec<String> = params.iter().map(|p| p.to_string()).collect();
                let ps = params.join(", ");
                &format!("{} ({})", return_ty, ps)
            },
            Type::Array(size, ty) => &format!("[{} x {}]", size, ty),
            Type::Struct { name: _name, packed, fields } => {
                let fields: Vec<String> = fields.iter().map(|f| f.to_string()).collect();
                let fs = fields.join(", ");
                if packed {
                    &format!("<{{{}}}>", fs)
                } else {
                    &format!("{{{}}}", fs)
                }
            },
            Type::Opaque => "opaque"
        };
        write!(f, "{}", s)
    }
}

pub enum Value {
    StringLiteral(String),
    IntegerLiteral(u32, u32),
    GlobalRef(String),
}
impl Display for Value {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match *self {
            Value::StringLiteral(s) => write!(f, "c\"{}\\0\"", s),
            Value::IntegerLiteral(_size, i) => write!(f, "{}", i),
            Value::GlobalRef(name) => write!(f, "@{}", name)
        }
    }
}

impl CodeGen for Form {
    fn generate(&self, module: &Module) -> Result<(), CodeGenError> {
        use syntax::ast::ast::form::*;
        match *self {
            Form::Fun(FunDecl { ref name, clauses: ref clauses, .. }) => {
                // Create and open function
                let fun = Function::new(name);
                fun.set_return_type(Type::Void);
                module.set_current_function(fun);
                // TODO: reduce multiple clauses to single clause + case
                for clause in clauses.iter() {
                    clause.generate(module)?;
                }
                module.end_current_function();
                Ok(())
            }
            _ => Ok(()),
        }
    }
}

impl CodeGen for Clause {
    fn generate(&self, module: &Module) -> Result<(), CodeGenError> {
        let block = Block::new("entry");
        let block = module.set_current_block(block);
        // TODO: Example body
        let greeting = "Hello, world!";
        Global::define_constant("greeting", array_type(greeting.len(), Type::Integer(8)), Value::StringLiteral(greeting.to_string()));
        block.call(module, "__lumen_system_io_puts", Type::Void, vec![Value::GlobalRef("greeting".to_string())]);
        block.ret(Value::IntegerLiteral(32, 0));
        module.end_current_block();
        Ok(())
    }
}

fn array_type(len: usize, ty: Type) -> Type {
    Type::Array(len as u32, Box::new(ty))
}

fn pointer_type(ty: Type) -> Type {
    Type::Pointer(Box::new(ty))
}
