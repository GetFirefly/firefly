use std::collections::HashMap;
use std::fs::File;
use std::hash::Hasher;
use std::path::Path;
use std::str;
use std::sync::Arc;

use firefly_bytecode::{Atom, AtomicStr, BytecodeWriter, Function, ModuleFunctionArity};
use firefly_linker::ModuleArtifacts;
use firefly_llvm::builder::ModuleBuilder;
use firefly_llvm::target::{OwnedTargetMachine, TargetMachine};
use firefly_llvm::{Align, GlobalValue, Linkage, Type, Value};
use firefly_pass::Pass;
use firefly_session::{Input, Options, OutputType};
use firefly_util::diagnostics::DiagnosticsHandler;

/// This pass lowers a bytecode module to a native object which exposes a global variable
/// containing the encoded bytecode, to be linked in to a compiled program so that the
/// emulator can find and load it on boot.
pub struct LowerBytecode {
    options: Arc<Options>,
    context: firefly_llvm::OwnedContext,
    target_machine: OwnedTargetMachine,
}
impl LowerBytecode {
    pub fn new(options: Arc<Options>, diagnostics: Arc<DiagnosticsHandler>) -> Self {
        let context = firefly_llvm::OwnedContext::new(options.clone(), diagnostics);
        let target_machine = TargetMachine::create(&options).unwrap();
        Self {
            options,
            context,
            target_machine,
        }
    }
}
impl Pass for LowerBytecode {
    type Input<'a> = firefly_bytecode::StandardByteCode;
    type Output<'a> = firefly_linker::ModuleArtifacts;

    fn run<'a>(&mut self, module: Self::Input<'a>) -> anyhow::Result<Self::Output<'a>> {
        // Encode bytecode as an array of bytes
        let mut buffer = Vec::<u8>::with_capacity(24 * 1024 * 1024);
        let writer = BytecodeWriter::new(&mut buffer);
        writer.write(&module)?;

        // Construct a module with the same name as the application being compiled
        let module_name = self.options.app.name.as_str().get();
        let builder = ModuleBuilder::new(
            module_name,
            &self.options,
            &self.context,
            self.target_machine.handle(),
        )?;

        let usize_type = builder.get_usize_type();
        let i8_ptr_type = builder.get_pointer_type(builder.get_i8_type());
        let atom_data_type =
            builder.get_struct_type(Some("AtomData"), &[usize_type.base(), i8_ptr_type.base()]);
        let is_simple_atom = |atom: &str| {
            for c in atom.chars() {
                match c {
                    'A'..='Z' => continue,
                    'a'..='z' => continue,
                    '0'..='9' => continue,
                    '_' => continue,
                    _ => return false,
                }
            }
            true
        };
        let mut atom_entries = HashMap::new();
        let mut insert_atom_table_entry = |atom: AtomicStr| {
            if let Some(gv) = atom_entries.get(&atom) {
                return *gv;
            }

            let name = str::from_utf8(atom.as_bytes()).unwrap();
            let is_simple = is_simple_atom(name);
            let symbol_name = if is_simple {
                format!("atom_{}", name)
            } else {
                let mut hasher = rustc_hash::FxHasher::default();
                hasher.write(name.as_bytes());
                let hash = hasher.finish();
                format!("atom_{}", hash)
            };
            let cstring_name = if is_simple {
                format!("cstr_{}", name)
            } else {
                let mut hasher = rustc_hash::FxHasher::default();
                hasher.write(name.as_bytes());
                let hash = hasher.finish();
                format!("cstr_{}", hash)
            };
            let section_name = if self.options.target.options.is_like_osx {
                "__DATA,__atoms"
            } else {
                "__atoms"
            };

            let cstring_bytes = builder.build_constant_bytes(name.as_bytes());
            let cstring_global = builder
                .define_global(&cstring_name, cstring_bytes.get_type())
                .unwrap();
            cstring_global.set_linkage(Linkage::LinkOnceODR);
            cstring_global.set_initializer(cstring_bytes);
            cstring_global.set_constant(true);

            let atom_size = builder.build_constant_uint(usize_type, name.as_bytes().len() as u64);

            let atom_data = builder.build_constant_named_struct(
                atom_data_type,
                &[atom_size.into(), cstring_global.try_into().unwrap()],
            );

            let global = builder.define_global(&symbol_name, atom_data_type).unwrap();
            global.set_linkage(Linkage::LinkOnceODR);
            global.set_alignment(8);
            global.set_section(section_name);
            global.set_initializer(atom_data);
            global.set_constant(true);
            atom_entries.insert(atom, global);
            global
        };

        // Insert symbols for all referenced native functions, this ensures they are linked
        let mut mapped_functions = HashMap::new();
        let atom_data_ptr_type = builder.get_pointer_type(atom_data_type);
        let i8_type = builder.get_i8_type();
        let void_type = builder.get_void_type();
        let void_function_type = builder.get_function_type(void_type, &[], false);
        let void_ptr_type = builder.get_pointer_type(void_function_type);
        let dispatch_entry_type = builder.get_struct_type(
            Some("FunctionSymbol"),
            &[
                atom_data_ptr_type.base(),
                atom_data_ptr_type.base(),
                i8_type.base(),
                void_ptr_type.base(),
            ],
        );
        let mut insert_dispatch_table_entry =
            |mfa: &ModuleFunctionArity<AtomicStr>, linkage: Linkage| {
                if let Some(gv) = mapped_functions.get(mfa) {
                    return *gv;
                }
                let module = insert_atom_table_entry(mfa.module);
                let function = insert_atom_table_entry(mfa.function);
                let name = mfa.to_string();
                let pointee = builder.build_function_with_attrs(
                    name.as_str(),
                    void_function_type,
                    linkage,
                    &[],
                );
                let mut hasher = rustc_hash::FxHasher::default();
                hasher.write(name.as_bytes());
                let hash = hasher.finish();
                let dispatch_entry_name = format!("firefly_dispatch_{:x}", hash);
                let section_name = if self.options.target.options.is_like_osx {
                    "__DATA,__dispatch"
                } else {
                    "__dispatch"
                };
                let global = builder
                    .define_global(&dispatch_entry_name, dispatch_entry_type)
                    .unwrap();
                global.set_linkage(Linkage::LinkOnceODR);
                global.set_alignment(8);
                global.set_section(section_name);

                let arity = builder.build_constant_uint(i8_type, mfa.arity as u64);
                let entry = builder.build_constant_named_struct(
                    dispatch_entry_type,
                    &[
                        module.try_into().unwrap(),
                        function.try_into().unwrap(),
                        arity.into(),
                        pointee.into(),
                    ],
                );

                global.set_initializer(entry);
                mapped_functions.insert(*mfa, global);
                global
            };

        let mut is_empty = true;
        for function in module.functions.iter() {
            match function {
                Function::Bytecode {
                    is_nif: true, mfa, ..
                } => {
                    is_empty = false;
                    // We must weakly link the native symbol, because the bytecode definition is
                    // used when the native version isn't available
                    insert_dispatch_table_entry(mfa, Linkage::ExternalWeak);
                }
                Function::Bytecode { .. } => continue,
                Function::Bif { mfa, .. } => {
                    is_empty = false;
                    // All BIFs must be available at link time
                    insert_dispatch_table_entry(mfa, Linkage::External);
                }
                Function::Native { name, .. } => {
                    // All referenced native symbols without a bytecode definition must also be
                    // available at link time
                    builder.build_external_function(name.as_bytes(), void_function_type);
                }
            }
        }

        // Make sure we have at least one function in the dispatch table
        //
        // This is very much an edge case, but linking will fail for the crt crate
        // on platforms other than macOS if we don't have anything in the section
        if is_empty {
            // We choose to use `erlang:display/1` here, since:
            //
            // * It is a BIF
            // * It is always natively-implemented
            // * It is almost always in real programs anyway
            insert_dispatch_table_entry(
                &ModuleFunctionArity {
                    module: "erlang".into(),
                    function: "display".into(),
                    arity: 1,
                },
                Linkage::External,
            );
        }

        // Insert a global constant value containing the size of the bytecode in bytes
        let constant = builder.build_constant_uint(usize_type, buffer.len() as u64);
        let gv_len = builder
            .define_global("__FIREFLY_BC_LEN", usize_type)
            .unwrap();
        gv_len.set_initializer(constant);
        gv_len.set_linkage(Linkage::LinkOnceODR);
        gv_len.set_alignment(8);

        // Insert a global constant value containing the bytecode bytes
        let constant = builder.build_constant_bytes(buffer.as_slice());
        let gv_bytes = builder
            .define_global("__FIREFLY_BC", constant.get_type())
            .unwrap();
        gv_bytes.set_initializer(constant);
        gv_bytes.set_linkage(Linkage::LinkOnceODR);
        gv_bytes.set_alignment(64);

        let module = builder.finish()?;

        // We need an input to represent the generated source
        let input = Input::from(Path::new(module_name));

        // Emit LLVM IR
        if let Some(ir_path) = self.options.maybe_emit(&input, OutputType::LLVMAssembly) {
            let mut file = File::create(ir_path.as_path())?;
            module.emit_ir(&mut file)?;
        }

        // Emit LLVM bitcode
        if let Some(bc_path) = self.options.maybe_emit(&input, OutputType::LLVMBitcode) {
            let mut file = File::create(bc_path.as_path())?;
            module.emit_bc(&mut file)?;
        }

        // Emit assembly
        if let Some(asm_path) = self.options.maybe_emit(&input, OutputType::Assembly) {
            let mut file = File::create(asm_path.as_path())?;
            module.emit_asm(&mut file, self.target_machine.handle())?;
        }

        // Emit object
        if let Some(obj_path) = self.options.maybe_emit(&input, OutputType::Object) {
            let mut file = File::create(obj_path.as_path())?;
            module.emit_obj(&mut file, self.target_machine.handle())?;

            Ok(ModuleArtifacts {
                name: self.options.app.name,
                object: Some(obj_path),
                dwarf_object: None,
                bytecode: None,
            })
        } else {
            Ok(ModuleArtifacts {
                name: self.options.app.name,
                object: None,
                dwarf_object: None,
                bytecode: None,
            })
        }
    }
}
