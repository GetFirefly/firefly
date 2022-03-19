use std::collections::HashSet;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use liblumen_intern::Symbol;
use liblumen_llvm::builder::ModuleBuilder;
use liblumen_llvm::ir::*;
use liblumen_llvm::target::TargetMachine;
use liblumen_llvm::StringRef;
use liblumen_session::{Input, Options, OutputType};

use crate::meta::CompiledModule;

/// Generates an LLVM module containing the raw atom table data for the current build
///
/// Process is as follows:
/// - Generate a constant for each atom string
/// - Generate a constant array containing `ConstantAtom` structs for all atoms:
///   - Has type `{ i64, i8* }`
///   - First field is the id of the `Symbol`
///   - Second field is the pointer to the string constant
/// - Generate the __LUMEN_ATOM_TABLE global as a pointer to the first element of the array
/// - Generate the __LUMEN_ATOM_TABLE_SIZE global with the number of elements in the array
pub fn generate(
    options: &Options,
    context: &Context,
    target_machine: TargetMachine,
    mut atoms: HashSet<Symbol>,
) -> anyhow::Result<Arc<CompiledModule>> {
    const NAME: &'static str = "liblumen_crt_atoms";

    let builder = ModuleBuilder::new(NAME, options, context, target_machine)?;

    // Ensure true/false are always present
    atoms.insert(Symbol::intern("false"));
    atoms.insert(Symbol::intern("true"));
    atoms.insert(Symbol::intern("error"));
    atoms.insert(Symbol::intern("exit"));
    atoms.insert(Symbol::intern("throw"));
    atoms.insert(Symbol::intern("nocatch"));
    atoms.insert(Symbol::intern("normal"));

    fn insert_atom<'ctx>(
        builder: &ModuleBuilder<'ctx>,
        atom: Symbol,
    ) -> anyhow::Result<GlobalVariable> {
        // We remap true/false to 1/0 respectively
        let id = atom.as_usize();
        // Each atom must be a null-terminated string
        let s = StringRef::from(atom);
        let null_terminated = s.to_cstr().into_owned();
        let constant =
            builder.build_named_constant_string(&format!("__atom{}.value", id), &null_terminated);
        // The atom constants are not accessible directly, only via the table
        constant.set_linkage(Linkage::Private);
        constant.set_alignment(8);
        Ok(constant)
    }

    // Generate globals/constants for each atom
    let mut values = Vec::with_capacity(atoms.len());
    for atom in atoms.iter().copied() {
        values.push((atom, insert_atom(&builder, atom)?));
    }

    // Generate constants array entries
    let i8_type = builder.get_i8_type();
    let i8ptr_type = builder.get_pointer_type(i8_type);
    let i64_type = builder.get_i64_type();
    let entry_type =
        builder.get_struct_type(Some("ConstantAtom"), &[i64_type.base(), i8ptr_type.base()]);

    let mut entries: Vec<ConstantValue> = Vec::with_capacity(values.len());
    for (sym, value) in values.iter() {
        let value = *value;
        let id = builder.build_constant_uint(i64_type, sym.as_usize() as u64);
        let value_ty = value.get_type();
        let constant: ConstantValue = value.try_into().unwrap();
        let ptr = builder.build_const_inbounds_gep(value_ty, constant, &[0, 0]);
        entries.push(
            builder
                .build_constant_named_struct(entry_type, &[id.into(), ptr.into()])
                .into(),
        );
    }

    // Generate constants array
    let entries_const_init = builder.build_constant_array(entry_type, entries.as_slice());
    let entries_const_ty = entries_const_init.get_type();
    let entries_const = builder.build_constant(
        entries_const_ty,
        "__LUMEN_ATOM_TABLE_ENTRIES",
        entries_const_init,
    );
    entries_const.set_linkage(Linkage::Private);
    entries_const.set_alignment(8);

    // Generate atom table global itself
    let entry_ptr_type = builder.get_pointer_type(entry_type);
    let entries_const: ConstantValue = entries_const.try_into().unwrap();
    let table_global_init =
        builder.build_const_inbounds_gep(entries_const_ty, entries_const, &[0, 0]);
    let table_global = builder.build_global(
        entry_ptr_type,
        "__LUMEN_ATOM_TABLE",
        Some(table_global_init.base()),
    );
    table_global.set_alignment(8);

    // Generate atom table size global
    let table_size_global_init = builder.build_constant_uint(i64_type, entries.len() as u64);
    let table_size_global = builder.build_global(
        i64_type,
        "__LUMEN_ATOM_TABLE_SIZE",
        Some(table_size_global_init.base()),
    );
    table_size_global.set_alignment(8);

    // Finalize module
    let module = builder.finish()?;

    // We need an input to represent the generated source
    let input = Input::from(Path::new(&format!("{}", NAME)));

    // Emit LLVM IR file
    if let Some(ir_path) = options.maybe_emit(&input, OutputType::LLVMAssembly) {
        let mut file = File::create(ir_path.as_path())?;
        module.emit_ir(&mut file)?;
    }

    // Emit LLVM bitcode file
    if let Some(bc_path) = options.maybe_emit(&input, OutputType::LLVMBitcode) {
        let mut file = File::create(bc_path.as_path())?;
        module.emit_bc(&mut file)?;
    }

    // Emit assembly file
    if let Some(asm_path) = options.maybe_emit(&input, OutputType::Assembly) {
        let mut file = File::create(asm_path.as_path())?;
        module.emit_asm(&mut file, target_machine)?;
    }

    // Emit object file
    let obj_path = if let Some(obj_path) = options.maybe_emit(&input, OutputType::Object) {
        let mut file = File::create(obj_path.as_path())?;
        module.emit_obj(&mut file, target_machine)?;
        Some(obj_path)
    } else {
        None
    };

    Ok(Arc::new(CompiledModule::new(
        NAME.to_string(),
        obj_path,
        None,
    )))
}
