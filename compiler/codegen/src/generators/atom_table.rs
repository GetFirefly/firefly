use std::collections::HashSet;
use std::ffi::CString;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use libeir_intern::Symbol;

use liblumen_llvm as llvm;
use liblumen_llvm::builder::ModuleBuilder;
use liblumen_llvm::enums::Linkage;
use liblumen_llvm::target::TargetMachine;
use liblumen_session::Options;

use crate::meta::CompiledModule;
use crate::Result;

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
    context: &llvm::Context,
    target_machine: &TargetMachine,
    mut atoms: HashSet<Symbol>,
    output_dir: &Path,
) -> Result<Arc<CompiledModule>> {
    const NAME: &'static str = "liblumen_crt_atoms";

    let builder = ModuleBuilder::new(NAME, options, context, target_machine)?;

    // Ensure true/false are always present
    atoms.insert(Symbol::intern("false"));
    atoms.insert(Symbol::intern("true"));
    atoms.insert(Symbol::intern("error"));
    atoms.insert(Symbol::intern("exit"));
    atoms.insert(Symbol::intern("throw"));
    atoms.insert(Symbol::intern("nocatch"));

    fn insert_atom<'ctx>(builder: &ModuleBuilder<'ctx>, atom: Symbol) -> Result<llvm::Value> {
        // We remap true/false to 1/0 respectively
        let id = atom.as_usize();
        // Each atom must be a null-terminated string
        let s = atom.as_str().get();
        let constant = builder.build_named_constant_string(
            &format!("__atom{}.value", id),
            s,
            /* null_terminated= */ false,
        );
        // The atom constants are not accessible directly, only via the table
        builder.set_linkage(constant, Linkage::Private);
        builder.set_alignment(constant, 8);
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
    let entry_type = builder.get_struct_type(Some("ConstantAtom"), &[i64_type, i8ptr_type]);

    let mut entries = Vec::with_capacity(values.len());
    for (sym, value) in values.iter() {
        let id = builder.build_constant_uint(i64_type, sym.as_usize() as u64);
        let ptr = builder.build_const_inbounds_gep(*value, &[0, 0]);
        entries.push(builder.build_constant_struct(entry_type, &[id, ptr]));
    }

    // Generate constants array
    let entries_const_init = builder.build_constant_array(entry_type, entries.as_slice());
    let entries_const_ty = builder.type_of(entries_const_init);
    let entries_const = builder.build_constant(
        entries_const_ty,
        "__LUMEN_ATOM_TABLE_ENTRIES",
        Some(entries_const_init),
    );
    builder.set_linkage(entries_const, Linkage::Private);
    builder.set_alignment(entries_const, 8);

    // Generate atom table global itself
    let entry_ptr_type = builder.get_pointer_type(entry_type);
    let table_global_init = builder.build_const_inbounds_gep(entries_const, &[0, 0]);
    let table_global = builder.build_global(
        entry_ptr_type,
        "__LUMEN_ATOM_TABLE",
        Some(table_global_init),
    );
    builder.set_alignment(table_global, 8);

    // Generate atom table size global
    let table_size_global_init = builder.build_constant_uint(i64_type, entries.len() as u64);
    let table_size_global = builder.build_global(
        i64_type,
        "__LUMEN_ATOM_TABLE_SIZE",
        Some(table_size_global_init),
    );
    builder.set_alignment(table_size_global, 8);

    // Finalize module
    let module = builder.finish()?;
    // Open ll file for writing
    let ir_path = output_dir.join(&format!("{}.ll", NAME));
    let mut file = File::create(ir_path.as_path())?;
    // Emit IR file
    module.emit_ir(&mut file)?;

    // Open object file for writing
    let obj_path = output_dir.join(&format!("{}.o", NAME));
    let mut file = File::create(obj_path.as_path())?;
    // Emit object file
    module.emit_obj(&mut file)?;

    Ok(Arc::new(CompiledModule::new(
        NAME.to_string(),
        Some(obj_path),
        None,
    )))
}
