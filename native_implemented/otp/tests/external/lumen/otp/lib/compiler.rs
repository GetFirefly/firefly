//! https://github.com/lumen/otp/tree/lumen/lib/compiler/src

use super::*;

test_compiles_lumen_otp!(beam_a);
test_compiles_lumen_otp!(beam_asm);
test_compiles_lumen_otp!(beam_block);
test_compiles_lumen_otp!(beam_call_types);
test_compiles_lumen_otp!(beam_clean);
test_compiles_lumen_otp!(beam_dict);
test_compiles_lumen_otp!(beam_digraph);
test_compiles_lumen_otp!(beam_disasm);
test_compiles_lumen_otp!(beam_flatten);
test_compiles_lumen_otp!(beam_jump);
test_compiles_lumen_otp!(beam_kernel_to_ssa);
test_compiles_lumen_otp!(beam_listing);
test_compiles_lumen_otp!(beam_peep);
test_compiles_lumen_otp!(beam_ssa);
test_compiles_lumen_otp!(beam_ssa_bool);
test_compiles_lumen_otp!(beam_ssa_bsm);
test_compiles_lumen_otp!(beam_ssa_codegen);
test_compiles_lumen_otp!(beam_ssa_dead);
test_compiles_lumen_otp!(beam_ssa_funs);
test_compiles_lumen_otp!(beam_ssa_lint);
test_compiles_lumen_otp!(beam_ssa_opt);
test_compiles_lumen_otp!(beam_ssa_pp);
test_compiles_lumen_otp!(beam_ssa_pre_codegen);
test_compiles_lumen_otp!(beam_ssa_recv);
test_compiles_lumen_otp!(beam_ssa_share);
test_compiles_lumen_otp!(beam_ssa_type);
test_compiles_lumen_otp!(beam_trim);
test_compiles_lumen_otp!(beam_types);
test_compiles_lumen_otp!(beam_utils);
test_compiles_lumen_otp!(beam_validator);
test_compiles_lumen_otp!(beam_z);
test_compiles_lumen_otp!(cerl);
test_compiles_lumen_otp!(cerl_clauses);
test_compiles_lumen_otp!(cerl_inline);
test_compiles_lumen_otp!(cerl_sets);
test_compiles_lumen_otp!(cerl_trees);
test_compiles_lumen_otp!(compile);
test_compiles_lumen_otp!(core_lib);
test_compiles_lumen_otp!(core_lint);
test_compiles_lumen_otp!(core_pp);
test_compiles_lumen_otp!(core_scan);
test_compiles_lumen_otp!(erl_bifs);
test_compiles_lumen_otp!(rec_env);
test_compiles_lumen_otp!(sys_core_alias);
test_compiles_lumen_otp!(sys_core_bsm);
test_compiles_lumen_otp!(sys_core_fold);
test_compiles_lumen_otp!(sys_core_fold_lists);
test_compiles_lumen_otp!(sys_core_inline);
test_compiles_lumen_otp!(sys_core_prepare);
test_compiles_lumen_otp!(sys_pre_attributes);
test_compiles_lumen_otp!(v3_core);
test_compiles_lumen_otp!(v3_kernel);
test_compiles_lumen_otp!(v3_kernel_pp);

fn includes() -> Vec<&'static str> {
    let mut includes = super::includes();
    includes.push("lib/compiler/src");

    includes
}

fn relative_directory_path() -> PathBuf {
    super::relative_directory_path().join("compiler/src")
}
