//! https://github.com/lumen/otp/tree/lumen/lib/observer/src

use super::*;

test_compiles_lumen_otp!(cdv_atom_cb);
test_compiles_lumen_otp!(cdv_bin_cb);
test_compiles_lumen_otp!(cdv_detail_wx);
test_compiles_lumen_otp!(cdv_dist_cb);
test_compiles_lumen_otp!(cdv_ets_cb);
test_compiles_lumen_otp!(cdv_fun_cb);
test_compiles_lumen_otp!(cdv_gen_cb);
test_compiles_lumen_otp!(cdv_html_wx);
test_compiles_lumen_otp!(cdv_info_wx);
test_compiles_lumen_otp!(cdv_int_tab_cb);
test_compiles_lumen_otp!(cdv_mem_cb);
test_compiles_lumen_otp!(cdv_mod_cb);
test_compiles_lumen_otp!(cdv_multi_wx);
test_compiles_lumen_otp!(cdv_persistent_cb imports "lib/observer/src/crashdump_viewer", "lib/stdlib/src/ets");
test_compiles_lumen_otp!(cdv_port_cb);
test_compiles_lumen_otp!(cdv_proc_cb);
test_compiles_lumen_otp!(cdv_sched_cb);
test_compiles_lumen_otp!(cdv_table_wx);
test_compiles_lumen_otp!(cdv_term_cb);
test_compiles_lumen_otp!(cdv_timer_cb);
test_compiles_lumen_otp!(cdv_virtual_list_wx);
test_compiles_lumen_otp!(cdv_wx);
test_compiles_lumen_otp!(crashdump_viewer);
test_compiles_lumen_otp!(etop);
test_compiles_lumen_otp!(etop_tr);
test_compiles_lumen_otp!(etop_txt);
test_compiles_lumen_otp!(multitrace);
test_compiles_lumen_otp!(observer imports "lib/observer/src/observer_wx");
test_compiles_lumen_otp!(observer_alloc_wx);
test_compiles_lumen_otp!(observer_app_wx);
test_compiles_lumen_otp!(observer_html_lib);
test_compiles_lumen_otp!(observer_lib);
test_compiles_lumen_otp!(observer_perf_wx);
test_compiles_lumen_otp!(observer_port_wx);
test_compiles_lumen_otp!(observer_pro_wx);
test_compiles_lumen_otp!(observer_procinfo);
test_compiles_lumen_otp!(observer_sys_wx);
test_compiles_lumen_otp!(observer_trace_wx);
test_compiles_lumen_otp!(observer_traceoptions_wx);
test_compiles_lumen_otp!(observer_tv_table);
test_compiles_lumen_otp!(observer_tv_wx);
test_compiles_lumen_otp!(observer_wx);
test_compiles_lumen_otp!(ttb);
test_compiles_lumen_otp!(ttb_et);

fn includes() -> Vec<&'static str> {
    let mut includes = super::includes();
    includes.extend(vec![
        "lib/et/include",
        "lib/observer/include",
        "lib/observer/src",
    ]);

    includes
}

fn relative_directory_path() -> PathBuf {
    super::relative_directory_path().join("observer/src")
}
