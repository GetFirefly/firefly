//! https://github.com/lumen/otp/tree/lumen/lib/stdlib/src

use super::*;

test_compiles_lumen_otp!(array);
test_compiles_lumen_otp!(base64);
test_compiles_lumen_otp!(beam_lib);
test_compiles_lumen_otp!(binary);
test_compiles_lumen_otp!(c);
test_compiles_lumen_otp!(calendar);
test_compiles_lumen_otp!(dets);
test_compiles_lumen_otp!(dets_server);
test_compiles_lumen_otp!(dets_sup);
test_compiles_lumen_otp!(dets_utils);
test_compiles_lumen_otp!(dets_v9);
test_compiles_lumen_otp!(dict);
test_compiles_lumen_otp!(digraph);
test_compiles_lumen_otp!(digraph_utils);
test_compiles_lumen_otp!(edlin);
test_compiles_lumen_otp!(edlin_expand);
test_compiles_lumen_otp!(epp);
test_compiles_lumen_otp!(erl_abstract_code);
test_compiles_lumen_otp!(erl_anno);
test_compiles_lumen_otp!(erl_bits);
test_compiles_lumen_otp!(erl_compile);
test_compiles_lumen_otp!(erl_error);
test_compiles_lumen_otp!(erl_eval);
test_compiles_lumen_otp!(erl_expand_records);
test_compiles_lumen_otp!(erl_internal);
test_compiles_lumen_otp!(erl_lint);
test_compiles_lumen_otp!(erl_posix_msg);
test_compiles_lumen_otp!(erl_pp);
test_compiles_lumen_otp!(erl_scan);
test_compiles_lumen_otp!(erl_tar);
test_compiles_lumen_otp!(error_logger_file_h);
test_compiles_lumen_otp!(error_logger_tty_h);
test_compiles_lumen_otp!(escript);
test_compiles_lumen_otp!(ets);
test_compiles_lumen_otp!(eval_bits);
test_compiles_lumen_otp!(file_sorter);
test_compiles_lumen_otp!(filelib);
test_compiles_lumen_otp!(filename);
test_compiles_lumen_otp!(gb_sets);
test_compiles_lumen_otp!(gb_trees);
test_compiles_lumen_otp!(gen);
test_compiles_lumen_otp!(gen_event);
test_compiles_lumen_otp!(gen_fsm);
test_compiles_lumen_otp!(gen_server);
test_compiles_lumen_otp!(gen_statem);
test_compiles_lumen_otp!(io);
test_compiles_lumen_otp!(io_lib);
test_compiles_lumen_otp!(io_lib_format);
test_compiles_lumen_otp!(io_lib_fread);
test_compiles_lumen_otp!(io_lib_pretty);
test_compiles_lumen_otp!(lists);
test_compiles_lumen_otp!(log_mf_h);
test_compiles_lumen_otp!(maps);
test_compiles_lumen_otp!(math);
test_compiles_lumen_otp!(ms_transform);
test_compiles_lumen_otp!(orddict);
test_compiles_lumen_otp!(ordsets imports "lib/stdlib/src/lists");
test_compiles_lumen_otp!(otp_internal);
test_compiles_lumen_otp!(pool);
test_compiles_lumen_otp!(proc_lib);
test_compiles_lumen_otp!(proplists);
test_compiles_lumen_otp!(qlc);
test_compiles_lumen_otp!(qlc_pt);
test_compiles_lumen_otp!(queue);
test_compiles_lumen_otp!(rand);
test_compiles_lumen_otp!(random);
test_compiles_lumen_otp!(re);
test_compiles_lumen_otp!(sets);
test_compiles_lumen_otp!(shell);
test_compiles_lumen_otp!(shell_default imports "lib/stdlib/src/c", "lib/stdlib/src/io");
test_compiles_lumen_otp!(shell_docs);
test_compiles_lumen_otp!(slave);
test_compiles_lumen_otp!(sofs);
test_compiles_lumen_otp!(string);
test_compiles_lumen_otp!(supervisor);
test_compiles_lumen_otp!(supervisor_bridge);
test_compiles_lumen_otp!(sys);
test_compiles_lumen_otp!(timer);
test_compiles_lumen_otp!(unicode);
test_compiles_lumen_otp!(uri_string);
test_compiles_lumen_otp!(win32reg);
test_compiles_lumen_otp!(zip);

fn includes() -> Vec<&'static str> {
    let mut includes = super::includes();
    includes.extend(vec![
        "lib/kernel/include",
        "lib/stdlib/include",
        "lib/stdlib/src",
    ]);

    includes
}

fn relative_directory_path() -> PathBuf {
    super::relative_directory_path().join("stdlib/src")
}
