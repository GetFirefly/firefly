//! https://github.com/lumen/otp/tree/lumen/lib/debugger/src

use super::*;

test_compiles_lumen_otp!(dbg_debugged);
test_compiles_lumen_otp!(dbg_icmd);
test_compiles_lumen_otp!(dbg_idb);
test_compiles_lumen_otp!(dbg_ieval);
test_compiles_lumen_otp!(dbg_iload);
test_compiles_lumen_otp!(dbg_iserver);
test_compiles_lumen_otp!(dbg_istk);
test_compiles_lumen_otp!(dbg_wx_break);
test_compiles_lumen_otp!(dbg_wx_break_win);
test_compiles_lumen_otp!(dbg_wx_code);
test_compiles_lumen_otp!(dbg_wx_filedialog_win);
test_compiles_lumen_otp!(dbg_wx_interpret);
test_compiles_lumen_otp!(dbg_wx_mon);
test_compiles_lumen_otp!(dbg_wx_mon_win);
test_compiles_lumen_otp!(dbg_wx_settings);
test_compiles_lumen_otp!(dbg_wx_src_view);
test_compiles_lumen_otp!(dbg_wx_trace);
test_compiles_lumen_otp!(dbg_wx_trace_win);
test_compiles_lumen_otp!(dbg_wx_view);
test_compiles_lumen_otp!(dbg_wx_win);
test_compiles_lumen_otp!(dbg_wx_winman);
test_compiles_lumen_otp!(debugger);
test_compiles_lumen_otp!(i);
test_compiles_lumen_otp!(int);

fn includes() -> Vec<&'static str> {
    let mut includes = super::includes();
    includes.push("lib/debugger/src");

    includes
}

fn relative_directory_path() -> PathBuf {
    super::relative_directory_path().join("debugger/src")
}
