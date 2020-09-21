//! https://github.com/lumen/otp/tree/lumen/lib/sasl/src

use super::*;

test_compiles_lumen_otp!(alarm_handler);
test_compiles_lumen_otp!(erlsrv);
test_compiles_lumen_otp!(format_lib_supp imports "lib/stdlib/src/io", "lib/stdlib/src/io_lib", "lib/stdlib/src/lists","lib/stdlib/src/string", "lib/sasl/src/misc_supp");
test_compiles_lumen_otp!(misc_supp);
test_compiles_lumen_otp!(rb);
test_compiles_lumen_otp!(rb_format_supp);
test_compiles_lumen_otp!(release_handler);
test_compiles_lumen_otp!(release_handler_1);
test_compiles_lumen_otp!(sasl);
test_compiles_lumen_otp!(sasl_report);
test_compiles_lumen_otp!(sasl_report_file_h);
test_compiles_lumen_otp!(sasl_report_tty_h);
test_compiles_lumen_otp!(systools imports "lib/sasl/src/systools_lib", "lib/sasl/src/systools_make", "lib/sasl/src/systools_relup");
test_compiles_lumen_otp!(systools_lib);
test_compiles_lumen_otp!(systools_make);
test_compiles_lumen_otp!(systools_rc);
test_compiles_lumen_otp!(systools_relup);

fn includes() -> Vec<&'static str> {
    let mut includes = super::includes();
    includes.extend(vec!["lib/sasl/src", "lib/stdlib/include/"]);

    includes
}

fn relative_directory_path() -> PathBuf {
    super::relative_directory_path().join("sasl/src")
}
