//! Tests compiling Erlang source from https://github.com/lumen/otp
macro_rules! test_compiles_lumen_otp {
    ($file_stem:ident) => {
        #[test]
        fn $file_stem() {
            setup();
            $crate::lumen::otp::compiles_lumen_otp(file!(), relative_directory_path(), stringify!($file_stem), includes(), vec![]);
        }
    };
    ($file_stem:ident imports $($dependency:literal),+) => {
        #[test]
        fn $file_stem() {
            setup();
            $crate::lumen::otp::compiles_lumen_otp(file!(), relative_directory_path(), stringify!($file_stem), includes(), vec![$($dependency),+]);
        }
    };
}

#[path = "otp/erts.rs"]
mod erts;
#[path = "otp/lib.rs"]
mod lib;

use std::path::PathBuf;

use crate::test::Compilation;

fn compiles_lumen_otp(
    file: &str,
    relative_directory_path: PathBuf,
    name: &str,
    includes: Vec<&str>,
    dependencies: Vec<&str>,
) {
    crate::test::compiled_path_buf(file, name, |Compilation { command, .. }| {
        let file_name = format!("{}.erl", name);
        let relative_path = relative_directory_path.join(file_name);
        let lumen_otp_directory = lumen_otp_directory();
        let input = lumen_otp_directory.join(relative_path);
        command.arg(input);
        command.args(
            dependencies
                .into_iter()
                .map(|dependency| lumen_otp_directory.join(format!("{}.erl", dependency))),
        );

        for include in includes {
            let include_path = lumen_otp_directory.join(include);
            command.arg("--include").arg(include_path);
        }
    });
}

fn lumen_otp_directory() -> PathBuf {
    std::env::var_os("ERL_TOP")
        .expect("ERL_TOP to be set to path to https://github.com/lumen/otp checkout")
        .into()
}

fn setup() {}
