use std::path::{Path, PathBuf};
use std::process::{Command, Output, Stdio};

#[allow(unused_macros)]
macro_rules! test_stdout {
    ($func_name:ident, $expected_stdout:literal) => {
        #[test]
        fn $func_name() {
            let output = $crate::test::output(file!(), stringify!($func_name));

            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);

            assert_eq!(
                stdout, $expected_stdout,
                "\nstdout = {}\nstderr = {}",
                stdout, stderr
            );
        }
    };
}

fn compile(file: &str, name: &str) -> PathBuf {
    // `file!()` starts with path relative to workspace root, but the `current_dir` will be inside
    // the crate root, so need to strip the relative crate root.
    let file_path = Path::new(file);
    let relative_file_path = file_path.strip_prefix("native_implemented/otp").unwrap();
    let directory_path = relative_file_path.parent().unwrap();
    let file_stem = file_path.file_stem().unwrap();
    let test_directory_path = directory_path.join(file_stem).join(name);

    let build_path_buf = test_directory_path.join("_build");
    std::fs::create_dir_all(&build_path_buf).unwrap();

    let mut command = Command::new("../../bin/lumen");

    let bin_path_buf = test_directory_path.join("bin");
    std::fs::create_dir_all(&bin_path_buf).unwrap();

    let output_path_buf = bin_path_buf.join(name);

    command
        .arg("compile")
        .arg("--output-dir")
        .arg(build_path_buf)
        .arg("-o")
        .arg(&output_path_buf)
        // Turn off optimizations as work-around for debug info bug in EIR
        .arg("-O0")
        .arg("-lc")
        .arg("--emit=all");

    let erlang_path = directory_path.join(file_stem).join(name).join("init.erl");

    let compile_output = command
        .arg(erlang_path)
        .stdin(Stdio::null())
        .output()
        .unwrap();

    assert!(
        compile_output.status.success(),
        "stdout = {}\nstderr = {}",
        String::from_utf8_lossy(&compile_output.stdout),
        String::from_utf8_lossy(&compile_output.stderr)
    );

    output_path_buf
}

pub fn output(file: &str, name: &str) -> Output {
    let bin_path_buf = compile(file, name);

    Command::new(bin_path_buf)
        .stdin(Stdio::null())
        .output()
        .unwrap()
}
