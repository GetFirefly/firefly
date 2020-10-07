use std::process::{Command, Stdio};
use std::sync::Once;

#[test]
fn without_arguments_calls_global_dynamic_functions() {
    ensure_compiled();

    let cli_output = Command::new("./global_dynamic")
        .stdin(Stdio::null())
        .output()
        .unwrap();

    let stdout = String::from_utf8_lossy(&cli_output.stdout);
    let stderr = String::from_utf8_lossy(&cli_output.stderr);

    assert_eq!(
        String::from_utf8_lossy(&cli_output.stdout),
        "{alice, says, hello, to, bob}\n{eve, overhears}\n",
        "\nstdout = {}\nstderr = {}",
        stdout,
        stderr
    );
}

static COMPILED: Once = Once::new();

fn ensure_compiled() {
    COMPILED.call_once(|| {
        compile();
    })
}

fn compile() {
    let mut command = Command::new("../bin/lumen");

    command
        .arg("compile")
        .arg("--output-dir")
        .arg("_build")
        .arg("--output")
        .arg("global_dynamic")
        // Turn off optimizations as work-around for debug info bug in EIR
        .arg("-O0");

    let compile_output = command
        .arg("tests/global_dynamic/init.erl")
        .stdin(Stdio::null())
        .output()
        .unwrap();

    assert!(
        compile_output.status.success(),
        "stdout = {}\nstderr = {}",
        String::from_utf8_lossy(&compile_output.stdout),
        String::from_utf8_lossy(&compile_output.stderr)
    );
}
