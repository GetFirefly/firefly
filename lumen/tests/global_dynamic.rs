use std::process::{Command, Stdio};
use std::sync::Once;

#[test]
fn without_arguments_calls_global_dynamic_functions() {
    ensure_compiled();

    let cli_output = Command::new("tests/_build/global_dynamic")
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
    std::fs::create_dir_all("tests/_build").unwrap();

    let mut command = Command::new("../bin/lumen");

    command
        .arg("compile")
        .arg("--output")
        .arg("tests/_build/global_dynamic");

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
