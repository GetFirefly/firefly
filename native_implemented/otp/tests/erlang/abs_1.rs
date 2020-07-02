use std::process::{Command, Stdio};

#[test]
fn without_number_errors_badarg() {

}

#[test]
fn with_number_returns_non_negative() {
    let name = "with_number_returns_non_negative";
    compile(name);

    let program = format!("./{}", name);

    let output = Command::new(program).stdin(Stdio::null()).output().unwrap();

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert_eq!(
        stdout, "\"Nothing to say.\"\n",
        "\nstdout = {}\nstderr = {}",
        stdout, stderr
    );
}

fn compile(name: &str) {
    std::fs::create_dir_all("_build").unwrap();

    let mut command = Command::new("../../bin/lumen");

    command
        .arg("compile")
        .arg("--output-dir")
        .arg("_build")
        .arg("-o")
        .arg(name)
        // Turn off optimizations as work-around for debug info bug in EIR
        .arg("-O0")
        .arg("-lc");

    let erlang_path = format!("tests/erlang/abs_1/{}/init.erl", name);

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
}
