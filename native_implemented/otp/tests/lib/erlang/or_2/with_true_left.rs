use super::*;

// `without_boolean_right_errors_badarg` in unit tests

#[test]
fn with_boolean_right_returns_true() {
    let name = "with_boolean_right_returns_true";
    let bin_path_buf = compile(file!(), name);

    let output = Command::new(bin_path_buf)
        .stdin(Stdio::null())
        .output()
        .unwrap();

    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    assert_eq!(
        stdout, ":'true'\n:'true'\n",
        "\nstdout = {}\nstderr = {}",
        stdout, stderr
    );
}
