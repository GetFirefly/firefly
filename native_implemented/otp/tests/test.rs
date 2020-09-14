use std::path::{Path, PathBuf};
use std::process::{Command, ExitStatus, Output, Stdio};

#[allow(unused_macros)]
macro_rules! test_stdout {
    ($func_name:ident, $expected_stdout:literal) => {
        #[test]
        fn $func_name() {
            let (command, output) = $crate::test::output(file!(), stringify!($func_name));

            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            let formatted_code = match output.status.code() {
                Some(code) => code.to_string(),
                None => "".to_string(),
            };
            let formatted_signal = $crate::test::signal(output.status);

            assert_eq!(
                stdout,
                $expected_stdout,
                "\nCommands:\ncd {}\n{:?}\nstdout: {}\nstderr: {}\nStatus code: {}\nSignal: {}",
                std::env::current_dir().unwrap().to_string_lossy(),
                command,
                stdout,
                stderr,
                formatted_code,
                formatted_signal
            );
        }
    };
}

#[allow(unused_macros)]
macro_rules! test_stdout_substrings {
    ($func_name:ident, $expected_stdout_substrings:expr) => {
        test_substrings!(
            $func_name,
            $expected_stdout_substrings,
            Vec::<&str>::default()
        );
    };
}

#[allow(unused_macros)]
macro_rules! test_stderr_substrings {
    ($func_name:ident, $expected_stderr_substrings:expr) => {
        test_substrings!(
            $func_name,
            Vec::<&str>::default(),
            $expected_stderr_substrings
        );
    };
}

#[allow(unused_macros)]
macro_rules! test_substrings {
    ($func_name:ident, $expected_stdout_substrings:expr, $expected_stderr_substrings:expr) => {
        #[test]
        fn $func_name() {
            let (command, output) = $crate::test::output(file!(), stringify!($func_name));

            let stdout = String::from_utf8_lossy(&output.stdout);

            let stderr = String::from_utf8_lossy(&output.stderr);
            let stripped_stderr_byte_vec = strip_ansi_escapes::strip(output.stderr.clone()).unwrap();
            let stripped_stderr = String::from_utf8_lossy(&stripped_stderr_byte_vec);

            let formatted_code = match output.status.code() {
                Some(code) => code.to_string(),
                None => "".to_string(),
            };
            let formatted_signal = $crate::test::signal(output.status);

            let expected_stdout_substrings: Vec<&str> = $expected_stdout_substrings;
            for expected_stdout_substring in expected_stdout_substrings {
                assert!(
                    stdout.contains(expected_stdout_substring),
                    "stdout does not contain substring\nCommands:\ncd {}\n{:?}\nsubstring: {}\nstdout: {}\nstderr: {}\nStatus code: {}\nSignal: {}",
                    std::env::current_dir().unwrap().to_string_lossy(),
                    command,
                    expected_stdout_substring,
                    stdout,
                    stderr,
                    formatted_code,
                    formatted_signal
                );
            }

            let expected_stderr_substrings: Vec<&str> = $expected_stderr_substrings;
            for expected_stderr_substring in expected_stderr_substrings {
                assert!(
                    stripped_stderr.contains(expected_stderr_substring),
                    "stderr does not contain substring\nCommands:\ncd {}\n{:?}\nsubstring: {}\nstdout: {}\nstderr: {}\nStatus code: {}\nSignal: {}",
                    std::env::current_dir().unwrap().to_string_lossy(),
                    command,
                    expected_stderr_substring,
                    stdout,
                    stderr,
                    formatted_code,
                    formatted_signal
                );
            }
        }
    }
}

fn compiled_path_buf(file: &str, name: &str) -> PathBuf {
    match compile(file, name) {
        Ok(path_buf) => path_buf,
        Err((command, output)) => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            let formatted_code = match output.status.code() {
                Some(code) => code.to_string(),
                None => "".to_string(),
            };
            let formatted_signal = signal(output.status);

            panic!(
                "Compilation failed\nCommands:\ncd {}\n{:?}\n\nstdout: {}\nstderr: {}\nStatus code: {}\nSignal: {}",
                std::env::current_dir().unwrap().to_string_lossy(),
                command,
                stdout,
                stderr,
                formatted_code,
                formatted_signal
            );
        }
    }
}

fn compile(file: &str, name: &str) -> Result<PathBuf, (Command, Output)> {
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
        .arg("--emit=all");

    let erlang_parent_path = directory_path.join(file_stem).join(name);
    let erlang_src_path = erlang_parent_path.join("src");

    let input_path = if erlang_src_path.is_dir() {
        erlang_src_path
    } else {
        erlang_parent_path.join("init.erl")
    };

    let compile_output = command
        .arg(input_path)
        .stdin(Stdio::null())
        .output()
        .unwrap();

    if compile_output.status.success() {
        Ok(output_path_buf)
    } else {
        Err((command, compile_output))
    }
}

pub fn output(file: &str, name: &str) -> (Command, Output) {
    let bin_path_buf = compiled_path_buf(file, name);
    let mut command = Command::new(bin_path_buf);

    let output = command.stdin(Stdio::null()).output().unwrap();

    (command, output)
}

#[cfg(unix)]
pub fn signal(exit_status: ExitStatus) -> String {
    use std::os::unix::process::ExitStatusExt;

    exit_status
        .signal()
        .map(|i| match i as libc::c_int {
            libc::SIGHUP => "hang up".to_string(),
            libc::SIGINT => "interrupt (Ctrl+C)".to_string(),
            libc::SIGQUIT => "quit (Ctrl+D)".to_string(),
            libc::SIGILL => "illegal instruction".to_string(),
            libc::SIGABRT => "abort program".to_string(),
            libc::SIGFPE => "floating point exception".to_string(),
            libc::SIGKILL => "killed".to_string(),
            libc::SIGSEGV => "segmentation fault (invalid address)".to_string(),
            libc::SIGBUS => "bus error (stack may not have enough pages)".to_string(),
            libc::SIGPIPE => "write on a pipe with no reader".to_string(),
            libc::SIGALRM => "alarm".to_string(),
            libc::SIGTERM => "terminated".to_string(),
            n => n.to_string(),
        })
        .unwrap_or("".to_string())
}

#[cfg(not(unix))]
pub fn signal(exit_status: ExitStatus) -> String {
    "".to_string()
}
