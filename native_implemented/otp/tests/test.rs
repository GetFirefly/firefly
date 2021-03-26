use std::env::current_dir;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Duration;

use process_control::{ChildExt, ExitStatus, Output, Timeout};

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

pub struct Compilation<'a> {
    pub command: &'a mut Command,
    pub test_directory_path: &'a Path,
}

pub fn command_failed(
    message: &'static str,
    working_directory: PathBuf,
    command: Command,
    output: Output,
) -> ! {
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    let formatted_code = match output.status.code() {
        Some(code) => code.to_string(),
        None => "".to_string(),
    };
    let formatted_signal = signal(output.status);

    panic!(
        "{} failed\nCommands:\ncd {}\n{:?}\n\nstdout: {}\nstderr: {}\nStatus code: {}\nSignal: {}",
        message,
        working_directory.display(),
        command,
        stdout,
        stderr,
        formatted_code,
        formatted_signal
    );
}

pub fn compiled_path_buf<F>(file: &str, name: &str, compilation_mutator: F) -> PathBuf
where
    F: FnOnce(Compilation),
{
    match compile(file, name, compilation_mutator) {
        Ok(path_buf) => path_buf,
        Err((command, output)) => command_failed(
            "Compilation",
            std::env::current_dir().unwrap(),
            command,
            output,
        ),
    }
}

fn compile<F>(file: &str, name: &str, compilation_mutator: F) -> Result<PathBuf, (Command, Output)>
where
    F: FnOnce(Compilation),
{
    let cwd = std::env::current_dir().unwrap();
    let build_path = cwd.join("tests/_build");

    // `file!()` starts with path relative to workspace root, but the `current_dir` will be inside
    // the crate root, so need to strip the relative crate root.
    let file_path = Path::new(file);
    let crate_relative_file_path = file_path.strip_prefix("native_implemented/otp").unwrap();
    let crate_relative_directory_path = crate_relative_file_path.parent().unwrap();
    let tests_relative_directory_path =
        crate_relative_directory_path.strip_prefix("tests").unwrap();
    let file_stem = file_path.file_stem().unwrap();
    let output_directory_path = build_path
        .join(tests_relative_directory_path)
        .join(file_stem)
        .join(name);
    let output_path = output_directory_path.join("bin").join(name);
    let test_directory_path = crate_relative_directory_path.join(file_stem).join(name);

    let mut command = Command::new("../../bin/lumen");

    command
        .arg("compile")
        .arg("--output")
        .arg(&output_path)
        .arg("--output-dir")
        .arg(&output_directory_path)
        .arg("-O0");

    if std::env::var_os("DEBUG").is_some() {
        command.arg("--emit=all");
    }

    compilation_mutator(Compilation {
        command: &mut command,
        test_directory_path: &test_directory_path,
    });

    timeout("Compilation", cwd, command, Duration::from_secs(30)).map(|_| output_path)
}

pub fn timeout(
    message: &'static str,
    working_directory: PathBuf,
    mut command: Command,
    time_limit: Duration,
) -> Result<(), (Command, Output)> {
    let process = command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();

    match process
        .with_output_timeout(time_limit)
        .terminating()
        .wait()
        .unwrap()
    {
        Some(output) => {
            if output.status.success() {
                Ok(())
            } else {
                Err((command, output))
            }
        }
        None => {
            panic!(
                "{} timed out after {:?}\nCommands:\ncd {}\n{:?}",
                message,
                time_limit,
                working_directory.display(),
                command,
            );
        }
    }
}

#[allow(dead_code)]
pub fn output(file: &str, name: &str) -> (Command, Output) {
    let bin_path_buf = compiled_path_buf(
        file,
        name,
        |Compilation {
             command,
             test_directory_path,
         }| {
            let erlang_src_path = test_directory_path.join("src");

            let input_path = if erlang_src_path.is_dir() {
                erlang_src_path
            } else {
                test_directory_path.join("init.erl")
            };

            let shared_path = current_dir().unwrap().join("tests/shared/src");

            command.arg(shared_path).arg(input_path);
        },
    );

    let mut command = Command::new(&bin_path_buf);

    let process = command
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap_or_else(|error| panic!("Could not run {:?}: {:?}", bin_path_buf, error));

    let time_limit = Duration::from_secs(10);

    let output = process
        .with_output_timeout(time_limit)
        .terminating()
        .wait()
        .unwrap()
        .unwrap();

    std::fs::remove_file(&bin_path_buf).ok();

    (command, output)
}

#[cfg(unix)]
pub fn signal(exit_status: ExitStatus) -> String {
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
