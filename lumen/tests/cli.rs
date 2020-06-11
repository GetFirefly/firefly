mod cli {
    use std::process::{Command, Stdio};
    use std::sync::Once;

    #[test]
    fn without_arguments_prints_nothing_to_say() {
        ensure_compiled();

        let cli_output = Command::new("./cli").stdin(Stdio::null()).output().unwrap();

        let stdout = String::from_utf8_lossy(&cli_output.stdout);
        let stderr = String::from_utf8_lossy(&cli_output.stderr);

        assert_eq!(
            String::from_utf8_lossy(&cli_output.stdout),
            "\"Nothing to say.\"\n",
            "\nstdout = {}\nstderr = {}",
            stdout,
            stderr
        );
    }

    #[test]
    fn with_false_argument_prints_nothing_to_say() {
        ensure_compiled();

        let cli_output = Command::new("./cli")
            .arg("false")
            .stdin(Stdio::null())
            .output()
            .unwrap();

        let stdout = String::from_utf8_lossy(&cli_output.stdout);
        let stderr = String::from_utf8_lossy(&cli_output.stderr);

        assert_eq!(
            stdout, "\"Nothing to say.\"\n",
            "\nstdout = {}\nstderr = {}",
            stdout, stderr
        );
    }

    #[test]
    fn with_true_argument_prints_nothing_to_say() {
        ensure_compiled();

        let cli_output = Command::new("./cli")
            .arg("true")
            .stdin(Stdio::null())
            .output()
            .unwrap();

        let stdout = String::from_utf8_lossy(&cli_output.stdout);
        let stderr = String::from_utf8_lossy(&cli_output.stderr);

        assert_eq!(
            stdout, "\"Hello, world!\"\n",
            "\nstdout = {}\nstderr = {}",
            stdout, stderr
        );
    }

    static COMPILED: Once = Once::new();

    fn ensure_compiled() {
        COMPILED.call_once(|| {
            compile();
        })
    }

    fn compile() {
        std::fs::create_dir_all("_build").unwrap();

        let mut command = Command::new("../bin/lumen");

        command
            .arg("compile")
            .arg("--output-dir")
            .arg("_build")
            .arg("-o")
            .arg("cli")
            .arg("-L../target/debug")
            // Turn off optimizations as work-around for debug info bug in EIR
            .arg("-O0")
            .arg("-lc");

        add_link_args(&mut command);

        let compile_output = command
            .arg("tests/cli/init.erl")
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

    #[cfg(not(target_os = "linux"))]
    fn add_link_args(_command: &mut Command) {}

    #[cfg(target_os = "linux")]
    fn add_link_args(command: &mut Command) {
        command
            .arg("-lunwind")
            .arg("-lpthread")
            .arg("-ldl")
            .arg("-lm");
    }
}
