mod cli {
    use std::process::{Command, Stdio};
    use std::sync::Once;

    #[test]
    fn without_arguments_prints_nothing_to_say() {
        ensure_compiled();

        let cli_output = Command::new("./cli").stdin(Stdio::null()).output().unwrap();

        assert_eq!(
            String::from_utf8_lossy(&cli_output.stdout),
            "\"Nothing to say.\"\n"
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

        assert_eq!(
            String::from_utf8_lossy(&cli_output.stdout),
            "\"Nothing to say.\"\n"
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

        assert_eq!(
            String::from_utf8_lossy(&cli_output.stdout),
            "\"Hello, world!\"\n"
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

        let compile_output = Command::new("../bin/lumen")
            .arg("compile")
            .arg("--output-dir")
            .arg("_build")
            .arg("-o")
            .arg("cli")
            .arg("-lc")
            .arg("tests/cli/init.erl")
            .output()
            .unwrap();

        assert!(
            compile_output.status.success(),
            "stdout = {}\nstderr = {}",
            String::from_utf8_lossy(&compile_output.stdout),
            String::from_utf8_lossy(&compile_output.stderr)
        );
    }
}
