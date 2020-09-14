mod hello_world {
    use std::process::{Command, Stdio};

    #[test]
    fn prints_hello_world() {
        let mut command = Command::new("../bin/lumen");

        command
            .arg("compile")
            .arg("--output-dir")
            .arg("_build")
            .arg("--output")
            .arg("hello_world")
            // Turn off optimizations as work-around for debug info bug in EIR
            .arg("-O0");

        let compile_output = command
            .arg("tests/hello_world/init.erl")
            .stdin(Stdio::null())
            .output()
            .unwrap();

        assert!(
            compile_output.status.success(),
            "stdout = {}\nstderr = {}",
            String::from_utf8_lossy(&compile_output.stdout),
            String::from_utf8_lossy(&compile_output.stderr)
        );

        let hello_world_output = Command::new("./hello_world").output().unwrap();
        let hello_world_stdout = String::from_utf8_lossy(&hello_world_output.stdout);
        let hello_world_stderr = String::from_utf8_lossy(&hello_world_output.stderr);

        assert_eq!(
            hello_world_stdout, "<<\"Hello, world!\">>\n",
            "\nstdout = {}\nstderr = {}",
            hello_world_stdout, hello_world_stderr
        );
    }
}
