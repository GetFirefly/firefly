mod hello_world {
    use std::process::{Command, Stdio};

    #[test]
    fn prints_hello_world() {
        std::fs::create_dir_all("_build").unwrap();

        let compile_output = Command::new("../bin/lumen")
            .arg("compile")
            .arg("--output-dir")
            .arg("_build")
            .arg("-o")
            .arg("hello_world")
            .arg("-lc")
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

        assert_eq!(
            String::from_utf8_lossy(&hello_world_output.stdout),
            "\"Hello, world!\"\n"
        );
    }
}
