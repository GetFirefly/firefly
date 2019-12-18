extern crate which;

use std::process::{Command, Stdio};

fn main() {
    let (hash, hash_date) = git_version();
    println!("cargo:rustc-env=LUMEN_COMMIT_HASH={}", hash);
    println!("cargo:rustc-env=LUMEN_COMMIT_DATE={}", hash_date);
    println!("cargo:rerun-if-env-changed=LUMEN_COMMIT_DATE");
    println!("cargo:rerun-if-env-changed=LUMEN_COMMIT_HASH");
    println!("cargo:rerun-if-changed=build.rs");
}

pub fn git_version() -> (String, String) {
    if let Err(_) = which::which("git") {
        fail("Unable to locate 'git'");
    }
    let mut cmd = Command::new("git");
    cmd.arg("log")
        .arg("-n1")
        .arg("--pretty=format:\"%h %cd\"")
        .arg("--date=human");

    let out = output(&mut cmd);
    let mut split = out.splitn(2, ' ');
    let hash = split
        .next()
        .expect("expected git hash")
        .trim_start_matches('"')
        .to_string();
    let date = split
        .next()
        .expect("expected git commit date")
        .trim_end_matches('"')
        .to_string();

    (hash, date)
}

pub fn output(cmd: &mut Command) -> String {
    let output = match cmd.stderr(Stdio::inherit()).output() {
        Ok(status) => status,
        Err(e) => fail(&format!(
            "failed to execute command: {:?}\nerror: {}",
            cmd, e
        )),
    };
    if !output.status.success() {
        panic!(
            "command did not execute successfully: {:?}\n\
             expected success, got: {}",
            cmd, output.status
        );
    }
    String::from_utf8(output.stdout).unwrap()
}

fn fail(s: &str) -> ! {
    panic!("\n{}\n\nbuild script failed, must exit now", s);
}
