use std::process::{Command, Stdio};
use std::sync::OnceLock;

use serde::Deserialize;

static LLVM_TARGET: OnceLock<String> = OnceLock::new();

#[derive(Deserialize)]
struct TargetSpec {
    #[serde(rename = "llvm-target")]
    llvm_target: String,
}

pub fn get_llvm_target(toolchain_name: &str, target: &str) -> &'static str {
    let target = LLVM_TARGET.get_or_init(|| {
        let mut rustc_cmd = Command::new("rustup");
        let rustc_cmd = rustc_cmd
            .arg("run")
            .arg(toolchain_name)
            .args(&["rustc"])
            .args(&["-Z", "unstable-options"])
            .args(&["--print", "target-spec-json", "--target"])
            .arg(target);

        let output = rustc_cmd
            .stdin(Stdio::null())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .output()
            .unwrap();

        if !output.status.success() {
            panic!(
                "unable to determine llvm target triple!: {}",
                String::from_utf8(output.stderr).unwrap()
            );
        }

        let spec: TargetSpec = serde_json::from_slice(output.stdout.as_slice()).unwrap();
        spec.llvm_target
    });
    target.as_str()
}
