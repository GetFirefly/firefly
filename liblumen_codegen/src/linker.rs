#![allow(dead_code)]

use std::fs::OpenOptions;
use std::path::{Path, PathBuf};
use std::process::Command;

use tempfile::NamedTempFile;

use super::CodeGenError;

pub fn link(obj: &Path, out: &Path) -> Result<(), CodeGenError> {
    // Crate temp file to hold linker output
    let mut temp_out = NamedTempFile::new().expect("could not create temp file");
    // Invoke linker
    ld(obj, temp_out.path())?;
    // Create a new file to hold final output
    let mut oo = OpenOptions::new();
    oo.create(true).write(true);
    // Make executable on *NIX
    if cfg!(unix) {
        use std::os::unix::fs::OpenOptionsExt;
        oo.mode(0o777);
    }
    // Write final output
    match oo.open(out) {
        Err(err) => Err(CodeGenError::LinkerError(err.to_string())),
        Ok(mut out) => {
            // Finally, copy the linked executable to its final destination
            match std::io::copy(&mut temp_out, &mut out) {
                Err(err) => Err(CodeGenError::LinkerError(err.to_string())),
                Ok(_) => Ok(()),
            }
        }
    }
}

fn ld(obj: &Path, out: &Path) -> Result<(), CodeGenError> {
    let result = Command::new("ld").arg("-o").arg(out).arg(obj).output();
    match result {
        Err(e) => {
            let msg = format!("Failed to execute linker: {}", e.to_string());
            Err(CodeGenError::LinkerError(msg.to_string()))
        }
        Ok(ref output) => {
            if output.status.success() {
                if cfg!(unix) {
                    let _ = Command::new("chown").arg("+x").arg(out).status().ok();
                }
                Ok(())
            } else {
                let stdout = String::from_utf8_lossy(&output.stdout);
                let stderr = String::from_utf8_lossy(&output.stderr);
                let msg = format!(
                    "Linker failed (status {}):\n{}\n{}",
                    output.status, stdout, stderr
                );
                Err(CodeGenError::LinkerError(msg.to_string()))
            }
        }
    }
}

fn with_linker_extension(path: &Path) -> PathBuf {
    if cfg!(target_os = "windows") {
        path.with_extension(".exe").as_path().to_owned()
    } else {
        let stem = path.file_stem().unwrap();
        path.parent().unwrap().join(stem).as_path().to_owned()
    }
}
