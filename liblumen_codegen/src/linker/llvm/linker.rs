use std::ffi::{CString, OsStr};
use std::path::Path;

use crate::linker::LinkerError;
use crate::config::Config;

use super::ffi;

#[derive(Debug)]
pub struct Linker {
    args: Vec<CString>,
}
impl Linker {
    pub fn new(config: &Config) -> Self {
        let output_dir = config.output_dir().join("a.bc");
        let args = vec![
            to_cstring("llvm-link"),
            to_cstring("-o"),
            to_cstring(output_dir.to_string_lossy().to_owned()),
        ];
        Linker {
            args,
        }
    }

    pub fn link(&self) -> Result<(), LinkerError> {
        // Map to Vec<*const c_char>
        let args = self.args().iter().map(|s| s.as_ptr()).collect::<Vec<_>>();
        if ffi::lumen_link(args.as_ptr(), args.len() as i32) {
            Ok(())
        } else {
            Err(LinkerError::LinkingFailed)
        }
    }

    pub fn cmd_like_arg<S>(&mut self, arg: S)
    where
        S: AsRef<OsStr>,
    {
        self.args
            .push(to_cstring(&arg.as_ref().to_string_lossy().to_owned()));
    }

    pub fn cmd_like_args<S>(&mut self, args: &[S])
    where
        S: AsRef<OsStr>,
    {
        for arg in args {
            self.cmd_like_arg(arg);
        }
    }

    pub fn args(&self) -> &[CString] {
        &self.args
    }

    pub fn link_file(&mut self, path: &Path) {
        self.cmd_like_arg(path)
    }

    pub fn link_files(&mut self, paths: &[&Path]) {
        self.cmd_like_args(paths)
    }
}

#[inline]
fn to_cstring(s: &str) -> CString {
    CString::new(s).unwrap()
}
