use std::ffi::{CString, OsStr};
use std::path::Path;

use crate::llvm::enums::OptimizationLevel;
use crate::linker::{LinkerError, OutputType};
use crate::config::Config;

use super::ffi::{self, LinkerFlavor};

#[derive(Debug)]
pub struct NativeLinker {
    flavor: LinkerFlavor,
    args: Vec<CString>,
    output_dir: PathBuf,
}
impl NativeLinker {
    pub fn new(config: &Config) -> Self {
        let flavor = LinkerFlavor::default();
        let args = vec![
            to_cstring("ld"),
            to_cstring(config.target_arch()),
        ];
        let build_type = config.build_type();
        args.push(to_cstring(build_type.to_linker_flag()));
        let output_dir = config.output_dir();

        Linker {
            flavor,
            args,
            output_dir,
        }
    }

    pub fn link(&self) -> Result<(), LinkerError> {
        // Map to Vec<*const c_char>
        let args = self.args().iter().map(|s| s.as_ptr()).collect::<Vec<_>>();
        if ffi::lumen_lld(self.flavor.into(), args.as_ptr(), args.len() as i32) {
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

    #[allow(unused)]
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

    #[allow(unused)]
    pub fn link_dylib(&mut self, lib: &str) {
        self.args.push(to_cstring("-l"));
        self.args.push(to_cstring(lib));
    }

    #[allow(unused)]
    pub fn link_staticlib(&mut self, lib: &str) {
        self.args.push(to_cstring("-l"));
        self.args.push(to_cstring(lib));
    }

    #[allow(unused)]
    pub fn include_path(&mut self, path: &Path) {
        self.args.push(to_cstring("-L"));
        self.cmd_like_arg(path);
    }

    #[allow(unused)]
    pub fn framework_path(&mut self, path: &Path) {
        self.args.push(to_cstring("-framework"));
        self.cmd_like_arg(path);
    }

    #[inline]
    pub fn add_input(&mut self, path: &Path) {
        self.cmd_like_arg(path);
    }

    pub fn add_inputs(&mut self, paths: &[PathBuf]) {
        for path in paths.iter() {
            self.add_input(path);
        }
    }

    #[allow(unused)]
    pub fn gc_sections(&mut self, _: bool) {
        self.args.push(to_cstring("--gc-sections"));
    }

    pub fn optimize(&mut self, level: OptimizationLevel) {
        let o = match level {
            OptimizationLevel::None => 0,
            OptimizationLevel::Less => 1,
            OptimizationLevel::Default => 2,
            OptimizationLevel::Aggressive => 3,
        };
        self.args.push(to_cstring(&format!("-lto-O{}", o)))
    }
}

#[inline]
fn to_cstring(s: &str) -> CString {
    CString::new(s).unwrap()
}
