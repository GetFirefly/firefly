use std::path::Path;
use std::ffi::{CString, OsStr};

use crate::llvm::enums::Optimization;

use super::ffi;

#[derive(Debug)]
pub enum LinkerFlavor {
    Elf,
    Coff,
    MingW,
    MachO,
    Wasm
}
impl LinkerFlavor {
    pub fn get() -> LinkerFlavor {
        if cfg!(target_arch = "wasm32") {
            LinkerFlavor::Wasm
        } else if cfg!(target_family = "windows") && cfg!(target_env = "gnu") {
            LinkerFlavor::MingW
        } else if cfg!(target_env = "ios") {
            LinkerFlavor::MachO
        } else if cfg!(target_family = "windows") {
            LinkerFlavor::Coff
        } else {
            LinkerFlavor::Elf
        }
    }
}
impl Default for LinkerFlavor {
    fn default() -> Self {
        LinkerFlavor::get()
    }
}

#[derive(Debug)]
pub enum LinkerError {
    LinkingFailed
}

#[derive(Debug)]
pub struct Linker {
    flavor: LinkerFlavor,
    args: Vec<CString>,
}
impl Linker {
    pub fn new() -> Self {
        Linker { flavor: LinkerFlavor::default(), args: vec![to_cstring("lld")] }
    }

    pub fn link(&self) -> Result<(), LinkerError> {
        let args = self.args().iter().map(|s| s.as_ptr()).collect::<Vec<_>>();
        let result = match self.flavor {
            LinkerFlavor::Elf => unsafe { ffi::lumen_lld_elf_link(args.as_ptr(), args.len() as u32) },
            LinkerFlavor::Coff => unsafe { ffi::lumen_lld_coff_link(args.as_ptr(), args.len() as u32) },
            LinkerFlavor::MingW => unsafe { ffi::lumen_lld_mingw_link(args.as_ptr(), args.len() as u32) },
            LinkerFlavor::MachO => unsafe { ffi::lumen_lld_mach_o_link(args.as_ptr(), args.len() as u32) },
            LinkerFlavor::Wasm => unsafe { ffi::lumen_lld_wasm_link(args.as_ptr(), args.len() as u32) },
        };
        if result {
            Ok(())
        } else {
            Err(LinkerError::LinkingFailed)
        }
    }

    pub fn cmd_like_arg<S>(&mut self, arg: S)
        where S: AsRef<OsStr>
    {
        self.args.push(to_cstring(&arg.as_ref().to_string_lossy().to_owned()));
    }

    #[allow(unused)]
    pub fn cmd_like_args<S>(&mut self, args: &[S])
        where S: AsRef<OsStr>
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

    pub fn add_object(&mut self, path: &Path) {
        self.cmd_like_arg(path);
    }

    #[allow(unused)]
    pub fn gc_sections(&mut self, _: bool) {
        self.args.push(to_cstring("--gc-sections"));
    }

    pub fn optimize(&mut self, level: Optimization) {
        let o = match level {
            Optimization::None => 0,
            Optimization::Less => 1,
            Optimization::Default => 2,
            Optimization::Aggressive => 3
        };
        self.args.push(to_cstring(&format!("-lto-O{}", o)))
    }
}

fn to_cstring(s: &str) -> CString {
    CString::new(s).unwrap()
}
