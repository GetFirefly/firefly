pub mod ffi;
mod linker;

use crate::compiler::codegen::Optimization;

use self::linker::{Linker, LinkerError};

pub fn link(objs: &[PathBuf], out: &Path, opt: Optimization) -> Result<(), LinkerError> {
    let mut lld = Linker::new();
    for obj in objs {
        lld.add_object(obj);
    }
    lld.cmd_like_arg(out);
    lld.optimize(opt);
    lld.link()
}
