mod context;
mod diagnostics;
mod dialect;
pub mod ir;
mod module;

pub use self::context::{Context, ContextRef};
pub use self::diagnostics::*;
pub use self::dialect::Dialect;
pub use self::module::{Module, ModuleRef};

use liblumen_session::Options;

pub fn init(_options: &Options) -> anyhow::Result<()> {
    unsafe {
        MLIRLumenInit();
    }

    Ok(())
}

extern "C" {
    fn MLIRLumenInit();
}
