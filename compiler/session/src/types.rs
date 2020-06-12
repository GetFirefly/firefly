use std::convert::{AsRef, From};
use std::fmt;
use std::ops::Deref;
use std::ptr;
use std::sync::Arc;

use libeir_ir as eir;
use libeir_syntax_erl as syntax;

use crate::config::{Emit, OutputType};

/// Holds a reference to a module in AST form
///
/// This is used by the incremental query engine,
/// as the AST module itself does not implement all
/// of the traits (nor can it), and is more expensive
/// to copy when retrieving.
///
/// In short, this acts like a shared reference, as long
/// as at least one instance is alive, the data is alive
#[derive(Clone)]
pub struct ParsedModule {
    module: Arc<syntax::ast::Module>,
}
impl fmt::Debug for ParsedModule {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let ident = self.module.name.clone();
        let ptr = self.module.as_ref() as *const _;
        write!(f, "ParsedModule({:?} at {:p})", ident, ptr)
    }
}
impl Eq for ParsedModule {}
impl PartialEq for ParsedModule {
    fn eq(&self, other: &Self) -> bool {
        self.module.name.eq(&other.module.name)
            && ptr::eq(
                self.module.as_ref() as *const _,
                other.module.as_ref() as *const _,
            )
    }
}
impl Deref for ParsedModule {
    type Target = syntax::ast::Module;

    fn deref(&self) -> &Self::Target {
        self.module.deref()
    }
}
impl AsRef<syntax::ast::Module> for ParsedModule {
    fn as_ref(&self) -> &syntax::ast::Module {
        self.module.deref()
    }
}
impl From<syntax::ast::Module> for ParsedModule {
    fn from(module: syntax::ast::Module) -> Self {
        Self {
            module: Arc::new(module),
        }
    }
}

/// Holds a reference to a module in EIR form
///
/// Like `ParsedModule`, this is used by the incremental
/// query engine to hold a reference to a EIR module in its
/// cache
#[derive(Clone)]
pub struct IRModule {
    module: Arc<eir::Module>,
}

unsafe impl Send for IRModule {}
// This implementation of Sync is safe ONLY because a module
// is read-only in the compiler; if that changes, then this
// will result in bugs.
unsafe impl Sync for IRModule {}

impl IRModule {
    #[inline]
    pub fn new(module: eir::Module) -> Self {
        Self {
            module: Arc::new(module),
        }
    }
}
impl fmt::Debug for IRModule {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let ident = self.module.name().clone();
        let ptr = self.module.as_ref() as *const _;
        write!(f, "IRModule({:?} at {:p})", ident, ptr)
    }
}
impl Eq for IRModule {}
impl PartialEq for IRModule {
    fn eq(&self, other: &Self) -> bool {
        self.module.name().eq(&other.module.name())
            && ptr::eq(
                self.module.as_ref() as *const _,
                other.module.as_ref() as *const _,
            )
    }
}
impl Deref for IRModule {
    type Target = eir::Module;

    fn deref(&self) -> &Self::Target {
        self.module.deref()
    }
}
impl AsRef<eir::Module> for IRModule {
    fn as_ref(&self) -> &eir::Module {
        self.module.as_ref()
    }
}
impl From<eir::Module> for IRModule {
    fn from(module: eir::Module) -> Self {
        Self {
            module: Arc::new(module),
        }
    }
}
impl Emit for IRModule {
    const TYPE: OutputType = OutputType::EIR;

    fn emit(&self, f: &mut std::fs::File) -> anyhow::Result<()> {
        self.module.emit(f)
    }
}
