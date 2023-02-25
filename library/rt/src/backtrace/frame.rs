use alloc::boxed::Box;
use alloc::string::ToString;
use core::cell::UnsafeCell;

#[cfg(feature = "std")]
use firefly_system::sync::OnceLock;

use super::{Symbol, Symbolication};
use crate::term::OpaqueTerm;

#[cfg(feature = "std")]
/// The current working directory in which the executable is running
static CWD: OnceLock<Option<std::path::PathBuf>> = OnceLock::new();

/// This trait allows us to abstract over the concrete implementation
/// of stack frames, which in the general case, requires libstd, which
/// we don't want to depend on in this crate, in order to allow supporting
/// platforms where libstd isn't available.
pub trait Frame {
    /// This function should resolve a frame to its symbolicated form,
    /// i.e. mfa+file+line
    fn resolve(&self) -> Option<Symbolication>;
    /// Returns the argument list term, if present in the frame metadata
    fn args(&self) -> Option<OpaqueTerm>;
}

#[cfg(feature = "std")]
impl Frame for backtrace::Frame {
    fn resolve(&self) -> Option<Symbolication> {
        use crate::function::ModuleFunctionArity;

        let mut result = None;

        let current_dir = CWD.get_or_init(|| std::env::current_dir().ok());

        backtrace::resolve_frame(self, |resolved_symbol| {
            let name = resolved_symbol.name();
            let symbol = if let Some(name) = name.and_then(|n| n.as_str()) {
                let mfa: Option<ModuleFunctionArity> = name.parse().ok();
                match mfa {
                    Some(mfa) => Some(Symbol::Erlang(mfa)),
                    None => Some(name.into()),
                }
            } else {
                None
            };
            let filename = resolved_symbol.filename().map(|p| match current_dir {
                None => p.to_string_lossy().into_owned(),
                Some(cwd) => match p.strip_prefix(cwd) {
                    Ok(stripped) => stripped.to_string_lossy().into_owned(),
                    Err(_) => p.to_string_lossy().into_owned(),
                },
            });
            let line = resolved_symbol.lineno();
            let column = resolved_symbol.colno();
            result = Some(Symbolication {
                symbol,
                filename,
                line,
                column,
            });
        });

        result
    }

    #[inline(always)]
    fn args(&self) -> Option<OpaqueTerm> {
        None
    }
}
impl Frame for firefly_bytecode::Symbol<crate::term::Atom> {
    fn resolve(&self) -> Option<Symbolication> {
        match self {
            Self::Erlang { mfa, loc } => {
                let symbol = Symbol::Erlang(mfa.clone().into());
                match loc {
                    None => Some(Symbolication::new(symbol, None, None, None)),
                    Some(loc) => {
                        let file = Some(loc.file.as_ref().to_string());
                        Some(Symbolication::new(
                            symbol,
                            file,
                            Some(loc.line),
                            Some(loc.column),
                        ))
                    }
                }
            }
            Self::Bif(mfa) => Some(Symbolication::new(
                Symbol::Erlang(mfa.clone().into()),
                None,
                None,
                None,
            )),
            Self::Native(name) => Some(Symbolication::new(
                Symbol::Native(name.as_str().to_string()),
                None,
                None,
                None,
            )),
        }
    }

    #[inline(always)]
    fn args(&self) -> Option<OpaqueTerm> {
        None
    }
}

/// This type is used to embellish a frame that has extra info, with that info
pub struct FrameWithExtraInfo<T: Frame> {
    pub frame: T,
    pub args: OpaqueTerm,
}
impl<F: Frame> FrameWithExtraInfo<F> {
    #[inline]
    pub fn new(frame: F, args: OpaqueTerm) -> Box<Self> {
        Box::new(Self { frame, args })
    }
}
impl<F: Frame> Frame for FrameWithExtraInfo<F> {
    #[inline]
    fn resolve(&self) -> Option<Symbolication> {
        self.frame.resolve()
    }

    #[inline]
    fn args(&self) -> Option<OpaqueTerm> {
        Some(self.args)
    }
}

/// This struct wraps the underlying concrete representation of a stack frame
/// and handles caching symbolication requests.
///
/// It is guaranteed that an exception will only ever be accessed by a single thread
/// at a time, but there is no restriction on handing off an exception to another thread,
/// however this would always be exclusive of concurrent reads/writes. As a result, we
/// use an UnsafeCell to store the cached symbolication result, since we don't want to incur
/// unnecessary overhead accessing it. Should the guarantees change around concurrent access,
/// this will need to be changed to some other Sync type like `RefCell`.
pub struct TraceFrame {
    pub(crate) frame: Box<dyn Frame>,
    symbol: UnsafeCell<Option<Symbolication>>,
}
impl TraceFrame {
    /// Produce the symbolication data for this frame
    pub fn symbolicate(&self) -> Option<&Symbolication> {
        // If we've already symbolicated, return the cached value
        if let Some(sym) = self.symbol() {
            return Some(sym);
        }

        // Otherwise resolve symbols for this frame
        self.set_symbol(self.frame.resolve())
    }

    /// Set `symbol` to the given value, and return a reference to it
    #[inline]
    fn set_symbol(&self, symbol: Option<Symbolication>) -> Option<&Symbolication> {
        let ptr = self.symbol.get();
        unsafe {
            ptr.write(symbol);
        }
        unsafe { &*ptr }.as_ref()
    }

    /// Returns `true` if the symbol cache for this frame is populated
    #[inline(always)]
    fn symbol(&self) -> Option<&Symbolication> {
        unsafe { &*self.symbol.get() }.as_ref()
    }
}
impl From<Box<dyn Frame>> for TraceFrame {
    fn from(frame: Box<dyn Frame>) -> Self {
        Self {
            frame,
            symbol: UnsafeCell::new(None),
        }
    }
}
#[cfg(feature = "std")]
impl From<backtrace::Frame> for TraceFrame {
    fn from(frame: backtrace::Frame) -> Self {
        Self {
            frame: Box::new(frame),
            symbol: UnsafeCell::new(None),
        }
    }
}
