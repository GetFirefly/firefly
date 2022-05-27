use alloc::boxed::Box;
use core::cell::UnsafeCell;

use super::Symbolication;

/// This trait allows us to abstract over the concrete implementation
/// of stack frames, which in the general case, requires libstd, which
/// we don't want to depend on in this crate, in order to allow supporting
/// platforms where libstd isn't available.
pub trait Frame {
    /// This function should resolve a frame to its symbolicated form,
    /// i.e. mfa+file+line
    fn resolve(&self) -> Option<Symbolication>;
}

#[cfg(feature = "std")]
impl Frame for backtrace::BacktraceFrame {
    fn resolve(&self) -> Option<Symbolication> {
        let mut result = None;

        backtrace::resolve_frame(self, |symbol| {
            let name = symbol.name();
            let mfa = if let Some(name) = name.as_str() {
                name.parse().ok()
            } else {
                None
            };
            let filename = symbol.filename().map(|p| p.to_string_lossy().into_owned());
            let line = symbol.lineno();
            result = Some(Symbolication {
                mfa,
                filename,
                line,
            });
        });

        result
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
    frame: Box<dyn Frame>,
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
impl From<backtrace::BacktraceFrame> for TraceFrame {
    fn from(frame: backtrace::BacktraceFrame) -> Self {
        Self {
            frame: Box::new(frame),
            symbol: UnsafeCell::new(None),
        }
    }
}
