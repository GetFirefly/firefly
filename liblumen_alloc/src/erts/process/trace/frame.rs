use liblumen_core::util::thread_local::ThreadLocalCell;

use super::{FrameImpl, Symbolication};

pub struct TraceFrame {
    frame: FrameImpl,
    symbol: ThreadLocalCell<Option<Symbolication>>,
}
impl TraceFrame {
    pub fn symbolicate(&self) -> Option<&Symbolication> {
        // If we've already symbolicated, return the cached value
        let sym = self.symbol.as_ref();
        if sym.is_some() {
            return sym.as_ref();
        }

        // Otherwise resolve symbols for this frame
        if let Some(symbol) = super::resolve_frame(&self.frame) {
            unsafe { self.symbol.set(Some(symbol)) }
        }

        let symbol = super::resolve_frame(&self.frame);
        if symbol.is_some() {
            unsafe {
                self.symbol.set(symbol);
            }
        } else {
            // Do not try to symbolicate again
            unsafe { self.symbol.set(Some(Symbolication::default())) };
        }

        self.symbol.as_ref().as_ref()
    }
}
impl From<&FrameImpl> for TraceFrame {
    fn from(frame: &FrameImpl) -> Self {
        Self {
            frame: frame.clone(),
            symbol: ThreadLocalCell::new(None),
        }
    }
}
impl From<FrameImpl> for TraceFrame {
    fn from(frame: FrameImpl) -> Self {
        Self {
            frame,
            symbol: ThreadLocalCell::new(None),
        }
    }
}
