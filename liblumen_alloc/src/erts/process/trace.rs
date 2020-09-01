use cfg_if::cfg_if;

cfg_if! {
    if #[cfg(not(target_arch = "wasm32"))] {
        mod backtrace;
        use self::backtrace as inner;
    } else {
        mod fallback;
        use self::fallback as inner;
    }
}

mod format;
mod frame;
mod utils;

pub use self::frame::TraceFrame;
pub use self::inner::Frame as FrameImpl;
pub use self::utils::Symbolication;

use self::inner::resolve_frame;
pub use self::inner::Trace;
