mod frame;
mod symbolication;
mod trace;

pub use self::frame::{Frame, TraceFrame};
pub use self::symbolication::Symbolication;
pub use self::trace::Trace;

/*
use alloc::string::String;
use alloc::sync::Arc;
use alloc::vec::Vec;

pub struct Trace(Vec<String>);
impl Trace {
    pub fn capture() -> Arc<Self> {
        Arc::new(Self(Vec::new()))
    }

    pub fn into_raw(trace: Arc<Trace>) -> *mut Trace {
        Arc::into_raw(trace) as *mut Trace
    }

    pub unsafe fn from_raw(ptr: *mut Trace) -> Arc<Trace> {
        Arc::from_raw(ptr)
    }
}
*/
