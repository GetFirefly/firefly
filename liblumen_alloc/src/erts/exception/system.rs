use core::alloc::AllocErr;

#[derive(Debug, PartialEq)]
pub enum Exception {
    // TODO include the needed size
    AllocErr,
}

impl From<AllocErr> for Exception {
    fn from(_: AllocErr) -> Exception {
        Exception::AllocErr
    }
}
