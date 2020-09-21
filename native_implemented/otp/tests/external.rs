//! Tests that use `liblumen_otp` with external dependencies to check that `lumen` can compile the
//! external dependencies and that `liblumen_otp` is supplying all the needed BIFs for those
//! external dependencies.  These would normally be tests in the dependencies since they are built
//! on top of OTP, but since the dependencies don't know that `lumen` exists, we test them here
//! instead.

#[macro_use]
#[path = "test.rs"]
mod test;

#[path = "external/lumen.rs"]
mod lumen;
