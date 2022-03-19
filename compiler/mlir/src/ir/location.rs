use std::ffi::c_void;
use std::fmt::{self, Display};

use crate::support::{self, MlirStringCallback, StringRef};
use crate::Context;

extern "C" {
    type MlirLocation;
}

/// Represents a source location in MLIR
///
/// MLIR has several different subtypes of locations, they are all
/// represented using this struct in Rust
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct Location(*mut MlirLocation);
impl Location {
    pub fn get(context: Context, filename: &str, line: u32, col: u32) -> Self {
        unsafe { mlir_location_file_line_col_get(context, filename.into(), line, col) }
    }

    pub fn fuse(context: Context, locs: &[Location]) -> Location {
        unsafe { mlir_location_fused_get(context, locs.as_ptr(), locs.len()) }
    }

    pub fn unknown(context: Context) -> Self {
        unsafe { mlir_location_unknown_get(context) }
    }

    pub fn get_context(self) -> Context {
        unsafe { mlir_location_get_context(self) }
    }
}
impl fmt::Pointer for Location {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:p}", self.0)
    }
}
impl Eq for Location {}
impl PartialEq for Location {
    fn eq(&self, other: &Self) -> bool {
        ::core::ptr::eq(self.0, other.0)
    }
}
impl Display for Location {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        unsafe {
            mlir_location_print(
                *self,
                support::write_to_formatter,
                f as *mut _ as *mut c_void,
            );
        }
        Ok(())
    }
}

extern "C" {
    #[link_name = "mlirLocationFileLineColGet"]
    fn mlir_location_file_line_col_get(
        context: Context,
        filename: StringRef,
        line: u32,
        col: u32,
    ) -> Location;
    #[link_name = "mlirLocationFusedGet"]
    fn mlir_location_fused_get(context: Context, locs: *const Location, len: usize) -> Location;
    #[link_name = "mlirLocationUnknownGet"]
    fn mlir_location_unknown_get(context: Context) -> Location;
    #[link_name = "mlirLocationGetContext"]
    fn mlir_location_get_context(loc: Location) -> Context;
    #[link_name = "mlirLocationPrint"]
    fn mlir_location_print(loc: Location, callback: MlirStringCallback, userdata: *mut c_void);
}
