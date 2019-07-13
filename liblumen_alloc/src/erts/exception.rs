/// Combined `System` and `User` exceptions.
pub mod runtime;
pub mod system;

use core::alloc::AllocErr;
use core::convert::Into;

use crate::erts::term::{BytesFromBinaryError, StrFromBinaryError, Term};
use crate::erts::process::alloc::heap_alloc::{HeapAlloc, MakePidError};

#[derive(Debug, PartialEq)]
pub enum Exception {
    System(system::Exception),
    Runtime(runtime::Exception),
}

impl Exception {
    pub fn badkey<A: HeapAlloc>(
        heap: &mut A,
        key: Term,
        #[cfg(debug_assertions)]
        file: &'static str,
        #[cfg(debug_assertions)]
        line: u32,
        #[cfg(debug_assertions)]
        column: u32,
    ) -> Self {
        match runtime::Exception::badkey(
            heap,
            key,
            #[cfg(debug_assertions)]
            file,
            #[cfg(debug_assertions)]
            line,
            #[cfg(debug_assertions)]
            column
        ) {
            Ok(runtime_error) => runtime_error.into(),
            Err(alloc_err) => alloc_err.into()
        }
    }

    pub fn badmap<A: HeapAlloc>(
        heap: &mut A,
        map: Term,
        #[cfg(debug_assertions)]
        file: &'static str,
        #[cfg(debug_assertions)]
        line: u32,
        #[cfg(debug_assertions)]
        column: u32,
    ) -> Self {
        match runtime::Exception::badmap(
            heap,
            map,
            #[cfg(debug_assertions)]
            file,
            #[cfg(debug_assertions)]
            line,
            #[cfg(debug_assertions)]
            column
        ) {
            Ok(runtime_error) => runtime_error.into(),
            Err(alloc_err) => alloc_err.into()
        }
    }

    pub fn undef<A: HeapAlloc>(
        heap: &mut A,
        module: Term,
        function: Term,
        arguments: Term,
        stacktrace_tail: Term,
        #[cfg(debug_assertions)]
        file: &'static str,
        #[cfg(debug_assertions)]
        line: u32,
        #[cfg(debug_assertions)]
        column: u32,
    ) -> Self {
        match runtime::Exception::undef(
            heap,
            module,
            function,
            arguments,
            stacktrace_tail,
         #[cfg(debug_assertions)]
            file,
        #[cfg(debug_assertions)]
            line,
         #[cfg(debug_assertions)]
            column,
        ) {
            Ok(runtime_error) => runtime_error.into(),
            Err(alloc_err) => alloc_err.into()
        }
    }
}

impl<R: Into<runtime::Exception>> From<R> for Exception {
    fn from(r: R) -> Self {
        Exception::Runtime(r.into())
    }
}

impl From<AllocErr> for Exception {
    fn from(alloc_err: AllocErr) -> Self {
        Exception::System(alloc_err.into())
    }
}

impl From<BytesFromBinaryError> for Exception {
    fn from(bytes_from_binary_error: BytesFromBinaryError) -> Self {
        use BytesFromBinaryError::*;

        match bytes_from_binary_error {
            Alloc(error) => error.into(),
            NotABinary | Type => badarg!().into(),
        }
    }
}

impl From<MakePidError> for Exception {
    fn from(make_pid_errror: MakePidError) -> Self {
        use MakePidError::*;

        match make_pid_errror {
            Alloc(error) => error.into(),
            Number | Serial => badarg!().into(),
        }
    }
}

impl From<StrFromBinaryError> for Exception {
    fn from(str_from_binary_error: StrFromBinaryError) -> Self {
        use StrFromBinaryError::*;

        match str_from_binary_error {
            Alloc(error) => error.into(),
            NotABinary | Type | Utf8Error(_) => badarg!().into(),
        }
    }
}

pub type Result = core::result::Result<Term, Exception>;
