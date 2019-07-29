/// Combined `System` and `User` exceptions.
pub mod runtime;
pub mod system;

use core::alloc::AllocErr;
use core::convert::Into;

use crate::erts::term::{BytesFromBinaryError, StrFromBinaryError, Term};
use crate::erts::process::alloc::heap_alloc::{MakePidError};
use crate::erts::process::ProcessControlBlock;

#[derive(Debug, PartialEq)]
pub enum Exception {
    System(system::Exception),
    Runtime(runtime::Exception),
}

impl Exception {
    pub fn badarity(
        process: &ProcessControlBlock,
        function: Term,
        arguments: Term,
        #[cfg(debug_assertions)]
        file: &'static str,
        #[cfg(debug_assertions)]
        line: u32,
        #[cfg(debug_assertions)]
        column: u32) -> Self {
        match runtime::Exception::badarity(
            process,
            function,
            arguments,
         #[cfg(debug_assertions)]
            file,
        #[cfg(debug_assertions)]
            line,
         #[cfg(debug_assertions)]
            column
        ) {
            Ok(runtime_exception) => runtime_exception.into(),
            Err(alloc_err) => alloc_err.into()
        }
    }

    pub fn badfun(
        process: &ProcessControlBlock,
        function: Term,
        #[cfg(debug_assertions)]
        file: &'static str,
        #[cfg(debug_assertions)]
        line: u32,
        #[cfg(debug_assertions)]
        column: u32) -> Self {
        match runtime::Exception::badfun(
            process,
            function,
            #[cfg(debug_assertions)]
            file,
            #[cfg(debug_assertions)]
            line,
            #[cfg(debug_assertions)]
            column
        ) {
            Ok(runtime_exception) => runtime_exception.into(),
            Err(alloc_err) => alloc_err.into()
        }
    }

    pub fn badkey(
        process: &ProcessControlBlock,
        key: Term,
        #[cfg(debug_assertions)]
        file: &'static str,
        #[cfg(debug_assertions)]
        line: u32,
        #[cfg(debug_assertions)]
        column: u32,
    ) -> Self {
        match runtime::Exception::badkey(
            process,
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

    pub fn badmap(
        process: &ProcessControlBlock,
        map: Term,
        #[cfg(debug_assertions)]
        file: &'static str,
        #[cfg(debug_assertions)]
        line: u32,
        #[cfg(debug_assertions)]
        column: u32,
    ) -> Self {
        match runtime::Exception::badmap(
            process,
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

    pub fn undef(
        process: &ProcessControlBlock,
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
            process,
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
