/// Combined `System` and `User` exceptions.
pub mod runtime;
pub mod system;

use core::convert::Into;
use core::num::TryFromIntError;

use crate::erts::process::alloc::heap_alloc::MakePidError;
use crate::erts::process::Process;

use super::term::index::IndexError;
use super::term::prelude::*;
use super::string::InvalidEncodingNameError;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Exception {
    System(system::Exception),
    Runtime(runtime::Exception),
}

impl Exception {
    pub fn badarity(
        process: &Process,
        function: Term,
        arguments: Term,
        file: &'static str,
        line: u32,
        column: u32,
    ) -> Self {
        match runtime::Exception::badarity(process, function, arguments, file, line, column) {
            Ok(runtime_exception) => Self::Runtime(runtime_exception.into()),
            Err(alloc) => Self::System(alloc.into()),
        }
    }

    pub fn badfun(
        process: &Process,
        function: Term,
        file: &'static str,
        line: u32,
        column: u32,
    ) -> Self {
        match runtime::Exception::badfun(process, function, file, line, column) {
            Ok(runtime_exception) => Self::Runtime(runtime_exception.into()),
            Err(alloc) => Self::System(alloc.into()),
        }
    }

    pub fn badkey(
        process: &Process,
        key: Term,
        file: &'static str,
        line: u32,
        column: u32,
    ) -> Self {
        match runtime::Exception::badkey(process, key, file, line, column) {
            Ok(runtime_error) => Self::Runtime(runtime_error.into()),
            Err(alloc) => Self::System(alloc.into()),
        }
    }

    pub fn badmap(
        process: &Process,
        map: Term,
        file: &'static str,
        line: u32,
        column: u32,
    ) -> Self {
        match runtime::Exception::badmap(process, map, file, line, column) {
            Ok(runtime_error) => Self::Runtime(runtime_error.into()),
            Err(alloc) => Self::System(alloc.into()),
        }
    }

    pub fn undef(
        process: &Process,
        module: Term,
        function: Term,
        arguments: Term,
        stacktrace_tail: Term,
        file: &'static str,
        line: u32,
        column: u32,
    ) -> Self {
        match runtime::Exception::undef(
            process,
            module,
            function,
            arguments,
            stacktrace_tail,
            file,
            line,
            column,
        ) {
            Ok(runtime_error) => Self::Runtime(runtime_error.into()),
            Err(alloc) => Self::System(alloc.into()),
        }
    }
}
impl From<system::Alloc> for Exception {
    fn from(alloc: system::Alloc) -> Self {
        Self::System(alloc.into())
    }
}

impl From<AtomError> for Exception {
    fn from(atom_error: AtomError) -> Self {
        Self::Runtime(atom_error.into())
    }
}

impl From<BoolError> for Exception {
    fn from(bool_error: BoolError) -> Self {
        Self::Runtime(bool_error.into())
    }
}

impl From<BytesFromBinaryError> for Exception {
    fn from(bytes_from_binary_error: BytesFromBinaryError) -> Self {
        use BytesFromBinaryError::*;

        match bytes_from_binary_error {
            NotABinary | Type => Self::Runtime(badarg!().into()),
            Alloc(error) => Self::System(error.into()),
        }
    }
}

impl From<InvalidEncodingNameError> for Exception {
    fn from(encoding_error: InvalidEncodingNameError) -> Self {
        Self::Runtime(encoding_error.into())
    }
}

impl From<runtime::Exception> for Exception {
    fn from(runtime: runtime::Exception) -> Self {
        Self::Runtime(runtime)
    }
}

impl From<ImproperList> for Exception {
    fn from(improper_list: ImproperList) -> Self {
        Self::Runtime(improper_list.into())
    }
}

impl From<IndexError> for Exception {
    fn from(index_error: IndexError) -> Self {
        Self::Runtime(index_error.into())
    }
}

impl From<MakePidError> for Exception {
    fn from(make_pid_error: MakePidError) -> Self {
        use MakePidError::*;

        match make_pid_error {
            Number | Serial => Self::Runtime(badarg!().into()),
            Alloc(error) => Self::System(error.into()),
        }
    }
}

impl From<StrFromBinaryError> for Exception {
    fn from(str_from_binary_error: StrFromBinaryError) -> Self {
        use StrFromBinaryError::*;

        match str_from_binary_error {
            NotABinary | Type | Utf8Error(_) => Self::Runtime(badarg!().into()),
            Alloc(error) => Self::System(error.into()),
        }
    }
}

impl From<TryFromIntError> for Exception {
    fn from(try_from_int_error: TryFromIntError) -> Self {
        Self::Runtime(try_from_int_error.into())
    }
}

impl From<TryIntoIntegerError> for Exception {
    fn from(try_into_integer_error: TryIntoIntegerError) -> Self {
        Self::Runtime(try_into_integer_error.into())
    }
}

impl From<TypeError> for Exception {
    fn from(type_error: TypeError) -> Self {
        Self::Runtime(type_error.into())
    }
}

pub type Result = core::result::Result<Term, Exception>;
