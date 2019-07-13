use core::alloc::AllocErr;
use core::convert::TryFrom;
use core::result::Result;
use core::num::TryFromIntError;

use crate::erts::term::{atom_unchecked, Term, TypedTerm, TypeError, BoolError, TryIntoIntegerError};
use crate::erts::term::atom::{AtomError, EncodingError};
use crate::erts::term::tuple::IndexError;
use crate::erts::term::list::ImproperList;
use crate::erts::HeapAlloc;

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
#[cfg_attr(debug_assertions, derive(Debug))]
pub enum Class {
    Error { arguments: Option<Term> },
    Exit,
    Throw,
}

impl TryFrom<Term> for Class {
    type Error = Exception;

    fn try_from(term: Term) -> Result<Class, Exception> {
        use self::Class::*;

        match term.to_typed_term().unwrap() {
            TypedTerm::Atom(atom) => match atom.name() {
                "error" => Ok(Error { arguments: None }),
                "exit" => Ok(Exit),
                "throw" => Ok(Throw),
                _ => Err(badarg!()),
            },
            _ => Err(badarg!()),
        }
    }
}

#[cfg_attr(debug_assertions, derive(Debug))]
#[cfg_attr(test, derive(Clone))]
pub struct Exception {
    pub class: Class,
    pub reason: Term,
    pub stacktrace: Option<Term>,
    #[cfg(debug_assertions)]
    pub file: &'static str,
    #[cfg(debug_assertions)]
    pub line: u32,
    #[cfg(debug_assertions)]
    pub column: u32,
}

impl Exception {
    pub fn badarg(
        #[cfg(debug_assertions)]
        file: &'static str,
        #[cfg(debug_assertions)]
        line: u32,
        #[cfg(debug_assertions)]
        column: u32
    ) -> Self {
        Self::error(Self::badarg_reason(), None, None, file, line, column)
    }

    #[cfg(debug_assertions)]
    pub fn badarith(file: &'static str, line: u32, column: u32) -> Self {
        Self::error(Self::badarith_reason(), None, None, file, line, column)
    }

    #[cfg(not(debug_assertions))]
    pub fn badarith() -> Self {
        Self::error(Self::badarith_reason(), None, None)
    }

    #[cfg(debug_assertions)]
    pub fn badarity<A: HeapAlloc>(
        heap: &mut A,
        function: Term,
        arguments: Term,
        file: &'static str,
        line: u32,
        column: u32,
    ) -> Result<Self, AllocErr> {
        let reason = Self::badarity_reason(heap, function, arguments)?;
        let error = Self::error(reason, None, None, file, line, column);

        Ok(error)
    }

    #[cfg(not(debug_assertions))]
    pub fn badarity<A: HeapAlloc>(heap: &mut A, function: Term, arguments: Term) -> Self {
        let reason = Self::badarity_reason(process_control_block, function, arguments)?;
        let error = Self::error(reason, None, None);

        Ok(error)
    }

    pub fn badkey<A: HeapAlloc>(
        heap: &mut A,
        key: Term,
        #[cfg(debug_assertions)]
        file: &'static str,
        #[cfg(debug_assertions)]
        line: u32,
        #[cfg(debug_assertions)]
        column: u32,
    ) -> Result<Self, AllocErr>{
        let tag = atom_unchecked("badkey");
        let reason = heap.tuple_from_slice(&[tag, key])?;
        let error = Self::error(reason,
                                None,
                                None,
                                #[cfg(debug_assertions)]
                                file,
                                #[cfg(debug_assertions)]
                                line,
                                #[cfg(debug_assertions)]
                                column,
        );

        Ok(error)
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
    ) -> Result<Self, AllocErr>{
        let tag = atom_unchecked("badmap");
        let reason = heap.tuple_from_slice(&[tag, map])?;
        let error = Self::error(reason,
                                None,
                                None,
                                #[cfg(debug_assertions)]
                                file,
                                #[cfg(debug_assertions)]
                                line,
                                #[cfg(debug_assertions)]
                                column,
        );

        Ok(error)
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
    ) -> Result<Self, AllocErr> {
        let reason = Self::undef_reason();
        let stacktrace =
            Self::undef_stacktrace(heap, module, function, arguments, stacktrace_tail)?;
        let exit = Self::exit(
            reason,
            Some(stacktrace),
            #[cfg(debug_assertions)]
            file,
            #[cfg(debug_assertions)]
            line,
            #[cfg(debug_assertions)]
            column
        );

        Ok(exit)
    }

    // Private

    fn badarg_reason() -> Term {
        atom_unchecked("badarg")
    }

    fn badarith_reason() -> Term {
        atom_unchecked("badarith")
    }

    fn badarity_reason<A: HeapAlloc>(
        heap: &mut A,
        function: Term,
        arguments: Term,
    ) -> Result<Term, AllocErr> {
        let function_arguments = heap.tuple_from_slice(&[function, arguments])?;

        heap.tuple_from_slice(&[Self::badarity_tag(), function_arguments])
    }

    fn badarity_tag() -> Term {
        atom_unchecked("badarity")
    }

    fn error(
        reason: Term,
        arguments: Option<Term>,
        stacktrace: Option<Term>,
        #[cfg(debug_assertions)]
        file: &'static str,
        #[cfg(debug_assertions)]
        line: u32,
        #[cfg(debug_assertions)]
        column: u32,
    ) -> Self {
        let class = Class::Error { arguments };
        Self::new(class,
                  reason,
                  stacktrace,
                  #[cfg(debug_assertions)]
                  file,
                  #[cfg(debug_assertions)]
                  line,
                  #[cfg(debug_assertions)]
                  column,
        )
    }

    #[cfg(debug_assertions)]
    fn exit(
        reason: Term,
        stacktrace: Option<Term>,
        file: &'static str,
        line: u32,
        column: u32,
    ) -> Self {
        let class = Class::Exit;
        Self::new(class, reason, stacktrace, file, line, column)
    }

    #[cfg(not(debug_assertions))]
    fn exit(reason: Term, stacktrace: Option<Term>) -> Self {
        let class = Class::Exit;
        Self::new(class, reason, stacktrace)
    }

    #[cfg(debug_assertions)]
    fn new(
        class: Class,
        reason: Term,
        stacktrace: Option<Term>,
        file: &'static str,
        line: u32,
        column: u32,
    ) -> Self {
        Exception {
            class,
            reason,
            stacktrace,
            file,
            line,
            column,
        }
    }

    #[cfg(not(debug_assertions))]
    fn new(class: Class, reason: Term, stacktrace: Option<Term>) -> Self {
        Exception {
            class,
            reason,
            stacktrace,
        }
    }

    fn undef_reason() -> Term {
        atom_unchecked("undef")
    }

    fn undef_stacktrace<A: HeapAlloc>(
        heap: &mut A,
        module: Term,
        function: Term,
        arguments: Term,
        tail: Term,
    ) -> Result<Term, AllocErr> {
        let top = heap.tuple_from_slice(
            &[
                module,
                function,
                arguments,
                // I'm not sure what this final empty list holds
                Term::NIL,
            ],
        )?;

        heap.cons(top, tail)
    }
}

impl Eq for Exception {}

impl From<AtomError> for Exception {
    fn from(_: AtomError) -> Self {
        badarg!()
    }
}

impl From<BoolError> for Exception {
    fn from(_: BoolError) -> Self {
        badarg!()
    }
}

impl From<EncodingError> for Exception {
    fn from(_: EncodingError) -> Self {
      badarg!()
    }
}

impl From<ImproperList> for Exception {
    fn from(_: ImproperList) -> Self {
        badarg!()
    }
}

impl From<IndexError> for Exception {
    fn from(_: IndexError) -> Self {
        badarg!()
    }
}

impl From<TryFromIntError> for Exception {
    fn from(_: TryFromIntError) -> Self {
        badarg!()
    }
}

impl From<TryIntoIntegerError> for Exception {
    fn from(_: TryIntoIntegerError) -> Self {
        badarg!()
    }
}

impl From<TypeError> for Exception {
    fn from(_: TypeError) -> Self {
        badarg!()
    }
}

impl PartialEq for Exception {
    /// `file`, `line`, and `column` don't count for equality as they are for `Debug` only to help
    /// track down exceptions.
    fn eq(&self, other: &Exception) -> bool {
        (self.class == other.class)
            & (self.reason == other.reason)
            & (self.stacktrace == other.stacktrace)
    }
}

impl TryFrom<super::Exception> for Exception {
    type Error = TypeError;

    fn try_from(exception: super::Exception) -> Result<Self, Self::Error> {
        match exception {
            super::Exception::Runtime(runtime_exception) => Ok(runtime_exception),
            _ => Err(TypeError)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod error {
        use super::Class::*;
        use super::*;

        use liblumen_alloc::atom_unchecked;

        #[test]
        fn without_arguments_stores_none() {
            let reason = atom_unchecked("badarg");

            let error = error!(reason);

            assert_eq!(error.reason, reason);
            assert_eq!(error.class, Error { arguments: None });
        }

        #[test]
        fn without_arguments_stores_some() {
            let reason = atom_unchecked("badarg");
            let arguments = Term::NIL;

            let error = error!(reason, Some(arguments));

            assert_eq!(error.reason, reason);
            assert_eq!(
                error.class,
                Error {
                    arguments: Some(arguments)
                }
            );
        }
    }
}
