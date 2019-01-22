use std::convert::From;
use std::fmt::{self, Display};

use failure::Fail;

use crate::beam::reader::ReadError;
use crate::serialization::etf;

#[derive(Debug)]
pub struct UnmatchedTerms(Vec<Unmatched>);
impl Display for UnmatchedTerms {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "[")?;
        let limit = 3;
        for (i, e) in self.0.iter().take(limit).enumerate() {
            if i != 0 {
                write!(f, ",")?;
            }
            write!(f, "{}", e)?;
        }
        if self.0.len() > limit {
            write!(f, " ..{}..", self.0.len() - limit)?;
        }
        write!(f, "]")?;

        Ok(())
    }
}

#[derive(Fail, Debug)]
pub enum FromBeamError {
    #[fail(display = "failed to load beam: {}", _0)]
    IO(#[fail(cause)] std::io::Error),

    #[fail(display = "invalid beam file: {}", _0)]
    BeamFile(#[fail(cause)] ReadError),

    #[fail(display = "unable to decode term: {}", _0)]
    TermDecode(#[fail(cause)] etf::DecodeError),

    #[fail(display = "debug info is required but not present")]
    NoDebugInfo,

    #[fail(display = "missing module attribute")]
    NoModuleAttribute,

    #[fail(display = "unexpected term: {}", _0)]
    UnexpectedTerm(UnmatchedTerms),
}
impl From<std::io::Error> for FromBeamError {
    fn from(x: std::io::Error) -> Self {
        FromBeamError::IO(x)
    }
}
impl From<ReadError> for FromBeamError {
    fn from(x: ReadError) -> Self {
        FromBeamError::BeamFile(x)
    }
}
impl From<etf::DecodeError> for FromBeamError {
    fn from(x: etf::DecodeError) -> Self {
        FromBeamError::TermDecode(x)
    }
}
impl<'a> From<etf::pattern::Unmatch<'a>> for FromBeamError {
    fn from(x: etf::pattern::Unmatch<'a>) -> Self {
        use std::ops::Deref;
        let mut trace = Vec::new();
        let mut curr = Some(&x);
        while let Some(x) = curr {
            trace.push(Unmatched {
                value: x.input.clone(),
                pattern: format!("{:?}", x.pattern),
            });
            curr = x.cause.as_ref().map(|x| x.deref());
        }
        trace.reverse();
        FromBeamError::UnexpectedTerm(UnmatchedTerms(trace))
    }
}

#[derive(Debug)]
pub struct Unmatched {
    pub value: etf::Term,
    pub pattern: String,
}
impl Display for Unmatched {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{{pattern:{}, value:{}}}", self.pattern, self.value)
    }
}
