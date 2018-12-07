use crate::beam::reader::ReadError;
use crate::serialization::etf;

#[derive(Debug)]
pub enum FromBeamError {
    Io(std::io::Error),
    BeamFile(ReadError),
    TermDecode(etf::DecodeError),
    NoDebugInfo,
    NoModuleAttribute,
    UnexpectedTerm(Vec<Unmatched>),
}
impl std::error::Error for FromBeamError {
    fn description(&self) -> &str {
        use self::FromBeamError::*;

        match *self {
            Io(ref x) => x.description(),
            BeamFile(ref x) => x.description(),
            TermDecode(ref x) => x.description(),
            NoDebugInfo => "No debug information",
            NoModuleAttribute => "No module attribute",
            UnexpectedTerm(_) => "Unexpected term",
        }
    }
    fn cause(&self) -> Option<&std::error::Error> {
        use self::FromBeamError::*;

        match *self {
            Io(ref x) => x.cause(),
            BeamFile(ref x) => x.cause(),
            TermDecode(ref x) => x.cause(),
            _ => None,
        }
    }
}
impl std::fmt::Display for FromBeamError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        use self::FromBeamError::*;
        match *self {
            Io(ref x) => x.fmt(f),
            BeamFile(ref x) => x.fmt(f),
            TermDecode(ref x) => x.fmt(f),
            NoDebugInfo => write!(f, "The beam has no debug information"),
            NoModuleAttribute => write!(f, "No module attribute"),
            UnexpectedTerm(ref trace) => {
                write!(f, "Unexpected term: [")?;
                let limit = 3;
                for (i, e) in trace.iter().take(limit).enumerate() {
                    if i != 0 {
                        write!(f, ",")?;
                    }
                    write!(f, "{}", e)?;
                }
                if trace.len() > limit {
                    write!(f, " ..{}..", trace.len() - limit)?;
                }
                write!(f, "]")?;
                Ok(())
            }
        }
    }
}
impl std::convert::From<std::io::Error> for FromBeamError {
    fn from(x: std::io::Error) -> Self {
        FromBeamError::Io(x)
    }
}
impl std::convert::From<crate::beam::reader::ReadError> for FromBeamError {
    fn from(x: crate::beam::reader::ReadError) -> Self {
        FromBeamError::BeamFile(x)
    }
}
impl std::convert::From<etf::DecodeError> for FromBeamError {
    fn from(x: etf::DecodeError) -> Self {
        FromBeamError::TermDecode(x)
    }
}
impl<'a> std::convert::From<etf::pattern::Unmatch<'a>> for FromBeamError {
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
        FromBeamError::UnexpectedTerm(trace)
    }
}

#[derive(Debug)]
pub struct Unmatched {
    pub value: etf::Term,
    pub pattern: String,
}
impl std::fmt::Display for Unmatched {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{{pattern:{}, value:{}}}", self.pattern, self.value)
    }
}
