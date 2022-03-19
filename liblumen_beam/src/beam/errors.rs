use std::fmt;

use thiserror::Error;

use crate::beam;
use crate::serialization::etf;

#[derive(Error, Debug)]
pub enum FromBeamError {
    #[error("failed to load beam: {0}")]
    IO(#[from] std::io::Error),

    #[error("invalid beam file: {0}")]
    BeamFile(#[from] beam::reader::ReadError),

    #[error("unable to decode term: {0}")]
    TermDecode(#[from] etf::DecodeError),

    #[error("debug info is required but not present")]
    NoDebugInfo,

    #[error("missing module attribute")]
    NoModuleAttribute,

    #[error("unexpected term: {0}")]
    UnexpectedTerm(UnmatchedTerms),
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
        Self::UnexpectedTerm(UnmatchedTerms(trace))
    }
}

#[derive(Debug)]
pub struct Unmatched {
    pub value: etf::Term,
    pub pattern: String,
}
impl fmt::Display for Unmatched {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{{pattern:{}, value:{}}}", self.pattern, self.value)
    }
}

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
