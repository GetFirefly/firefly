use liblumen_diagnostics::{Diagnostic, SourceIndex, SourceSpan, ToDiagnostic};
use liblumen_intern::Ident;

use super::escape_stm as escape;

#[derive(Debug, thiserror::Error)]
pub enum StringTokenizeError {
    #[error("invalid string escape")]
    InvalidStringEscape {
        span: SourceSpan,
        source: escape::EscapeStmError<SourceIndex>,
    },
}
impl ToDiagnostic for StringTokenizeError {
    fn to_diagnostic(&self) -> Diagnostic {
        match self {
            StringTokenizeError::InvalidStringEscape { source, .. } =>
                source.to_diagnostic(),
        }
    }
}

pub struct StringTokenizer {
    ident: Ident,
    chars: std::str::Chars<'static>,
    byte_idx: usize,
    stm: escape::EscapeStm<SourceIndex>,
    again: Option<(Option<char>, SourceIndex)>,
    finished: bool,
}

impl StringTokenizer {
    pub fn new(string: Ident) -> Self {
        StringTokenizer {
            ident: string,
            chars: string.name.as_str().get().chars(),
            byte_idx: 0,
            stm: escape::EscapeStm::new(),
            again: None,
            finished: false,
        }
    }
}

impl Iterator for StringTokenizer {
    type Item = Result<(u64, SourceSpan), StringTokenizeError>;
    fn next(&mut self) -> Option<Self::Item> {
        use escape::EscapeStmAction;

        if self.finished {
            return None;
        }

        loop {
            if let Some((again_chr, idx)) = self.again.take() {
                let res = self.stm.transition(again_chr, idx);

                let out = match res {
                    Ok((EscapeStmAction::Next, out)) => {
                        match again_chr {
                            Some(c) => self.byte_idx += c.len_utf8(),
                            // If this is None, it means we have reached the
                            // end of the string.
                            None => self.finished = true,
                        }
                        out
                    },
                    Ok((EscapeStmAction::Again, out)) => {
                        self.again = Some((again_chr, idx));
                        out
                    },
                    Err(err) => {
                        self.finished = true;
                        return Some(Err(StringTokenizeError::InvalidStringEscape {
                            span: err.span(),
                            source: err,
                        }));
                    },
                };

                if let Some(result) = out {
                    // If we have a result, we return that
                    let span = SourceSpan::new(result.range.0, result.range.1);
                    return Some(Ok((result.cp, span)));
                } else if self.finished {
                    // If there is no result, and the iterator has been marked
                    // as finished, we return nothing.
                    return None;
                } else {
                    // Otherwise, the state machine needs more data.
                    continue;
                }
            }

            let idx = self.ident.span.start() + self.byte_idx;
            let chr = self.chars.next();
            self.again = Some((chr, idx));
        }
    }
}

#[cfg(test)]
mod tests {
    use liblumen_intern::Ident;
    use liblumen_diagnostics::SourceSpan;

    use super::*;

    #[test]
    fn tokenize_plaintext() {
        let mut tokenizer = StringTokenizer::new(Ident::from_str("abc"));
        assert_eq!(tokenizer.next().unwrap().unwrap(), ('a' as u64, SourceSpan::UNKNOWN));
        assert_eq!(tokenizer.next().unwrap().unwrap(), ('b' as u64, SourceSpan::UNKNOWN));
        assert_eq!(tokenizer.next().unwrap().unwrap(), ('c' as u64, SourceSpan::UNKNOWN));
        assert!(tokenizer.next().is_none());
    }

    #[test]
    fn tokenize_unicode_codepoint() {
        let mut tokenizer = StringTokenizer::new(Ident::from_str("\\x{afaf}"));
        assert_eq!(tokenizer.next().unwrap().unwrap(), (0xafaf, SourceSpan::UNKNOWN));
        assert!(tokenizer.next().is_none());
    }

}
