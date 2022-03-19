/// A result of parsing. Can have one of three states:
/// - Fail: Error only
/// - Warn: Error and result
/// - Ok: Result only
pub enum ParseResult<R, E> {
    Fail(E),
    Warn(E, R),
    Ok(R),
}

pub trait Print {
    fn print(&self);
}

impl<R, E> ParseResult<R, E> {
    pub fn unwrap(self) -> R {
        match self {
            ParseResult::Fail(_err) => {
                panic!();
            }
            ParseResult::Warn(_err, res) => res,
            ParseResult::Ok(res) => res,
        }
    }

    pub fn unwrap_print(self) -> R
    where
        E: Print,
    {
        match self {
            ParseResult::Fail(err) => {
                err.print();
                panic!();
            }
            ParseResult::Warn(err, res) => {
                err.print();
                res
            }
            ParseResult::Ok(res) => res,
        }
    }
}
