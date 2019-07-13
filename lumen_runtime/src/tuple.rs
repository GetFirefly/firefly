use core::convert::{TryFrom, TryInto};

use liblumen_alloc::erts::exception::runtime::Exception;
use liblumen_alloc::{badarg, Term};

#[repr(transparent)]
pub(crate) struct ZeroBasedIndex(pub usize);

impl TryFrom<Term> for ZeroBasedIndex {
    type Error = Exception;

    fn try_from(term: Term) -> Result<ZeroBasedIndex, Exception> {
        let OneBasedIndex(one_based_index) = term.try_into()?;

        Ok(ZeroBasedIndex(one_based_index - 1))
    }
}

#[repr(transparent)]
pub(crate) struct OneBasedIndex(usize);

impl TryFrom<Term> for OneBasedIndex {
    type Error = Exception;

    fn try_from(term: Term) -> Result<OneBasedIndex, Exception> {
        let one_based_index_usize: usize = term.try_into()?;

        if 1 <= one_based_index_usize {
            Ok(OneBasedIndex(one_based_index_usize))
        } else {
            Err(badarg!())
        }
    }
}
