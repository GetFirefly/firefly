use crate::syntax::tokenizer::tokens::AtomToken;

use super::commons;
use super::exprs;
use super::GuardTest;

pub type Tuple = commons::Tuple<GuardTest>;
pub type Map = commons::Map<GuardTest>;
pub type Record = commons::Record<GuardTest>;
pub type RecordFieldIndex = commons::RecordFieldIndex;
pub type RecordFieldAccess = exprs::RecordFieldAccess<GuardTest>;
pub type List = commons::List<GuardTest>;
pub type Bits = commons::Bits<GuardTest>;
pub type Parenthesized = commons::Parenthesized<GuardTest>;
pub type FunCall = commons::Call<AtomToken, GuardTest>;
pub type UnaryOpCall = commons::UnaryOpCall<GuardTest>;
pub type BinaryOpCall = commons::BinaryOpCall<GuardTest>;
