use super::commons;
use super::Pattern;

pub type Tuple = commons::Tuple<Pattern>;
pub type Map = commons::Map<Pattern>;
pub type Record = commons::Record<Pattern>;
pub type RecordFieldIndex = commons::RecordFieldIndex;
pub type List = commons::List<Pattern>;
pub type Bits = commons::Bits<Pattern>;
pub type Parenthesized = commons::Parenthesized<Pattern>;
pub type UnaryOpCall = commons::UnaryOpCall<Pattern>;
pub type BinaryOpCall = commons::BinaryOpCall<Pattern>;
pub type Match = commons::Match<Pattern>;
