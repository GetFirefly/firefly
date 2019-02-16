#![cfg_attr(not(test), allow(dead_code))]

use std::cmp::Ordering;
use std::convert::{TryFrom, TryInto};
use std::fmt::{self, Debug, Display};

use liblumen_arena::TypedArena;

use crate::atom;
use crate::list::Cons;
use crate::process::{DebugInProcess, IntoProcess, OrderInProcess, Process};
use crate::tuple::{Element, Tuple};

impl From<&Term> for atom::Index {
    fn from(term: &Term) -> atom::Index {
        assert_eq!(term.tag(), Tag::Atom);

        atom::Index(term.tagged >> Tag::ATOM_BIT_COUNT)
    }
}

#[derive(Debug, PartialEq)]
// MUST be `repr(u*)` so that size and layout is fixed for direct LLVM IR checking of tags
#[repr(usize)]
pub enum Tag {
    Arity = 0b0000_00,
    BinaryAggregate = 0b0001_00,
    PositiveBigNumber = 0b0010_00,
    NegativeBigNumber = 0b0011_00,
    Reference = 0b0100_00,
    Function = 0b0101_00,
    Float = 0b0110_00,
    Export = 0b0111_00,
    ReferenceCountedBinary = 0b1000_00,
    HeapBinary = 0b1001_00,
    Subbinary = 0b1010_00,
    ExternalPid = 0b1100_00,
    ExternalPort = 0b1101_00,
    ExternalReference = 0b1110_00,
    Map = 0b1111_00,
    List = 0b01,
    Boxed = 0b10,
    LocalPid = 0b00_11,
    LocalPort = 0b01_11,
    Atom = 0b00_10_11,
    CatchPointer = 0b01_10_11,
    EmptyList = 0b11_10_11,
    SmallInteger = 0b11_11,
}

impl Tag {
    const PRIMARY_MASK: usize = 0b11;
    const BOXED_MASK: usize = Self::PRIMARY_MASK;
    const LIST_MASK: usize = Self::PRIMARY_MASK;
    const ATOM_BIT_COUNT: u8 = 6;
    const ARITY_BIT_COUNT: u8 = 6;
    const SMALL_INTEGER_BIT_COUNT: u8 = 4;
}

pub struct TagError {
    tag: usize,
    bit_count: usize,
}

impl Display for TagError {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(
            formatter,
            "{tag:0bit_count$b} is not a valid Term tag",
            tag = self.tag,
            bit_count = self.bit_count
        )
    }
}

const HEADER_PRIMARY_TAG: usize = 0b00;
const HEADER_PRIMARY_TAG_MASK: usize = 0b1111_11;
const IMMEDIATE_PRIMARY_TAG_MASK: usize = 0b11_11;
const IMMEDIATE_IMMEDIATE_PRIMARY_TAG_MASK: usize = 0b11_11_11;

impl TryFrom<usize> for Tag {
    type Error = TagError;

    fn try_from(bits: usize) -> Result<Self, Self::Error> {
        match bits & Tag::PRIMARY_MASK {
            HEADER_PRIMARY_TAG => match bits & HEADER_PRIMARY_TAG_MASK {
                0b0000_00 => Ok(Tag::Arity),
                0b0001_00 => Ok(Tag::BinaryAggregate),
                0b0010_00 => Ok(Tag::PositiveBigNumber),
                0b0011_00 => Ok(Tag::NegativeBigNumber),
                0b0100_00 => Ok(Tag::Reference),
                0b0101_00 => Ok(Tag::Function),
                0b0110_00 => Ok(Tag::Float),
                0b0111_00 => Ok(Tag::Export),
                0b1000_00 => Ok(Tag::ReferenceCountedBinary),
                0b1001_00 => Ok(Tag::HeapBinary),
                0b1010_00 => Ok(Tag::Subbinary),
                0b1100_00 => Ok(Tag::ExternalPid),
                0b1101_00 => Ok(Tag::ExternalPort),
                0b1110_00 => Ok(Tag::ExternalReference),
                0b1111_00 => Ok(Tag::Map),
                tag => Err(TagError { tag, bit_count: 6 }),
            },
            0b01 => Ok(Tag::List),
            0b10 => Ok(Tag::Boxed),
            0b11 => match bits & IMMEDIATE_PRIMARY_TAG_MASK {
                0b00_11 => Ok(Tag::LocalPid),
                0b01_11 => Ok(Tag::LocalPort),
                0b10_11 => match bits & IMMEDIATE_IMMEDIATE_PRIMARY_TAG_MASK {
                    0b00_10_11 => Ok(Tag::Atom),
                    0b01_10_11 => Ok(Tag::CatchPointer),
                    0b11_10_11 => Ok(Tag::EmptyList),
                    tag => Err(TagError { tag, bit_count: 6 }),
                },
                0b11_11 => Ok(Tag::SmallInteger),
                tag => Err(TagError { tag, bit_count: 4 }),
            },
            tag => Err(TagError { tag, bit_count: 2 }),
        }
    }
}

#[derive(Clone, Copy)]
// MUST be `repr(C)` so that size and layout is fixed for direct LLVM IR checking of tags
#[repr(C)]
pub struct Term {
    pub tagged: usize,
}

#[derive(PartialEq)]
pub struct BadArgument;

impl Debug for BadArgument {
    fn fmt(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
        write!(formatter, "bad argument")
    }
}

impl Term {
    const MAX_ARITY: usize = std::usize::MAX >> Tag::ARITY_BIT_COUNT;

    pub const EMPTY_LIST: Term = Term {
        tagged: Tag::EmptyList as usize,
    };

    pub fn arity(arity: usize) -> Term {
        if Term::MAX_ARITY < arity {
            panic!(
                "Arity ({}) exceeds max arity ({}) that can fit in a Term",
                arity,
                Term::MAX_ARITY
            );
        }

        Term {
            tagged: (arity << Tag::ARITY_BIT_COUNT) | Tag::Arity as usize,
        }
    }

    pub fn arity_to_integer(&self) -> Term {
        const TAG_ARITY: usize = Tag::Arity as usize;

        assert_eq!(
            self.tagged & TAG_ARITY,
            TAG_ARITY,
            "Term ({:#b}) is not a tuple arity",
            self.tagged
        );

        // Tag::ARITY_BIT_COUNT > Tag::SMALL_INTEGER_BIT_COUNT, so any arity MUST fit into a term
        // and not need `into_process` for allocation.
        ((self.tagged & !(TAG_ARITY)) >> Tag::ARITY_BIT_COUNT).into()
    }

    pub fn alloc_slice(slice: &[Term], term_arena: &mut TypedArena<Term>) -> *const Term {
        term_arena.alloc_slice(slice).as_ptr()
    }

    pub fn cons(head: Term, tail: Term, process: &mut Process) -> Term {
        let pointer_bits = process.cons(head, tail) as usize;

        assert_eq!(
            pointer_bits & Tag::LIST_MASK,
            0,
            "List tag bit ({:#b}) would overwrite pointer bits ({:#b})",
            Tag::LIST_MASK,
            pointer_bits
        );

        Term {
            tagged: pointer_bits | (Tag::List as usize),
        }
    }

    pub fn tag(&self) -> Tag {
        match (self.tagged as usize).try_into() {
            Ok(tag) => tag,
            Err(tag_error) => panic!(tag_error),
        }
    }

    pub fn abs(&self) -> Result<Term, BadArgument> {
        match self.tag() {
            Tag::SmallInteger => {
                if unsafe { self.small_integer_is_negative() } {
                    // cast first so that sign bit is extended on shift
                    let signed = (self.tagged as isize) >> SMALL_INTEGER_TAG_BIT_COUNT;
                    let positive = -signed;
                    Ok(Term {
                        tagged: ((positive << SMALL_INTEGER_TAG_BIT_COUNT) as usize)
                            | (Tag::SmallInteger as usize),
                    })
                } else {
                    Ok(Term {
                        tagged: self.tagged,
                    })
                }
            }
            _ => Err(BadArgument),
        }
    }

    pub fn head(&self) -> Result<Term, BadArgument> {
        match self.tag() {
            Tag::List => {
                let cons: &Cons = (*self).into();
                Ok(cons.head())
            }
            _ => Err(BadArgument),
        }
    }

    pub fn tail(&self) -> Result<Term, BadArgument> {
        match self.tag() {
            Tag::List => {
                let cons: &Cons = (*self).into();
                Ok(cons.tail())
            }
            _ => Err(BadArgument),
        }
    }

    pub fn is_atom(&self, mut process: &mut Process) -> Term {
        (self.tag() == Tag::Atom).into_process(&mut process)
    }

    pub fn is_empty_list(&self, mut process: &mut Process) -> Term {
        (self.tag() == Tag::EmptyList).into_process(&mut process)
    }

    pub fn is_integer(&self, mut process: &mut Process) -> Term {
        match self.tag() {
            Tag::SmallInteger => true,
            _ => false,
        }
        .into_process(&mut process)
    }

    pub fn is_list(&self, mut process: &mut Process) -> Term {
        match self.tag() {
            Tag::EmptyList | Tag::List => true,
            _ => false,
        }
        .into_process(&mut process)
    }

    pub fn is_tuple(&self, mut process: &mut Process) -> Term {
        (self.tag() == Tag::Boxed && self.unbox().tag() == Tag::Arity).into_process(&mut process)
    }

    pub fn length(&self, mut process: &mut Process) -> Result<Term, BadArgument> {
        let mut length: usize = 0;
        let mut tail = *self;

        loop {
            match tail.tag() {
                Tag::EmptyList => break Ok(length.into_process(&mut process)),
                Tag::List => {
                    tail = tail.tail().unwrap();
                    length += 1;
                }
                _ => break Err(BadArgument),
            }
        }
    }

    pub fn size(&self) -> Result<Term, BadArgument> {
        Ok(<&Tuple>::try_from(self)?.size())
    }

    pub fn slice_to_tuple(slice: &[Term], process: &mut Process) -> Term {
        process.slice_to_tuple(slice).into()
    }

    fn unbox(&self) -> &Term {
        match self.tag() {
            Tag::Boxed => {
                let pointer = (self.tagged & !(Tag::Boxed as usize)) as *const Term;
                unsafe { pointer.as_ref() }.unwrap()
            }
            tag => panic!(
                "Tagged ({:?}) term ({:#b}) cannot be unboxed",
                tag, self.tagged
            ),
        }
    }

    const SMALL_INTEGER_SIGN_BIT_MASK: usize = std::isize::MIN as usize;

    /// Only call if verified `tag` is `Tag::SmallInteger`.
    unsafe fn small_integer_is_negative(&self) -> bool {
        self.tagged & Term::SMALL_INTEGER_SIGN_BIT_MASK == Term::SMALL_INTEGER_SIGN_BIT_MASK
    }
}

impl DebugInProcess for Term {
    fn format_in_process(&self, process: &Process) -> String {
        match self.tag() {
            Tag::Arity => format!("Term::arity({})", usize::from(Term::arity_to_integer(self))),
            Tag::Boxed => {
                let unboxed = self.unbox();

                match unboxed.tag() {
                    Tag::Arity => {
                        let tuple: &Tuple = unboxed.try_into().unwrap();

                        let mut strings: Vec<String> = Vec::new();

                        strings.push("Term::slice_to_tuple(&[".to_string());

                        let mut iter = tuple.iter();

                        if let Some(first_element) = iter.next() {
                            strings.push(first_element.format_in_process(process));

                            for element in iter {
                                strings.push(", ".to_string());
                                strings.push(element.format_in_process(process));
                            }
                        }

                        strings.push("], &mut process)".to_string());

                        strings.join("")
                    }
                    _ => unimplemented!(),
                }
            }
            Tag::EmptyList => "Term::EMPTY_LIST".to_string(),
            Tag::List => {
                let cons: &Cons = self.try_into().unwrap();
                format!(
                    "Term::cons({}, {}, &mut process)",
                    cons.head().format_in_process(&process),
                    cons.tail().format_in_process(&process)
                )
            }
            Tag::SmallInteger => format!("{:?}.into()", isize::from(self)),
            _ => format!(
                "Term {{ tagged: 0b{tagged:0bit_count$b} }}",
                tagged = self.tagged,
                bit_count = std::mem::size_of::<usize>() * 8
            ),
        }
    }
}

trait DeleteElement<T> {
    fn delete_element(&self, index: T, process: &mut Process) -> Result<Term, BadArgument>;
}

impl DeleteElement<Term> for Term {
    fn delete_element(&self, index: Term, mut process: &mut Process) -> Result<Term, BadArgument> {
        DeleteElement::delete_element(<&Tuple>::try_from(self)?, index, &mut process)
    }
}

impl DeleteElement<Term> for Tuple {
    fn delete_element(&self, index: Term, process: &mut Process) -> Result<Term, BadArgument> {
        match index.tag() {
            Tag::SmallInteger => self
                .delete_element(usize::from(index), &mut process.term_arena)
                .map(|tuple| tuple.into()),
            _ => Err(BadArgument),
        }
    }
}

impl Element<Term> for Term {
    fn element(&self, index: Term) -> Result<Term, BadArgument> {
        <&Tuple>::try_from(self)?.element(index)
    }
}

impl Element<Term> for Tuple {
    fn element(&self, index: Term) -> Result<Term, BadArgument> {
        match index.tag() {
            Tag::SmallInteger => self.element(usize::from(index)),
            _ => Err(BadArgument),
        }
    }
}

impl From<Term> for *const Cons {
    fn from(term: Term) -> Self {
        (term.tagged & !(Tag::List as usize)) as *const Cons
    }
}

impl From<Term> for &Cons {
    fn from(term: Term) -> Self {
        let pointer: *const Cons = term.into();
        unsafe { &*pointer }
    }
}

impl From<&Term> for isize {
    fn from(term: &Term) -> Self {
        match term.tag() {
            Tag::SmallInteger => (term.tagged as isize) >> SMALL_INTEGER_TAG_BIT_COUNT,
            tag => panic!(
                "{:?} tagged term {:#b} cannot be converted to isize",
                tag, term.tagged
            ),
        }
    }
}

impl From<&Tuple> for Term {
    fn from(tuple: &Tuple) -> Self {
        let pointer_bits = tuple as *const Tuple as usize;

        assert_eq!(
            pointer_bits & Tag::BOXED_MASK,
            0,
            "Boxed tag bit ({:#b}) would overwrite pointer bits ({:#b})",
            Tag::BOXED_MASK,
            pointer_bits
        );

        Term {
            tagged: pointer_bits | (Tag::Boxed as usize),
        }
    }
}

const SMALL_INTEGER_TAG_BIT_COUNT: u8 = 4;
const MIN_SMALL_INTEGER: isize = std::isize::MIN >> SMALL_INTEGER_TAG_BIT_COUNT;
const MAX_SMALL_INTEGER: isize = std::isize::MAX >> SMALL_INTEGER_TAG_BIT_COUNT;

impl From<Term> for usize {
    fn from(term: Term) -> Self {
        match term.tag() {
            Tag::Arity => term.tagged >> Tag::ARITY_BIT_COUNT,
            Tag::SmallInteger => {
                let term_isize = (term.tagged as isize) >> Tag::SMALL_INTEGER_BIT_COUNT;

                if 0 <= term_isize {
                    term_isize as usize
                } else {
                    panic!("Term ({:#b}) contains negative small integer ({}) that can't be converted to usize", term.tagged, term_isize);
                }
            }
            _ => unimplemented!(),
        }
    }
}

impl From<usize> for Term {
    fn from(u: usize) -> Self {
        if u <= (MAX_SMALL_INTEGER as usize) {
            Term {
                tagged: (u << SMALL_INTEGER_TAG_BIT_COUNT) | (Tag::SmallInteger as usize),
            }
        } else {
            panic!(
                "usize ({}) is greater than max small integer ({})",
                u, MAX_SMALL_INTEGER
            );
        }
    }
}

trait InsertElement<T> {
    fn insert_element(
        &self,
        index: T,
        element: Term,
        process: &mut Process,
    ) -> Result<Term, BadArgument>;
}

impl InsertElement<Term> for Term {
    fn insert_element(
        &self,
        index: Term,
        element: Term,
        mut process: &mut Process,
    ) -> Result<Term, BadArgument> {
        InsertElement::insert_element(<&Tuple>::try_from(self)?, index, element, &mut process)
    }
}

impl InsertElement<Term> for Tuple {
    fn insert_element(
        &self,
        index: Term,
        element: Term,
        process: &mut Process,
    ) -> Result<Term, BadArgument> {
        match index.tag() {
            Tag::SmallInteger => self
                .insert_element(usize::from(index), element, &mut process.term_arena)
                .map(|tuple| tuple.into()),
            _ => Err(BadArgument),
        }
    }
}

impl IntoProcess<Term> for isize {
    fn into_process(self: Self, _process: &mut Process) -> Term {
        if MIN_SMALL_INTEGER <= self && self <= MAX_SMALL_INTEGER {
            Term {
                tagged: ((self as usize) << SMALL_INTEGER_TAG_BIT_COUNT)
                    | (Tag::SmallInteger as usize),
            }
        } else {
            panic!("isize ({}) is not between the min small integer ({}) and max small integer ({}), inclusive", self, MIN_SMALL_INTEGER, MAX_SMALL_INTEGER);
        }
    }
}

impl IntoProcess<Term> for usize {
    fn into_process(self: Self, _process: &mut Process) -> Term {
        self.into()
    }
}

const MAX_ATOM_INDEX: usize = (std::usize::MAX << Tag::ATOM_BIT_COUNT) >> Tag::ATOM_BIT_COUNT;

impl From<atom::Index> for Term {
    fn from(atom_index: atom::Index) -> Self {
        if atom_index.0 <= MAX_ATOM_INDEX {
            Term {
                tagged: (atom_index.0 << Tag::ATOM_BIT_COUNT) | (Tag::Atom as usize),
            }
        } else {
            panic!("index ({}) in atom table exceeds max index that can be tagged as an atom in a Term ({})", atom_index.0, MAX_ATOM_INDEX)
        }
    }
}

impl<'a> TryFrom<&'a Term> for &'a Tuple {
    type Error = BadArgument;

    fn try_from(term: &Term) -> Result<&Tuple, BadArgument> {
        match term.tag() {
            Tag::Arity => {
                let pointer = term as *const Term as *const Tuple;
                Ok(unsafe { pointer.as_ref() }.unwrap())
            }
            Tag::Boxed => term.unbox().try_into(),
            _ => Err(BadArgument),
        }
    }
}

impl<'a> TryFrom<&'a Term> for &'a Cons {
    type Error = BadArgument;

    fn try_from(term: &Term) -> Result<&Cons, BadArgument> {
        match term.tag() {
            Tag::List => {
                let untagged = term.tagged & !(Tag::List as usize);
                let pointer = untagged as *const Term as *const Cons;
                Ok(unsafe { pointer.as_ref() }.unwrap())
            }
            _ => Err(BadArgument),
        }
    }
}

/// All terms in Erlang and Elixir are completely ordered.
///
/// number < atom < reference < function < port < pid < tuple < map < list < bitstring
///
/// > When comparing two numbers of different types (a number being either an integer or a float), a
/// > conversion to the type with greater precision will always occur, unless the comparison
/// > operator used is either === or !==. A float will be considered more precise than an integer,
/// > unless the float is greater/less than +/-9007199254740992.0 respectively, at which point all
/// > the significant figures of the float are to the left of the decimal point. This behavior
/// > exists so that the comparison of large numbers remains transitive.
/// >
/// > The collection types are compared using the following rules:
/// >
/// > * Tuples are compared by size, then element by element.
/// > * Maps are compared by size, then by keys in ascending term order, then by values in key
/// order. >   In the specific case of maps' key ordering, integers are always considered to be less
/// than >   floats.
/// > * Lists are compared element by element.
/// > * Bitstrings are compared byte by byte, incomplete bytes are compared bit by bit.
/// > -- https://hexdocs.pm/elixir/operators.html#term-ordering
impl OrderInProcess for Term {
    fn cmp_in_process(&self, other: &Self, process: &Process) -> Ordering {
        // in ascending order
        match (self.tag(), other.tag()) {
            (Tag::Arity, Tag::Arity) => self.tagged.cmp(&other.tagged),
            (Tag::SmallInteger, Tag::SmallInteger) => {
                if self.tagged == other.tagged {
                    Ordering::Equal
                } else {
                    let self_isize: isize = self.try_into().unwrap();
                    let other_isize: isize = other.try_into().unwrap();
                    self_isize.cmp(&other_isize)
                }
            }
            (Tag::SmallInteger, Tag::Atom) => Ordering::Less,
            (Tag::Atom, Tag::SmallInteger) => Ordering::Greater,
            (Tag::Atom, Tag::Atom) => {
                if self.tagged == other.tagged {
                    Ordering::Equal
                } else {
                    let self_name = process.atom_to_string(self);
                    let other_name = process.atom_to_string(other);

                    self_name.cmp(&other_name)
                }
            }
            (Tag::Atom, Tag::Boxed) => {
                let other_unboxed = other.unbox();

                match other_unboxed.tag() {
                    Tag::Arity => Ordering::Less,
                    other_unboxed_tag => unimplemented!("Atom cmp unboxed {:?}", other_unboxed_tag),
                }
            }
            (Tag::Boxed, Tag::Atom) => {
                let self_unboxed = self.unbox();

                match self_unboxed.tag() {
                    Tag::Arity => Ordering::Greater,
                    self_unboxed_tag => unimplemented!("unboxed {:?} cmp Atom", self_unboxed_tag),
                }
            }
            (Tag::Boxed, Tag::Boxed) => {
                let self_unboxed = self.unbox();
                let other_unboxed = other.unbox();

                // in ascending order
                match (self_unboxed.tag(), other_unboxed.tag()) {
                    (Tag::Arity, Tag::Arity) => {
                        let self_tuple: &Tuple = self_unboxed.try_into().unwrap();
                        let other_tuple: &Tuple = other_unboxed.try_into().unwrap();

                        self_tuple.cmp_in_process(other_tuple, process)
                    }
                    (self_unboxed_tag, other_unboxed_tag) => unimplemented!(
                        "unboxed {:?} cmp unboxed {:?}",
                        self_unboxed_tag,
                        other_unboxed_tag
                    ),
                }
            }
            (Tag::Boxed, Tag::EmptyList) | (Tag::Boxed, Tag::List) => {
                let self_unboxed = self.unbox();

                match self_unboxed.tag() {
                    Tag::Arity => Ordering::Less,
                    self_unboxed_tag => unimplemented!("unboxed {:?} cmp []", self_unboxed_tag),
                }
            }
            (Tag::EmptyList, Tag::Boxed) | (Tag::List, Tag::Boxed) => {
                let other_unboxed = other.unbox();

                match other_unboxed.tag() {
                    Tag::Arity => Ordering::Greater,
                    other_unboxed_tag => unimplemented!("[] cmp unboxed {:?}", other_unboxed_tag),
                }
            }
            (Tag::EmptyList, Tag::EmptyList) => Ordering::Equal,
            (Tag::EmptyList, Tag::List) => {
                // Empty list is shorter than all lists, so it is lesser.
                Ordering::Less
            }
            (Tag::List, Tag::EmptyList) => {
                // Any list is longer than empty lit
                Ordering::Greater
            }
            (Tag::List, Tag::List) => {
                let self_cons: &Cons = self.try_into().unwrap();
                let other_cons: &Cons = other.try_into().unwrap();

                self_cons.cmp_in_process(other_cons, process)
            }
            (self_tag, other_tag) => unimplemented!("{:?} cmp {:?}", self_tag, other_tag),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::RwLock;

    mod abs {
        use super::*;

        #[test]
        fn with_atom_is_bad_argument() {
            let mut process = process();
            let atom_term = process.find_or_insert_atom("atom");

            assert_eq_in_process!(atom_term.abs(), Err(BadArgument), process);
        }

        #[test]
        fn with_empty_list_is_bad_argument() {
            assert_eq_in_process!(Term::EMPTY_LIST.abs(), Err(BadArgument), Default::default());
        }

        #[test]
        fn with_list_is_bad_argument() {
            let mut process = process();
            let list_term = list_term(&mut process);

            assert_eq_in_process!(list_term.abs(), Err(BadArgument), process);
        }

        #[test]
        fn with_negative_is_positive() {
            let mut process = process();

            let negative: isize = -1;
            let negative_term = negative.into_process(&mut process);

            let positive = -negative;
            let positive_term = positive.into_process(&mut process);

            assert_eq_in_process!(negative_term.abs(), Ok(positive_term), process);
        }

        #[test]
        fn with_positive_is_self() {
            let mut process = process();
            let positive_term = 1usize.into_process(&mut process);

            assert_eq_in_process!(positive_term.abs(), Ok(positive_term), process);
        }

        #[test]
        fn with_tuple_is_bad_argument() {
            let mut process = process();
            let tuple_term = tuple_term(&mut process);

            assert_eq_in_process!(tuple_term.abs(), Err(BadArgument), process);
        }
    }

    mod cmp_in_process {
        use super::*;

        mod less {
            use super::*;

            #[test]
            fn number_is_less_than_atom() {
                let mut process = process();
                let number_term: Term = 0.into();
                let atom_term = process.find_or_insert_atom("0");

                assert_cmp_in_process!(number_term, Ordering::Less, atom_term, process);
                refute_cmp_in_process!(atom_term, Ordering::Less, number_term, process);
            }

            #[test]
            fn atom_is_less_than_tuple() {
                let mut process = process();
                let atom_term = process.find_or_insert_atom("0");
                let tuple_term = Term::slice_to_tuple(&[], &mut process);

                assert_cmp_in_process!(atom_term, Ordering::Less, tuple_term, process);
                refute_cmp_in_process!(tuple_term, Ordering::Less, atom_term, process);
            }

            #[test]
            fn atom_is_less_than_atom_if_name_is_less_than() {
                let mut process = process();
                let greater_name = "b";
                let greater_term = process.find_or_insert_atom(greater_name);
                let lesser_name = "a";
                let lesser_term = process.find_or_insert_atom(lesser_name);

                assert!(lesser_name < greater_name);
                assert_cmp_in_process!(lesser_term, Ordering::Less, greater_term, process);
                // it isn't just comparing the atom index
                assert!(!(lesser_term.tagged < greater_term.tagged));

                assert!(!(greater_name < lesser_name));
                refute_cmp_in_process!(greater_term, Ordering::Less, lesser_term, process);
                assert!(greater_term.tagged < lesser_term.tagged);
            }

            #[test]
            fn shorter_tuple_is_less_than_longer_tuple() {
                let mut process = process();
                let shorter_tuple = Term::slice_to_tuple(&[], &mut process);
                let longer_tuple = Term::slice_to_tuple(&[0.into()], &mut process);

                assert_cmp_in_process!(shorter_tuple, Ordering::Less, longer_tuple, process);
                refute_cmp_in_process!(longer_tuple, Ordering::Less, shorter_tuple, process);
            }

            #[test]
            fn same_length_tuples_with_lesser_elements_is_lesser() {
                let mut process = process();
                let lesser_tuple = Term::slice_to_tuple(&[0.into()], &mut process);
                let greater_tuple = Term::slice_to_tuple(&[1.into()], &mut process);

                assert_cmp_in_process!(lesser_tuple, Ordering::Less, greater_tuple, process);
                refute_cmp_in_process!(greater_tuple, Ordering::Less, lesser_tuple, process);
            }

            #[test]
            fn tuple_is_less_than_empty_list() {
                let mut process = process();
                let tuple_term = Term::slice_to_tuple(&[], &mut process);
                let empty_list_term = Term::EMPTY_LIST;

                assert_cmp_in_process!(tuple_term, Ordering::Less, empty_list_term, process);
                refute_cmp_in_process!(empty_list_term, Ordering::Less, tuple_term, process);
            }

            #[test]
            fn tuple_is_less_than_list() {
                let mut process = process();
                let tuple_term = Term::slice_to_tuple(&[], &mut process);
                let list_term = list_term(&mut process);

                assert_cmp_in_process!(tuple_term, Ordering::Less, list_term, process);
                refute_cmp_in_process!(list_term, Ordering::Less, tuple_term, process);
            }
        }

        mod equal {
            use super::*;

            #[test]
            fn with_improper_list() {
                let mut process = process();
                let list_term = Term::cons(0.into(), 1.into(), &mut process);
                let equal_list_term = Term::cons(0.into(), 1.into(), &mut process);
                let unequal_list_term = Term::cons(1.into(), 0.into(), &mut process);

                assert_eq_in_process!(list_term, list_term, process);
                assert_eq_in_process!(equal_list_term, equal_list_term, process);
                assert_ne_in_process!(list_term, unequal_list_term, process);
            }

            #[test]
            fn with_proper_list() {
                let mut process = process();
                let list_term = Term::cons(0.into(), Term::EMPTY_LIST, &mut process);
                let equal_list_term = Term::cons(0.into(), Term::EMPTY_LIST, &mut process);
                let unequal_list_term = Term::cons(1.into(), Term::EMPTY_LIST, &mut process);

                assert_eq_in_process!(list_term, list_term, process);
                assert_eq_in_process!(list_term, equal_list_term, process);
                assert_ne_in_process!(list_term, unequal_list_term, process);
            }

            #[test]
            fn with_nested_list() {
                let mut process = process();
                let list_term = Term::cons(
                    0.into(),
                    Term::cons(1.into(), Term::EMPTY_LIST, &mut process),
                    &mut process,
                );
                let equal_list_term = Term::cons(
                    0.into(),
                    Term::cons(1.into(), Term::EMPTY_LIST, &mut process),
                    &mut process,
                );
                let unequal_list_term = Term::cons(
                    1.into(),
                    Term::cons(0.into(), Term::EMPTY_LIST, &mut process),
                    &mut process,
                );

                assert_eq_in_process!(list_term, list_term, process);
                assert_eq_in_process!(list_term, equal_list_term, process);
                assert_ne_in_process!(list_term, unequal_list_term, process);
            }

            #[test]
            fn with_lists_of_unequal_length() {
                let mut process = process();
                let list_term = Term::cons(
                    0.into(),
                    Term::cons(1.into(), Term::EMPTY_LIST, &mut process),
                    &mut process,
                );
                let equal_list_term = Term::cons(
                    0.into(),
                    Term::cons(1.into(), Term::EMPTY_LIST, &mut process),
                    &mut process,
                );
                let shorter_list_term = Term::cons(0.into(), Term::EMPTY_LIST, &mut process);
                let longer_list_term = Term::cons(
                    0.into(),
                    Term::cons(
                        1.into(),
                        Term::cons(2.into(), Term::EMPTY_LIST, &mut process),
                        &mut process,
                    ),
                    &mut process,
                );

                assert_eq_in_process!(list_term, list_term, process);
                assert_eq_in_process!(list_term, equal_list_term, process);
                assert_ne_in_process!(list_term, shorter_list_term, process);
                assert_ne_in_process!(list_term, longer_list_term, process);
            }

            #[test]
            fn with_tuples_of_unequal_length() {
                let mut process = process();
                let tuple_term = Term::slice_to_tuple(&[0.into()], &mut process);
                let equal_term = Term::slice_to_tuple(&[0.into()], &mut process);
                let unequal_term = Term::slice_to_tuple(&[0.into(), 1.into()], &mut process);

                assert_eq_in_process!(tuple_term, tuple_term, process);
                assert_eq_in_process!(tuple_term, equal_term, process);
                assert_ne_in_process!(tuple_term, unequal_term, process);
            }
        }

    }

    mod delete_element {
        use super::*;

        #[test]
        fn with_atom_is_bad_argument() {
            let mut process = process();
            let atom_term = process.find_or_insert_atom("atom");

            assert_eq_in_process!(
                atom_term.delete_element(0.into(), &mut process),
                Err(BadArgument),
                process
            );
        }

        #[test]
        fn with_empty_list_is_bad_argument() {
            let mut process = process();

            assert_eq_in_process!(
                Term::EMPTY_LIST.delete_element(0.into(), &mut process),
                Err(BadArgument),
                process
            );
        }

        #[test]
        fn with_list_is_bad_argument() {
            let mut process = process();
            let list_term = list_term(&mut process);

            assert_eq_in_process!(
                list_term.delete_element(0.into(), &mut process),
                Err(BadArgument),
                process
            );
        }

        #[test]
        fn with_small_integer_is_bad_argument() {
            let mut process = process();
            let small_integer_term = small_integer_term(&mut process, 0);

            assert_eq_in_process!(
                small_integer_term.delete_element(0.into(), &mut process),
                Err(BadArgument),
                process
            );
        }

        #[test]
        fn with_tuple_without_small_integer_index_is_bad_argument() {
            let mut process = process();
            let tuple_term = Term::slice_to_tuple(&[0.into(), 1.into(), 2.into()], &mut process);
            let index = 1usize;
            let invalid_index_term = Term::arity(index);

            assert_ne!(invalid_index_term.tag(), Tag::SmallInteger);
            assert_eq_in_process!(
                tuple_term.delete_element(invalid_index_term, &mut process),
                Err(BadArgument),
                process
            );

            let valid_index_term = Term::from(index);

            assert_eq!(valid_index_term.tag(), Tag::SmallInteger);
            assert_eq_in_process!(
                tuple_term.delete_element(valid_index_term, &mut process),
                Ok(Term::slice_to_tuple(&[0.into(), 2.into()], &mut process)),
                process
            );
        }

        #[test]
        fn with_tuple_without_index_in_range_is_bad_argument() {
            let mut process = process();
            let empty_tuple_term = tuple_term(&mut process);

            assert_eq_in_process!(
                empty_tuple_term.delete_element(0.into(), &mut process),
                Err(BadArgument),
                process
            );
        }

        #[test]
        fn with_tuple_with_index_in_range_returns_tuple_without_element() {
            let mut process = process();
            let tuple_term = Term::slice_to_tuple(&[0.into(), 1.into(), 2.into()], &mut process);

            assert_eq_in_process!(
                tuple_term.delete_element(1.into(), &mut process),
                Ok(Term::slice_to_tuple(&[0.into(), 2.into()], &mut process)),
                process
            );
        }
    }

    mod element {
        use super::*;

        #[test]
        fn with_atom_is_bad_argument() {
            let mut process = process();
            let atom_term = process.find_or_insert_atom("atom");

            assert_eq_in_process!(atom_term.element(0.into()), Err(BadArgument), process);
        }

        #[test]
        fn with_empty_list_is_bad_argument() {
            assert_eq_in_process!(
                Term::EMPTY_LIST.element(0.into()),
                Err(BadArgument),
                Default::default()
            );
        }

        #[test]
        fn with_list_is_bad_argument() {
            let mut process = process();
            let list_term = list_term(&mut process);

            assert_eq_in_process!(list_term.element(0.into()), Err(BadArgument), process);
        }

        #[test]
        fn with_small_integer_is_bad_argument() {
            let mut process = process();
            let small_integer_term = small_integer_term(&mut process, 0);

            assert_eq_in_process!(
                small_integer_term.element(0.into()),
                Err(BadArgument),
                process
            );
        }

        #[test]
        fn with_tuple_without_small_integer_index_is_bad_argument() {
            let mut process = process();
            let element_term = 1.into();
            let tuple_term = Term::slice_to_tuple(&[element_term], &mut process);
            let index = 0usize;
            let invalid_index_term = Term::arity(index);

            assert_ne!(invalid_index_term.tag(), Tag::SmallInteger);
            assert_eq_in_process!(
                tuple_term.element(invalid_index_term),
                Err(BadArgument),
                process
            );

            let valid_index_term = Term::from(index);

            assert_eq!(valid_index_term.tag(), Tag::SmallInteger);
            assert_eq_in_process!(
                tuple_term.element(valid_index_term),
                Ok(element_term),
                process
            );
        }

        #[test]
        fn with_tuple_without_index_in_range_is_bad_argument() {
            let mut process = process();
            let empty_tuple_term = tuple_term(&mut process);

            assert_eq_in_process!(
                empty_tuple_term.element(0.into()),
                Err(BadArgument),
                process
            );
        }

        #[test]
        fn with_tuple_with_index_in_range_is_element() {
            let mut process = process();
            let element_term = 1.into();
            let tuple_term = Term::slice_to_tuple(&[element_term], &mut process);

            assert_eq_in_process!(tuple_term.element(0.into()), Ok(element_term), process);
        }
    }
    mod head {
        use super::*;

        #[test]
        fn with_atom_is_bad_argument() {
            let mut process = process();
            let atom_term = process.find_or_insert_atom("atom");

            assert_eq_in_process!(atom_term.head(), Err(BadArgument), process);
        }

        #[test]
        fn with_empty_list_is_bad_argument() {
            let empty_list_term = Term::EMPTY_LIST;

            assert_eq_in_process!(empty_list_term.head(), Err(BadArgument), Default::default());
        }

        #[test]
        fn with_list_returns_head() {
            let mut process = process();
            let head_term = process.find_or_insert_atom("head");
            let list_term = Term::cons(head_term, Term::EMPTY_LIST, &mut process);

            assert_eq_in_process!(list_term.head(), Ok(head_term), process);
        }

        #[test]
        fn with_small_integer_is_bad_argument() {
            let mut process = process();
            let small_integer_term = small_integer_term(&mut process, 0);

            assert_eq_in_process!(small_integer_term.head(), Err(BadArgument), process);
        }

        #[test]
        fn with_tuple_is_bad_argument() {
            let mut process = process();
            let tuple_term = tuple_term(&mut process);

            assert_eq_in_process!(tuple_term.head(), Err(BadArgument), process);
        }
    }

    mod tail {
        use super::*;

        #[test]
        fn with_atom_is_bad_argument() {
            let mut process = process();
            let atom_term = process.find_or_insert_atom("atom");

            assert_eq_in_process!(atom_term.tail(), Err(BadArgument), process);
        }

        #[test]
        fn with_empty_list_is_bad_argument() {
            let empty_list_term = Term::EMPTY_LIST;

            assert_eq_in_process!(empty_list_term.tail(), Err(BadArgument), Default::default());
        }

        #[test]
        fn with_list_returns_tail() {
            let mut process = process();
            let head_term = process.find_or_insert_atom("head");
            let list_term = Term::cons(head_term, Term::EMPTY_LIST, &mut process);

            assert_eq_in_process!(list_term.tail(), Ok(Term::EMPTY_LIST), process);
        }

        #[test]
        fn with_small_integer_is_bad_argument() {
            let mut process = process();
            let small_integer_term = small_integer_term(&mut process, 0);

            assert_eq_in_process!(small_integer_term.tail(), Err(BadArgument), process);
        }

        #[test]
        fn with_tuple_is_bad_argument() {
            let mut process = process();
            let tuple_term = tuple_term(&mut process);

            assert_eq_in_process!(tuple_term.tail(), Err(BadArgument), process);
        }
    }

    mod insert_element {
        use super::*;

        #[test]
        fn with_atom_is_bad_argument() {
            let mut process = process();
            let atom_term = process.find_or_insert_atom("atom");

            assert_eq_in_process!(
                atom_term.insert_element(0.into(), 0.into(), &mut process),
                Err(BadArgument),
                process
            );
        }

        #[test]
        fn with_empty_list_is_bad_argument() {
            let mut process = process();

            assert_eq_in_process!(
                Term::EMPTY_LIST.insert_element(0.into(), 0.into(), &mut process),
                Err(BadArgument),
                process
            );
        }

        #[test]
        fn with_list_is_bad_argument() {
            let mut process = process();
            let list_term = list_term(&mut process);

            assert_eq_in_process!(
                list_term.insert_element(0.into(), 0.into(), &mut process),
                Err(BadArgument),
                process
            );
        }

        #[test]
        fn with_small_integer_is_bad_argument() {
            let mut process = process();
            let small_integer_term = small_integer_term(&mut process, 0);

            assert_eq_in_process!(
                small_integer_term.insert_element(0.into(), 0.into(), &mut process),
                Err(BadArgument),
                process
            );
        }

        #[test]
        fn with_tuple_without_small_integer_index_is_bad_argument() {
            let mut process = process();
            let tuple_term = Term::slice_to_tuple(&[0.into(), 2.into()], &mut process);
            let index = 1usize;
            let invalid_index_term = Term::arity(index);

            assert_ne!(invalid_index_term.tag(), Tag::SmallInteger);
            assert_eq_in_process!(
                tuple_term.insert_element(invalid_index_term, 0.into(), &mut process),
                Err(BadArgument),
                process
            );

            let valid_index_term = Term::from(index);

            assert_eq!(valid_index_term.tag(), Tag::SmallInteger);
            assert_eq_in_process!(
                tuple_term.insert_element(valid_index_term, 1.into(), &mut process),
                Ok(Term::slice_to_tuple(
                    &[0.into(), 1.into(), 2.into()],
                    &mut process
                )),
                process
            );
        }

        #[test]
        fn with_tuple_without_index_in_range_is_bad_argument() {
            let mut process = process();
            let empty_tuple_term = tuple_term(&mut process);

            assert_eq_in_process!(
                empty_tuple_term.insert_element(1.into(), 0.into(), &mut process),
                Err(BadArgument),
                process
            );
        }

        #[test]
        fn with_tuple_with_index_in_range_returns_tuple_with_new_element_at_index() {
            let mut process = process();
            let tuple_term = Term::slice_to_tuple(&[0.into(), 2.into()], &mut process);

            assert_eq_in_process!(
                tuple_term.insert_element(1.into(), 1.into(), &mut process),
                Ok(Term::slice_to_tuple(
                    &[0.into(), 1.into(), 2.into()],
                    &mut process
                )),
                process
            );
        }

        #[test]
        fn with_tuple_with_index_at_size_return_tuples_with_new_element_at_end() {
            let mut process = process();
            let tuple_term = Term::slice_to_tuple(&[0.into()], &mut process);

            assert_eq_in_process!(
                tuple_term.insert_element(1.into(), 1.into(), &mut process),
                Ok(Term::slice_to_tuple(&[0.into(), 1.into()], &mut process)),
                process
            )
        }
    }

    mod is_atom {
        use super::*;

        #[test]
        fn with_atom_is_true() {
            let mut process = process();
            let atom_term = process.find_or_insert_atom("atom");

            assert_eq_in_process!(
                atom_term.is_atom(&mut process),
                true.into_process(&mut process),
                process
            );
        }

        #[test]
        fn with_booleans_is_true() {
            let mut process = process();
            let true_term = true.into_process(&mut process);
            let false_term = false.into_process(&mut process);

            assert_eq_in_process!(true_term.is_atom(&mut process), true_term, process);
            assert_eq_in_process!(false_term.is_atom(&mut process), true_term, process);
        }

        #[test]
        fn with_nil_is_true() {
            let mut process = process();
            let nil_term = process.find_or_insert_atom("nil");
            let true_term = true.into_process(&mut process);

            assert_eq_in_process!(nil_term.is_atom(&mut process), true_term, process);
        }

        #[test]
        fn with_empty_list_is_false() {
            let mut process = process();
            let empty_list_term = Term::EMPTY_LIST;
            let false_term = false.into_process(&mut process);

            assert_eq_in_process!(empty_list_term.is_atom(&mut process), false_term, process);
        }

        #[test]
        fn with_list_is_false() {
            let mut process = process();
            let head_term = process.find_or_insert_atom("head");
            let list_term = Term::cons(head_term, Term::EMPTY_LIST, &mut process);
            let false_term = false.into_process(&mut process);

            assert_eq_in_process!(list_term.is_atom(&mut process), false_term, process);
        }

        #[test]
        fn with_small_integer_is_false() {
            let mut process = process();
            let small_integer_term = small_integer_term(&mut process, 0);
            let false_term = false.into_process(&mut process);

            assert_eq_in_process!(
                small_integer_term.is_atom(&mut process),
                false_term,
                process
            );
        }

        #[test]
        fn with_tuple_is_false() {
            let mut process = process();
            let tuple_term = tuple_term(&mut process);
            let false_term = false.into_process(&mut process);

            assert_eq_in_process!(tuple_term.is_atom(&mut process), false_term, process);
        }
    }

    mod is_empty_list {
        use super::*;

        #[test]
        fn with_atom_is_false() {
            let mut process = process();
            let atom_term = process.find_or_insert_atom("atom");
            let false_term = false.into_process(&mut process);

            assert_eq_in_process!(atom_term.is_empty_list(&mut process), false_term, process);
        }

        #[test]
        fn with_empty_list_is_true() {
            let mut process = process();
            let empty_list_term = Term::EMPTY_LIST;
            let true_term = true.into_process(&mut process);

            assert_eq_in_process!(
                empty_list_term.is_empty_list(&mut process),
                true_term,
                process
            );
        }

        #[test]
        fn with_list_is_false() {
            let mut process = process();
            let head_term = process.find_or_insert_atom("head");
            let list_term = Term::cons(head_term, Term::EMPTY_LIST, &mut process);
            let false_term = false.into_process(&mut process);

            assert_eq_in_process!(list_term.is_empty_list(&mut process), false_term, process);
        }

        #[test]
        fn with_small_integer_is_false() {
            let mut process = process();
            let small_integer_term = small_integer_term(&mut process, 0);
            let false_term = false.into_process(&mut process);

            assert_eq_in_process!(
                small_integer_term.is_empty_list(&mut process),
                false_term,
                process
            );
        }

        #[test]
        fn with_tuple_is_false() {
            let mut process = process();
            let tuple_term = tuple_term(&mut process);
            let false_term = false.into_process(&mut process);

            assert_eq_in_process!(tuple_term.is_empty_list(&mut process), false_term, process);
        }
    }

    mod is_integer {
        use super::*;

        #[test]
        fn with_atom_is_false() {
            let mut process = process();
            let atom_term = process.find_or_insert_atom("atom");
            let false_term = false.into_process(&mut process);

            assert_eq_in_process!(atom_term.is_integer(&mut process), false_term, process);
        }

        #[test]
        fn with_empty_list_is_false() {
            let mut process = process();
            let empty_list_term = Term::EMPTY_LIST;
            let false_term = false.into_process(&mut process);

            assert_eq_in_process!(
                empty_list_term.is_integer(&mut process),
                false_term,
                process
            );
        }

        #[test]
        fn with_list_is_false() {
            let mut process = process();
            let list_term = list_term(&mut process);
            let false_term = false.into_process(&mut process);

            assert_eq_in_process!(list_term.is_integer(&mut process), false_term, process);
        }

        #[test]
        fn with_small_integer_is_true() {
            let mut process = process();
            let zero_term = 0usize.into_process(&mut process);
            let true_term = true.into_process(&mut process);

            assert_eq_in_process!(zero_term.is_integer(&mut process), true_term, process);
        }

        #[test]
        fn with_tuple_is_false() {
            let mut process = process();
            let tuple_term = tuple_term(&mut process);
            let false_term = false.into_process(&mut process);

            assert_eq_in_process!(tuple_term.is_integer(&mut process), false_term, process);
        }
    }

    mod is_list {
        use super::*;

        #[test]
        fn with_atom_is_false() {
            let mut process = process();
            let atom_term = process.find_or_insert_atom("atom");
            let false_term = false.into_process(&mut process);

            assert_eq_in_process!(atom_term.is_list(&mut process), false_term, process);
        }

        #[test]
        fn with_empty_list_is_true() {
            let mut process = process();
            let empty_list_term = Term::EMPTY_LIST;
            let true_term = true.into_process(&mut process);

            assert_eq_in_process!(empty_list_term.is_list(&mut process), true_term, process);
        }

        #[test]
        fn with_list_is_true() {
            let mut process = process();
            let list_term = list_term(&mut process);
            let true_term = true.into_process(&mut process);

            assert_eq_in_process!(list_term.is_list(&mut process), true_term, process);
        }

        #[test]
        fn with_small_integer_is_false() {
            let mut process = process();
            let small_integer_term = small_integer_term(&mut process, 0);
            let false_term = false.into_process(&mut process);

            assert_eq_in_process!(
                small_integer_term.is_list(&mut process),
                false_term,
                process
            );
        }

        #[test]
        fn with_tuple_is_false() {
            let mut process = process();
            let tuple_term = tuple_term(&mut process);
            let false_term = false.into_process(&mut process);

            assert_eq_in_process!(tuple_term.is_list(&mut process), false_term, process);
        }
    }

    mod is_tuple {
        use super::*;

        #[test]
        fn with_atom_is_false() {
            let mut process = process();
            let atom_term = process.find_or_insert_atom("atom");
            let false_term = false.into_process(&mut process);

            assert_eq_in_process!(atom_term.is_tuple(&mut process), false_term, process);
        }

        #[test]
        fn with_empty_list_is_false() {
            let mut process = process();
            let empty_list_term = Term::EMPTY_LIST;
            let false_term = false.into_process(&mut process);

            assert_eq_in_process!(empty_list_term.is_tuple(&mut process), false_term, process);
        }

        #[test]
        fn with_list_is_false() {
            let mut process = process();
            let list_term = list_term(&mut process);
            let false_term = false.into_process(&mut process);

            assert_eq_in_process!(list_term.is_tuple(&mut process), false_term, process);
        }

        #[test]
        fn with_small_integer_is_false() {
            let mut process = process();
            let small_integer_term = small_integer_term(&mut process, 0);
            let false_term = false.into_process(&mut process);

            assert_eq_in_process!(
                small_integer_term.is_tuple(&mut process),
                false_term,
                process
            );
        }

        #[test]
        fn with_tuple_is_true() {
            let mut process = process();
            let tuple_term = tuple_term(&mut process);
            let true_term = true.into_process(&mut process);

            assert_eq_in_process!(tuple_term.is_tuple(&mut process), true_term, process);
        }
    }

    mod length {
        use super::*;

        #[test]
        fn with_atom_is_bad_argument() {
            let mut process = process();
            let atom_term = process.find_or_insert_atom("atom");

            assert_eq_in_process!(atom_term.length(&mut process), Err(BadArgument), process);
        }

        #[test]
        fn with_empty_list_is_zero() {
            let mut process = process();
            let zero_term = small_integer_term(&mut process, 0);

            assert_eq_in_process!(
                Term::EMPTY_LIST.length(&mut process),
                Ok(zero_term),
                process
            );
        }

        #[test]
        fn with_improper_list_is_bad_argument() {
            let mut process = process();
            let head_term = process.find_or_insert_atom("head");
            let tail_term = process.find_or_insert_atom("tail");
            let improper_list_term = Term::cons(head_term, tail_term, &mut process);

            assert_eq_in_process!(
                improper_list_term.length(&mut process),
                Err(BadArgument),
                process
            );
        }

        #[test]
        fn with_list_is_length() {
            let mut process = process();
            let list_term = (0..=2).rfold(Term::EMPTY_LIST, |acc, i| {
                Term::cons(small_integer_term(&mut process, i), acc, &mut process)
            });

            assert_eq_in_process!(
                list_term.length(&mut process),
                Ok(small_integer_term(&mut process, 3)),
                process
            );
        }

        #[test]
        fn with_small_integer_is_bad_argument() {
            let mut process = process();
            let small_integer_term = small_integer_term(&mut process, 0);

            assert_eq_in_process!(
                small_integer_term.length(&mut process),
                Err(BadArgument),
                process
            );
        }

        #[test]
        fn with_tuple_is_bad_argument() {
            let mut process = process();
            let tuple_term = tuple_term(&mut process);

            assert_eq_in_process!(tuple_term.length(&mut process), Err(BadArgument), process);
        }
    }

    mod size {
        use super::*;

        #[test]
        fn with_atom_is_bad_argument() {
            let mut process = process();
            let atom_term = process.find_or_insert_atom("atom");

            assert_eq_in_process!(atom_term.size(), Err(BadArgument), process);
        }

        #[test]
        fn with_empty_list_is_bad_argument() {
            let process: Process = Default::default();

            assert_eq_in_process!(Term::EMPTY_LIST.size(), Err(BadArgument), process);
        }

        #[test]
        fn with_list_is_bad_argument() {
            let mut process = process();
            let list_term = list_term(&mut process);

            assert_eq_in_process!(list_term.size(), Err(BadArgument), process);
        }

        #[test]
        fn with_small_integer_is_bad_argument() {
            let mut process = process();
            let small_integer_term = small_integer_term(&mut process, 0);

            assert_eq_in_process!(small_integer_term.size(), Err(BadArgument), process);
        }

        #[test]
        fn with_tuple_without_elements_is_zero() {
            let mut process = process();
            let empty_tuple_term = tuple_term(&mut process);
            let zero_term = 0usize.into_process(&mut process);

            assert_eq_in_process!(empty_tuple_term.size(), Ok(zero_term), process);
        }

        #[test]
        fn with_tuple_with_elements_is_element_count() {
            let mut process = process();
            let element_vec: Vec<Term> =
                (0..=2usize).map(|i| i.into_process(&mut process)).collect();
            let element_slice: &[Term] = element_vec.as_slice();
            let tuple_term = Term::slice_to_tuple(element_slice, &mut process);
            let arity_term = 3usize.into_process(&mut process);

            assert_eq_in_process!(tuple_term.size(), Ok(arity_term), process);
        }
    }

    fn process() -> Process {
        use crate::environment::Environment;

        Process::new(Arc::new(RwLock::new(Environment::new())))
    }

    fn small_integer_term(mut process: &mut Process, signed_size: isize) -> Term {
        signed_size.into_process(&mut process)
    }

    fn list_term(process: &mut Process) -> Term {
        let head_term = process.find_or_insert_atom("head");
        Term::cons(head_term, Term::EMPTY_LIST, process)
    }

    fn tuple_term(process: &mut Process) -> Term {
        Term::slice_to_tuple(&[], process)
    }
}
