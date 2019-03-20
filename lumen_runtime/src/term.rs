#![cfg_attr(not(test), allow(dead_code))]

use std::cmp::Ordering;
use std::convert::{TryFrom, TryInto};
use std::fmt::{self, Display};
use std::hash::{Hash, Hasher};
use std::mem::size_of;
use std::str::Chars;

use num_bigint::BigInt;

use liblumen_arena::TypedArena;

use crate::atom::{self, Encoding, Existence};
use crate::binary::{self, heap, sub, Part, PartToList};
use crate::exception::{self, Exception};
use crate::float::Float;
use crate::integer::Integer::{self, Big, Small};
use crate::integer::{big, small};
use crate::list::Cons;
use crate::map::Map;
use crate::process::{
    self, DebugInProcess, IntoProcess, OrderInProcess, Process, TryFromInProcess, TryIntoInProcess,
};
use crate::tuple::Tuple;

pub mod external_format;

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
    BigInteger = 0b0010_00,
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
    const HEADER_BIT_COUNT: u8 = 6;
    const ATOM_BIT_COUNT: u8 = 6;
    const ARITY_BIT_COUNT: u8 = Self::HEADER_BIT_COUNT;
    const HEAP_BINARY_BIT_COUNT: u8 = Self::HEADER_BIT_COUNT;

    pub const LOCAL_PID_BIT_COUNT: u8 = 4;
    pub const SMALL_INTEGER_BIT_COUNT: u8 = 4;
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
                0b0010_00 => Ok(Tag::BigInteger),
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

impl Term {
    pub const BIT_COUNT: u8 = (size_of::<Term>() * 8) as u8;

    const MAX_ARITY: usize = std::usize::MAX >> Tag::ARITY_BIT_COUNT;
    const MAX_HEAP_BINARY_BYTE_COUNT: usize = std::usize::MAX >> Tag::HEAP_BINARY_BIT_COUNT;

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

    pub fn arity_to_integer(&self, mut process: &mut Process) -> Term {
        self.arity_to_usize().into_process(&mut process)
    }

    pub fn arity_to_usize(&self) -> usize {
        assert_eq!(
            self.tag(),
            Tag::Arity,
            "Term ({:#b}) is not a tuple arity",
            self.tagged
        );

        ((self.tagged & !(Tag::Arity as usize)) >> Tag::ARITY_BIT_COUNT)
    }

    pub unsafe fn as_ref_cons_unchecked(&self) -> &'static Cons {
        let untagged = self.tagged & !(Tag::List as usize);
        let pointer = untagged as *const Term as *const Cons;

        &*pointer
    }

    pub fn atom_to_encoding(&self, mut process: &mut Process) -> Result<Encoding, Exception> {
        match self.tag() {
            Tag::Atom => {
                let unicode_atom =
                    Term::str_to_atom("unicode", Existence::DoNotCare, &mut process).unwrap();
                let tagged = self.tagged;

                if tagged == unicode_atom.tagged {
                    Ok(Encoding::Unicode)
                } else {
                    let utf8_atom =
                        Term::str_to_atom("utf8", Existence::DoNotCare, &mut process).unwrap();

                    if tagged == utf8_atom.tagged {
                        Ok(Encoding::Utf8)
                    } else {
                        let latin1_atom =
                            Term::str_to_atom("latin1", Existence::DoNotCare, &mut process)
                                .unwrap();

                        if tagged == latin1_atom.tagged {
                            Ok(Encoding::Latin1)
                        } else {
                            Err(bad_argument!(&mut process))
                        }
                    }
                }
            }
            _ => Err(bad_argument!(&mut process)),
        }
    }

    pub fn atom_to_string(&self, process: &Process) -> String {
        process.atom_index_to_string(self.try_into().unwrap())
    }

    pub fn alloc_slice(slice: &[Term], term_arena: &mut TypedArena<Term>) -> *const Term {
        term_arena.alloc_slice(slice).as_ptr()
    }

    pub fn byte(&self, index: usize) -> u8 {
        match self.tag() {
            Tag::Boxed => {
                let unboxed: &Term = self.unbox_reference();

                match unboxed.tag() {
                    Tag::HeapBinary => {
                        let heap_binary: &heap::Binary = self.unbox_reference();

                        heap_binary.byte(index)
                    }
                    Tag::ReferenceCountedBinary => unimplemented!(),
                    unboxed_tag => panic!("Cannot get bytes of unboxed {:?}", unboxed_tag),
                }
            }
            tag => panic!("Cannot get bytes of {:?}", tag),
        }
    }

    pub fn chars_to_list(chars: Chars, mut process: &mut Process) -> Term {
        chars.rfold(Self::EMPTY_LIST, |acc, character| {
            Term::cons(character.into_process(&mut process), acc, &mut process)
        })
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

    pub fn external_pid(
        node: usize,
        number: usize,
        serial: usize,
        mut process: &mut Process,
    ) -> exception::Result {
        if (number <= process::identifier::NUMBER_MAX)
            && (serial <= process::identifier::SERIAL_MAX)
        {
            Ok(Term::box_reference(
                process.external_pid(node, number, serial),
            ))
        } else {
            Err(bad_argument!(&mut process))
        }
    }

    pub fn heap_binary(byte_count: usize) -> Term {
        assert!(
            byte_count <= Self::MAX_HEAP_BINARY_BYTE_COUNT,
            "byte_count ({}) is greater than max heap binary byte count ({})",
            byte_count,
            Self::MAX_HEAP_BINARY_BYTE_COUNT,
        );

        Term {
            tagged: ((byte_count << Tag::HEAP_BINARY_BIT_COUNT) as usize)
                | (Tag::HeapBinary as usize),
        }
    }

    pub fn heap_binary_to_byte_count(&self) -> usize {
        const TAG_HEAP_BINARY: usize = Tag::HeapBinary as usize;

        assert_eq!(
            self.tag(),
            Tag::HeapBinary,
            "Term ({:#b}) is not a heap binary",
            self.tagged
        );

        (self.tagged & !(TAG_HEAP_BINARY)) >> Tag::HEAP_BINARY_BIT_COUNT
    }

    pub fn local_pid(number: usize, serial: usize, mut process: &mut Process) -> exception::Result {
        if (number <= process::identifier::NUMBER_MAX)
            && (serial <= process::identifier::SERIAL_MAX)
        {
            Ok(Term {
                tagged: (serial
                    << (process::identifier::NUMBER_BIT_COUNT + Tag::LOCAL_PID_BIT_COUNT))
                    | (number << (Tag::LOCAL_PID_BIT_COUNT))
                    | (Tag::LocalPid as usize),
            })
        } else {
            Err(bad_argument!(&mut process))
        }
    }

    pub unsafe fn local_pid_unchecked(number: usize, serial: usize) -> Term {
        Term {
            tagged: (serial << (process::identifier::NUMBER_BIT_COUNT + Tag::LOCAL_PID_BIT_COUNT))
                | (number << (Tag::LOCAL_PID_BIT_COUNT))
                | (Tag::LocalPid as usize),
        }
    }

    pub fn tag(&self) -> Tag {
        match (self.tagged as usize).try_into() {
            Ok(tag) => tag,
            Err(tag_error) => panic!(tag_error),
        }
    }

    pub fn is_empty_list(&self) -> bool {
        (self.tag() == Tag::EmptyList)
    }

    pub fn pid(
        node: usize,
        number: usize,
        serial: usize,
        process: &mut Process,
    ) -> exception::Result {
        if node == 0 {
            Self::local_pid(number, serial, process)
        } else {
            Self::external_pid(node, number, serial, process)
        }
    }

    pub fn slice_to_binary(slice: &[u8], process: &mut Process) -> Term {
        process.slice_to_binary(slice).into()
    }

    pub fn slice_to_map(slice: &[(Term, Term)], process: &mut Process) -> Term {
        process.slice_to_map(slice).into()
    }

    pub fn slice_to_tuple(slice: &[Term], process: &mut Process) -> Term {
        process.slice_to_tuple(slice).into()
    }

    pub fn str_to_atom(
        name: &str,
        existence: Existence,
        process: &mut Process,
    ) -> exception::Result {
        process
            .str_to_atom_index(name, existence)
            .map(|atom_index| atom_index.into())
    }

    pub fn str_to_char_list(name: &str, mut process: &mut Process) -> Term {
        name.chars().rfold(Term::EMPTY_LIST, |acc, c| {
            Term::cons(c.into_process(&mut process), acc, &mut process)
        })
    }

    pub fn subbinary(
        original: Term,
        byte_offset: usize,
        bit_offset: u8,
        byte_count: usize,
        bit_count: u8,
        process: &mut Process,
    ) -> Term {
        process
            .subbinary(original, byte_offset, bit_offset, byte_count, bit_count)
            .into()
    }

    fn box_reference<T>(reference: &T) -> Term {
        let pointer_bits = reference as *const T as usize;

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

    pub fn unbox_reference<T>(&self) -> &'static T {
        const TAG_BOXED: usize = Tag::Boxed as usize;

        assert_eq!(
            self.tagged & TAG_BOXED,
            TAG_BOXED,
            "Term ({:#b}) is not tagged as boxed ({:#b})",
            self.tagged,
            TAG_BOXED
        );

        let pointer_bits = self.tagged & !TAG_BOXED;
        let pointer = pointer_bits as *const T;

        unsafe { pointer.as_ref() }.unwrap()
    }

    const SMALL_INTEGER_SIGN_BIT_MASK: usize = std::isize::MIN as usize;

    pub unsafe fn small_integer_to_usize(&self) -> usize {
        assert_eq!(
            self.tag(),
            Tag::SmallInteger,
            "Term ({:#b}) is not a small integer",
            self.tagged
        );

        ((self.tagged & !(Tag::SmallInteger as usize)) >> Tag::SMALL_INTEGER_BIT_COUNT)
    }

    /// Only call if verified `tag` is `Tag::SmallInteger`.
    pub unsafe fn small_integer_is_negative(&self) -> bool {
        self.tagged & Term::SMALL_INTEGER_SIGN_BIT_MASK == Term::SMALL_INTEGER_SIGN_BIT_MASK
    }

    pub const unsafe fn isize_to_small_integer(i: isize) -> Term {
        Term {
            tagged: ((i << Tag::SMALL_INTEGER_BIT_COUNT) as usize) | (Tag::SmallInteger as usize),
        }
    }

    pub unsafe fn small_integer_to_isize(&self) -> isize {
        (self.tagged as isize) >> Tag::SMALL_INTEGER_BIT_COUNT
    }
}

impl DebugInProcess for Term {
    fn format_in_process(&self, process: &Process) -> String {
        match self.tag() {
            Tag::Arity => format!("Term::arity({})", self.arity_to_usize()),
            Tag::Atom => format!(
                "Term::str_to_atom(\"{}\", Existence::DoNotCare, &mut process).unwrap()",
                self.atom_to_string(process)
            ),
            Tag::Boxed => {
                let unboxed: &Term = self.unbox_reference();

                match unboxed.tag() {
                    Tag::Arity => {
                        let tuple: &Tuple = self.unbox_reference();

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
                    Tag::BigInteger => {
                        let big_integer: &big::Integer = self.unbox_reference();

                        format!(
                            "BigInt::parse_bytes(b\"{}\", 10).unwrap().into_process(&mut process)",
                            big_integer.inner
                        )
                    }
                    Tag::Float => {
                        let float: &Float = self.unbox_reference();

                        format!("{}_f64.into_process(&mut process)", float.inner)
                    }
                    Tag::HeapBinary => {
                        let binary: &heap::Binary = self.unbox_reference();

                        let mut strings: Vec<String> = Vec::new();

                        strings.push("Term::slice_to_binary(&[".to_string());

                        let mut iter = binary.iter();

                        if let Some(first_byte) = iter.next() {
                            strings.push(first_byte.to_string());

                            for byte in iter {
                                strings.push(", ".to_string());
                                strings.push(byte.to_string());
                            }
                        }

                        strings.push("], &mut process)".to_string());

                        strings.join("")
                    }
                    Tag::Subbinary => {
                        let subbinary: &sub::Binary = self.unbox_reference();

                        let mut strings: Vec<String> = Vec::new();

                        strings.push("Term::subbinary(".to_string());
                        strings.push(subbinary.original.format_in_process(process));
                        strings.push(", ".to_string());
                        strings.push(subbinary.byte_offset.to_string());
                        strings.push(", ".to_string());
                        strings.push(subbinary.bit_offset.to_string());
                        strings.push(", ".to_string());
                        strings.push(subbinary.byte_count.to_string());
                        strings.push(", ".to_string());
                        strings.push(subbinary.bit_count.to_string());
                        strings.push(", &mut process)".to_string());

                        strings.join("")
                    }
                    unboxed_tag => unimplemented!("unboxed {:?}", unboxed_tag),
                }
            }
            Tag::EmptyList => "Term::EMPTY_LIST".to_string(),
            Tag::List => {
                let cons: &Cons = unsafe { self.as_ref_cons_unchecked() };

                format!(
                    "Term::cons({}, {}, &mut process)",
                    cons.head().format_in_process(&process),
                    cons.tail().format_in_process(&process)
                )
            }
            Tag::SmallInteger => format!("{:?}.into_process(&mut process)", isize::from(self)),
            _ => format!(
                "Term {{ tagged: 0b{tagged:0bit_count$b} }}",
                tagged = self.tagged,
                bit_count = std::mem::size_of::<usize>() * 8
            ),
        }
    }
}

impl DebugInProcess for Option<Term> {
    fn format_in_process(&self, process: &Process) -> String {
        match self {
            Some(term) => format!("Some({})", term.format_in_process(process)),
            None => "None".to_string(),
        }
    }
}

impl Eq for Term {}

impl<'a> From<binary::Binary<'a>> for Term {
    fn from(binary: binary::Binary<'a>) -> Self {
        match binary {
            binary::Binary::Heap(heap_binary) => Term::box_reference(heap_binary),
            binary::Binary::Sub(sub_binary) => Term::box_reference(sub_binary),
        }
    }
}

impl From<u8> for Term {
    fn from(u: u8) -> Term {
        let untagged: isize = u as isize;

        Term {
            tagged: ((untagged << Tag::SMALL_INTEGER_BIT_COUNT) as usize)
                | (Tag::SmallInteger as usize),
        }
    }
}

impl From<&Term> for isize {
    fn from(term: &Term) -> Self {
        match term.tag() {
            Tag::SmallInteger => (term.tagged as isize) >> Tag::SMALL_INTEGER_BIT_COUNT,
            tag => panic!(
                "{:?} tagged term {:#b} cannot be converted to isize",
                tag, term.tagged
            ),
        }
    }
}

impl<T> From<&T> for Term {
    fn from(reference: &T) -> Self {
        Term::box_reference(reference)
    }
}

impl Hash for Term {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self.tag() {
            Tag::Atom => self.tagged.hash(state),
            tag => unimplemented!("tag {:?}", tag),
        }
    }
}

impl IntoProcess<Term> for char {
    fn into_process(self, mut process: &mut Process) -> Term {
        let integer: Integer = self.into();

        integer.into_process(&mut process)
    }
}

impl IntoProcess<Term> for f64 {
    fn into_process(self, process: &mut Process) -> Term {
        let process_float: &Float = process.f64_to_float(self);

        Term::box_reference(process_float)
    }
}

impl IntoProcess<Term> for i32 {
    fn into_process(self, mut process: &mut Process) -> Term {
        let integer: Integer = self.into();

        integer.into_process(&mut process)
    }
}

impl IntoProcess<Term> for isize {
    fn into_process(self, mut process: &mut Process) -> Term {
        let integer: Integer = self.into();

        integer.into_process(&mut process)
    }
}

impl IntoProcess<Term> for u8 {
    fn into_process(self, mut process: &mut Process) -> Term {
        let integer: Integer = self.into();

        integer.into_process(&mut process)
    }
}

impl IntoProcess<Term> for usize {
    fn into_process(self, mut process: &mut Process) -> Term {
        let integer: Integer = self.into();

        integer.into_process(&mut process)
    }
}

impl IntoProcess<Term> for Integer {
    fn into_process(self, process: &mut Process) -> Term {
        match self {
            Small(small::Integer(untagged)) => Term {
                tagged: ((untagged << Tag::SMALL_INTEGER_BIT_COUNT) as usize)
                    | (Tag::SmallInteger as usize),
            },
            Big(big_int) => {
                let process_integer: &big::Integer =
                    process.num_bigint_big_in_to_big_integer(big_int);

                Term::box_reference(process_integer)
            }
        }
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

impl<'a> Part<'a, Term, Term, Term> for heap::Binary {
    fn part(&'a self, start: Term, length: Term, mut process: &mut Process) -> exception::Result {
        let start_usize: usize = start.try_into_in_process(&mut process)?;
        let length_isize: isize = length.try_into_in_process(&mut process)?;

        let binary = self.part(start_usize, length_isize, &mut process)?;

        match binary {
            // a heap binary is only returned if it is the same
            binary::Binary::Heap(_) => Ok(self.into()),
            binary::Binary::Sub(subbinary) => Ok(subbinary.into()),
        }
    }
}

impl<'a> Part<'a, Term, Term, Term> for sub::Binary {
    fn part(&'a self, start: Term, length: Term, mut process: &mut Process) -> exception::Result {
        let start_usize: usize = start.try_into_in_process(&mut process)?;
        let length_isize: isize = length.try_into_in_process(&mut process)?;
        let new_subbinary = self.part(start_usize, length_isize, &mut process)?;

        Ok(new_subbinary.into())
    }
}

impl PartToList<Term, Term> for heap::Binary {
    fn part_to_list(
        &self,
        start: Term,
        length: Term,
        mut process: &mut Process,
    ) -> exception::Result {
        let start_usize: usize = start.try_into_in_process(&mut process)?;
        let length_isize: isize = length.try_into_in_process(&mut process)?;

        self.part_to_list(start_usize, length_isize, process)
    }
}

impl PartToList<Term, Term> for sub::Binary {
    fn part_to_list(
        &self,
        start: Term,
        length: Term,
        mut process: &mut Process,
    ) -> exception::Result {
        let start_usize: usize = start.try_into_in_process(&mut process)?;
        let length_isize: isize = length.try_into_in_process(&mut process)?;

        self.part_to_list(start_usize, length_isize, process)
    }
}

impl PartialEq for Term {
    fn eq(&self, other: &Term) -> bool {
        let tag = self.tag();

        if tag == other.tag() {
            match tag {
                Tag::Atom => self.tagged == other.tagged,
                _ => unimplemented!("tag ({:?})", tag),
            }
        } else {
            false
        }
    }

    fn ne(&self, other: &Term) -> bool {
        !self.eq(other)
    }
}

impl TryFromInProcess<Term> for isize {
    fn try_from_in_process(term: Term, mut process: &mut Process) -> Result<isize, Exception> {
        match term.tag() {
            Tag::SmallInteger => {
                let term_isize = (term.tagged as isize) >> Tag::SMALL_INTEGER_BIT_COUNT;
                Ok(term_isize)
            }
            _ => Err(bad_argument!(&mut process)),
        }
    }
}

impl TryFromInProcess<Term> for usize {
    fn try_from_in_process(term: Term, mut process: &mut Process) -> Result<usize, Exception> {
        match term.tag() {
            Tag::SmallInteger => {
                let term_isize = (term.tagged as isize) >> Tag::SMALL_INTEGER_BIT_COUNT;

                if term_isize < 0 {
                    Err(bad_argument!(&mut process))
                } else {
                    Ok(term_isize as usize)
                }
            }
            _ => Err(bad_argument!(&mut process)),
        }
    }
}

impl TryFromInProcess<Term> for &'static Cons {
    fn try_from_in_process(
        term: Term,
        mut process: &mut Process,
    ) -> Result<&'static Cons, Exception> {
        match term.tag() {
            Tag::List => Ok(unsafe { term.as_ref_cons_unchecked() }),
            _ => Err(bad_argument!(&mut process)),
        }
    }
}

impl TryFromInProcess<Term> for BigInt {
    fn try_from_in_process(term: Term, mut process: &mut Process) -> Result<BigInt, Exception> {
        match term.tag() {
            Tag::SmallInteger => {
                let term_isize = (term.tagged as isize) >> Tag::SMALL_INTEGER_BIT_COUNT;

                Ok(term_isize.into())
            }
            Tag::Boxed => {
                let unboxed: &Term = term.unbox_reference();

                match unboxed.tag() {
                    Tag::BigInteger => {
                        let big_integer: &big::Integer = term.unbox_reference();

                        Ok(big_integer.inner.clone())
                    }
                    _ => Err(bad_argument!(&mut process)),
                }
            }
            _ => Err(bad_argument!(&mut process)),
        }
    }
}

impl TryFromInProcess<Term> for String {
    fn try_from_in_process(term: Term, mut process: &mut Process) -> Result<String, Exception> {
        match term.tag() {
            Tag::Boxed => {
                let unboxed: &Term = term.unbox_reference();

                match unboxed.tag() {
                    Tag::HeapBinary => {
                        let heap_binary: &heap::Binary = term.unbox_reference();

                        heap_binary.try_into_in_process(&mut process)
                    }
                    Tag::Subbinary => {
                        let subbinary: &sub::Binary = term.unbox_reference();

                        subbinary.try_into_in_process(&mut process)
                    }
                    // TODO ReferenceCountedBinary
                    _ => Err(bad_argument!(&mut process)),
                }
            }
            _ => Err(bad_argument!(&mut process)),
        }
    }
}

impl TryFromInProcess<Term> for &'static Tuple {
    fn try_from_in_process(
        term: Term,
        mut process: &mut Process,
    ) -> Result<&'static Tuple, Exception> {
        match term.tag() {
            Tag::Boxed => term
                .unbox_reference::<Term>()
                .try_into_in_process(&mut process),
            _ => Err(bad_argument!(&mut process)),
        }
    }
}

impl TryFromInProcess<&Term> for BigInt {
    fn try_from_in_process(
        term_ref: &Term,
        mut process: &mut Process,
    ) -> Result<BigInt, Exception> {
        (*term_ref).try_into_in_process(&mut process)
    }
}

impl<'a> TryFromInProcess<&'a Term> for &'a Tuple {
    fn try_from_in_process(
        term: &'a Term,
        mut process: &mut Process,
    ) -> Result<&'a Tuple, Exception> {
        match term.tag() {
            Tag::Arity => {
                let pointer = term as *const Term as *const Tuple;
                Ok(unsafe { pointer.as_ref() }.unwrap())
            }
            Tag::Boxed => term
                .unbox_reference::<Term>()
                .try_into_in_process(&mut process),
            _ => Err(bad_argument!(&mut process)),
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
/// >   order.   In the specific case of maps' key ordering, integers are always considered to be
/// >   less than floats.
/// > * Lists are compared element by element.
/// > * Bitstrings are compared byte by byte, incomplete bytes are compared bit by bit.
/// > -- https://hexdocs.pm/elixir/operators.html#term-ordering
impl OrderInProcess for Term {
    fn cmp_in_process(&self, other: &Self, process: &Process) -> Ordering {
        // in ascending order
        match (self.tag(), other.tag()) {
            (Tag::Arity, Tag::Arity) | (Tag::HeapBinary, Tag::HeapBinary) => {
                self.tagged.cmp(&other.tagged)
            }
            (Tag::SmallInteger, Tag::SmallInteger) => {
                if self.tagged == other.tagged {
                    Ordering::Equal
                } else {
                    let self_isize: isize = self.try_into().unwrap();
                    let other_isize: isize = other.try_into().unwrap();
                    self_isize.cmp(&other_isize)
                }
            }
            (Tag::SmallInteger, Tag::Boxed) => {
                let other_unboxed: &Term = other.unbox_reference();

                match other_unboxed.tag() {
                    Tag::BigInteger => {
                        let self_big_int: BigInt = unsafe { self.small_integer_to_isize() }.into();

                        let other_big_integer: &big::Integer = other.unbox_reference();
                        let other_big_int = &other_big_integer.inner;

                        self_big_int.cmp(other_big_int)
                    }
                    other_unboxed_tag => {
                        unimplemented!("SmallInteger cmp unboxed {:?}", other_unboxed_tag)
                    }
                }
            }
            (Tag::SmallInteger, Tag::Atom) => Ordering::Less,
            (Tag::SmallInteger, Tag::List) => Ordering::Less,
            (Tag::Atom, Tag::SmallInteger) => Ordering::Greater,
            (Tag::Atom, Tag::Atom) => {
                if self.tagged == other.tagged {
                    Ordering::Equal
                } else {
                    let self_name = self.atom_to_string(process);
                    let other_name = other.atom_to_string(process);

                    self_name.cmp(&other_name)
                }
            }
            (Tag::Atom, Tag::Boxed) => {
                let other_unboxed: &Term = other.unbox_reference();

                match other_unboxed.tag() {
                    Tag::Arity => Ordering::Less,
                    Tag::Subbinary => Ordering::Less,
                    other_unboxed_tag => unimplemented!("Atom cmp unboxed {:?}", other_unboxed_tag),
                }
            }
            (Tag::Boxed, Tag::SmallInteger) => {
                let self_unboxed: &Term = self.unbox_reference();

                match self_unboxed.tag() {
                    Tag::BigInteger => {
                        let self_big_integer: &big::Integer = self.unbox_reference();
                        let self_big_int = &self_big_integer.inner;

                        let other_big_int: BigInt =
                            unsafe { other.small_integer_to_isize() }.into();

                        self_big_int.cmp(&other_big_int)
                    }
                    Tag::Subbinary => Ordering::Greater,
                    self_unboxed_tag => {
                        unimplemented!("unboxed {:?} cmp SmallInteger", self_unboxed_tag)
                    }
                }
            }
            (Tag::Boxed, Tag::Atom) => {
                let self_unboxed: &Term = self.unbox_reference();

                match self_unboxed.tag() {
                    Tag::Arity => Ordering::Greater,
                    self_unboxed_tag => unimplemented!("unboxed {:?} cmp Atom", self_unboxed_tag),
                }
            }
            (Tag::Boxed, Tag::Boxed) => {
                let self_unboxed: &Term = self.unbox_reference();
                let other_unboxed: &Term = other.unbox_reference();

                // in ascending order
                match (self_unboxed.tag(), other_unboxed.tag()) {
                    (Tag::BigInteger, Tag::BigInteger) => {
                        let self_big_integer: &big::Integer = self.unbox_reference();
                        let other_big_integer: &big::Integer = other.unbox_reference();

                        self_big_integer.inner.cmp(&other_big_integer.inner)
                    }
                    (Tag::Float, Tag::Float) => {
                        let self_float: &Float = self.unbox_reference();
                        let self_inner = self_float.inner;

                        let other_float: &Float = other.unbox_reference();
                        let other_inner = other_float.inner;

                        // Erlang doesn't support the floats that can't be compared
                        self_inner.partial_cmp(&other_inner).unwrap_or_else(|| {
                            panic!(
                                "Comparing these floats ({} and {}) is not supported",
                                self_inner, other_inner
                            )
                        })
                    }
                    (Tag::ExternalPid, Tag::ExternalPid) => {
                        let self_external_pid: &process::identifier::External =
                            self.unbox_reference();
                        let other_external_pid: &process::identifier::External =
                            other.unbox_reference();

                        self_external_pid.cmp_in_process(other_external_pid, process)
                    }
                    (Tag::Arity, Tag::Arity) => {
                        let self_tuple: &Tuple = self.unbox_reference();
                        let other_tuple: &Tuple = other.unbox_reference();

                        self_tuple.cmp_in_process(other_tuple, process)
                    }
                    (Tag::Arity, Tag::Float) => Ordering::Greater,
                    (Tag::Map, Tag::Map) => {
                        let self_map: &Map = self.unbox_reference();
                        let other_map: &Map = other.unbox_reference();

                        self_map.cmp_in_process(other_map, process)
                    }
                    (Tag::HeapBinary, Tag::HeapBinary) => {
                        let self_binary: &heap::Binary = self.unbox_reference();
                        let other_binary: &heap::Binary = other.unbox_reference();

                        self_binary.cmp_in_process(other_binary, process)
                    }
                    (Tag::Subbinary, Tag::HeapBinary) => {
                        let self_subbinary: &sub::Binary = self.unbox_reference();
                        let other_heap_binary: &heap::Binary = other.unbox_reference();

                        self_subbinary.cmp_in_process(other_heap_binary, process)
                    }
                    (Tag::Subbinary, Tag::Subbinary) => {
                        let self_subbinary: &sub::Binary = self.unbox_reference();
                        let other_subbinary: &sub::Binary = other.unbox_reference();

                        self_subbinary.cmp_in_process(other_subbinary, process)
                    }
                    (self_unboxed_tag, other_unboxed_tag) => unimplemented!(
                        "unboxed {:?} cmp unboxed {:?}",
                        self_unboxed_tag,
                        other_unboxed_tag
                    ),
                }
            }
            (Tag::Boxed, Tag::EmptyList) | (Tag::Boxed, Tag::List) => {
                let self_unboxed: &Term = self.unbox_reference();

                match self_unboxed.tag() {
                    Tag::Arity => Ordering::Less,
                    Tag::HeapBinary => Ordering::Greater,
                    self_unboxed_tag => unimplemented!("unboxed {:?} cmp list()", self_unboxed_tag),
                }
            }
            (Tag::LocalPid, Tag::LocalPid) => self.tagged.cmp(&other.tagged),
            (Tag::EmptyList, Tag::Boxed) | (Tag::List, Tag::Boxed) => {
                let other_unboxed: &Term = other.unbox_reference();

                match other_unboxed.tag() {
                    Tag::Arity => Ordering::Greater,
                    Tag::HeapBinary => Ordering::Less,
                    other_unboxed_tag => {
                        unimplemented!("list() cmp unboxed {:?}", other_unboxed_tag)
                    }
                }
            }
            (Tag::EmptyList, Tag::EmptyList) => Ordering::Equal,
            (Tag::EmptyList, Tag::List) => {
                // Empty list is shorter than all lists, so it is lesser.
                Ordering::Less
            }
            (Tag::List, Tag::SmallInteger) => Ordering::Greater,
            (Tag::List, Tag::EmptyList) => {
                // Any list is longer than empty lit
                Ordering::Greater
            }
            (Tag::List, Tag::List) => {
                let self_cons: &Cons = unsafe { self.as_ref_cons_unchecked() };
                let other_cons: &Cons = unsafe { other.as_ref_cons_unchecked() };

                self_cons.cmp_in_process(other_cons, process)
            }
            (self_tag, other_tag) => unimplemented!("{:?} cmp {:?}", self_tag, other_tag),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    mod cmp_in_process {
        use super::*;

        use std::sync::{Arc, RwLock};

        use crate::environment::{self, Environment};

        mod less {
            use super::*;

            #[test]
            fn number_is_less_than_atom() {
                let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
                let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
                let mut process = process_rw_lock.write().unwrap();
                let number_term: Term = 0.into_process(&mut process);
                let atom_term = Term::str_to_atom("0", Existence::DoNotCare, &mut process).unwrap();

                assert_cmp_in_process!(number_term, Ordering::Less, atom_term, process);
                refute_cmp_in_process!(atom_term, Ordering::Less, number_term, process);
            }

            #[test]
            fn atom_is_less_than_tuple() {
                let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
                let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
                let mut process = process_rw_lock.write().unwrap();
                let atom_term = Term::str_to_atom("0", Existence::DoNotCare, &mut process).unwrap();
                let tuple_term = Term::slice_to_tuple(&[], &mut process);

                assert_cmp_in_process!(atom_term, Ordering::Less, tuple_term, process);
                refute_cmp_in_process!(tuple_term, Ordering::Less, atom_term, process);
            }

            #[test]
            fn atom_is_less_than_atom_if_name_is_less_than() {
                let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
                let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
                let mut process = process_rw_lock.write().unwrap();
                let greater_name = "b";
                let greater_term =
                    Term::str_to_atom(greater_name, Existence::DoNotCare, &mut process).unwrap();
                let lesser_name = "a";
                let lesser_term =
                    Term::str_to_atom(lesser_name, Existence::DoNotCare, &mut process).unwrap();

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
                let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
                let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
                let mut process = process_rw_lock.write().unwrap();
                let shorter_tuple = Term::slice_to_tuple(&[], &mut process);
                let longer_tuple =
                    Term::slice_to_tuple(&[0.into_process(&mut process)], &mut process);

                assert_cmp_in_process!(shorter_tuple, Ordering::Less, longer_tuple, process);
                refute_cmp_in_process!(longer_tuple, Ordering::Less, shorter_tuple, process);
            }

            #[test]
            fn same_length_tuples_with_lesser_elements_is_lesser() {
                let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
                let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
                let mut process = process_rw_lock.write().unwrap();
                let lesser_tuple =
                    Term::slice_to_tuple(&[0.into_process(&mut process)], &mut process);
                let greater_tuple =
                    Term::slice_to_tuple(&[1.into_process(&mut process)], &mut process);

                assert_cmp_in_process!(lesser_tuple, Ordering::Less, greater_tuple, process);
                refute_cmp_in_process!(greater_tuple, Ordering::Less, lesser_tuple, process);
            }

            #[test]
            fn tuple_is_less_than_empty_list() {
                let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
                let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
                let mut process = process_rw_lock.write().unwrap();
                let tuple_term = Term::slice_to_tuple(&[], &mut process);
                let empty_list_term = Term::EMPTY_LIST;

                assert_cmp_in_process!(tuple_term, Ordering::Less, empty_list_term, process);
                refute_cmp_in_process!(empty_list_term, Ordering::Less, tuple_term, process);
            }

            #[test]
            fn tuple_is_less_than_list() {
                let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
                let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
                let mut process = process_rw_lock.write().unwrap();
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
                let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
                let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
                let mut process = process_rw_lock.write().unwrap();
                let list_term = Term::cons(
                    0.into_process(&mut process),
                    1.into_process(&mut process),
                    &mut process,
                );
                let equal_list_term = Term::cons(
                    0.into_process(&mut process),
                    1.into_process(&mut process),
                    &mut process,
                );
                let unequal_list_term = Term::cons(
                    1.into_process(&mut process),
                    0.into_process(&mut process),
                    &mut process,
                );

                assert_eq_in_process!(list_term, list_term, process);
                assert_eq_in_process!(equal_list_term, equal_list_term, process);
                assert_ne_in_process!(list_term, unequal_list_term, process);
            }

            #[test]
            fn with_proper_list() {
                let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
                let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
                let mut process = process_rw_lock.write().unwrap();
                let list_term =
                    Term::cons(0.into_process(&mut process), Term::EMPTY_LIST, &mut process);
                let equal_list_term =
                    Term::cons(0.into_process(&mut process), Term::EMPTY_LIST, &mut process);
                let unequal_list_term =
                    Term::cons(1.into_process(&mut process), Term::EMPTY_LIST, &mut process);

                assert_eq_in_process!(list_term, list_term, process);
                assert_eq_in_process!(list_term, equal_list_term, process);
                assert_ne_in_process!(list_term, unequal_list_term, process);
            }

            #[test]
            fn with_nested_list() {
                let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
                let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
                let mut process = process_rw_lock.write().unwrap();
                let list_term = Term::cons(
                    0.into_process(&mut process),
                    Term::cons(1.into_process(&mut process), Term::EMPTY_LIST, &mut process),
                    &mut process,
                );
                let equal_list_term = Term::cons(
                    0.into_process(&mut process),
                    Term::cons(1.into_process(&mut process), Term::EMPTY_LIST, &mut process),
                    &mut process,
                );
                let unequal_list_term = Term::cons(
                    1.into_process(&mut process),
                    Term::cons(0.into_process(&mut process), Term::EMPTY_LIST, &mut process),
                    &mut process,
                );

                assert_eq_in_process!(list_term, list_term, process);
                assert_eq_in_process!(list_term, equal_list_term, process);
                assert_ne_in_process!(list_term, unequal_list_term, process);
            }

            #[test]
            fn with_lists_of_unequal_length() {
                let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
                let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
                let mut process = process_rw_lock.write().unwrap();
                let list_term = Term::cons(
                    0.into_process(&mut process),
                    Term::cons(1.into_process(&mut process), Term::EMPTY_LIST, &mut process),
                    &mut process,
                );
                let equal_list_term = Term::cons(
                    0.into_process(&mut process),
                    Term::cons(1.into_process(&mut process), Term::EMPTY_LIST, &mut process),
                    &mut process,
                );
                let shorter_list_term =
                    Term::cons(0.into_process(&mut process), Term::EMPTY_LIST, &mut process);
                let longer_list_term = Term::cons(
                    0.into_process(&mut process),
                    Term::cons(
                        1.into_process(&mut process),
                        Term::cons(2.into_process(&mut process), Term::EMPTY_LIST, &mut process),
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
                let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
                let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
                let mut process = process_rw_lock.write().unwrap();
                let tuple_term =
                    Term::slice_to_tuple(&[0.into_process(&mut process)], &mut process);
                let equal_term =
                    Term::slice_to_tuple(&[0.into_process(&mut process)], &mut process);
                let unequal_term = Term::slice_to_tuple(
                    &[0.into_process(&mut process), 1.into_process(&mut process)],
                    &mut process,
                );

                assert_eq_in_process!(tuple_term, tuple_term, process);
                assert_eq_in_process!(tuple_term, equal_term, process);
                assert_ne_in_process!(tuple_term, unequal_term, process);
            }

            #[test]
            fn with_heap_binaries_of_unequal_length() {
                let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
                let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
                let mut process = process_rw_lock.write().unwrap();
                let heap_binary_term = Term::slice_to_binary(&[0, 1], &mut process);
                let equal_heap_binary_term = Term::slice_to_binary(&[0, 1], &mut process);
                let shorter_heap_binary_term = Term::slice_to_binary(&[0], &mut process);
                let longer_heap_binary_term = Term::slice_to_binary(&[0, 1, 2], &mut process);

                assert_eq_in_process!(heap_binary_term, heap_binary_term, process);
                assert_eq_in_process!(heap_binary_term, equal_heap_binary_term, process);
                assert_ne_in_process!(heap_binary_term, shorter_heap_binary_term, process);
                assert_ne_in_process!(heap_binary_term, longer_heap_binary_term, process);
            }
        }
    }

    mod is_empty_list {
        use super::*;

        use std::sync::{Arc, RwLock};

        use crate::environment::{self, Environment};

        #[test]
        fn with_atom_is_false() {
            let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
            let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
            let mut process = process_rw_lock.write().unwrap();
            let atom_term = Term::str_to_atom("atom", Existence::DoNotCare, &mut process).unwrap();

            assert_eq!(atom_term.is_empty_list(), false);
        }

        #[test]
        fn with_empty_list_is_true() {
            assert_eq!(Term::EMPTY_LIST.is_empty_list(), true);
        }

        #[test]
        fn with_list_is_false() {
            let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
            let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
            let mut process = process_rw_lock.write().unwrap();
            let head_term = Term::str_to_atom("head", Existence::DoNotCare, &mut process).unwrap();
            let list_term = Term::cons(head_term, Term::EMPTY_LIST, &mut process);

            assert_eq!(list_term.is_empty_list(), false);
        }

        #[test]
        fn with_small_integer_is_false() {
            let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
            let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
            let mut process = process_rw_lock.write().unwrap();
            let small_integer_term = small_integer_term(&mut process, 0);

            assert_eq!(small_integer_term.is_empty_list(), false);
        }

        #[test]
        fn with_tuple_is_false() {
            let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
            let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
            let mut process = process_rw_lock.write().unwrap();
            let tuple_term = tuple_term(&mut process);

            assert_eq!(tuple_term.is_empty_list(), false);
        }

        #[test]
        fn with_heap_binary_is_false() {
            let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
            let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
            let mut process = process_rw_lock.write().unwrap();
            let heap_binary_term = Term::slice_to_binary(&[], &mut process);

            assert_eq!(heap_binary_term.is_empty_list(), false);
        }
    }

    fn small_integer_term(mut process: &mut Process, signed_size: isize) -> Term {
        signed_size.into_process(&mut process)
    }

    fn list_term(mut process: &mut Process) -> Term {
        let head_term = Term::str_to_atom("head", Existence::DoNotCare, &mut process).unwrap();
        Term::cons(head_term, Term::EMPTY_LIST, process)
    }

    fn tuple_term(process: &mut Process) -> Term {
        Term::slice_to_tuple(&[], process)
    }
}
