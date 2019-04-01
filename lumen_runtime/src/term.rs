#![cfg_attr(not(test), allow(dead_code))]

use std::cmp::Ordering::{self, *};
use std::convert::{TryFrom, TryInto};
use std::fmt::{self, Debug, Display};
use std::hash::{Hash, Hasher};
use std::mem::size_of;
use std::str::Chars;
use std::sync::Arc;

use num_bigint::BigInt;
use num_traits::cast::ToPrimitive;

use liblumen_arena::TypedArena;

use crate::atom::{self, Encoding, Existence, Existence::*, Index};
use crate::binary::{self, heap, sub, Part, PartToList};
use crate::exception::{self, Exception};
use crate::float::Float;
use crate::integer::Integer::{self, Big, Small};
use crate::integer::{big, small};
use crate::list::Cons;
use crate::map::Map;
use crate::process::{self, IntoProcess, Process, TryFromInProcess, TryIntoInProcess};
use crate::reference::local;
use crate::tuple::Tuple;

pub mod external_format;

#[derive(Debug, PartialEq)]
// MUST be `repr(u*)` so that size and layout is fixed for direct LLVM IR checking of tags
#[repr(usize)]
pub enum Tag {
    Arity = 0b0000_00,
    BinaryAggregate = 0b0001_00,
    BigInteger = 0b0010_00,
    LocalReference = 0b0100_00,
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

use self::Tag::*;

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
                0b0000_00 => Ok(Arity),
                0b0001_00 => Ok(BinaryAggregate),
                0b0010_00 => Ok(BigInteger),
                0b0100_00 => Ok(LocalReference),
                0b0101_00 => Ok(Function),
                0b0110_00 => Ok(Float),
                0b0111_00 => Ok(Export),
                0b1000_00 => Ok(ReferenceCountedBinary),
                0b1001_00 => Ok(HeapBinary),
                0b1010_00 => Ok(Subbinary),
                0b1100_00 => Ok(ExternalPid),
                0b1101_00 => Ok(ExternalPort),
                0b1110_00 => Ok(ExternalReference),
                0b1111_00 => Ok(Map),
                tag => Err(TagError { tag, bit_count: 6 }),
            },
            0b01 => Ok(List),
            0b10 => Ok(Boxed),
            0b11 => match bits & IMMEDIATE_PRIMARY_TAG_MASK {
                0b00_11 => Ok(LocalPid),
                0b01_11 => Ok(LocalPort),
                0b10_11 => match bits & IMMEDIATE_IMMEDIATE_PRIMARY_TAG_MASK {
                    0b00_10_11 => Ok(Atom),
                    0b01_10_11 => Ok(CatchPointer),
                    0b11_10_11 => Ok(EmptyList),
                    tag => Err(TagError { tag, bit_count: 6 }),
                },
                0b11_11 => Ok(SmallInteger),
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
        tagged: EmptyList as usize,
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
            tagged: (arity << Tag::ARITY_BIT_COUNT) | Arity as usize,
        }
    }

    pub unsafe fn arity_to_integer(&self, mut process: &mut Process) -> Term {
        self.arity_to_usize().into_process(&mut process)
    }

    pub unsafe fn arity_to_usize(&self) -> usize {
        ((self.tagged & !(Arity as usize)) >> Tag::ARITY_BIT_COUNT)
    }

    pub unsafe fn as_ref_cons_unchecked(&self) -> &'static Cons {
        let untagged = self.tagged & !(List as usize);
        let pointer = untagged as *const Term as *const Cons;

        &*pointer
    }

    pub fn atom_to_encoding(&self) -> Result<Encoding, Exception> {
        match self.tag() {
            Atom => {
                let unicode_atom = Term::str_to_atom("unicode", DoNotCare).unwrap();
                let tagged = self.tagged;

                if tagged == unicode_atom.tagged {
                    Ok(Encoding::Unicode)
                } else {
                    let utf8_atom = Term::str_to_atom("utf8", DoNotCare).unwrap();

                    if tagged == utf8_atom.tagged {
                        Ok(Encoding::Utf8)
                    } else {
                        let latin1_atom = Term::str_to_atom("latin1", DoNotCare).unwrap();

                        if tagged == latin1_atom.tagged {
                            Ok(Encoding::Latin1)
                        } else {
                            Err(bad_argument!())
                        }
                    }
                }
            }
            _ => Err(bad_argument!()),
        }
    }

    pub unsafe fn atom_to_index(&self) -> Index {
        Index(self.tagged >> Tag::ATOM_BIT_COUNT)
    }

    pub unsafe fn atom_to_string(&self) -> Arc<String> {
        atom::index_to_string(self.atom_to_index()).unwrap()
    }

    pub fn alloc_slice(slice: &[Term], term_arena: &mut TypedArena<Term>) -> *const Term {
        term_arena.alloc_slice(slice).as_ptr()
    }

    pub fn byte(&self, index: usize) -> u8 {
        match self.tag() {
            Boxed => {
                let unboxed: &Term = self.unbox_reference();

                match unboxed.tag() {
                    HeapBinary => {
                        let heap_binary: &heap::Binary = self.unbox_reference();

                        heap_binary.byte(index)
                    }
                    ReferenceCountedBinary => unimplemented!(),
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
        let pointer_bits = process.cons(head, tail) as *const Cons as usize;

        assert_eq!(
            pointer_bits & Tag::LIST_MASK,
            0,
            "List tag bit ({:#b}) would overwrite pointer bits ({:#b})",
            Tag::LIST_MASK,
            pointer_bits
        );

        Term {
            tagged: pointer_bits | (List as usize),
        }
    }

    pub fn external_pid(
        node: usize,
        number: usize,
        serial: usize,
        process: &mut Process,
    ) -> exception::Result {
        if (number <= process::identifier::NUMBER_MAX)
            && (serial <= process::identifier::SERIAL_MAX)
        {
            Ok(Term::box_reference(
                process.external_pid(node, number, serial),
            ))
        } else {
            Err(bad_argument!())
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
            tagged: ((byte_count << Tag::HEAP_BINARY_BIT_COUNT) as usize) | (HeapBinary as usize),
        }
    }

    pub unsafe fn heap_binary_to_byte_count(&self) -> usize {
        (self.tagged & !(HeapBinary as usize)) >> Tag::HEAP_BINARY_BIT_COUNT
    }

    pub fn local_pid(number: usize, serial: usize) -> exception::Result {
        if (number <= process::identifier::NUMBER_MAX)
            && (serial <= process::identifier::SERIAL_MAX)
        {
            Ok(unsafe { Self::local_pid_unchecked(number, serial) })
        } else {
            Err(bad_argument!())
        }
    }

    pub unsafe fn local_pid_unchecked(number: usize, serial: usize) -> Term {
        Term {
            tagged: (serial << (process::identifier::NUMBER_BIT_COUNT + Tag::LOCAL_PID_BIT_COUNT))
                | (number << (Tag::LOCAL_PID_BIT_COUNT))
                | (LocalPid as usize),
        }
    }

    pub fn local_reference(process: &mut Process) -> Term {
        Term::box_reference(process.local_reference())
    }

    pub fn tag(&self) -> Tag {
        match (self.tagged as usize).try_into() {
            Ok(tag) => tag,
            Err(tag_error) => panic!(tag_error),
        }
    }

    pub fn is_empty_list(&self) -> bool {
        (self.tag() == EmptyList)
    }

    pub fn pid(
        node: usize,
        number: usize,
        serial: usize,
        process: &mut Process,
    ) -> exception::Result {
        if node == 0 {
            Self::local_pid(number, serial)
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

    pub fn str_to_atom(name: &str, existence: Existence) -> Option<Term> {
        atom::str_to_index(name, existence).map(|atom_index| atom_index.into())
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

    pub fn vec_to_list(vec: &Vec<Term>, initial_tail: Term, mut process: &mut Process) -> Term {
        vec.iter().rfold(initial_tail, |acc, element| {
            Term::cons(element.clone(), acc, &mut process)
        })
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
            tagged: pointer_bits | (Boxed as usize),
        }
    }

    pub fn unbox_reference<T>(&self) -> &'static T {
        const TAG_BOXED: usize = Boxed as usize;

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
            SmallInteger,
            "Term ({:#b}) is not a small integer",
            self.tagged
        );

        ((self.tagged & !(SmallInteger as usize)) >> Tag::SMALL_INTEGER_BIT_COUNT)
    }

    /// Only call if verified `tag` is `SmallInteger`.
    pub unsafe fn small_integer_is_negative(&self) -> bool {
        self.tagged & Term::SMALL_INTEGER_SIGN_BIT_MASK == Term::SMALL_INTEGER_SIGN_BIT_MASK
    }

    pub const unsafe fn isize_to_small_integer(i: isize) -> Term {
        Term {
            tagged: ((i << Tag::SMALL_INTEGER_BIT_COUNT) as usize) | (SmallInteger as usize),
        }
    }

    pub unsafe fn small_integer_to_isize(&self) -> isize {
        (self.tagged as isize) >> Tag::SMALL_INTEGER_BIT_COUNT
    }

    pub fn u64_to_local_reference(number: u64, process: &mut Process) -> Term {
        Term::box_reference(process.u64_to_local_reference(number))
    }
}

impl Debug for Term {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self.tag() {
            Arity => write!(f, "Term::arity({})", unsafe { self.arity_to_usize() }),
            Atom => write!(f, "Term::str_to_atom(\"{}\", DoNotCare).unwrap()", unsafe {
                self.atom_to_string()
            }),
            Boxed => {
                let unboxed: &Term = self.unbox_reference();

                match unboxed.tag() {
                    Arity => {
                        let tuple: &Tuple = self.unbox_reference();

                        write!(f, "Term::slice_to_tuple(&[")?;

                        let mut iter = tuple.iter();

                        if let Some(first_element) = iter.next() {
                            write!(f, "{:?}", first_element)?;

                            for element in iter {
                                write!(f, ", {:?}", element)?;
                            }
                        }

                        write!(f, "], &mut process)")
                    }
                    BigInteger => {
                        let big_integer: &big::Integer = self.unbox_reference();

                        write!(
                            f,
                            "BigInt::parse_bytes(b\"{}\", 10).unwrap().into_process(&mut process)",
                            big_integer.inner
                        )
                    }
                    ExternalPid => {
                        let external_pid: &process::identifier::External = self.unbox_reference();

                        write!(
                            f,
                            "Term::external_pid({:?}, {:?}, {:?}, &mut process)",
                            external_pid.node, external_pid.number, external_pid.serial
                        )
                    }
                    Float => {
                        let float: &Float = self.unbox_reference();

                        write!(f, "{:?}_f64.into_process(&mut process)", float.inner)
                    }
                    HeapBinary => {
                        let binary: &heap::Binary = self.unbox_reference();

                        write!(f, "Term::slice_to_binary(&[")?;

                        let mut iter = binary.iter();

                        if let Some(first_byte) = iter.next() {
                            write!(f, "{:?}", first_byte)?;

                            for byte in iter {
                                write!(f, ", {:?}", byte)?;
                            }
                        }

                        write!(f, "], &mut process)")
                    }
                    LocalReference => {
                        let local_reference: &local::Reference = self.unbox_reference();

                        write!(
                            f,
                            "Term::u64_to_local_reference({:?}, &mut process)",
                            local_reference.number
                        )
                    }
                    Subbinary => {
                        let subbinary: &sub::Binary = self.unbox_reference();

                        write!(
                            f,
                            "Term::subbinary({:?}, {:?}, {:?}, {:?}, {:?}, &mut process)",
                            subbinary.original,
                            subbinary.byte_offset,
                            subbinary.bit_offset,
                            subbinary.byte_count,
                            subbinary.bit_count
                        )
                    }
                    unboxed_tag => unimplemented!("unboxed {:?}", unboxed_tag),
                }
            }
            EmptyList => write!(f, "Term::EMPTY_LIST"),
            List => {
                let cons: &Cons = unsafe { self.as_ref_cons_unchecked() };

                write!(
                    f,
                    "Term::cons({:?}, {:?}, &mut process)",
                    cons.head(),
                    cons.tail()
                )
            }
            SmallInteger => write!(f, "{:?}.into_process(&mut process)", isize::from(self)),
            _ => write!(
                f,
                "Term {{ tagged: 0b{tagged:0bit_count$b} }}",
                tagged = self.tagged,
                bit_count = std::mem::size_of::<usize>() * 8
            ),
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

impl From<bool> for Term {
    fn from(b: bool) -> Term {
        Term::str_to_atom(&b.to_string(), DoNotCare).unwrap()
    }
}

impl From<u8> for Term {
    fn from(u: u8) -> Term {
        let untagged: isize = u as isize;

        Term {
            tagged: ((untagged << Tag::SMALL_INTEGER_BIT_COUNT) as usize) | (SmallInteger as usize),
        }
    }
}

impl From<&Term> for isize {
    fn from(term: &Term) -> Self {
        match term.tag() {
            SmallInteger => (term.tagged as isize) >> Tag::SMALL_INTEGER_BIT_COUNT,
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
            Atom | EmptyList | LocalPid | SmallInteger => self.tagged.hash(state),
            Boxed => {
                let unboxed: &Term = self.unbox_reference();

                match unboxed.tag() {
                    Arity => {
                        let tuple: &Tuple = self.unbox_reference();

                        tuple.hash(state)
                    }
                    BigInteger => {
                        let big_integer: &big::Integer = self.unbox_reference();

                        big_integer.hash(state)
                    }
                    ExternalPid => {
                        let external_pid: &process::identifier::External = self.unbox_reference();

                        external_pid.hash(state)
                    }
                    Float => {
                        let float: &Float = self.unbox_reference();

                        float.hash(state)
                    }
                    HeapBinary => {
                        let heap_binary: &heap::Binary = self.unbox_reference();

                        heap_binary.hash(state)
                    }
                    LocalReference => {
                        let local_reference: &local::Reference = self.unbox_reference();

                        local_reference.hash(state)
                    }
                    Subbinary => {
                        let subbinary: &sub::Binary = self.unbox_reference();

                        subbinary.hash(state)
                    }
                    unboxed_tag => unimplemented!("unboxed tag {:?}", unboxed_tag),
                }
            }
            List => {
                let cons: &Cons = unsafe { self.as_ref_cons_unchecked() };

                cons.hash(state)
            }
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
                    | (SmallInteger as usize),
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
                tagged: (atom_index.0 << Tag::ATOM_BIT_COUNT) | (Atom as usize),
            }
        } else {
            panic!("index ({}) in atom table exceeds max index that can be tagged as an atom in a Term ({})", atom_index.0, MAX_ATOM_INDEX)
        }
    }
}

impl Ord for Term {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl<'a> Part<'a, Term, Term, Term> for heap::Binary {
    fn part(&'a self, start: Term, length: Term, mut process: &mut Process) -> exception::Result {
        let start_usize: usize = start.try_into()?;
        let length_isize: isize = length.try_into()?;

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
        let start_usize: usize = start.try_into()?;
        let length_isize: isize = length.try_into()?;
        let new_subbinary = self.part(start_usize, length_isize, &mut process)?;

        Ok(new_subbinary.into())
    }
}

impl PartToList<Term, Term> for heap::Binary {
    fn part_to_list(&self, start: Term, length: Term, process: &mut Process) -> exception::Result {
        let start_usize: usize = start.try_into()?;
        let length_isize: isize = length.try_into()?;

        self.part_to_list(start_usize, length_isize, process)
    }
}

impl PartToList<Term, Term> for sub::Binary {
    fn part_to_list(&self, start: Term, length: Term, process: &mut Process) -> exception::Result {
        let start_usize: usize = start.try_into()?;
        let length_isize: isize = length.try_into()?;

        self.part_to_list(start_usize, length_isize, process)
    }
}

impl PartialEq for Term {
    fn eq(&self, other: &Term) -> bool {
        match (self.tag(), other.tag()) {
            (Atom, Atom) | (LocalPid, LocalPid) | (SmallInteger, SmallInteger) => {
                self.tagged == other.tagged
            }
            (Boxed, Boxed) => {
                let self_unboxed: &Term = self.unbox_reference();
                let other_unboxed: &Term = other.unbox_reference();

                match (self_unboxed.tag(), other_unboxed.tag()) {
                    (Arity, Arity) => {
                        let self_tuple: &Tuple = self.unbox_reference();
                        let other_tuple: &Tuple = other.unbox_reference();

                        self_tuple == other_tuple
                    }
                    (BigInteger, BigInteger) => {
                        let self_big_integer: &big::Integer = self.unbox_reference();
                        let other_big_integer: &big::Integer = other.unbox_reference();

                        self_big_integer == other_big_integer
                    }
                    (ExternalPid, ExternalPid) => {
                        let self_external_pid: &process::identifier::External =
                            self.unbox_reference();
                        let other_external_pid: &process::identifier::External =
                            other.unbox_reference();

                        self_external_pid == other_external_pid
                    }
                    (Float, Float) => {
                        let self_float: &Float = self.unbox_reference();
                        let other_float: &Float = other.unbox_reference();

                        self_float == other_float
                    }
                    (HeapBinary, HeapBinary) => {
                        let self_heap_binary: &heap::Binary = self.unbox_reference();
                        let other_heap_binary: &heap::Binary = other.unbox_reference();

                        self_heap_binary == other_heap_binary
                    }
                    (HeapBinary, Subbinary) => {
                        let self_heap_binary: &heap::Binary = self.unbox_reference();
                        let other_subbinary: &sub::Binary = other.unbox_reference();

                        self_heap_binary == other_subbinary
                    }
                    (LocalReference, LocalReference) => {
                        let self_local_reference: &local::Reference = self.unbox_reference();
                        let other_local_reference: &local::Reference = other.unbox_reference();

                        self_local_reference == other_local_reference
                    }
                    (Map, Map) => {
                        let self_map: &Map = self.unbox_reference();
                        let other_map: &Map = other.unbox_reference();

                        self_map == other_map
                    }
                    (Subbinary, HeapBinary) => {
                        let self_subbinary: &sub::Binary = self.unbox_reference();
                        let other_heap_binary: &heap::Binary = other.unbox_reference();

                        self_subbinary == other_heap_binary
                    }
                    (Subbinary, Subbinary) => {
                        let self_subbinary: &sub::Binary = self.unbox_reference();
                        let other_subbinary: &sub::Binary = other.unbox_reference();

                        self_subbinary == other_subbinary
                    }
                    (self_unboxed_tag, other_unboxed_tag)
                        if self_unboxed_tag == other_unboxed_tag =>
                    {
                        unimplemented!(
                            "unboxed {:?} == unboxed {:?}",
                            self_unboxed_tag,
                            other_unboxed_tag
                        )
                    }
                    _ => false,
                }
            }
            (EmptyList, EmptyList) => true,
            (List, List) => {
                let self_cons: &Cons = unsafe { self.as_ref_cons_unchecked() };
                let other_cons: &Cons = unsafe { other.as_ref_cons_unchecked() };

                self_cons == other_cons
            }
            (self_tag, other_tag) if self_tag == other_tag => {
                unimplemented!("{:?} == {:?}", self_tag, other_tag)
            }
            _ => false,
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
impl PartialOrd for Term {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        // in ascending order
        match (self.tag(), other.tag()) {
            (Arity, Arity) | (HeapBinary, HeapBinary) => self.tagged.partial_cmp(&other.tagged),
            (SmallInteger, SmallInteger) => {
                if self.tagged == other.tagged {
                    Some(Equal)
                } else {
                    let self_isize: isize = unsafe { self.small_integer_to_isize() };
                    let other_isize: isize = unsafe { other.small_integer_to_isize() };

                    self_isize.partial_cmp(&other_isize)
                }
            }
            (SmallInteger, Boxed) => {
                let other_unboxed: &Term = other.unbox_reference();

                match other_unboxed.tag() {
                    BigInteger => {
                        let self_big_int: BigInt = unsafe { self.small_integer_to_isize() }.into();

                        let other_big_integer: &big::Integer = other.unbox_reference();
                        let other_big_int = &other_big_integer.inner;

                        self_big_int.partial_cmp(other_big_int)
                    }
                    other_unboxed_tag => {
                        unimplemented!("SmallInteger cmp unboxed {:?}", other_unboxed_tag)
                    }
                }
            }
            (SmallInteger, Atom) => Some(Less),
            (SmallInteger, List) => Some(Less),
            (Atom, SmallInteger) => Some(Greater),
            (Atom, Atom) => {
                if self.tagged == other.tagged {
                    Some(Equal)
                } else {
                    let self_index = unsafe { self.atom_to_index() };
                    let other_index = unsafe { other.atom_to_index() };

                    self_index.partial_cmp(&other_index)
                }
            }
            (Atom, Boxed) => {
                let other_unboxed: &Term = other.unbox_reference();

                match other_unboxed.tag() {
                    Arity => Some(Less),
                    Subbinary => Some(Less),
                    other_unboxed_tag => unimplemented!("Atom cmp unboxed {:?}", other_unboxed_tag),
                }
            }
            (Boxed, SmallInteger) => {
                let self_unboxed: &Term = self.unbox_reference();

                match self_unboxed.tag() {
                    BigInteger => {
                        let self_big_integer: &big::Integer = self.unbox_reference();
                        let self_big_int = &self_big_integer.inner;

                        let other_big_int: BigInt =
                            unsafe { other.small_integer_to_isize() }.into();

                        self_big_int.partial_cmp(&other_big_int)
                    }
                    Subbinary => Some(Greater),
                    self_unboxed_tag => {
                        unimplemented!("unboxed {:?} cmp SmallInteger", self_unboxed_tag)
                    }
                }
            }
            (Boxed, Atom) => {
                let self_unboxed: &Term = self.unbox_reference();

                match self_unboxed.tag() {
                    Arity => Some(Greater),
                    self_unboxed_tag => unimplemented!("unboxed {:?} cmp Atom", self_unboxed_tag),
                }
            }
            (Boxed, Boxed) => {
                let self_unboxed: &Term = self.unbox_reference();
                let other_unboxed: &Term = other.unbox_reference();

                // in ascending order
                match (self_unboxed.tag(), other_unboxed.tag()) {
                    (BigInteger, BigInteger) => {
                        let self_big_integer: &big::Integer = self.unbox_reference();
                        let other_big_integer: &big::Integer = other.unbox_reference();

                        self_big_integer.inner.partial_cmp(&other_big_integer.inner)
                    }
                    (Float, Float) => {
                        let self_float: &Float = self.unbox_reference();
                        let self_inner = self_float.inner;

                        let other_float: &Float = other.unbox_reference();
                        let other_inner = other_float.inner;

                        // Erlang doesn't support the floats that can't be compared
                        self_inner.partial_cmp(&other_inner)
                    }
                    (LocalReference, LocalReference) => {
                        let self_local_reference: &local::Reference = self.unbox_reference();
                        let other_local_reference: &local::Reference = other.unbox_reference();

                        self_local_reference
                            .number
                            .partial_cmp(&other_local_reference.number)
                    }
                    (ExternalPid, ExternalPid) => {
                        let self_external_pid: &process::identifier::External =
                            self.unbox_reference();
                        let other_external_pid: &process::identifier::External =
                            other.unbox_reference();

                        self_external_pid.partial_cmp(other_external_pid)
                    }
                    (Arity, Arity) => {
                        let self_tuple: &Tuple = self.unbox_reference();
                        let other_tuple: &Tuple = other.unbox_reference();

                        self_tuple.partial_cmp(other_tuple)
                    }
                    (Arity, Float) => Some(Greater),
                    (Map, Map) => {
                        let self_map: &Map = self.unbox_reference();
                        let other_map: &Map = other.unbox_reference();

                        self_map.partial_cmp(other_map)
                    }
                    (HeapBinary, HeapBinary) => {
                        let self_binary: &heap::Binary = self.unbox_reference();
                        let other_binary: &heap::Binary = other.unbox_reference();

                        self_binary.partial_cmp(other_binary)
                    }
                    (Subbinary, HeapBinary) => {
                        let self_subbinary: &sub::Binary = self.unbox_reference();
                        let other_heap_binary: &heap::Binary = other.unbox_reference();

                        self_subbinary.partial_cmp(other_heap_binary)
                    }
                    (Subbinary, Subbinary) => {
                        let self_subbinary: &sub::Binary = self.unbox_reference();
                        let other_subbinary: &sub::Binary = other.unbox_reference();

                        self_subbinary.partial_cmp(other_subbinary)
                    }
                    (self_unboxed_tag, other_unboxed_tag) => unimplemented!(
                        "unboxed {:?} cmp unboxed {:?}",
                        self_unboxed_tag,
                        other_unboxed_tag
                    ),
                }
            }
            (Boxed, EmptyList) | (Boxed, List) => {
                let self_unboxed: &Term = self.unbox_reference();

                match self_unboxed.tag() {
                    Arity => Some(Less),
                    HeapBinary => Some(Greater),
                    self_unboxed_tag => unimplemented!("unboxed {:?} cmp list()", self_unboxed_tag),
                }
            }
            (LocalPid, LocalPid) => self.tagged.partial_cmp(&other.tagged),
            (EmptyList, Boxed) | (List, Boxed) => {
                let other_unboxed: &Term = other.unbox_reference();

                match other_unboxed.tag() {
                    Arity => Some(Greater),
                    HeapBinary => Some(Less),
                    other_unboxed_tag => {
                        unimplemented!("list() cmp unboxed {:?}", other_unboxed_tag)
                    }
                }
            }
            (EmptyList, EmptyList) => Some(Equal),
            (EmptyList, List) => {
                // Empty list is shorter than all lists, so it is lesser.
                Some(Less)
            }
            (List, SmallInteger) => Some(Greater),
            (List, Atom) => Some(Greater),
            (List, EmptyList) => {
                // Any list is longer than empty list
                Some(Greater)
            }
            (List, List) => {
                let self_cons: &Cons = unsafe { self.as_ref_cons_unchecked() };
                let other_cons: &Cons = unsafe { other.as_ref_cons_unchecked() };

                self_cons.partial_cmp(other_cons)
            }
            (self_tag, other_tag) => unimplemented!("{:?} cmp {:?}", self_tag, other_tag),
        }
    }
}

impl TryFrom<Term> for BigInt {
    type Error = Exception;

    fn try_from(term: Term) -> Result<BigInt, Exception> {
        match term.tag() {
            SmallInteger => {
                let term_isize = (term.tagged as isize) >> Tag::SMALL_INTEGER_BIT_COUNT;

                Ok(term_isize.into())
            }
            Boxed => {
                let unboxed: &Term = term.unbox_reference();

                match unboxed.tag() {
                    BigInteger => {
                        let big_integer: &big::Integer = term.unbox_reference();

                        Ok(big_integer.inner.clone())
                    }
                    _ => Err(bad_argument!()),
                }
            }
            _ => Err(bad_argument!()),
        }
    }
}

impl TryFrom<Term> for char {
    type Error = Exception;

    fn try_from(term: Term) -> Result<char, Exception> {
        let term_u32: u32 = term.try_into()?;

        match std::char::from_u32(term_u32) {
            Some(c) => Ok(c),
            None => Err(bad_argument!()),
        }
    }
}

impl TryFrom<Term> for isize {
    type Error = Exception;

    fn try_from(term: Term) -> Result<isize, Exception> {
        match term.tag() {
            SmallInteger => {
                let term_isize = unsafe { term.small_integer_to_isize() };

                Ok(term_isize)
            }
            _ => Err(bad_argument!()),
        }
    }
}

impl TryFrom<Term> for u32 {
    type Error = Exception;

    fn try_from(term: Term) -> Result<u32, Exception> {
        match term.tag() {
            SmallInteger => {
                let term_isize = unsafe { term.small_integer_to_isize() };

                term_isize.try_into().map_err(|_| bad_argument!())
            }
            Boxed => {
                let unboxed: &Term = term.unbox_reference();

                match unboxed.tag() {
                    BigInteger => {
                        let big_integer: &big::Integer = term.unbox_reference();

                        // does not implement `to_u32` directly
                        match big_integer.inner.to_u64() {
                            Some(term_u64) => term_u64.try_into().map_err(|_| bad_argument!()),
                            None => Err(bad_argument!()),
                        }
                    }
                    _ => Err(bad_argument!()),
                }
            }
            _ => Err(bad_argument!()),
        }
    }
}

impl TryFrom<Term> for u64 {
    type Error = Exception;

    fn try_from(term: Term) -> Result<u64, Exception> {
        match term.tag() {
            Boxed => {
                let unboxed: &Term = term.unbox_reference();

                match unboxed.tag() {
                    LocalReference => {
                        let local_reference: &local::Reference = term.unbox_reference();

                        Ok(local_reference.number)
                    }
                    _ => Err(bad_argument!()),
                }
            }
            _ => Err(bad_argument!()),
        }
    }
}

impl TryFrom<Term> for usize {
    type Error = Exception;

    fn try_from(term: Term) -> Result<usize, Exception> {
        match term.tag() {
            SmallInteger => {
                let term_isize = unsafe { term.small_integer_to_isize() };

                if term_isize < 0 {
                    Err(bad_argument!())
                } else {
                    Ok(term_isize as usize)
                }
            }
            _ => Err(bad_argument!()),
        }
    }
}

impl TryFrom<Term> for &'static Cons {
    type Error = Exception;

    fn try_from(term: Term) -> Result<&'static Cons, Exception> {
        match term.tag() {
            List => Ok(unsafe { term.as_ref_cons_unchecked() }),
            _ => Err(bad_argument!()),
        }
    }
}

impl TryFromInProcess<Term> for &'static Map {
    fn try_from_in_process(
        term: Term,
        mut process: &mut Process,
    ) -> Result<&'static Map, Exception> {
        match term.tag() {
            Boxed => {
                let unboxed: &Term = term.unbox_reference();

                match unboxed.tag() {
                    Map => {
                        let map: &Map = term.unbox_reference();

                        Some(map)
                    }
                    _ => None,
                }
            }
            _ => None,
        }
        .ok_or_else(|| {
            let badmap = Term::str_to_atom("badmap", DoNotCare).unwrap();
            let reason = Term::slice_to_tuple(&[badmap, term], &mut process);

            error!(reason)
        })
    }
}

impl TryFrom<Term> for String {
    type Error = Exception;

    fn try_from(term: Term) -> Result<String, Exception> {
        match term.tag() {
            Boxed => {
                let unboxed: &Term = term.unbox_reference();

                match unboxed.tag() {
                    HeapBinary => {
                        let heap_binary: &heap::Binary = term.unbox_reference();

                        heap_binary.try_into()
                    }
                    Subbinary => {
                        let subbinary: &sub::Binary = term.unbox_reference();

                        subbinary.try_into()
                    }
                    // TODO ReferenceCountedBinary
                    _ => Err(bad_argument!()),
                }
            }
            _ => Err(bad_argument!()),
        }
    }
}

impl TryFromInProcess<Term> for &'static Tuple {
    fn try_from_in_process(
        term: Term,
        mut process: &mut Process,
    ) -> Result<&'static Tuple, Exception> {
        match term.tag() {
            Boxed => term
                .unbox_reference::<Term>()
                .try_into_in_process(&mut process),
            _ => Err(bad_argument!()),
        }
    }
}

impl TryFrom<&Term> for BigInt {
    type Error = Exception;

    fn try_from(term_ref: &Term) -> Result<BigInt, Exception> {
        (*term_ref).try_into()
    }
}

impl<'a> TryFromInProcess<&'a Term> for &'a Tuple {
    fn try_from_in_process(
        term: &'a Term,
        mut process: &mut Process,
    ) -> Result<&'a Tuple, Exception> {
        match term.tag() {
            Arity => {
                let pointer = term as *const Term as *const Tuple;
                Ok(unsafe { pointer.as_ref() }.unwrap())
            }
            Boxed => term
                .unbox_reference::<Term>()
                .try_into_in_process(&mut process),
            _ => Err(bad_argument!()),
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
                let atom_term = Term::str_to_atom("0", DoNotCare).unwrap();

                assert!(number_term < atom_term);
                assert!(!(atom_term < number_term));
            }

            #[test]
            fn atom_is_less_than_tuple() {
                let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
                let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
                let mut process = process_rw_lock.write().unwrap();
                let atom_term = Term::str_to_atom("0", DoNotCare).unwrap();
                let tuple_term = Term::slice_to_tuple(&[], &mut process);

                assert!(atom_term < tuple_term);
                assert!(!(tuple_term < atom_term));
            }

            #[test]
            fn atom_is_less_than_atom_if_name_is_less_than() {
                let greater_name = "bbbbbbbbbbb";
                let greater_term = Term::str_to_atom(greater_name, DoNotCare).unwrap();
                let lesser_name = "aaaaaaaaaa";
                let lesser_term = Term::str_to_atom(lesser_name, DoNotCare).unwrap();

                assert!(lesser_name < greater_name);
                assert!(lesser_term < greater_term);
                // it isn't just comparing the atom index
                assert!(!(lesser_term.tagged < greater_term.tagged));

                assert!(!(greater_name < lesser_name));
                assert!(!(greater_term < lesser_term));
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

                assert!(shorter_tuple < longer_tuple);
                assert!(!(longer_tuple < shorter_tuple));
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

                assert!(lesser_tuple < greater_tuple);
                assert!(!(greater_tuple < lesser_tuple));
            }

            #[test]
            fn tuple_is_less_than_empty_list() {
                let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
                let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
                let mut process = process_rw_lock.write().unwrap();
                let tuple_term = Term::slice_to_tuple(&[], &mut process);
                let empty_list_term = Term::EMPTY_LIST;

                assert!(tuple_term < empty_list_term);
                assert!(!(empty_list_term < tuple_term));
            }

            #[test]
            fn tuple_is_less_than_list() {
                let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
                let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
                let mut process = process_rw_lock.write().unwrap();
                let tuple_term = Term::slice_to_tuple(&[], &mut process);
                let list_term = list_term(&mut process);

                assert!(tuple_term < list_term);
                assert!(!(list_term < tuple_term));
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

                assert_eq!(list_term, list_term);
                assert_eq!(equal_list_term, equal_list_term);
                assert_ne!(list_term, unequal_list_term);
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

                assert_eq!(list_term, list_term);
                assert_eq!(list_term, equal_list_term);
                assert_ne!(list_term, unequal_list_term);
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

                assert_eq!(list_term, list_term);
                assert_eq!(list_term, equal_list_term);
                assert_ne!(list_term, unequal_list_term);
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

                assert_eq!(list_term, list_term);
                assert_eq!(list_term, equal_list_term);
                assert_ne!(list_term, shorter_list_term);
                assert_ne!(list_term, longer_list_term);
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

                assert_eq!(tuple_term, tuple_term);
                assert_eq!(tuple_term, equal_term);
                assert_ne!(tuple_term, unequal_term);
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

                assert_eq!(heap_binary_term, heap_binary_term);
                assert_eq!(heap_binary_term, equal_heap_binary_term);
                assert_ne!(heap_binary_term, shorter_heap_binary_term);
                assert_ne!(heap_binary_term, longer_heap_binary_term);
            }
        }
    }

    mod is_empty_list {
        use super::*;

        use std::sync::{Arc, RwLock};

        use crate::environment::{self, Environment};

        #[test]
        fn with_atom_is_false() {
            let atom_term = Term::str_to_atom("atom", DoNotCare).unwrap();

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
            let head_term = Term::str_to_atom("head", DoNotCare).unwrap();
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

    mod u64_to_local_reference {
        use super::*;

        use std::sync::{Arc, RwLock};

        use crate::environment::{self, Environment};

        #[test]
        fn round_trips_with_local_reference() {
            let environment_rw_lock: Arc<RwLock<Environment>> = Default::default();
            let process_rw_lock = environment::process(Arc::clone(&environment_rw_lock));
            let mut process = process_rw_lock.write().unwrap();

            let original = Term::local_reference(&mut process);
            let original_u64: u64 = original.try_into().unwrap();
            let from_u64 = Term::u64_to_local_reference(original_u64, &mut process);

            assert_eq!(original, from_u64);
        }
    }

    fn small_integer_term(mut process: &mut Process, signed_size: isize) -> Term {
        signed_size.into_process(&mut process)
    }

    fn list_term(process: &mut Process) -> Term {
        let head_term = Term::str_to_atom("head", DoNotCare).unwrap();
        Term::cons(head_term, Term::EMPTY_LIST, process)
    }

    fn tuple_term(process: &mut Process) -> Term {
        Term::slice_to_tuple(&[], process)
    }
}
