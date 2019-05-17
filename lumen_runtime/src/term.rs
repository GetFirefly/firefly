use std::cmp::Ordering::{self, *};
use std::convert::{TryFrom, TryInto};
#[cfg(test)]
use std::fmt::Display;
use std::fmt::{self, Debug};
use std::hash::{Hash, Hasher};
use std::mem::size_of;
use std::str::Chars;
use std::sync::Arc;

use num_bigint::{BigInt, Sign::*};
use num_traits::cast::ToPrimitive;

use crate::atom::{self, Encoding, Existence, Existence::*, Index};
use crate::binary::{self, heap, sub, Part, PartToList};
use crate::code::Code;
use crate::exception::{self, Class, Exception};
use crate::float::{self, Float};
use crate::function::Function;
use crate::heap::{CloneIntoHeap, Heap};
use crate::integer::Integer::{self, Big, Small};
use crate::integer::{
    big::{self, integral_f64_to_big_int},
    small,
};
use crate::list::Cons;
use crate::map::Map;
use crate::process::{
    self, IntoProcess, ModuleFunctionArity, Process, TryFromInProcess, TryIntoInProcess,
};
use crate::reference::local;
use crate::scheduler::Scheduler;
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
use std::hint::unreachable_unchecked;

pub struct TagError {
    #[cfg_attr(not(test), allow(dead_code))]
    tag: usize,
    #[cfg_attr(not(test), allow(dead_code))]
    bit_count: usize,
}

#[cfg(test)]
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

    pub unsafe fn arity_to_integer(&self, process: &Process) -> Term {
        self.arity_to_usize().into_process(&process)
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
                            Err(badarg!())
                        }
                    }
                }
            }
            _ => Err(badarg!()),
        }
    }

    pub unsafe fn atom_to_index(&self) -> Index {
        Index(self.tagged >> Tag::ATOM_BIT_COUNT)
    }

    #[cfg(not(test))]
    pub unsafe fn atom_to_string(&self) -> Arc<String> {
        // bypass need to define `Debug` Term
        match atom::index_to_string(self.atom_to_index()) {
            Ok(string) => string,
            Err(_) => panic!("Atom not in table"),
        }
    }

    #[cfg(test)]
    pub unsafe fn atom_to_string(&self) -> Arc<String> {
        atom::index_to_string(self.atom_to_index()).unwrap()
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

    pub fn chars_to_list(chars: Chars, process: &Process) -> Term {
        chars.rfold(Self::EMPTY_LIST, |acc, character| {
            Term::cons(character.into_process(&process), acc, &process)
        })
    }

    pub fn cons(head: Term, tail: Term, process: &Process) -> Term {
        Self::heap_cons(head, tail, &process.heap.lock().unwrap())
    }

    pub fn heap_cons(head: Term, tail: Term, heap: &Heap) -> Term {
        let pointer_bits = heap.cons(head, tail) as *const Cons as usize;

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

    /// Counts the number of elements in the Term if it is a list
    pub fn count(self) -> Option<usize> {
        let mut count: usize = 0;
        let mut tail = self;

        loop {
            match tail.tag() {
                EmptyList => break Some(count),
                List => {
                    let cons: &Cons = unsafe { tail.as_ref_cons_unchecked() };
                    tail = cons.tail();
                    count += 1;
                }
                _ => break None,
            }
        }
    }

    /// Converts integers and floats and only do conversion when not `eq`.
    pub fn eq_after_conversion(&self, right: &Term) -> bool {
        (self.eq(right) | {
            match (self.tag(), right.tag()) {
                (SmallInteger, Boxed) => {
                    let right_unboxed: &Term = right.unbox_reference();

                    match right_unboxed.tag() {
                        Float => unsafe {
                            Self::small_integer_eq_float_after_conversion(self, right)
                        },
                        _ => false,
                    }
                }
                (Boxed, SmallInteger) => {
                    let left_unboxed: &Term = self.unbox_reference();

                    match left_unboxed.tag() {
                        Float => unsafe {
                            Self::small_integer_eq_float_after_conversion(right, self)
                        },
                        _ => false,
                    }
                }
                (Boxed, Boxed) => {
                    let left_unboxed: &Term = self.unbox_reference();
                    let right_unbox: &Term = right.unbox_reference();

                    match (left_unboxed.tag(), right_unbox.tag()) {
                        (BigInteger, Float) => unsafe {
                            Self::big_integer_eq_float_after_conversion(self, right)
                        },
                        (Float, BigInteger) => unsafe {
                            Self::big_integer_eq_float_after_conversion(right, self)
                        },
                        _ => false,
                    }
                }
                _ => false,
            }
        })
    }

    pub fn function(
        module_function_arity: Arc<ModuleFunctionArity>,
        code: Code,
        process: &Process,
    ) -> Term {
        Term::box_reference(process.function(module_function_arity, code))
    }

    unsafe fn small_integer_eq_float_after_conversion(
        small_integer: &Term,
        float_term: &Term,
    ) -> bool {
        let float: &Float = float_term.unbox_reference();
        let float_f64 = float.inner;

        (float_f64.fract() == 0.0) & {
            let small_integer_isize = small_integer.small_integer_to_isize();

            // float is out-of-range of SmallInteger, so it can't be equal
            if (float_f64 < (small::MIN as f64)) | ((small::MAX as f64) < float_f64) {
                false
            } else {
                let float_isize = float_f64 as isize;

                small_integer_isize == float_isize
            }
        }
    }

    unsafe fn big_integer_partial_cmp_float(
        big_integer_term: &Term,
        float_term: &Term,
    ) -> Option<Ordering> {
        let big_integer: &big::Integer = big_integer_term.unbox_reference();
        let big_integer_big_int = &big_integer.inner;
        let float: &Float = float_term.unbox_reference();
        let float_f64 = float.inner;

        match big_integer_big_int.sign() {
            Minus => {
                if float_f64 < 0.0 {
                    // fits in small integer so the big integer must be lesser
                    if (small::MIN as f64) <= float_f64 {
                        Some(Less)
                    // big_int can't fit in float, so it must be less than any float
                    } else if (std::f64::MAX_EXP as usize) < big_integer_big_int.bits() {
                        Some(Less)
                    // > A float is more precise than an integer until all
                    // > significant figures of the float are to the left of the
                    // > decimal point.
                    } else if float::INTEGRAL_MIN <= float_f64 {
                        let big_integer_f64: f64 = big_integer.into();

                        big_integer_f64.partial_cmp(&float_f64)
                    } else {
                        let float_integral_f64 = float_f64.trunc();
                        let float_big_int = integral_f64_to_big_int(float_integral_f64);

                        match big_integer_big_int.partial_cmp(&float_big_int) {
                            Some(Equal) => {
                                let float_fract = float_f64 - float_integral_f64;

                                if float_fract == 0.0 {
                                    Some(Equal)
                                } else {
                                    // BigInt Is -N while float is -N.M
                                    Some(Greater)
                                }
                            }
                            partial_ordering => partial_ordering,
                        }
                    }
                } else {
                    Some(Less)
                }
            }
            // BigInt does not have a zero because zero is a SmallInteger
            NoSign => unreachable_unchecked(),
            Plus => {
                if 0.0 < float_f64 {
                    // fits in small integer, so the big integer must be greater
                    if float_f64 <= (small::MAX as f64) {
                        Some(Greater)
                    // big_int can't fit in float, so it must be greater than any float
                    } else if (std::f64::MAX_EXP as usize) < big_integer_big_int.bits() {
                        Some(Greater)
                    // > A float is more precise than an integer until all
                    // > significant figures of the float are to the left of the
                    // > decimal point.
                    } else if float_f64 <= float::INTEGRAL_MAX {
                        let big_integer_f64: f64 = big_integer.into();

                        big_integer_f64.partial_cmp(&float_f64)
                    } else {
                        let float_integral_f64 = float_f64.trunc();
                        let float_big_int = integral_f64_to_big_int(float_integral_f64);

                        match big_integer_big_int.partial_cmp(&float_big_int) {
                            Some(Equal) => {
                                let float_fract = float_f64 - float_integral_f64;

                                if float_fract == 0.0 {
                                    Some(Equal)
                                } else {
                                    // BigInt is N while float is N.M
                                    Some(Less)
                                }
                            }
                            partial_ordering => partial_ordering,
                        }
                    }
                } else {
                    Some(Greater)
                }
            }
        }
    }

    // See https://github.com/erlang/otp/blob/741c5a5e1dbffd32d0478d4941ab0f725d709086/erts/emulator/beam/utils.c#L3196-L3221
    unsafe fn big_integer_eq_float_after_conversion(
        big_integer_term: &Term,
        float_term: &Term,
    ) -> bool {
        let float: &Float = float_term.unbox_reference();
        let float_f64 = float.inner;

        (float_f64.fract() == 0.0) & {
            // Float fits in small integer range, so it can't be a BigInt
            // https://github.com/erlang/otp/blob/741c5a5e1dbffd32d0478d4941ab0f725d709086/erts/emulator/beam/utils.c#L3199-L3202
            if (((small::MIN - 1) as f64) < float_f64) | (float_f64 < ((small::MAX + 1) as f64)) {
                false
            } else {
                let big_integer: &big::Integer = big_integer_term.unbox_reference();
                let big_integer_big_int = &big_integer.inner;

                // big_int can't fit in float
                if (std::f64::MAX_EXP as usize) < big_integer_big_int.bits() {
                    false
                // > A float is more precise than an integer until all
                // > significant figures of the float are to the left of the
                // > decimal point.
                } else if (float::INTEGRAL_MIN <= float_f64) & (float_f64 <= float::INTEGRAL_MAX) {
                    let big_integer_f64: f64 = big_integer.into();

                    big_integer_f64 == float_f64
                } else {
                    let float_big_int = integral_f64_to_big_int(float_f64);

                    big_integer_big_int == &float_big_int
                }
            }
        }
    }

    pub fn external_pid(
        node: usize,
        number: usize,
        serial: usize,
        process: &Process,
    ) -> exception::Result {
        if (number <= process::identifier::NUMBER_MAX)
            && (serial <= process::identifier::SERIAL_MAX)
        {
            Ok(Term::box_reference(
                process.external_pid(node, number, serial),
            ))
        } else {
            Err(badarg!())
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
            Err(badarg!())
        }
    }

    pub unsafe fn local_pid_unchecked(number: usize, serial: usize) -> Term {
        Term {
            tagged: (serial << (process::identifier::NUMBER_BIT_COUNT + Tag::LOCAL_PID_BIT_COUNT))
                | (number << (Tag::LOCAL_PID_BIT_COUNT))
                | (LocalPid as usize),
        }
    }

    pub unsafe fn decompose_local_pid(&self) -> (usize, usize) {
        let untagged = self.tagged >> Tag::LOCAL_PID_BIT_COUNT;

        let number = untagged & process::identifier::NUMBER_MASK;
        let serial = untagged >> process::identifier::NUMBER_BIT_COUNT;

        (number, serial)
    }

    pub fn local_reference(number: local::Number, process: &Process) -> Term {
        Term::box_reference(Scheduler::current().reference(number, process))
    }

    pub fn next_local_reference(process: &Process) -> Term {
        Term::box_reference(Scheduler::current().next_reference(process))
    }

    pub fn tag(&self) -> Tag {
        match (self.tagged as usize).try_into() {
            Ok(tag) => tag,
            Err(tag_error) => panic!(tag_error),
        }
    }

    pub fn is_atom(&self) -> bool {
        self.tag() == Atom
    }

    pub fn is_char_list(&self) -> bool {
        match self.tag() {
            EmptyList => true,
            List => {
                let cons: &Cons = unsafe { self.as_ref_cons_unchecked() };

                cons.is_char_list()
            }
            _ => false,
        }
    }

    pub fn is_empty_list(&self) -> bool {
        (self.tag() == EmptyList)
    }

    pub fn is_function(&self) -> bool {
        match self.tag() {
            Boxed => {
                let unboxed: &Term = self.unbox_reference();

                unboxed.tag() == Function
            }
            _ => false,
        }
    }

    pub fn is_integer(&self) -> bool {
        match self.tag() {
            SmallInteger => true,
            Boxed => {
                let unboxed: &Term = self.unbox_reference();

                unboxed.tag() == BigInteger
            }
            _ => false,
        }
    }

    pub fn is_local_pid(&self) -> bool {
        self.tag() == LocalPid
    }

    pub fn is_number(&self) -> bool {
        match self.tag() {
            SmallInteger => true,
            Boxed => {
                let unboxed: &Term = self.unbox_reference();

                match unboxed.tag() {
                    BigInteger | Float => true,
                    _ => false,
                }
            }
            _ => false,
        }
    }

    pub fn is_proper_list(&self) -> bool {
        match self.tag() {
            EmptyList => true,
            List => unsafe { self.as_ref_cons_unchecked() }.is_proper(),
            _ => false,
        }
    }

    pub fn pid(node: usize, number: usize, serial: usize, process: &Process) -> exception::Result {
        if node == 0 {
            Self::local_pid(number, serial)
        } else {
            Self::external_pid(node, number, serial, process)
        }
    }

    pub fn slice_to_binary(slice: &[u8], process: &Process) -> Term {
        process.slice_to_binary(slice).into()
    }

    pub fn slice_to_list(slice: &[Term], process: &Process) -> Term {
        Self::slice_to_improper_list(slice, Term::EMPTY_LIST, &process)
    }

    pub fn slice_to_improper_list(slice: &[Term], tail: Term, process: &Process) -> Term {
        slice.iter().rfold(tail, |acc, element| {
            Term::cons(element.clone(), acc, &process)
        })
    }

    pub fn slice_to_map(slice: &[(Term, Term)], process: &Process) -> Term {
        process.slice_to_map(slice).into()
    }

    pub fn slice_to_tuple(slice: &[Term], process: &Process) -> Term {
        process.slice_to_tuple(slice).into()
    }

    pub fn str_to_atom(name: &str, existence: Existence) -> Option<Term> {
        atom::str_to_index(name, existence).map(|atom_index| atom_index.into())
    }

    pub fn str_to_char_list(name: &str, process: &Process) -> Term {
        name.chars().rfold(Term::EMPTY_LIST, |acc, c| {
            Term::cons(c.into_process(&process), acc, &process)
        })
    }

    pub fn subbinary(
        original: Term,
        byte_offset: usize,
        bit_offset: u8,
        byte_count: usize,
        bit_count: u8,
        process: &Process,
    ) -> Term {
        process
            .subbinary(original, byte_offset, bit_offset, byte_count, bit_count)
            .into()
    }

    pub fn vec_to_list(vec: &Vec<Term>, initial_tail: Term, process: &Process) -> Term {
        vec.iter().rfold(initial_tail, |acc, element| {
            Term::cons(element.clone(), acc, &process)
        })
    }

    pub fn box_reference<T>(reference: &T) -> Term {
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

    unsafe fn small_integer_partial_cmp_boxed(
        small_integer: &Term,
        boxed: &Term,
    ) -> Option<Ordering> {
        let unboxed: &Term = boxed.unbox_reference();

        match unboxed.tag() {
            Float => {
                let small_integer_f64: f64 = small_integer.small_integer_to_isize() as f64;

                let boxed_float: &Float = boxed.unbox_reference();
                let boxed_f64 = boxed_float.inner;

                small_integer_f64.partial_cmp(&boxed_f64)
            }
            BigInteger => {
                let small_integer_big_int: BigInt = small_integer.small_integer_to_isize().into();

                let boxed_big_integer: &big::Integer = boxed.unbox_reference();
                let boxed_big_int = &boxed_big_integer.inner;

                small_integer_big_int.partial_cmp(boxed_big_int)
            }
            LocalReference | ExternalPid | Arity | Map | HeapBinary | Subbinary => Some(Less),
            other_unboxed_tag => unimplemented!("SmallInteger cmp unboxed {:?}", other_unboxed_tag),
        }
    }

    pub const unsafe fn isize_to_small_integer(i: isize) -> Term {
        Term {
            tagged: ((i << Tag::SMALL_INTEGER_BIT_COUNT) as usize) | (SmallInteger as usize),
        }
    }

    pub unsafe fn small_integer_to_isize(&self) -> isize {
        (self.tagged as isize) >> Tag::SMALL_INTEGER_BIT_COUNT
    }

    pub unsafe fn small_integer_to_big_int(&self) -> BigInt {
        self.small_integer_to_isize().into()
    }
}

impl CloneIntoHeap for Term {
    fn clone_into_heap(&self, heap: &Heap) -> Term {
        match self.tag() {
            Boxed => {
                let unboxed: &Term = self.unbox_reference();

                match unboxed.tag() {
                    Arity => {
                        let tuple: &Tuple = self.unbox_reference();
                        let heap_tuple = tuple.clone_into_heap(heap);

                        Term::box_reference(heap_tuple)
                    }
                    BigInteger => {
                        let big_integer: &big::Integer = self.unbox_reference();
                        let heap_big_integer = big_integer.clone_into_heap(heap);

                        Term::box_reference(heap_big_integer)
                    }
                    ExternalPid => {
                        let external_pid: &process::identifier::External = self.unbox_reference();
                        let heap_external_pid = external_pid.clone_into_heap(heap);

                        Term::box_reference(heap_external_pid)
                    }
                    Float => {
                        let float: &Float = self.unbox_reference();
                        let heap_float = float.clone_into_heap(heap);

                        Term::box_reference(heap_float)
                    }
                    HeapBinary => {
                        let heap_binary: &heap::Binary = self.unbox_reference();
                        let heap_heap_binary = heap_binary.clone_into_heap(heap);

                        Term::box_reference(heap_heap_binary)
                    }
                    LocalReference => {
                        let local_reference: &local::Reference = self.unbox_reference();
                        let heap_local_reference = local_reference.clone_into_heap(heap);

                        Term::box_reference(heap_local_reference)
                    }
                    Map => {
                        let map: &Map = self.unbox_reference();
                        let heap_map = map.clone_into_heap(heap);

                        Term::box_reference(heap_map)
                    }
                    Subbinary => {
                        let subbinary: &sub::Binary = self.unbox_reference();
                        let heap_subbinary = subbinary.clone_into_heap(heap);

                        Term::box_reference(heap_subbinary)
                    }
                    unboxed_tag => unimplemented!("Cloning unboxed {:?} into Heap", unboxed_tag),
                }
            }
            Atom | EmptyList | LocalPid | SmallInteger => self.clone(),
            List => {
                let cons: &Cons = unsafe { self.as_ref_cons_unchecked() };

                cons.clone_into_heap(heap)
            }
            tag => unimplemented!("Cloning {:?} into Heap", tag),
        }
    }
}

impl CloneIntoHeap for Vec<Term> {
    fn clone_into_heap(&self, heap: &Heap) -> Vec<Term> {
        self.iter().map(|term| term.clone_into_heap(heap)).collect()
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

                        write!(f, "], &process)")
                    }
                    BigInteger => {
                        let big_integer: &big::Integer = self.unbox_reference();

                        write!(
                            f,
                            "BigInt::parse_bytes(b\"{}\", 10).unwrap().into_process(&process)",
                            big_integer.inner
                        )
                    }
                    ExternalPid => {
                        let external_pid: &process::identifier::External = self.unbox_reference();

                        write!(
                            f,
                            "Term::external_pid({:?}, {:?}, {:?}, &process)",
                            external_pid.node, external_pid.number, external_pid.serial
                        )
                    }
                    Float => {
                        let float: &Float = self.unbox_reference();

                        write!(f, "{:?}_f64.into_process(&process)", float.inner)
                    }
                    Function => {
                        let function: &Function = self.unbox_reference();

                        write!(
                            f,
                            "Term::function({:?}, code, &process)",
                            function.module_function_arity()
                        )
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

                        write!(f, "], &process)")
                    }
                    LocalReference => {
                        let local_reference: &local::Reference = self.unbox_reference();

                        write!(
                            f,
                            "Term::local_reference({:?}, {:?}, &process)",
                            local_reference.scheduler_id(),
                            local_reference.number()
                        )
                    }
                    Map => {
                        let map: &Map = self.unbox_reference();

                        write!(f, "Term::slice_to_map(&[")?;

                        let mut iter = map.iter();

                        if let Some(item) = iter.next() {
                            write!(f, "{:?}", item)?;

                            for item in iter {
                                write!(f, ", {:?}", item)?;
                            }
                        }

                        write!(f, "], &process)")
                    }
                    Subbinary => {
                        let subbinary: &sub::Binary = self.unbox_reference();

                        write!(
                            f,
                            "Term::subbinary({:?}, {:?}, {:?}, {:?}, {:?}, &process)",
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
                    "Term::cons({:?}, {:?}, &process)",
                    cons.head(),
                    cons.tail()
                )
            }
            LocalPid => {
                let (number, serial) = unsafe { self.decompose_local_pid() };

                write!(f, "Term::local_pid({:?}, {:?}).unwrap()", number, serial)
            }
            SmallInteger => write!(f, "{:?}.into_process(&process)", isize::from(self)),
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
    fn into_process(self, process: &Process) -> Term {
        let integer: Integer = self.into();

        integer.into_process(&process)
    }
}

impl IntoProcess<Term> for f64 {
    fn into_process(self, process: &Process) -> Term {
        let process_float: &Float = process.f64_to_float(self);

        Term::box_reference(process_float)
    }
}

impl IntoProcess<Term> for i32 {
    fn into_process(self, process: &Process) -> Term {
        let integer: Integer = self.into();

        integer.into_process(&process)
    }
}

impl IntoProcess<Term> for isize {
    fn into_process(self, process: &Process) -> Term {
        let integer: Integer = self.into();

        integer.into_process(&process)
    }
}

impl IntoProcess<Term> for u8 {
    fn into_process(self, process: &Process) -> Term {
        let integer: Integer = self.into();

        integer.into_process(&process)
    }
}

impl IntoProcess<Term> for usize {
    fn into_process(self, process: &Process) -> Term {
        let integer: Integer = self.into();

        integer.into_process(process)
    }
}

impl IntoProcess<Term> for u64 {
    fn into_process(self, process: &Process) -> Term {
        let big_int: BigInt = self.into();

        big_int.into_process(process)
    }
}

impl IntoProcess<Term> for u128 {
    fn into_process(self, process: &Process) -> Term {
        let big_int: BigInt = self.into();

        big_int.into_process(process)
    }
}

impl IntoProcess<Term> for Integer {
    fn into_process(self, process: &Process) -> Term {
        match self {
            Small(small::Integer(untagged)) => Term {
                tagged: ((untagged << Tag::SMALL_INTEGER_BIT_COUNT) as usize)
                    | (SmallInteger as usize),
            },
            Big(big_int) => {
                let process_integer: &big::Integer = process.num_bigint_big_to_big_integer(big_int);

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
    fn part(&'a self, start: Term, length: Term, process: &Process) -> exception::Result {
        let start_usize: usize = start.try_into()?;
        let length_isize: isize = length.try_into()?;

        let binary = self.part(start_usize, length_isize, &process)?;

        match binary {
            // a heap binary is only returned if it is the same
            binary::Binary::Heap(_) => Ok(self.into()),
            binary::Binary::Sub(subbinary) => Ok(subbinary.into()),
        }
    }
}

impl<'a> Part<'a, Term, Term, Term> for sub::Binary {
    fn part(&'a self, start: Term, length: Term, process: &Process) -> exception::Result {
        let start_usize: usize = start.try_into()?;
        let length_isize: isize = length.try_into()?;
        let new_subbinary = self.part(start_usize, length_isize, &process)?;

        Ok(new_subbinary.into())
    }
}

impl PartToList<Term, Term> for heap::Binary {
    fn part_to_list(&self, start: Term, length: Term, process: &Process) -> exception::Result {
        let start_usize: usize = start.try_into()?;
        let length_isize: isize = length.try_into()?;

        self.part_to_list(start_usize, length_isize, process)
    }
}

impl PartToList<Term, Term> for sub::Binary {
    fn part_to_list(&self, start: Term, length: Term, process: &Process) -> exception::Result {
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
            (SmallInteger, Boxed) => unsafe {
                Term::small_integer_partial_cmp_boxed(&self, &other)
            },
            (SmallInteger, Atom)
            | (SmallInteger, LocalPid)
            | (SmallInteger, EmptyList)
            | (SmallInteger, List) => Some(Less),
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
                    LocalReference | ExternalPid | Arity | Map | HeapBinary | Subbinary => {
                        Some(Less)
                    }
                    BigInteger | Float => Some(Greater),
                    other_unboxed_tag => unimplemented!("Atom cmp unboxed {:?}", other_unboxed_tag),
                }
            }
            (Atom, LocalPid) | (Atom, EmptyList) | (Atom, List) => Some(Less),
            (Boxed, SmallInteger) => {
                unsafe { Term::small_integer_partial_cmp_boxed(&other, &self) }
                    .map(|ordering| ordering.reverse())
            }
            (Boxed, Atom) => {
                let self_unboxed: &Term = self.unbox_reference();

                match self_unboxed.tag() {
                    BigInteger | Float => Some(Less),
                    LocalReference | ExternalPid | Arity | Map | HeapBinary | Subbinary => {
                        Some(Greater)
                    }
                    self_unboxed_tag => unimplemented!("unboxed {:?} cmp Atom", self_unboxed_tag),
                }
            }
            (Boxed, Boxed) => {
                let self_unboxed: &Term = self.unbox_reference();
                let other_unboxed: &Term = other.unbox_reference();

                // in ascending order
                match (self_unboxed.tag(), other_unboxed.tag()) {
                    (Float, BigInteger) => {
                        unsafe { Term::big_integer_partial_cmp_float(&other, &self) }
                            .map(|ordering| ordering.reverse())
                    }
                    (Float, Float) => {
                        let self_float: &Float = self.unbox_reference();
                        let self_inner = self_float.inner;

                        let other_float: &Float = other.unbox_reference();
                        let other_inner = other_float.inner;

                        // Erlang doesn't support the floats that can't be compared
                        self_inner.partial_cmp(&other_inner)
                    }
                    (Float, LocalReference)
                    | (Float, ExternalPid)
                    | (Float, Arity)
                    | (Float, Map)
                    | (Float, HeapBinary)
                    | (Float, Subbinary) => Some(Less),
                    (BigInteger, BigInteger) => {
                        let self_big_integer: &big::Integer = self.unbox_reference();
                        let other_big_integer: &big::Integer = other.unbox_reference();

                        self_big_integer.inner.partial_cmp(&other_big_integer.inner)
                    }
                    (BigInteger, Float) => unsafe {
                        Term::big_integer_partial_cmp_float(&self, &other)
                    },
                    (BigInteger, LocalReference)
                    | (BigInteger, ExternalPid)
                    | (BigInteger, Arity)
                    | (BigInteger, Map)
                    | (BigInteger, HeapBinary)
                    | (BigInteger, Subbinary) => Some(Less),
                    (LocalReference, BigInteger) | (LocalReference, Float) => Some(Greater),
                    (LocalReference, LocalReference) => {
                        let self_local_reference: &local::Reference = self.unbox_reference();
                        let other_local_reference: &local::Reference = other.unbox_reference();

                        self_local_reference
                            .number()
                            .partial_cmp(&other_local_reference.number())
                    }
                    (LocalReference, LocalPid)
                    | (LocalReference, ExternalPid)
                    | (LocalReference, Arity)
                    | (LocalReference, Map)
                    | (LocalReference, HeapBinary)
                    | (LocalReference, Subbinary) => Some(Less),
                    (ExternalPid, Float)
                    | (ExternalPid, BigInteger)
                    | (ExternalPid, LocalReference) => Some(Greater),
                    (ExternalPid, ExternalPid) => {
                        let self_external_pid: &process::identifier::External =
                            self.unbox_reference();
                        let other_external_pid: &process::identifier::External =
                            other.unbox_reference();

                        self_external_pid.partial_cmp(other_external_pid)
                    }
                    (ExternalPid, Arity)
                    | (ExternalPid, Map)
                    | (ExternalPid, HeapBinary)
                    | (ExternalPid, Subbinary) => Some(Less),
                    (Arity, Float)
                    | (Arity, BigInteger)
                    | (Arity, LocalReference)
                    | (Arity, ExternalPid) => Some(Greater),
                    (Arity, Arity) => {
                        let self_tuple: &Tuple = self.unbox_reference();
                        let other_tuple: &Tuple = other.unbox_reference();

                        self_tuple.partial_cmp(other_tuple)
                    }
                    (Arity, Map) | (Arity, HeapBinary) | (Arity, Subbinary) => Some(Less),
                    (Map, Float)
                    | (Map, BigInteger)
                    | (Map, LocalReference)
                    | (Map, ExternalPid)
                    | (Map, Arity) => Some(Greater),
                    (Map, Map) => {
                        let self_map: &Map = self.unbox_reference();
                        let other_map: &Map = other.unbox_reference();

                        self_map.partial_cmp(other_map)
                    }
                    (Map, HeapBinary) | (Map, Subbinary) => Some(Less),
                    (HeapBinary, Float)
                    | (HeapBinary, BigInteger)
                    | (HeapBinary, LocalReference)
                    | (HeapBinary, ExternalPid)
                    | (HeapBinary, Arity)
                    | (HeapBinary, Map) => Some(Greater),
                    (HeapBinary, HeapBinary) => {
                        let self_binary: &heap::Binary = self.unbox_reference();
                        let other_binary: &heap::Binary = other.unbox_reference();

                        self_binary.partial_cmp(other_binary)
                    }
                    (HeapBinary, Subbinary) => {
                        let self_heap_binary: &heap::Binary = self.unbox_reference();
                        let other_subbinary: &sub::Binary = other.unbox_reference();

                        self_heap_binary.partial_cmp(other_subbinary)
                    }
                    (Subbinary, HeapBinary) => {
                        let self_subbinary: &sub::Binary = self.unbox_reference();
                        let other_heap_binary: &heap::Binary = other.unbox_reference();

                        self_subbinary.partial_cmp(other_heap_binary)
                    }
                    (Subbinary, Float)
                    | (Subbinary, BigInteger)
                    | (Subbinary, LocalReference)
                    | (Subbinary, ExternalPid)
                    | (Subbinary, Arity)
                    | (Subbinary, Map) => Some(Greater),
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
            (Boxed, LocalPid) => {
                let self_unboxed: &Term = self.unbox_reference();

                match self_unboxed.tag() {
                    Float | BigInteger | LocalReference => Some(Less),
                    // local pid has node 0 while all external pids have node > 0
                    ExternalPid | Arity | Map | HeapBinary | Subbinary => Some(Greater),
                    self_unboxed_tag => {
                        unimplemented!("unboxed {:?} cmp LocalPid", self_unboxed_tag)
                    }
                }
            }
            (Boxed, EmptyList) | (Boxed, List) => {
                let self_unboxed: &Term = self.unbox_reference();

                match self_unboxed.tag() {
                    Float | BigInteger | LocalReference | ExternalPid | Arity | Map => Some(Less),
                    HeapBinary | Subbinary => Some(Greater),
                    self_unboxed_tag => unimplemented!("unboxed {:?} cmp list()", self_unboxed_tag),
                }
            }
            (LocalPid, SmallInteger) | (LocalPid, Atom) => Some(Greater),
            (LocalPid, Boxed) => {
                let other_unboxed: &Term = other.unbox_reference();

                match other_unboxed.tag() {
                    Float | BigInteger | LocalReference => Some(Greater),
                    ExternalPid | Arity | Map | HeapBinary | Subbinary => Some(Less),
                    other_unboxed_tag => {
                        unimplemented!("LocalPid cmp unboxed {:?}", other_unboxed_tag)
                    }
                }
            }
            (LocalPid, LocalPid) => self.tagged.partial_cmp(&other.tagged),
            (LocalPid, EmptyList) | (LocalPid, List) => Some(Less),
            (EmptyList, SmallInteger) | (EmptyList, Atom) | (EmptyList, LocalPid) => Some(Greater),
            (EmptyList, Boxed) | (List, Boxed) => {
                let other_unboxed: &Term = other.unbox_reference();

                match other_unboxed.tag() {
                    Float | BigInteger | LocalReference | ExternalPid | Arity | Map => {
                        Some(Greater)
                    }
                    HeapBinary | Subbinary => Some(Less),
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
            (List, SmallInteger) | (List, Atom) | (List, LocalPid) => Some(Greater),
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
                    _ => Err(badarg!()),
                }
            }
            _ => Err(badarg!()),
        }
    }
}

impl TryFrom<Term> for Class {
    type Error = Exception;

    fn try_from(term: Term) -> std::result::Result<Class, Exception> {
        use self::Class::*;

        match term.tag() {
            Atom => match unsafe { term.atom_to_string() }.as_ref().as_ref() {
                "error" => Ok(Error { arguments: None }),
                "exit" => Ok(Exit),
                "throw" => Ok(Throw),
                _ => Err(badarg!()),
            },
            _ => Err(badarg!()),
        }
    }
}

impl TryFrom<Term> for bool {
    type Error = Exception;

    fn try_from(term: Term) -> Result<bool, Exception> {
        match term.tag() {
            Atom => match unsafe { term.atom_to_string() }.as_ref().as_ref() {
                "false" => Ok(false),
                "true" => Ok(true),
                _ => Err(badarg!()),
            },
            _ => Err(badarg!()),
        }
    }
}

impl TryFrom<Term> for char {
    type Error = Exception;

    fn try_from(term: Term) -> Result<char, Exception> {
        let term_u32: u32 = term.try_into()?;

        match std::char::from_u32(term_u32) {
            Some(c) => Ok(c),
            None => Err(badarg!()),
        }
    }
}

impl TryFrom<Term> for f64 {
    type Error = Exception;

    fn try_from(term: Term) -> Result<f64, Exception> {
        match term.tag() {
            SmallInteger => {
                let term_isize: isize = unsafe { term.small_integer_to_isize() };
                let term_f64: f64 = term_isize as f64;

                Ok(term_f64)
            }
            Boxed => {
                let unboxed: &Term = term.unbox_reference();

                match unboxed.tag() {
                    BigInteger => {
                        let big_integer: &big::Integer = term.unbox_reference();
                        let term_f64: f64 = big_integer.into();

                        Ok(term_f64)
                    }
                    Float => {
                        let float: &Float = term.unbox_reference();

                        Ok(float.inner)
                    }
                    _ => Err(badarith!()),
                }
            }
            _ => Err(badarith!()),
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
            _ => Err(badarg!()),
        }
    }
}

impl TryFrom<Term> for u32 {
    type Error = Exception;

    fn try_from(term: Term) -> Result<u32, Exception> {
        match term.tag() {
            SmallInteger => {
                let term_isize = unsafe { term.small_integer_to_isize() };

                term_isize.try_into().map_err(|_| badarg!())
            }
            Boxed => {
                let unboxed: &Term = term.unbox_reference();

                match unboxed.tag() {
                    BigInteger => {
                        let big_integer: &big::Integer = term.unbox_reference();

                        // does not implement `to_u32` directly
                        match big_integer.inner.to_u64() {
                            Some(term_u64) => term_u64.try_into().map_err(|_| badarg!()),
                            None => Err(badarg!()),
                        }
                    }
                    _ => Err(badarg!()),
                }
            }
            _ => Err(badarg!()),
        }
    }
}

impl TryFrom<Term> for u64 {
    type Error = Exception;

    fn try_from(term: Term) -> Result<u64, Exception> {
        match term.tag() {
            SmallInteger => {
                let i: isize = unsafe { term.small_integer_to_isize() };

                if 0 <= i {
                    Ok(i as u64)
                } else {
                    Err(badarg!())
                }
            }
            _ => Err(badarg!()),
        }
    }
}

impl TryFrom<Term> for u128 {
    type Error = Exception;

    fn try_from(term: Term) -> Result<u128, Exception> {
        match term.tag() {
            SmallInteger => {
                let i: isize = unsafe { term.small_integer_to_isize() };

                if 0 <= i {
                    Ok(i as u128)
                } else {
                    Err(badarg!())
                }
            }
            Boxed => {
                let unboxed: &Term = term.unbox_reference();

                match unboxed.tag() {
                    BigInteger => unimplemented!(),
                    _ => Err(badarg!()),
                }
            }
            _ => Err(badarg!()),
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
                    Err(badarg!())
                } else {
                    Ok(term_isize as usize)
                }
            }
            _ => Err(badarg!()),
        }
    }
}

impl TryFrom<Term> for Vec<Term> {
    type Error = Exception;

    fn try_from(term: Term) -> Result<Vec<Term>, Exception> {
        match term.tag() {
            EmptyList => Ok(Vec::new()),
            List => {
                let cons: &Cons = unsafe { term.as_ref_cons_unchecked() };

                cons.try_into()
            }
            _ => Err(badarg!()),
        }
    }
}

impl TryFrom<Term> for &'static Cons {
    type Error = Exception;

    fn try_from(term: Term) -> Result<&'static Cons, Exception> {
        match term.tag() {
            List => Ok(unsafe { term.as_ref_cons_unchecked() }),
            _ => Err(badarg!()),
        }
    }
}

impl TryFromInProcess<Term> for &'static Map {
    fn try_from_in_process(term: Term, process: &Process) -> Result<&'static Map, Exception> {
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
        .ok_or_else(|| badmap!(term, &process))
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
                    _ => Err(badarg!()),
                }
            }
            _ => Err(badarg!()),
        }
    }
}

impl TryFromInProcess<Term> for &'static Tuple {
    fn try_from_in_process(term: Term, process: &Process) -> Result<&'static Tuple, Exception> {
        match term.tag() {
            Boxed => term.unbox_reference::<Term>().try_into_in_process(&process),
            _ => Err(badarg!()),
        }
    }
}

impl TryFrom<&Term> for BigInt {
    type Error = Exception;

    fn try_from(term_ref: &Term) -> Result<BigInt, Exception> {
        (*term_ref).try_into()
    }
}

impl TryFrom<&Term> for usize {
    type Error = Exception;

    fn try_from(term_ref: &Term) -> Result<usize, Exception> {
        match term_ref.tag() {
            SmallInteger => {
                let i = unsafe { term_ref.small_integer_to_isize() };

                if 0 <= i {
                    Ok(i as usize)
                } else {
                    Err(badarg!())
                }
            }
            _ => Err(badarg!()),
        }
    }
}

impl<'a> TryFromInProcess<&'a Term> for &'a Tuple {
    fn try_from_in_process(term: &'a Term, process: &Process) -> Result<&'a Tuple, Exception> {
        match term.tag() {
            Arity => {
                let pointer = term as *const Term as *const Tuple;
                Ok(unsafe { pointer.as_ref() }.unwrap())
            }
            Boxed => term.unbox_reference::<Term>().try_into_in_process(&process),
            _ => Err(badarg!()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::scheduler::with_process;

    mod cmp_in_process {
        use super::*;

        mod less {
            use super::*;

            #[test]
            fn number_is_less_than_atom() {
                with_process(|process| {
                    let number_term: Term = 0.into_process(&process);
                    let atom_term = Term::str_to_atom("0", DoNotCare).unwrap();

                    assert!(number_term < atom_term);
                    assert!(!(atom_term < number_term));
                });
            }

            #[test]
            fn atom_is_less_than_tuple() {
                with_process(|process| {
                    let atom_term = Term::str_to_atom("0", DoNotCare).unwrap();
                    let tuple_term = Term::slice_to_tuple(&[], &process);

                    assert!(atom_term < tuple_term);
                    assert!(!(tuple_term < atom_term));
                });
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
                with_process(|process| {
                    let shorter_tuple = Term::slice_to_tuple(&[], &process);
                    let longer_tuple = Term::slice_to_tuple(&[0.into_process(&process)], &process);

                    assert!(shorter_tuple < longer_tuple);
                    assert!(!(longer_tuple < shorter_tuple));
                });
            }

            #[test]
            fn same_length_tuples_with_lesser_elements_is_lesser() {
                with_process(|process| {
                    let lesser_tuple = Term::slice_to_tuple(&[0.into_process(&process)], &process);
                    let greater_tuple = Term::slice_to_tuple(&[1.into_process(&process)], &process);

                    assert!(lesser_tuple < greater_tuple);
                    assert!(!(greater_tuple < lesser_tuple));
                });
            }

            #[test]
            fn tuple_is_less_than_empty_list() {
                with_process(|process| {
                    let tuple_term = Term::slice_to_tuple(&[], &process);
                    let empty_list_term = Term::EMPTY_LIST;

                    assert!(tuple_term < empty_list_term);
                    assert!(!(empty_list_term < tuple_term));
                });
            }

            #[test]
            fn tuple_is_less_than_list() {
                with_process(|process| {
                    let tuple_term = Term::slice_to_tuple(&[], &process);
                    let list_term = list_term(&process);

                    assert!(tuple_term < list_term);
                    assert!(!(list_term < tuple_term));
                });
            }
        }

        mod equal {
            use super::*;

            #[test]
            fn with_improper_list() {
                with_process(|process| {
                    let list_term =
                        Term::cons(0.into_process(&process), 1.into_process(&process), &process);
                    let equal_list_term =
                        Term::cons(0.into_process(&process), 1.into_process(&process), &process);
                    let unequal_list_term =
                        Term::cons(1.into_process(&process), 0.into_process(&process), &process);

                    assert_eq!(list_term, list_term);
                    assert_eq!(equal_list_term, equal_list_term);
                    assert_ne!(list_term, unequal_list_term);
                });
            }

            #[test]
            fn with_proper_list() {
                with_process(|process| {
                    let list_term =
                        Term::cons(0.into_process(&process), Term::EMPTY_LIST, &process);
                    let equal_list_term =
                        Term::cons(0.into_process(&process), Term::EMPTY_LIST, &process);
                    let unequal_list_term =
                        Term::cons(1.into_process(&process), Term::EMPTY_LIST, &process);

                    assert_eq!(list_term, list_term);
                    assert_eq!(list_term, equal_list_term);
                    assert_ne!(list_term, unequal_list_term);
                });
            }

            #[test]
            fn with_nested_list() {
                with_process(|process| {
                    let list_term = Term::cons(
                        0.into_process(&process),
                        Term::cons(1.into_process(&process), Term::EMPTY_LIST, &process),
                        &process,
                    );
                    let equal_list_term = Term::cons(
                        0.into_process(&process),
                        Term::cons(1.into_process(&process), Term::EMPTY_LIST, &process),
                        &process,
                    );
                    let unequal_list_term = Term::cons(
                        1.into_process(&process),
                        Term::cons(0.into_process(&process), Term::EMPTY_LIST, &process),
                        &process,
                    );

                    assert_eq!(list_term, list_term);
                    assert_eq!(list_term, equal_list_term);
                    assert_ne!(list_term, unequal_list_term);
                });
            }

            #[test]
            fn with_lists_of_unequal_length() {
                with_process(|process| {
                    let list_term = Term::cons(
                        0.into_process(&process),
                        Term::cons(1.into_process(&process), Term::EMPTY_LIST, &process),
                        &process,
                    );
                    let equal_list_term = Term::cons(
                        0.into_process(&process),
                        Term::cons(1.into_process(&process), Term::EMPTY_LIST, &process),
                        &process,
                    );
                    let shorter_list_term =
                        Term::cons(0.into_process(&process), Term::EMPTY_LIST, &process);
                    let longer_list_term = Term::cons(
                        0.into_process(&process),
                        Term::cons(
                            1.into_process(&process),
                            Term::cons(2.into_process(&process), Term::EMPTY_LIST, &process),
                            &process,
                        ),
                        &process,
                    );

                    assert_eq!(list_term, list_term);
                    assert_eq!(list_term, equal_list_term);
                    assert_ne!(list_term, shorter_list_term);
                    assert_ne!(list_term, longer_list_term);
                });
            }

            #[test]
            fn with_tuples_of_unequal_length() {
                with_process(|process| {
                    let tuple_term = Term::slice_to_tuple(&[0.into_process(&process)], &process);
                    let equal_term = Term::slice_to_tuple(&[0.into_process(&process)], &process);
                    let unequal_term = Term::slice_to_tuple(
                        &[0.into_process(&process), 1.into_process(&process)],
                        &process,
                    );

                    assert_eq!(tuple_term, tuple_term);
                    assert_eq!(tuple_term, equal_term);
                    assert_ne!(tuple_term, unequal_term);
                });
            }

            #[test]
            fn with_heap_binaries_of_unequal_length() {
                with_process(|process| {
                    let heap_binary_term = Term::slice_to_binary(&[0, 1], &process);
                    let equal_heap_binary_term = Term::slice_to_binary(&[0, 1], &process);
                    let shorter_heap_binary_term = Term::slice_to_binary(&[0], &process);
                    let longer_heap_binary_term = Term::slice_to_binary(&[0, 1, 2], &process);

                    assert_eq!(heap_binary_term, heap_binary_term);
                    assert_eq!(heap_binary_term, equal_heap_binary_term);
                    assert_ne!(heap_binary_term, shorter_heap_binary_term);
                    assert_ne!(heap_binary_term, longer_heap_binary_term);
                });
            }
        }
    }

    mod is_empty_list {
        use super::*;

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
            with_process(|process| {
                let head_term = Term::str_to_atom("head", DoNotCare).unwrap();
                let list_term = Term::cons(head_term, Term::EMPTY_LIST, &process);

                assert_eq!(list_term.is_empty_list(), false);
            });
        }

        #[test]
        fn with_small_integer_is_false() {
            with_process(|process| {
                let small_integer_term = small_integer_term(&process, 0);

                assert_eq!(small_integer_term.is_empty_list(), false);
            });
        }

        #[test]
        fn with_tuple_is_false() {
            with_process(|process| {
                let tuple_term = tuple_term(&process);

                assert_eq!(tuple_term.is_empty_list(), false);
            });
        }

        #[test]
        fn with_heap_binary_is_false() {
            with_process(|process| {
                let heap_binary_term = Term::slice_to_binary(&[], &process);

                assert_eq!(heap_binary_term.is_empty_list(), false);
            });
        }
    }

    fn small_integer_term(process: &Process, signed_size: isize) -> Term {
        signed_size.into_process(&process)
    }

    fn list_term(process: &Process) -> Term {
        let head_term = Term::str_to_atom("head", DoNotCare).unwrap();
        Term::cons(head_term, Term::EMPTY_LIST, process)
    }

    fn tuple_term(process: &Process) -> Term {
        Term::slice_to_tuple(&[], process)
    }
}
