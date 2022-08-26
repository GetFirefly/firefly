use alloc::alloc::{AllocError, Allocator, Layout};
use alloc::boxed::Box;
use alloc::string::String;
use core::any::TypeId;
use core::fmt::{self, Debug, Display};
use core::hash::{Hash, Hasher};
use core::marker::PhantomData;
use core::ptr::NonNull;

use firefly_alloc::gc::GcBox;
use firefly_alloc::heap::Heap;
use firefly_alloc::rc::Rc;
use firefly_binary::{BinaryFlags, BitVec, Bitstring, Encoding};

use super::{BinaryData, OpaqueTerm, Term, TupleIndex};

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum CharlistToBinaryError {
    /// The list isn't a charlist and/or is an improper list
    InvalidList,
    /// Could not allocate enough memory to store the binary
    AllocError,
}

#[repr(C)]
#[derive(Copy, Clone)]
pub struct Cons {
    pub head: OpaqueTerm,
    pub tail: OpaqueTerm,
}
impl Cons {
    pub const TYPE_ID: TypeId = TypeId::of::<Cons>();

    /// Allocates a new cons cell via Box<T>
    pub fn new<H: Into<OpaqueTerm>, T: Into<OpaqueTerm>>(head: H, tail: T) -> Box<Self> {
        Box::new(Self {
            head: head.into(),
            tail: tail.into(),
        })
    }

    /// Allocates a new cons cell in the given allocator
    ///
    /// NOTE: The returned cell is wrapped in `MaybeUninit<T>` because the head/tail require
    /// initialization.
    pub fn new_in<A: Allocator>(alloc: A) -> Result<NonNull<Cons>, AllocError> {
        alloc.allocate(Layout::new::<Cons>()).map(|ptr| ptr.cast())
    }

    /// Constructs a list from the given slice, the output of which will be in the same order as the slice.
    pub fn from_slice<H: Heap>(
        slice: &[Term],
        heap: H,
    ) -> Result<Option<NonNull<Cons>>, AllocError> {
        let mut builder = ListBuilder::new(&heap);
        for value in slice.iter().rev() {
            builder.push(*value)?;
        }
        Ok(builder.finish())
    }
    /// During garbage collection, when a list cell is moved to the new heap, a
    /// move marker is left in the original location. For a cons cell, the move
    /// marker sets the first word to None, and the second word to a pointer to
    /// the new location.
    #[inline]
    pub fn is_move_marker(&self) -> bool {
        self.head.is_none()
    }

    /// Returns the head of this list as a Term
    pub fn head(&self) -> Term {
        self.head.into()
    }

    /// Returns the tail of this list as a Term
    ///
    /// NOTE: If the tail of this cell is _not_ Nil or Cons, it represents an improper list
    pub fn tail(&self) -> Term {
        self.tail.into()
    }

    /// Constructs a new cons cell with the given head/tail values
    #[inline]
    pub fn cons(head: Term, tail: Term) -> Cons {
        Self {
            head: head.into(),
            tail: tail.into(),
        }
    }

    /// Traverse the list, producing a `Result<Term, ImproperList>` for each element.
    ///
    /// If the list is proper, all elements will be `Ok(Term)`, but if the list is improper,
    /// the last element produced will be `Err(ImproperList)`. This can be unwrapped to get at
    /// the contained value, or treated as an error, depending on the context.
    #[inline]
    pub fn iter(&self) -> Iter<'_> {
        Iter::new(self)
    }

    /// Returns true if this cell is the head of a proper list.
    ///
    /// NOTE: The cost of this function is linear in the length of the list (i.e. `O(N)`)
    pub fn is_proper(&self) -> bool {
        self.iter().all(|result| result.is_ok())
    }

    /// Searches this keyword list for the first element which has a matching key
    /// at the given index.
    ///
    /// If no key is found, returns 'badarg'
    pub fn keyfind<I, K: Into<Term>>(&self, index: I, key: K) -> Result<Option<Term>, ImproperList>
    where
        I: TupleIndex + Copy,
    {
        let key = key.into();
        for result in self.iter() {
            let Term::Tuple(ptr) = result? else { continue; };
            let tuple = unsafe { ptr.as_ref() };
            let Ok(candidate) = tuple.get_element(index) else { continue; };
            if candidate == key {
                return Ok(Some(Term::Tuple(ptr)));
            }
        }

        Ok(None)
    }
}

// Charlists
impl Cons {
    /// Traverses this list and determines if every element is a valid latin1/utf8 character
    pub fn is_charlist(&self) -> bool {
        for result in self.iter() {
            let Ok(Term::Int(i)) = result else { return false; };
            let Ok(i) = i.try_into() else { return false; };
            if char::from_u32(i).is_none() {
                return false;
            }
        }

        true
    }

    /// Traverses the list, constructing a String from each codepoint in the list
    ///
    /// If the list is improper, or any element is not a valid latin1/utf8 codepoint, this function returns None
    pub fn to_string(&self) -> Option<String> {
        let mut buffer = String::with_capacity(10);
        for result in self.iter() {
            let Ok(Term::Int(i)) = result else { return None; };
            let Ok(i) = i.try_into() else { return None; };
            match char::from_u32(i) {
                Some(c) => buffer.push(c),
                None => return None,
            }
        }
        Some(buffer)
    }

    /// Constructs a charlist from the given string
    pub fn from_bytes<H: Heap>(bytes: &[u8], heap: H) -> Result<Option<NonNull<Cons>>, AllocError> {
        let mut builder = ListBuilder::new(&heap);
        for byte in bytes.iter().rev().copied() {
            builder.push(Term::Int(byte as i64))?;
        }
        Ok(builder.finish())
    }

    /// Constructs a charlist from the given string
    pub fn charlist_from_str<H: Heap>(
        s: &str,
        heap: H,
    ) -> Result<Option<NonNull<Cons>>, AllocError> {
        let mut builder = ListBuilder::new(&heap);
        for c in s.chars().rev() {
            builder.push(Term::Int((c as u32) as i64))?;
        }
        Ok(builder.finish())
    }

    /// Converts a charlist to a binary value.
    ///
    /// NOTE: This function will return an error if the list is not a charlist. It will also return
    /// an error if we are unable to allocate memory for the binary.
    pub fn charlist_to_binary<H: Heap>(&self, heap: H) -> Result<Term, CharlistToBinaryError> {
        // We need to know whether or not the resulting binary should be allocated in `alloc`,
        // or on the global heap as a reference-counted binary. We also want to determine the target
        // encoding. So we'll scan the list twice, once to gather the size in bytes + encoding, the second
        // to write each byte to the allocated region.
        let (len, encoding) = self
            .get_charlist_size_and_encoding()
            .ok_or_else(|| CharlistToBinaryError::InvalidList)?;
        if len < 64 {
            self.charlist_to_heap_binary(len, encoding, heap)
        } else {
            self.charlist_to_refc_binary(len, encoding)
        }
    }

    /// Writes this charlist to a GcBox, i.e. allocates on a process heap
    fn charlist_to_heap_binary<H: Heap>(
        &self,
        len: usize,
        encoding: Encoding,
        heap: H,
    ) -> Result<Term, CharlistToBinaryError> {
        let mut buf = BitVec::with_capacity(len);
        if encoding == Encoding::Utf8 {
            self.write_unicode_charlist_to_buffer(&mut buf)?;
        } else {
            self.write_raw_charlist_to_buffer(&mut buf)?;
        }
        let mut gcbox = GcBox::<BinaryData>::with_capacity_in(buf.byte_size(), heap)
            .map_err(|_| CharlistToBinaryError::AllocError)?;
        {
            unsafe {
                gcbox.set_flags(BinaryFlags::new(len, encoding));
            }
            gcbox.copy_from_slice(unsafe { buf.as_bytes_unchecked() });
        }
        Ok(gcbox.into())
    }

    /// Writes this charlist to an Rc, i.e. allocates on the global heap
    fn charlist_to_refc_binary(
        &self,
        len: usize,
        encoding: Encoding,
    ) -> Result<Term, CharlistToBinaryError> {
        let mut buf = BitVec::with_capacity(len);
        if encoding == Encoding::Utf8 {
            self.write_unicode_charlist_to_buffer(&mut buf)?;
        } else {
            self.write_raw_charlist_to_buffer(&mut buf)?;
        }
        let mut rc = Rc::<BinaryData>::with_capacity(buf.byte_size());
        {
            let value = unsafe { Rc::get_mut_unchecked(&mut rc) };
            unsafe {
                value.set_flags(BinaryFlags::new(len, encoding));
            }

            value.copy_from_slice(unsafe { buf.as_bytes_unchecked() });
        }
        Ok(Rc::into_weak(rc).into())
    }

    /// Writes this charlist codepoint-by-codepoint to a buffer via the provided writer
    ///
    /// By the time this has called, we should already have validated that the list is valid unicode codepoints,
    /// and that the binary we've allocated has enough raw bytes to hold the contents of this charlist. This
    /// should not be called directly otherwise.
    fn write_unicode_charlist_to_buffer<W: fmt::Write>(
        &self,
        writer: &mut W,
    ) -> Result<(), CharlistToBinaryError> {
        for element in self.iter() {
            let Ok(Term::Int(codepoint)) = element else { return Err(CharlistToBinaryError::InvalidList); };
            let codepoint = codepoint.try_into().unwrap();
            let c = unsafe { char::from_u32_unchecked(codepoint) };
            writer.write_char(c).unwrap()
        }
        Ok(())
    }

    /// Same as `write_unicode_charlist_to_buffer`, but for ASCII charlists, which is slightly more efficient
    /// since we can skip the unicode conversion overhead.
    fn write_raw_charlist_to_buffer<A: Allocator>(
        &self,
        buf: &mut BitVec<A>,
    ) -> Result<(), CharlistToBinaryError> {
        for element in self.iter() {
            let Ok(Term::Int(byte)) = element else { return Err(CharlistToBinaryError::InvalidList); };
            buf.push_byte(byte.try_into().unwrap());
        }
        Ok(())
    }

    /// This function walks the entire list, calculating the total bytes required to hold all of the characters,
    /// as well as what encoding is suitable for the charlist.
    ///
    /// If this list is not a charlist, or is an improper list, None is returned.
    fn get_charlist_size_and_encoding(&self) -> Option<(usize, Encoding)> {
        let mut len = 0;
        let mut encoding = Encoding::Utf8;
        for element in self.iter() {
            match element.ok()? {
                Term::Int(codepoint) => match encoding {
                    // If we think we have a valid utf-8 charlist, we do some extra validation
                    Encoding::Utf8 => {
                        match codepoint.try_into() {
                            Ok(codepoint) => match char::from_u32(codepoint) {
                                Some(_) => {
                                    len += len_utf8(codepoint);
                                }
                                None if codepoint > 255 => {
                                    // Invalid UTF-8 codepoint and not a valid byte value, this isn't a charlist
                                    return None;
                                }
                                None => {
                                    // This is either a valid latin1 codepoint, or a plain byte, determine which,
                                    // as in both cases we need to update the encoding
                                    len += 1;
                                    if Encoding::is_latin1_byte(codepoint.try_into().unwrap()) {
                                        encoding = Encoding::Latin1;
                                    } else {
                                        encoding = Encoding::Raw;
                                    }
                                }
                            },
                            // The codepoint exceeds the valid range for u32, cannot be a charlist
                            Err(_) => return None,
                        }
                    }
                    // Likewise for Latin1
                    Encoding::Latin1 => {
                        if codepoint > 255 {
                            return None;
                        }
                        len += 1;
                        if !Encoding::is_latin1_byte(codepoint.try_into().unwrap()) {
                            encoding = Encoding::Raw;
                        }
                    }
                    Encoding::Raw => {
                        if codepoint > 255 {
                            return None;
                        }
                        len += 1;
                    }
                },
                _ => return None,
            }
        }

        Some((len, encoding))
    }

    // See https://github.com/erlang/otp/blob/b8e11b6abe73b5f6306e8833511fcffdb9d252b5/erts/emulator/beam/erl_printf_term.c#L117-L140
    fn is_printable_string(&self) -> bool {
        self.iter().all(|result| match result {
            Ok(element) => {
                // See https://github.com/erlang/otp/blob/b8e11b6abe73b5f6306e8833511fcffdb9d252b5/erts/emulator/beam/erl_printf_term.c#L128-L129
                let Ok(c) = element.as_char() else { return false; };
                // https://github.com/erlang/otp/blob/b8e11b6abe73b5f6306e8833511fcffdb9d252b5/erts/emulator/beam/erl_printf_term.c#L132
                c.is_ascii_graphic() || c.is_ascii_whitespace()
            }
            _ => false,
        })
    }
}

impl Eq for Cons {}
impl PartialEq for Cons {
    fn eq(&self, other: &Self) -> bool {
        self.head().eq(&other.head()) && self.tail().eq(&other.tail())
    }
}
impl PartialOrd for Cons {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Cons {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.iter().cmp(other.iter())
    }
}
impl Hash for Cons {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for item in self.iter() {
            item.hash(state);
        }
    }
}
impl Debug for Cons {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use core::fmt::Write;

        f.write_char('[')?;
        for (i, value) in self.iter().enumerate() {
            match value {
                Ok(value) if i > 0 => write!(f, ", {:?}", value)?,
                Ok(value) => write!(f, "{:?}", value)?,
                Err(improper) if i > 0 => write!(f, " | {:?}", improper)?,
                Err(improper) => write!(f, "{:?}", improper)?,
            }
        }
        f.write_char(']')
    }
}
impl Display for Cons {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use core::fmt::Write;

        // See https://github.com/erlang/otp/blob/b8e11b6abe73b5f6306e8833511fcffdb9d252b5/erts/emulator/beam/erl_printf_term.c#L423-443
        if self.is_printable_string() {
            f.write_char('\"')?;

            for result in self.iter() {
                // `is_printable_string` guarantees all Ok
                let element = result.unwrap();
                match element.try_into().unwrap() {
                    '\n' => f.write_str("\\\n")?,
                    '\"' => f.write_str("\\\"")?,
                    c => f.write_char(c)?,
                }
            }

            f.write_char('\"')
        } else {
            f.write_char('[')?;

            for (i, value) in self.iter().enumerate() {
                match value {
                    Ok(value) if i > 0 => write!(f, ", {}", value)?,
                    Ok(value) => write!(f, "{}", value)?,
                    Err(improper) if i > 0 => write!(f, " | {}", improper)?,
                    Err(improper) => write!(f, "{}", improper)?,
                }
            }

            f.write_char(']')
        }
    }
}

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct ImproperList {
    pub tail: Term,
}
impl Debug for ImproperList {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Debug::fmt(&self.tail, f)
    }
}
impl Display for ImproperList {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        Display::fmt(&self.tail, f)
    }
}

pub struct Iter<'a> {
    head: Option<Result<Term, ImproperList>>,
    tail: Option<OpaqueTerm>,
    _marker: PhantomData<&'a Cons>,
}
impl Iter<'_> {
    fn new(cons: &Cons) -> Self {
        Self {
            head: Some(Ok(cons.head())),
            tail: Some(cons.tail),
            _marker: PhantomData,
        }
    }
}

impl core::iter::FusedIterator for Iter<'_> {}

impl Iterator for Iter<'_> {
    type Item = Result<Term, ImproperList>;

    fn next(&mut self) -> Option<Self::Item> {
        let next = self.head.take();

        match next {
            None => next,
            Some(Err(_)) => {
                self.head = None;
                self.tail = None;
                next
            }
            Some(Ok(Term::Nil)) if self.tail.is_none() => {
                self.head = None;
                self.tail = None;
                None
            }
            next => {
                let tail = self.tail.unwrap();
                match tail.into() {
                    Term::Nil => {
                        self.head = Some(Ok(Term::Nil));
                        self.tail = None;
                        next
                    }
                    Term::Cons(ptr) => {
                        let cons = unsafe { ptr.as_ref() };
                        self.head = Some(Ok(cons.head()));
                        self.tail = Some(cons.tail);
                        next
                    }
                    Term::None => panic!("invalid none value found in list"),
                    tail => {
                        self.head = Some(Err(ImproperList { tail }));
                        self.tail = None;
                        next
                    }
                }
            }
        }
    }
}

pub struct ListBuilder<'a, H: Heap> {
    heap: &'a H,
    tail: Option<NonNull<Cons>>,
}
impl<'a, H: Heap> ListBuilder<'a, H> {
    pub fn new(heap: &'a H) -> Self {
        Self { heap, tail: None }
    }

    pub fn push(&mut self, value: Term) -> Result<(), AllocError> {
        let value = value.clone_to_heap(self.heap)?.into();
        match self.tail.take() {
            None => {
                // This is the first value pushed, so we need to allocate a new cell
                let cell = Cons::new_in(self.heap)?;
                unsafe {
                    cell.as_ptr().write(Cons {
                        head: value,
                        tail: OpaqueTerm::NIL,
                    });
                }
                self.tail = Some(cell.cast());
            }
            Some(tail) => {
                // We're consing a new element to an existing cell
                let cell = Cons::new_in(self.heap)?;
                unsafe {
                    cell.as_ptr().write(Cons {
                        head: value,
                        tail: tail.into(),
                    });
                }
                self.tail = Some(cell.cast());
            }
        }
        Ok(())
    }

    pub fn finish(mut self) -> Option<NonNull<Cons>> {
        self.tail.take()
    }
}

#[inline]
fn len_utf8(code: u32) -> usize {
    const MAX_ONE_B: u32 = 0x80;
    const MAX_TWO_B: u32 = 0x800;
    const MAX_THREE_B: u32 = 0x10000;

    if code < MAX_ONE_B {
        1
    } else if code < MAX_TWO_B {
        2
    } else if code < MAX_THREE_B {
        3
    } else {
        4
    }
}

#[cfg(test)]
mod test {
    use super::*;

    use crate::process::Process;
    use crate::term::ProcessId;

    #[test]
    fn list_builder_builds_proper_lists() {
        let process = Process::new(None, ProcessId::next(), "root:init/0".parse().unwrap());
        let mut builder = ListBuilder::new(&process);
        builder.push(Term::Int(3)).unwrap();
        builder.push(Term::Int(2)).unwrap();
        builder.push(Term::Int(1)).unwrap();
        builder.push(Term::Int(0)).unwrap();
        let ptr = builder.finish().unwrap();
        let list = unsafe { ptr.as_ref() };

        let mut iter = list.iter();
        assert_eq!(iter.next(), Some(Ok(Term::Int(0))));
        assert_eq!(iter.next(), Some(Ok(Term::Int(1))));
        assert_eq!(iter.next(), Some(Ok(Term::Int(2))));
        assert_eq!(iter.next(), Some(Ok(Term::Int(3))));
        assert_eq!(iter.next(), None);
        assert_eq!(iter.next(), None);
    }
}
