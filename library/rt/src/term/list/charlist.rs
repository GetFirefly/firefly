use alloc::string::String;

use firefly_alloc::heap::Heap;
use firefly_binary::{BinaryFlags, BitVec, Bitstring, Encoding};

use crate::gc::Gc;
use crate::term::{BinaryData, Term};

use super::*;

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum CharlistToBinaryError {
    /// The list isn't a charlist and/or is an improper list
    InvalidList,
    /// Could not allocate enough memory to store the binary
    AllocError,
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
    pub fn from_bytes<H: ?Sized + Heap>(
        bytes: &[u8],
        heap: &H,
    ) -> Result<Option<Gc<Self>>, AllocError> {
        let mut builder = ListBuilder::new(&heap);
        for byte in bytes.iter().rev().copied() {
            builder.push(Term::Int(byte as i64))?;
        }
        Ok(builder.finish())
    }

    /// Constructs a charlist from the given string
    pub fn charlist_from_str<H: ?Sized + Heap>(
        s: &str,
        heap: &H,
    ) -> Result<Option<Gc<Self>>, AllocError> {
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
    pub fn charlist_to_binary<H: ?Sized + Heap>(
        &self,
        heap: &H,
    ) -> Result<Term, CharlistToBinaryError> {
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

    /// Writes this charlist to a Gc<T>, i.e. allocates on a process heap
    fn charlist_to_heap_binary<H: ?Sized + Heap>(
        &self,
        len: usize,
        encoding: Encoding,
        heap: &H,
    ) -> Result<Term, CharlistToBinaryError> {
        let mut buf = BitVec::with_capacity(len);
        if encoding == Encoding::Utf8 {
            self.write_unicode_charlist_to_buffer(&mut buf)?;
        } else {
            self.write_raw_charlist_to_buffer(&mut buf)?;
        }
        let byte_size = buf.byte_size();
        let mut bin = BinaryData::with_capacity_small(byte_size, heap)
            .map_err(|_| CharlistToBinaryError::AllocError)?;
        {
            unsafe {
                bin.set_flags(BinaryFlags::new(byte_size, encoding));
            }
            bin.copy_from_slice(unsafe { buf.as_bytes_unchecked() });
        }
        Ok(Term::HeapBinary(bin))
    }

    /// Writes this charlist to an Arc, i.e. allocates on the global heap
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
        Ok(Term::RcBinary(BinaryData::from_bytes(unsafe {
            buf.as_bytes_unchecked()
        })))
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
                    _ => unreachable!(),
                },
                _ => return None,
            }
        }

        Some((len, encoding))
    }

    // See https://github.com/erlang/otp/blob/b8e11b6abe73b5f6306e8833511fcffdb9d252b5/erts/emulator/beam/erl_printf_term.c#L117-L140
    pub fn is_printable_string(&self) -> bool {
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
