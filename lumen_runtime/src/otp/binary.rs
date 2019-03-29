use crate::binary::{heap, sub, PartToList};
use crate::exception::Exception;
use crate::process::Process;
use crate::term::{Tag::*, Term};

/// Converts `binary` to a list of bytes, each representing the value of one byte.
///
/// ## Arguments
///
/// * `binary` - a heap, reference counted, or subbinary.
/// * `position` - 0-based index into the bytes of `binary`.  `position` can be +1 the last index in
///   the binary if `length` is negative.
/// * `length` - the length of the part.  A negative length will begin the part from the end of the
///   of the binary.
///
/// ## Returns
///
/// * `Ok(Term)` - the list of bytes
/// * `Err(BadArgument)` - binary is not a binary; position is invalid; length is invalid.
pub fn bin_to_list(
    binary: Term,
    position: Term,
    length: Term,
    mut process: &mut Process,
) -> Result<Term, Exception> {
    match binary.tag() {
        Boxed => {
            let unboxed: &Term = binary.unbox_reference();

            match unboxed.tag() {
                HeapBinary => {
                    let heap_binary: &heap::Binary = binary.unbox_reference();

                    heap_binary.part_to_list(position, length, &mut process)
                }
                Subbinary => {
                    let subbinary: &sub::Binary = binary.unbox_reference();

                    subbinary.part_to_list(position, length, &mut process)
                }
                _ => Err(bad_argument!()),
            }
        }
        _ => Err(bad_argument!()),
    }
}
