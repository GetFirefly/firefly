use alloc::collections::vec_deque::VecDeque;
use alloc::vec::Vec;
use core::intrinsics::{likely, unlikely};
use core::mem;
use core::str::FromStr;

use firefly_alloc::heap::Heap;
use firefly_binary::{BinaryFlags, BitVec, Bitstring, Encoding, Selection};

use crate::function::ErlangResult;
use crate::gc::{garbage_collect, Gc, RootSet};
use crate::process::ProcessLock;
use crate::term::*;

#[export_name = "erlang:binary_part/2"]
pub extern "C-unwind" fn binary_part2(
    process: &mut ProcessLock,
    binary: OpaqueTerm,
    start_length: OpaqueTerm,
) -> ErlangResult {
    let start_length_tuple = tuple_with_arity_or_badarg!(process, start_length.into(), 2);

    binary_part3(
        process,
        binary,
        start_length_tuple[0],
        start_length_tuple[1],
    )
}

#[export_name = "erlang:binary_part/3"]
pub extern "C-unwind" fn binary_part3(
    process: &mut ProcessLock,
    mut binary: OpaqueTerm,
    start_term: OpaqueTerm,
    length_term: OpaqueTerm,
) -> ErlangResult {
    let needed = mem::size_of::<BitSlice>();
    let heap_available = process.heap_available();
    if heap_available < needed {
        let mut roots = RootSet::default();
        roots += &mut binary as *mut OpaqueTerm;
        assert!(garbage_collect(process, roots).is_ok());
    }

    let start = usize_or_badarg!(process, start_term.into());
    let len = isize_or_badarg!(process, length_term.into());
    let bin: Term = binary.into();
    let bin = binary_or_badarg!(process, bin);

    match bin.select_binary_part(start, len) {
        Ok(selection) => {
            let selection = unsafe { mem::transmute::<_, Selection<'static>>(selection) };
            let slice = BitSlice::from_selection(binary, selection);
            let part = Gc::new_in(slice, process).unwrap();
            ErlangResult::Ok(part.into())
        }
        Err(_) => badarg!(process, binary),
    }
}

#[export_name = "erlang:atom_to_binary/1"]
pub extern "C-unwind" fn atom_to_binary1(
    process: &mut ProcessLock,
    atom: OpaqueTerm,
) -> ErlangResult {
    atom_to_binary2(process, atom, atoms::Utf8.into())
}

#[export_name = "erlang:atom_to_binary/2"]
pub extern "C-unwind" fn atom_to_binary2(
    process: &mut ProcessLock,
    atom: OpaqueTerm,
    encoding_term: OpaqueTerm,
) -> ErlangResult {
    let atom = atom_or_badarg!(process, atom.into());
    let encoding = atom_or_badarg!(process, encoding_term.into());
    if let Err(_) = Encoding::from_str(encoding.as_str()) {
        badarg!(process, encoding_term);
    }

    // TODO: If we generate AtomData to point to a constant BinaryData,
    // this can be converted to a Term::ConstantBinary which is much more
    // efficient and avoids any allocation. To keep things simple right
    // now, we're allocating a copy of the data to the process heap
    let s = atom.as_str();
    let bytes = s.as_bytes();
    if bytes.len() > BinaryData::MAX_HEAP_BYTES {
        let bin = BinaryData::from_str(s);
        ErlangResult::Ok(bin.into())
    } else {
        let mut layout = LayoutBuilder::new();
        layout.build_heap_binary(bytes.len());
        let needed = layout.finish().size();
        if needed > process.heap_available() {
            process.gc_needed = needed;
            assert!(garbage_collect(process, Default::default()).is_ok());
        }

        let bin = BinaryData::from_small_str(s, process).unwrap();
        ErlangResult::Ok(bin.into())
    }
}

#[export_name = "erlang:binary_to_atom/2"]
pub extern "C-unwind" fn binary_to_atom2(
    process: &mut ProcessLock,
    binary: OpaqueTerm,
    encoding_term: OpaqueTerm,
) -> ErlangResult {
    let bin: Term = binary.into();
    let bin = binary_or_badarg!(process, bin);
    let encoding = atom_or_badarg!(process, encoding_term.into());
    if let Err(_) = Encoding::from_str(encoding.as_str()) {
        badarg!(process, encoding_term);
    }

    if bin.is_aligned() {
        if let Some(s) = bin.as_str() {
            if let Ok(atom) = Atom::try_from(s) {
                return ErlangResult::Ok(atom.into());
            }
        }
    } else {
        let bytes = bin.bytes().collect::<Vec<_>>();
        if let Ok(atom) = Atom::try_from(bytes.as_slice()) {
            return ErlangResult::Ok(atom.into());
        }
    }
    badarg!(process, binary);
}

#[export_name = "erlang:binary_to_existing_atom/2"]
pub extern "C-unwind" fn binary_to_existing_atom2(
    process: &mut ProcessLock,
    binary: OpaqueTerm,
    encoding_term: OpaqueTerm,
) -> ErlangResult {
    let bin: Term = binary.into();
    let bin = binary_or_badarg!(process, bin);
    let encoding = atom_or_badarg!(process, encoding_term.into());
    if let Err(_) = Encoding::from_str(encoding.as_str()) {
        badarg!(process, encoding_term);
    }

    if bin.is_aligned() {
        if let Some(s) = bin.as_str() {
            if let Ok(atom) = Atom::try_from_str_existing(s) {
                return ErlangResult::Ok(atom.into());
            }
        }
    } else {
        let bytes = bin.bytes().collect::<Vec<_>>();
        if let Ok(atom) = Atom::try_from_latin1_bytes_existing(bytes.as_slice()) {
            return ErlangResult::Ok(atom.into());
        }
    }
    badarg!(process, binary);
}

#[export_name = "erlang:binary_to_float/1"]
pub extern "C-unwind" fn binary_to_float1(
    process: &mut ProcessLock,
    binary: OpaqueTerm,
) -> ErlangResult {
    let bin: Term = binary.into();
    let bin = binary_or_badarg!(process, bin);

    if let Some(s) = bin.as_str() {
        if let Ok(f) = s.parse::<Float>() {
            return ErlangResult::Ok(f.into());
        }
    }
    badarg!(process, binary);
}

#[export_name = "erlang:binary_to_integer/1"]
pub extern "C-unwind" fn binary_to_integer1(
    process: &mut ProcessLock,
    binary: OpaqueTerm,
) -> ErlangResult {
    binary_to_integer2(process, binary, Term::Int(10).into())
}

#[export_name = "erlang:binary_to_integer/2"]
pub extern "C-unwind" fn binary_to_integer2(
    process: &mut ProcessLock,
    binary: OpaqueTerm,
    base_term: OpaqueTerm,
) -> ErlangResult {
    use core::num::IntErrorKind;

    let base = usize_or_badarg!(process, base_term.into());
    if !(2..=36).contains(&base) {
        badarg!(process, base_term);
    }
    let base = base as u32;

    let bin: Term = binary.into();
    let bin = binary_or_badarg!(process, bin);
    if let Some(s) = bin.as_str() {
        match i64::from_str_radix(s, base) {
            Ok(i) if OpaqueTerm::is_small_integer(i) => {
                return ErlangResult::Ok(Term::Int(i).into())
            }
            Ok(i) => match Gc::<BigInt>::new_uninit_in(process) {
                Ok(mut boxed) => unsafe {
                    boxed.write(BigInt::from(i));
                    return ErlangResult::Ok(boxed.assume_init().into());
                },
                Err(_) => {
                    assert!(garbage_collect(process, Default::default()).is_ok());
                    let boxed = Gc::new_in(BigInt::from(i), process).unwrap();
                    return ErlangResult::Ok(boxed.into());
                }
            },
            Err(e) => match e.kind() {
                IntErrorKind::PosOverflow | IntErrorKind::NegOverflow => {
                    use firefly_number::Num;
                    if let Ok(i) = BigInt::from_str_radix(s, base) {
                        match Gc::<BigInt>::new_uninit_in(process) {
                            Ok(mut boxed) => unsafe {
                                boxed.write(i);
                                return ErlangResult::Ok(boxed.assume_init().into());
                            },
                            Err(_) => {
                                assert!(garbage_collect(process, Default::default()).is_ok());
                                let boxed = Gc::new_in(i, process).unwrap();
                                return ErlangResult::Ok(boxed.into());
                            }
                        }
                    }
                }
                _ => badarg!(process, binary),
            },
        }
    }

    badarg!(process, binary);
}

#[export_name = "erlang:bitstring_to_list/1"]
pub extern "C-unwind" fn bitstring_to_list1(
    process: &mut ProcessLock,
    mut bitstring: OpaqueTerm,
) -> ErlangResult {
    let Some(flags) = bitstring.binary_flags() else { badarg!(process, bitstring); };
    let needed = {
        let mut layout = LayoutBuilder::new();
        if flags.is_bitstring() {
            layout.build_heap_binary(1);
        }
        layout.build_list(flags.size());
        layout.finish().size()
    };
    let available = process.heap_available();
    if available < needed {
        process.gc_needed = needed - available;
        let mut roots = RootSet::default();
        roots += &mut bitstring as *mut OpaqueTerm;
        assert!(garbage_collect(process, roots).is_ok());
    }

    let bits: Term = bitstring.into();
    let bits = bits.as_bitstring().unwrap();

    let selection = bits.select_all();
    let (bytes, maybe_partial) = selection.to_maybe_partial_bytes();
    match maybe_partial {
        None => {
            let list = Cons::from_bytes(&bytes, process)
                .unwrap()
                .map(Term::Cons)
                .unwrap_or(Term::Nil);
            ErlangResult::Ok(list.into())
        }
        Some(partial_byte) => {
            let mut rest = BinaryData::with_capacity_small(1, process).unwrap();
            rest[0] = partial_byte.byte();
            unsafe {
                let flags = BinaryFlags::new(0, Encoding::Raw)
                    .with_trailing_bits(partial_byte.size as usize);
                rest.set_flags(flags);
            }
            let mut builder = ListBuilder::new_improper(rest.into(), process);
            for byte in bytes.iter().rev().copied() {
                builder.push(Term::Int(byte as i64)).unwrap();
            }
            let list = builder.finish().map(Term::Cons).unwrap_or(Term::Nil);
            ErlangResult::Ok(list.into())
        }
    }
}

#[export_name = "erlang:binary_to_list/1"]
pub extern "C-unwind" fn binary_to_list1(
    process: &mut ProcessLock,
    mut binary: OpaqueTerm,
) -> ErlangResult {
    let Some(flags) = binary.binary_flags() else { badarg!(process, binary); };
    if flags.is_bitstring() {
        badarg!(process, binary);
    }

    let needed = {
        let mut layout = LayoutBuilder::new();
        layout.build_list(flags.size());
        layout.finish().size()
    };
    let available = process.heap_available();
    if available < needed {
        process.gc_needed = needed - available;
        let mut roots = RootSet::default();
        roots += &mut binary as *mut OpaqueTerm;
        assert!(garbage_collect(process, roots).is_ok());
    }

    let bin: Term = binary.into();
    let bin = bin.as_binary().unwrap();

    let selection = bin.select_all();
    let bytes = selection.to_bytes();
    let list = Cons::from_bytes(&bytes, process)
        .unwrap()
        .map(Term::Cons)
        .unwrap_or(Term::Nil);
    ErlangResult::Ok(list.into())
}

#[export_name = "erlang:binary_to_list/3"]
pub extern "C-unwind" fn binary_to_list3(
    process: &mut ProcessLock,
    mut binary: OpaqueTerm,
    start_term: OpaqueTerm,
    end_term: OpaqueTerm,
) -> ErlangResult {
    let mut start = usize_or_badarg!(process, start_term.into());
    if start < 1 {
        badarg!(process, start_term);
    }
    let mut end = usize_or_badarg!(process, end_term.into());
    if end < 1 {
        badarg!(process, start_term);
    }
    if start > end {
        badarg!(process, end_term);
    }
    start -= 1;
    end -= 1;
    let len = end - start;
    let needed = {
        let mut layout = LayoutBuilder::new();
        layout.build_list(len);
        layout.finish().size()
    };
    let available = process.heap_available();
    if available < needed {
        process.gc_needed = needed - available;
        let mut roots = RootSet::default();
        roots += &mut binary as *mut OpaqueTerm;
        assert!(garbage_collect(process, roots).is_ok());
    }

    let bin: Term = binary.into();
    let bin = binary_or_badarg!(process, bin);
    if let Ok(selection) = bin.select_bytes_at(start, len) {
        let bytes = selection.to_bytes();
        let list = Cons::from_bytes(&bytes, process)
            .unwrap()
            .map(Term::Cons)
            .unwrap_or(Term::Nil);
        ErlangResult::Ok(list.into())
    } else {
        let byte_size = bin.byte_size();
        let blame = if start > byte_size {
            start_term
        } else {
            end_term
        };
        badarg!(process, blame);
    }
}

#[export_name = "erlang:bit_size/1"]
pub extern "C-unwind" fn bit_size1(
    process: &mut ProcessLock,
    bitstring: OpaqueTerm,
) -> ErlangResult {
    let bits: Term = bitstring.into();
    let bits = bitstring_or_badarg!(process, bits);
    let bit_size: i64 = bits.bit_size().try_into().unwrap();
    ErlangResult::Ok(Term::Int(bit_size).into())
}

#[export_name = "erlang:byte_size/1"]
pub extern "C-unwind" fn byte_size1(
    process: &mut ProcessLock,
    bitstring: OpaqueTerm,
) -> ErlangResult {
    let bits: Term = bitstring.into();
    let bits = bitstring_or_badarg!(process, bits);
    let byte_size: i64 = bits.byte_size().try_into().unwrap();
    ErlangResult::Ok(Term::Int(byte_size).into())
}

#[export_name = "erlang:split_binary/2"]
pub extern "C-unwind" fn split_binary2(
    process: &mut ProcessLock,
    mut binary: OpaqueTerm,
    pos_term: OpaqueTerm,
) -> ErlangResult {
    let pos = usize_or_badarg!(process, pos_term.into());
    let Some(flags) = binary.binary_flags() else { badarg!(process, binary); };
    if flags.is_bitstring() {
        badarg!(process, binary);
    }
    let byte_size = flags.size();

    let mut layout = LayoutBuilder::new();
    match (pos, byte_size) {
        (0, 0) => (),
        (0, _) => {
            layout.build_ref_binary();
        }
        (n, byte_size) if n > byte_size => badarg!(process, pos_term),
        (_, _) => {
            layout.build_ref_binary();
            layout.build_ref_binary();
        }
    }
    layout.build_tuple(2);

    let needed = layout.finish().size();
    if needed > process.heap_available() {
        let mut roots = RootSet::default();
        roots += &mut binary as *mut OpaqueTerm;
        assert!(garbage_collect(process, Default::default()).is_ok());
    }

    let bin: Term = binary.into();
    let bin = binary_or_badarg!(process, bin);

    let (first, second) = match bin.select_split(pos).unwrap() {
        (Selection::Empty, Selection::Empty) => (
            Term::ConstantBinary(EMPTY_BIN),
            Term::ConstantBinary(EMPTY_BIN),
        ),
        (Selection::Empty, selection) => {
            let selection = unsafe { mem::transmute::<_, Selection<'static>>(selection) };
            let second = Gc::new_in(BitSlice::from_selection(binary, selection), process).unwrap();
            (Term::ConstantBinary(EMPTY_BIN), Term::RefBinary(second))
        }
        (selection, Selection::Empty) => {
            let selection = unsafe { mem::transmute::<_, Selection<'static>>(selection) };
            let first = Gc::new_in(BitSlice::from_selection(binary, selection), process).unwrap();
            (Term::RefBinary(first), Term::ConstantBinary(EMPTY_BIN))
        }
        (first_selection, second_selection) => {
            let first_selection =
                unsafe { mem::transmute::<_, Selection<'static>>(first_selection) };
            let second_selection =
                unsafe { mem::transmute::<_, Selection<'static>>(second_selection) };
            let first =
                Gc::new_in(BitSlice::from_selection(binary, first_selection), process).unwrap();
            let second =
                Gc::new_in(BitSlice::from_selection(binary, second_selection), process).unwrap();
            (Term::RefBinary(first), Term::RefBinary(second))
        }
    };

    let tuple = Tuple::from_slice(&[first.into(), second.into()], process).unwrap();
    ErlangResult::Ok(tuple.into())
}

#[export_name = "erlang:iolist_size/1"]
pub extern "C-unwind" fn iolist_size1(process: &mut ProcessLock, item: OpaqueTerm) -> ErlangResult {
    let Ok(size) = iolist_size(item.into()) else { badarg!(process, item); };

    let result: Result<i64, _> = size.try_into();
    if unlikely(result.is_err()) {
        return ErlangResult::Ok(convert_to_bigint(process, size).into());
    }

    let result = unsafe { result.unwrap_unchecked() };
    if likely(OpaqueTerm::is_small_integer(result)) {
        return ErlangResult::Ok(Term::Int(result).into());
    }

    ErlangResult::Ok(convert_to_bigint(process, result).into())
}

#[export_name = "erlang:iolist_to_binary/1"]
pub extern "C-unwind" fn iolist_to_binary1(
    process: &mut ProcessLock,
    item: OpaqueTerm,
) -> ErlangResult {
    let mut bitvec = BitVec::with_capacity(1024);
    let mut worklist = VecDeque::new();
    worklist.push_back(item.into());
    while let Some(term) = worklist.pop_front() {
        match term {
            Term::Nil => continue,
            Term::Cons(cons) => match cons.tail.into() {
                Term::Nil => {
                    worklist.push_front(cons.head.into());
                }
                tail => {
                    worklist.push_front(tail);
                    worklist.push_front(cons.head.into());
                }
            },
            Term::Int(i) if (0..256).contains(&i) => {
                bitvec.push_byte(i as u8);
            }
            term @ (Term::HeapBinary(_)
            | Term::RcBinary(_)
            | Term::RefBinary(_)
            | Term::ConstantBinary(_)) => {
                let bin = term.as_binary().ok_or(())?;
                bitvec.push_selection(bin.select_all());
            }
            _ => badarg!(process, item),
        }
    }

    let byte_size = bitvec.byte_size();

    if unlikely(byte_size == 0) {
        return ErlangResult::Ok(Term::ConstantBinary(EMPTY_BIN).into());
    }

    let mut layout = LayoutBuilder::new();
    layout.build_binary(byte_size);
    let needed = layout.finish().size();
    if needed > process.heap_available() {
        assert!(garbage_collect(process, Default::default()).is_ok());
    }

    let bytes = unsafe { bitvec.as_bytes_unchecked() };
    if byte_size > BinaryData::MAX_HEAP_BYTES {
        ErlangResult::Ok(BinaryData::from_bytes(bytes).into())
    } else {
        ErlangResult::Ok(BinaryData::from_small_bytes(bytes, process).unwrap().into())
    }
}

#[export_name = "erlang:iolist_to_iovec/1"]
pub extern "C-unwind" fn iolist_to_iovec(
    process: &mut ProcessLock,
    item: OpaqueTerm,
) -> ErlangResult {
    let mut run: Option<BitVec> = None;
    let mut parts = Vec::new();
    match item.into() {
        Term::Nil => return ErlangResult::Ok(OpaqueTerm::NIL),
        Term::Cons(cons) => {
            if extend_iovec_parts(&cons, &mut run, &mut parts).is_err() {
                badarg!(process, item);
            }
        }
        bin @ (Term::HeapBinary(_)
        | Term::RcBinary(_)
        | Term::RefBinary(_)
        | Term::ConstantBinary(_)) => {
            parts.push(IoVecPart::Term(bin));
        }
        _ => badarg!(process, item),
    }

    let mut layout = LayoutBuilder::new();
    for part in parts.iter() {
        if let IoVecPart::Raw(bitvec) = part {
            layout.build_binary(bitvec.byte_size());
        }
    }

    layout.build_list(parts.len());
    let needed = layout.finish().size();
    if needed > process.heap_available() {
        let root_parts = parts.as_mut_slice();
        let mut roots = RootSet::default();
        for root in root_parts.iter_mut() {
            if let IoVecPart::Term(ref mut term) = root {
                roots += term as *mut Term;
            }
        }
        assert!(garbage_collect(process, roots).is_ok());
    }

    for part in parts.iter_mut() {
        match part {
            IoVecPart::Term(_) => continue,
            IoVecPart::Raw(bitvec) => {
                if bitvec.byte_size() > BinaryData::MAX_HEAP_BYTES {
                    let bin = BinaryData::from_bytes(unsafe { bitvec.as_bytes_unchecked() });
                    *part = IoVecPart::Term(Term::RcBinary(bin));
                } else {
                    let bin = BinaryData::from_small_bytes(
                        unsafe { bitvec.as_bytes_unchecked() },
                        process,
                    )
                    .unwrap();
                    *part = IoVecPart::Term(Term::HeapBinary(bin));
                }
            }
        }
    }

    let mut builder = ListBuilder::new(process);
    for part in parts.drain(..).rev() {
        let IoVecPart::Term(term) = part else { unreachable!() };
        unsafe {
            builder.push_unsafe(term).unwrap();
        }
    }

    let result = builder.finish().map(Term::Cons).unwrap_or(Term::Nil);
    ErlangResult::Ok(result.into())
}

enum IoVecPart {
    Raw(BitVec),
    Term(Term),
}

fn extend_iovec_parts(
    cons: &Cons,
    run: &mut Option<BitVec>,
    parts: &mut Vec<IoVecPart>,
) -> Result<(), ()> {
    for maybe_improper in cons.iter() {
        match maybe_improper.map_err(|improper| improper.tail) {
            Ok(term) | Err(term) => match term {
                Term::Int(i) if (0..256).contains(&i) => {
                    if let Some(bits) = run.as_mut() {
                        bits.push_byte(i as u8);
                    } else {
                        let mut bits = BitVec::new();
                        bits.push_byte(i as u8);
                        *run = Some(bits);
                    }
                }
                Term::HeapBinary(bin) => {
                    if let Some(bits) = run.as_mut() {
                        bits.push_bytes(unsafe { bin.as_bytes_unchecked() });
                    } else {
                        let mut bits = BitVec::with_capacity(bin.byte_size());
                        bits.push_bytes(unsafe { bin.as_bytes_unchecked() });
                        *run = Some(bits);
                    }
                }
                bin @ Term::RcBinary(_) => {
                    if let Some(bits) = run.take() {
                        parts.push(IoVecPart::Raw(bits));
                    }
                    parts.push(IoVecPart::Term(bin));
                }
                Term::RefBinary(bin) if bin.byte_size() > BinaryData::MAX_HEAP_BYTES => {
                    if let Some(bits) = run.take() {
                        parts.push(IoVecPart::Raw(bits));
                    }
                    parts.push(IoVecPart::Term(Term::RefBinary(bin)));
                }
                Term::RefBinary(bin) => {
                    if let Some(bits) = run.as_mut() {
                        if bin.is_aligned() {
                            bits.push_bytes(unsafe { bin.as_bytes_unchecked() });
                        } else {
                            for byte in bin.bytes() {
                                bits.push_byte(byte);
                            }
                        }
                    } else {
                        let mut bits = BitVec::with_capacity(bin.byte_size());
                        if bin.is_aligned() {
                            bits.push_bytes(unsafe { bin.as_bytes_unchecked() });
                        } else {
                            for byte in bin.bytes() {
                                bits.push_byte(byte);
                            }
                        }
                        *run = Some(bits);
                    }
                }
                Term::ConstantBinary(bin) if bin.byte_size() > BinaryData::MAX_HEAP_BYTES => {
                    if let Some(bits) = run.take() {
                        parts.push(IoVecPart::Raw(bits));
                    }
                    parts.push(IoVecPart::Term(Term::ConstantBinary(bin)));
                }
                Term::ConstantBinary(bin) => {
                    if let Some(bits) = run.as_mut() {
                        bits.push_bytes(unsafe { bin.as_bytes_unchecked() });
                    } else {
                        let mut bits = BitVec::with_capacity(bin.byte_size());
                        bits.push_bytes(unsafe { bin.as_bytes_unchecked() });
                        *run = Some(bits);
                    }
                }
                Term::Nil => continue,
                Term::Cons(nested) => {
                    extend_iovec_parts(&nested, run, parts)?;
                }
                _ => return Err(()),
            },
        }
    }

    Ok(())
}

fn iolist_size(item: Term) -> Result<usize, ()> {
    let mut size = 0;

    let mut worklist = VecDeque::new();
    worklist.push_back(item);
    while let Some(term) = worklist.pop_front() {
        match term {
            Term::Nil => continue,
            Term::Cons(cons) => {
                for maybe_improper in cons.iter() {
                    match maybe_improper {
                        Ok(element) => worklist.push_back(element),
                        Err(improper) => worklist.push_back(improper.tail),
                    }
                }
            }
            Term::Int(i) if (0..256).contains(&i) => {
                size += 1;
            }
            term @ (Term::HeapBinary(_)
            | Term::RcBinary(_)
            | Term::RefBinary(_)
            | Term::ConstantBinary(_)) => {
                let bin = term.as_binary().ok_or(())?;
                size += bin.byte_size();
            }
            _ => return Err(()),
        }
    }

    Ok(size)
}

#[inline(never)]
#[cold]
fn convert_to_bigint<I: Into<BigInt>>(process: &mut ProcessLock, integer: I) -> Gc<BigInt> {
    let mut layout = LayoutBuilder::new();
    layout.build_bigint();
    let needed = layout.finish().size();
    if needed > process.heap_available() {
        assert!(garbage_collect(process, Default::default()).is_ok());
    }
    Gc::new_in(integer.into(), process).unwrap()
}
