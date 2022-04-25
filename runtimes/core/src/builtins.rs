use std::convert::TryInto;
use std::panic;

use hashbrown::HashMap;

use liblumen_alloc::erts::exception::{self, badmap, ErlangException, RuntimeException};
use liblumen_alloc::erts::process::ffi::ErlangResult;
use liblumen_alloc::erts::process::trace::Trace;
use liblumen_alloc::erts::term::{binary, prelude::*};
use liblumen_core::sys::Endianness;

use crate::process::current_process;

#[export_name = "__lumen_builtin_raise/2"]
pub extern "C" fn capture_and_raise(kind: Term, reason: Term) -> *mut ErlangException {
    let kind: Atom = kind.decode().unwrap().try_into().unwrap();
    let trace = Trace::capture();
    let exception = match kind.name() {
        "throw" => RuntimeException::Throw(exception::Throw::new(reason, trace, None)),
        "error" => RuntimeException::Error(exception::Error::new(reason, None, trace, None)),
        "exit" => RuntimeException::Exit(exception::Exit::new(reason, trace, None)),
        other => panic!("invalid exception kind: {}", &other),
    };
    current_process().raise(exception)
}

#[export_name = "__lumen_builtin_raise/3"]
pub extern "C" fn raise(kind: Term, reason: Term, trace: *mut Trace) -> *mut ErlangException {
    debug_assert!(!trace.is_null());
    let trace = unsafe { Trace::from_raw(trace) };
    let kind: Atom = kind.decode().unwrap().try_into().unwrap();
    let exception = match kind.name() {
        "throw" => RuntimeException::Throw(exception::Throw::new(reason, trace, None)),
        "error" => RuntimeException::Error(exception::Error::new(reason, None, trace, None)),
        "exit" => RuntimeException::Exit(exception::Exit::new(reason, trace, None)),
        other => panic!("invalid exception kind: {}", &other),
    };
    current_process().raise(exception)
}

#[export_name = "__lumen_build_stacktrace"]
pub extern "C" fn capture_trace() -> *mut Trace {
    let trace = Trace::capture();
    Trace::into_raw(trace)
}

#[export_name = "__lumen_stacktrace_to_term"]
pub extern "C" fn trace_to_term(trace: *mut Trace) -> Term {
    if trace.is_null() {
        return Term::NIL;
    }
    let trace = unsafe { Trace::from_raw(trace) };
    if let Ok(term) = trace.as_term() {
        term
    } else {
        Term::NIL
    }
}

#[export_name = "__lumen_cleanup_exception"]
pub unsafe extern "C" fn cleanup(ptr: *mut ErlangException) {
    let _ = Box::from_raw(ptr);
}

#[export_name = "__lumen_builtin_bigint_from_cstr"]
pub extern "C" fn builtin_bigint_from_cstr(ptr: *const u8, size: usize) -> Term {
    let bytes = unsafe { core::slice::from_raw_parts(ptr, size) };
    let value = BigInteger::from_bytes(bytes).unwrap();
    current_process().integer(value)
}

#[export_name = "__lumen_builtin_map.new"]
pub extern "C" fn builtin_map_new() -> Term {
    current_process().map_from_hash_map(HashMap::default())
}

#[export_name = "__lumen_builtin_map.insert"]
pub extern "C" fn builtin_map_insert(map: Term, key: Term, value: Term) -> ErlangResult {
    let decoded_map: Result<Boxed<Map>, _> = map.decode().unwrap().try_into();
    if let Ok(m) = decoded_map {
        if let Some(new_map) = m.put(key, value) {
            ErlangResult::ok(current_process().map_from_hash_map(new_map))
        } else {
            ErlangResult::ok(map)
        }
    } else {
        let arc_process = current_process();
        let exception = badmap(&arc_process, map, Trace::capture(), None);
        ErlangResult::error(arc_process.raise(exception))
    }
}

#[export_name = "__lumen_builtin_map.update"]
pub extern "C" fn builtin_map_update(map: Term, key: Term, value: Term) -> Term {
    let decoded_map: Result<Boxed<Map>, _> = map.decode().unwrap().try_into();
    if let Ok(m) = decoded_map {
        if let Some(new_map) = m.update(key, value) {
            current_process().map_from_hash_map(new_map)
        } else {
            // TODO: Trigger badkey error
            Term::NONE
        }
    } else {
        Term::NONE
    }
}

#[export_name = "__lumen_builtin_map.is_key"]
pub extern "C" fn builtin_map_is_key(map: Term, key: Term) -> bool {
    let decoded_map: Result<Boxed<Map>, _> = map.decode().unwrap().try_into();
    let m = decoded_map.unwrap();
    m.is_key(key)
}

#[export_name = "__lumen_builtin_map.get"]
pub extern "C" fn builtin_map_get(map: Term, key: Term) -> Term {
    let decoded_map: Result<Boxed<Map>, _> = map.decode().unwrap().try_into();
    let m = decoded_map.unwrap();
    m.get(key).unwrap_or(Term::NONE)
}

#[export_name = "__lumen_builtin_fatal_error"]
pub extern "C" fn builtin_fatal_error() -> ! {
    core::intrinsics::abort();
}

/// Binary Construction
#[export_name = "__lumen_builtin_binary_start"]
pub extern "C" fn builtin_binary_start() -> *mut BinaryBuilder {
    let builder = Box::new(BinaryBuilder::new());
    Box::into_raw(builder)
}

#[export_name = "__lumen_builtin_binary_finish"]
pub extern "C" fn builtin_binary_finish(builder: *mut BinaryBuilder) -> Term {
    let builder = unsafe { Box::from_raw(builder) };
    let bytes = builder.finish();
    current_process().binary_from_bytes(bytes.as_slice())
}

#[export_name = "__lumen_builtin_binary_push_integer"]
pub extern "C" fn builtin_binary_push_integer(
    builder: &mut BinaryBuilder,
    value: Term,
    size: Term,
    unit: u8,
    signed: bool,
    endianness: Endianness,
) -> BinaryPushResult {
    let tt = value.decode().unwrap();
    let val: Result<Integer, _> = tt.try_into();
    let result = if let Ok(i) = val {
        let flags = BinaryPushFlags::new(signed, endianness);
        let bit_size = calculate_bit_size(size, unit, flags).unwrap();
        builder.push_integer(i, bit_size, flags)
    } else {
        Err(())
    };
    BinaryPushResult {
        builder,
        success: result.is_ok(),
    }
}

#[export_name = "__lumen_builtin_binary_push_float"]
pub extern "C" fn builtin_binary_push_float(
    builder: &mut BinaryBuilder,
    value: Term,
    size: Term,
    unit: u8,
    signed: bool,
    endianness: Endianness,
) -> BinaryPushResult {
    let tt = value.decode().unwrap();
    let val: Result<Float, _> = tt.try_into();
    let result = if let Ok(f) = val {
        let flags = BinaryPushFlags::new(signed, endianness);
        let bit_size = calculate_bit_size(size, unit, flags).unwrap();
        builder.push_float(f.into(), bit_size, flags)
    } else {
        Err(())
    };
    BinaryPushResult {
        builder,
        success: result.is_ok(),
    }
}

#[export_name = "__lumen_builtin_binary_push_utf8"]
pub extern "C" fn builtin_binary_push_utf8(
    builder: &mut BinaryBuilder,
    value: Term,
) -> BinaryPushResult {
    let tt = value.decode().unwrap();
    let val: Result<SmallInteger, _> = tt.try_into();
    let result = if let Ok(small) = val {
        builder.push_utf8(small.into())
    } else {
        Err(())
    };
    BinaryPushResult {
        builder,
        success: result.is_ok(),
    }
}

#[export_name = "__lumen_builtin_binary_push_utf16"]
pub extern "C" fn builtin_binary_push_utf16(
    builder: &mut BinaryBuilder,
    value: Term,
    signed: bool,
    endianness: Endianness,
) -> BinaryPushResult {
    let tt = value.decode().unwrap();
    let val: Result<SmallInteger, _> = tt.try_into();
    let result = if let Ok(small) = val {
        let flags = BinaryPushFlags::new(signed, endianness);
        builder.push_utf16(small.into(), flags)
    } else {
        Err(())
    };
    BinaryPushResult {
        builder,
        success: result.is_ok(),
    }
}

#[export_name = "__lumen_builtin_binary_push_utf32"]
pub extern "C" fn builtin_binary_push_utf32(
    builder: &mut BinaryBuilder,
    value: Term,
    size: Term,
    unit: u8,
    signed: bool,
    endianness: Endianness,
) -> BinaryPushResult {
    let tt = value.decode().unwrap();
    let result: Result<SmallInteger, _> = tt.try_into();
    if let Ok(small) = result {
        let i: isize = small.into();
        if i > 0x10FFFF || (0xD800 <= i && i <= 0xDFFF) {
            // Invalid utf32 integer
            return BinaryPushResult {
                builder,
                success: false,
            };
        }
        let flags = BinaryPushFlags::new(signed, endianness);
        let bit_size = calculate_bit_size(size, unit, flags).unwrap();
        let success = builder.push_integer(small.into(), bit_size, flags).is_ok();
        BinaryPushResult { builder, success }
    } else {
        BinaryPushResult {
            builder,
            success: false,
        }
    }
}

#[export_name = "__lumen_builtin_binary_push_byte_size_unit"]
pub extern "C" fn builtin_binary_push_byte_size_unit(
    _builder: &mut BinaryBuilder,
    _value: Term,
    _size: Term,
    _unit: u8,
) -> BinaryPushResult {
    unimplemented!()
}

#[export_name = "__lumen_builtin_binary_push_byte_unit"]
pub extern "C" fn builtin_binary_push_byte_unit(
    builder: &mut BinaryBuilder,
    value: Term,
    unit: u8,
) -> BinaryPushResult {
    BinaryPushResult {
        builder,
        success: builder.push_byte_unit(value, unit).is_ok(),
    }
}

#[export_name = "__lumen_builtin_binary_push_bits_size_unit"]
pub extern "C" fn builtin_binary_push_bits_size_unit(
    _builder: &mut BinaryBuilder,
    _value: Term,
    _size: Term,
    _unit: u8,
) -> BinaryPushResult {
    unimplemented!();
}

#[export_name = "__lumen_builtin_binary_push_bits_unit"]
pub extern "C" fn builtin_binary_push_bits_unit(
    builder: &mut BinaryBuilder,
    value: Term,
    unit: u8,
) -> BinaryPushResult {
    BinaryPushResult {
        builder,
        success: builder.push_bits_unit(value, unit).is_ok(),
    }
}

#[export_name = "__lumen_builtin_binary_push_string"]
pub extern "C" fn builtin_binary_push_string(
    builder: &mut BinaryBuilder,
    buffer: *const u8,
    len: usize,
) -> BinaryPushResult {
    let bytes = unsafe { core::slice::from_raw_parts(buffer, len) };
    let success = builder.push_string(bytes).is_ok();
    BinaryPushResult { builder, success }
}

#[export_name = "__lumen_builtin_binary_match.raw"]
pub extern "C" fn builtin_binary_match_raw(bin: Term, unit: u8, size: Term) -> BinaryMatchResult {
    let size_opt = if size.is_none() {
        None
    } else {
        let size_decoded: Result<SmallInteger, _> = size.decode().unwrap().try_into();
        // TODO: Should throw badarg
        let size: usize = size_decoded.unwrap().try_into().unwrap();
        Some(size)
    };
    let result = match bin.decode().unwrap() {
        TypedTerm::HeapBinary(bin) => binary::matcher::match_raw(bin, unit, size_opt),
        TypedTerm::ProcBin(bin) => binary::matcher::match_raw(bin, unit, size_opt),
        TypedTerm::BinaryLiteral(bin) => binary::matcher::match_raw(bin, unit, size_opt),
        TypedTerm::SubBinary(bin) => binary::matcher::match_raw(bin, unit, size_opt),
        TypedTerm::MatchContext(bin) => binary::matcher::match_raw(bin, unit, size_opt),
        _ => Err(()),
    };
    result.unwrap_or_else(|_| BinaryMatchResult::failed())
}

#[export_name = "__lumen_builtin_binary_match.integer"]
pub extern "C" fn builtin_binary_match_integer(
    _bin: Term,
    _signed: bool,
    _endianness: Endianness,
    _unit: u8,
    _size: Term,
) -> BinaryMatchResult {
    unimplemented!()
}

#[export_name = "__lumen_builtin_binary_match.float"]
pub extern "C" fn builtin_binary_match_float(
    _bin: Term,
    _endianness: Endianness,
    _unit: u8,
    _size: Term,
) -> BinaryMatchResult {
    unimplemented!()
}

#[export_name = "__lumen_builtin_binary_match.utf8"]
pub extern "C" fn builtin_binary_match_utf8(_bin: Term, _size: Term) -> BinaryMatchResult {
    unimplemented!()
}

#[export_name = "__lumen_builtin_binary_match.utf16"]
pub extern "C" fn builtin_binary_match_utf16(
    _bin: Term,
    _endianness: Endianness,
    _size: Term,
) -> BinaryMatchResult {
    unimplemented!()
}

#[export_name = "__lumen_builtin_binary_match.utf32"]
pub extern "C" fn builtin_binary_match_utf32(
    _bin: Term,
    _endianness: Endianness,
    _size: Term,
) -> BinaryMatchResult {
    unimplemented!();
}
