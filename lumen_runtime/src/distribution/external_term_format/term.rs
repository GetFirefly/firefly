use super::*;

pub fn decode_tagged<'a>(
    process: &Process,
    safe: bool,
    bytes: &'a [u8],
) -> InternalResult<(Term, &'a [u8])> {
    let (tag, after_tag_bytes) = Tag::decode(bytes)?;

    match tag {
        Tag::Atom => atom::decode_term(safe, after_tag_bytes),
        Tag::AtomCacheReference => unimplemented!("{:?}", tag),
        Tag::AtomUTF8 => atom_utf8::decode_term(safe, after_tag_bytes),
        Tag::Binary => binary::decode(process, after_tag_bytes),
        Tag::BitBinary => bit_binary::decode(process, after_tag_bytes),
        Tag::Export => export::decode(process, safe, after_tag_bytes),
        Tag::Float => unimplemented!("{:?}", tag),
        Tag::Function => unimplemented!("{:?}", tag),
        Tag::Integer => integer::decode(process, after_tag_bytes),
        Tag::LargeBig => big::large::decode(process, after_tag_bytes),
        Tag::LargeTuple => tuple::large::decode(process, safe, after_tag_bytes),
        Tag::List => list::decode(process, safe, after_tag_bytes),
        Tag::Map => map::decode(process, safe, after_tag_bytes),
        Tag::NewFloat => new_float::decode(process, after_tag_bytes),
        Tag::NewFunction => new_function::decode(process, safe, after_tag_bytes),
        Tag::NewPID => new_pid::decode_term(process, safe, after_tag_bytes),
        Tag::NewPort => unimplemented!("{:?}", tag),
        Tag::NewReference => unimplemented!("{:?}", tag),
        Tag::NewerReference => newer_reference::decode(process, safe, after_tag_bytes),
        Tag::Nil => Ok((Term::NIL, after_tag_bytes)),
        Tag::PID => pid::decode_term(process, safe, after_tag_bytes),
        Tag::Port => unimplemented!("{:?}", tag),
        Tag::Reference => unimplemented!("{:?}", tag),
        Tag::SmallAtom => small_atom::decode(safe, after_tag_bytes),
        Tag::SmallAtomUTF8 => small_atom_utf8::decode_term(safe, after_tag_bytes),
        Tag::SmallBig => big::small::decode(process, after_tag_bytes),
        Tag::SmallInteger => small_integer::decode(process, after_tag_bytes),
        Tag::SmallTuple => tuple::small::decode(process, safe, after_tag_bytes),
        Tag::String => string::decode(process, after_tag_bytes),
    }
}
