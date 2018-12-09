///! Data types:
///!
///! Eterm: A tagged erlang term (possibly 64 bits)
///! UInt:  An unsigned integer exactly as large as an Eterm.
///! SInt:  A signed integer exactly as large as an eterm and therefor large
///!        enough to hold the return value of the signed_val() macro.
///! UWord: An unsigned integer at least as large as a void * and also as large
///!          or larger than an Eterm
///! SWord: A signed integer at least as large as a void * and also as large
///!          or larger than an Eterm
///! Uint32: An unsigned integer of 32 bits exactly
///! Sint32: A signed integer of 32 bits exactly
///! Uint16: An unsigned integer of 16 bits exactly
///! Sint16: A signed integer of 16 bits exactly.
use lumen_macros::*;

#[tag_type]
pub enum EtermType {
    Undefined,
    Integer,
    Float
}

#[tagged_with(EtermType)]
#[derive(PartialEq, Debug)]
pub enum Eterm {
    Integer(u64),
    Float(f64)
}

#[cfg(test)]
mod test {
    use super::*;

    fn type_tagging() {
        let term = &Eterm::Integer(1423);
        assert_eq!(EtermType::Undefined, Eterm::tag_of(term as *const Eterm));
        let term_tagged = Eterm::tag(term as *const Eterm, EtermType::Integer);
        assert_eq!(EtermType::Integer, Eterm::tag_of(term_tagged));
        let term_tagged = unsafe { std::mem::transmute::<*const Eterm, &Eterm>(term_tagged) };
        assert_eq!(term, term_tagged);
    }
}
