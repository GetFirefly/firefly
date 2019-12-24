#[repr(C)]
pub enum Type {
    Unknown = 0,
    Term,
    Atom,
    Boolean,
    Fixnum,
    BigInt,
    Float,
    FloatPacked,
    Nil,
    Cons,
    Tuple,
    Map,
    Closure,
    HeapBin,
    Box,
    Ref,
}

#[repr(C)]
pub struct Arg {
    pub ty: Type,
    pub span_start: u32,
    pub span_end: u32,
}
