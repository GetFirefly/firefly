use core::fmt;

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Tag<T> 
where
    T: Clone + Copy + PartialEq + Eq,
{
    Atom,
    Pid,
    Port,
    SmallInteger,
    BigInteger,
    Float,
    Tuple,
    List,
    Map,
    Closure,
    ProcBin,
    HeapBinary,
    SubBinary,
    MatchContext,
    ExternalPid,
    ExternalPort,
    ExternalReference,
    Reference,
    ResourceReference,
    Nil,
    None,
    Box,
    Literal,
    Unknown(T)
}

impl<T> fmt::Debug for Tag<T>
where
    T: Clone + Copy + PartialEq + Eq + fmt::Debug + fmt::Binary,
{
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        use Tag::*;

        match self {
            Atom => write!(f, "Atom"),
            Pid => write!(f, "Pid"),
            Port => write!(f, "Port"),
            SmallInteger => write!(f, "SmallInteger"),
            BigInteger => write!(f, "BigInteger"),
            Float => write!(f, "Float"),
            Tuple => write!(f, "Tuple"),
            List => write!(f, "List"),
            Map => write!(f, "Map"),
            Closure => write!(f, "Closure"),
            ProcBin => write!(f, "ProcBin"),
            HeapBinary => write!(f, "HeapBinary"),
            SubBinary => write!(f, "SubBinary"),
            MatchContext => write!(f, "MatchContext"),
            ExternalPid => write!(f, "ExternalPid"),
            ExternalPort => write!(f, "ExternalPort"),
            ExternalReference => write!(f, "ExternalReference"),
            Reference => write!(f, "Reference"),
            ResourceReference => write!(f, "ResourceReference"),
            Nil => write!(f, "Nil"),
            None => write!(f, "None"),
            Box => write!(f, "Box"),
            Literal => write!(f, "Literal"),
            #[cfg(target_pointer_width = "32")]
            Unknown(unknown) => write!(f, "Unknown({:032b})", unknown),
            #[cfg(target_pointer_width = "64")]
            Unknown(unknown) => write!(f, "Unknown({:064b})", unknown),
        }
    }
}
