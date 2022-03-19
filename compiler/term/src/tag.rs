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
    Unknown(T),
}
impl<T> Tag<T>
where
    T: Clone + Copy + PartialEq + Eq,
{
    #[inline]
    pub fn is_term(&self) -> bool {
        match self {
            Self::None | Self::Unknown(_) => false,
            _ => true,
        }
    }

    #[inline]
    pub fn is_box(&self) -> bool {
        match self {
            Self::Box | Self::Literal => true,
            _ => false,
        }
    }

    #[inline]
    pub fn is_boxable(&self) -> bool {
        match self {
            Self::BigInteger
            | Self::Float
            | Self::Tuple
            | Self::List
            | Self::Map
            | Self::Closure
            | Self::ProcBin
            | Self::HeapBinary
            | Self::SubBinary
            | Self::MatchContext
            | Self::ExternalPid
            | Self::ExternalPort
            | Self::ExternalReference
            | Self::ResourceReference
            | Self::Literal => true,
            _ => false,
        }
    }

    #[inline]
    pub fn is_list(&self) -> bool {
        match self {
            Self::Nil | Self::List => true,
            _ => false,
        }
    }

    #[inline]
    pub fn is_atom(&self) -> bool {
        match self {
            Self::Atom => true,
            _ => false,
        }
    }

    #[inline]
    pub fn is_number(&self) -> bool {
        match self {
            Self::SmallInteger | Self::BigInteger | Self::Float => true,
            _ => false,
        }
    }

    #[inline]
    pub fn is_boxed_number(&self) -> bool {
        match self {
            Self::BigInteger | Self::Float => true,
            _ => false,
        }
    }

    #[inline]
    pub fn is_integer(&self) -> bool {
        match self {
            Self::SmallInteger | Self::BigInteger => true,
            _ => false,
        }
    }

    #[inline]
    pub fn is_big_integer(&self) -> bool {
        match self {
            Self::BigInteger => true,
            _ => false,
        }
    }

    #[inline]
    pub fn is_binary(&self) -> bool {
        match self {
            Self::ProcBin | Self::HeapBinary | Self::SubBinary => true,
            _ => false,
        }
    }

    #[inline]
    pub fn is_pid(&self) -> bool {
        match self {
            Self::Pid | Self::ExternalPid => true,
            _ => false,
        }
    }

    #[inline]
    pub fn is_external_pid(&self) -> bool {
        match self {
            Self::ExternalPid => true,
            _ => false,
        }
    }

    #[inline]
    pub fn is_port(&self) -> bool {
        match self {
            Self::Port | Self::ExternalPort => true,
            _ => false,
        }
    }

    #[inline]
    pub fn is_external_port(&self) -> bool {
        match self {
            Self::ExternalPort => true,
            _ => false,
        }
    }

    #[inline]
    pub fn is_reference(&self) -> bool {
        match self {
            Self::Reference | Self::ExternalReference => true,
            _ => false,
        }
    }

    #[inline]
    pub fn is_external_reference(&self) -> bool {
        match self {
            Self::ExternalReference => true,
            _ => false,
        }
    }
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
