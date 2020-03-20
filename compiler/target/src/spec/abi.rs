use core::fmt;

#[derive(PartialEq, Eq, PartialOrd, Ord, Hash, Clone, Copy, Debug)]
pub enum Abi {
    // N.B., this ordering MUST match the AbiDatas array below.
    // (This is ensured by the test indices_are_correct().)

    // Single platform ABIs
    Cdecl,
    Stdcall,
    Fastcall,
    Vectorcall,
    Thiscall,
    Aapcs,
    Win64,
    SysV64,
    PtxKernel,
    Msp430Interrupt,
    X86Interrupt,
    AmdGpuKernel,
    EfiApi,

    // Multiplatform / generic ABIs
    Erlang,
    C,
    System,
    PlatformIntrinsic,
    Unadjusted,
}

#[derive(Copy, Clone)]
pub struct AbiData {
    abi: Abi,

    /// Name of this ABI as we like it called.
    name: &'static str,

    /// A generic ABI is supported on all platforms.
    generic: bool,
}

#[allow(non_upper_case_globals)]
const AbiDatas: &[AbiData] = &[
    // Platform-specific ABIs
    AbiData {
        abi: Abi::Cdecl,
        name: "cdecl",
        generic: false,
    },
    AbiData {
        abi: Abi::Stdcall,
        name: "stdcall",
        generic: false,
    },
    AbiData {
        abi: Abi::Fastcall,
        name: "fastcall",
        generic: false,
    },
    AbiData {
        abi: Abi::Vectorcall,
        name: "vectorcall",
        generic: false,
    },
    AbiData {
        abi: Abi::Thiscall,
        name: "thiscall",
        generic: false,
    },
    AbiData {
        abi: Abi::Aapcs,
        name: "aapcs",
        generic: false,
    },
    AbiData {
        abi: Abi::Win64,
        name: "win64",
        generic: false,
    },
    AbiData {
        abi: Abi::SysV64,
        name: "sysv64",
        generic: false,
    },
    AbiData {
        abi: Abi::PtxKernel,
        name: "ptx-kernel",
        generic: false,
    },
    AbiData {
        abi: Abi::Msp430Interrupt,
        name: "msp430-interrupt",
        generic: false,
    },
    AbiData {
        abi: Abi::X86Interrupt,
        name: "x86-interrupt",
        generic: false,
    },
    AbiData {
        abi: Abi::AmdGpuKernel,
        name: "amdgpu-kernel",
        generic: false,
    },
    AbiData {
        abi: Abi::EfiApi,
        name: "efiapi",
        generic: false,
    },
    // Cross-platform ABIs
    AbiData {
        abi: Abi::Erlang,
        name: "Erlang",
        generic: true,
    },
    AbiData {
        abi: Abi::C,
        name: "C",
        generic: true,
    },
    AbiData {
        abi: Abi::System,
        name: "system",
        generic: true,
    },
    AbiData {
        abi: Abi::PlatformIntrinsic,
        name: "platform-intrinsic",
        generic: true,
    },
    AbiData {
        abi: Abi::Unadjusted,
        name: "unadjusted",
        generic: true,
    },
];

/// Returns the ABI with the given name (if any).
pub fn lookup(name: &str) -> Option<Abi> {
    AbiDatas
        .iter()
        .find(|abi_data| name == abi_data.name)
        .map(|&x| x.abi)
}

pub fn all_names() -> Vec<&'static str> {
    AbiDatas.iter().map(|d| d.name).collect()
}

impl Abi {
    #[inline]
    pub fn index(self) -> usize {
        self as usize
    }

    #[inline]
    pub fn data(self) -> &'static AbiData {
        &AbiDatas[self.index()]
    }

    pub fn name(self) -> &'static str {
        self.data().name
    }

    pub fn generic(self) -> bool {
        self.data().generic
    }
}

impl fmt::Display for Abi {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "\"{}\"", self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_erlang() {
        let abi = lookup("Erlang");
        assert!(abi.is_some() && abi.unwrap().data().name == "Erlang");
    }

    #[test]
    fn lookup_cdecl() {
        let abi = lookup("cdecl");
        assert!(abi.is_some() && abi.unwrap().data().name == "cdecl");
    }

    #[test]
    fn lookup_baz() {
        let abi = lookup("baz");
        assert!(abi.is_none());
    }

    #[test]
    fn indices_are_correct() {
        for (i, abi_data) in AbiDatas.iter().enumerate() {
            assert_eq!(i, abi_data.abi.index());
        }
    }
}
