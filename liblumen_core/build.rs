use std::fmt::{self, Display};
use std::ops::Range;
use std::str::FromStr;

// Emit custom cfg types:
//     cargo:rustc-cfg=has_foo
// Can then be used as `#[cfg(has_foo)]` when emitted

// Emit custom env data:
//     cargo:rustc-env=foo=bar
// Can then be fetched with `env!("foo")`

fn main() {
    let target = target::triple();
    // wasm32 doesn't have mmap primitives
    if target.arch() != "wasm32" {
        println!("cargo:rustc-cfg=has_mmap");
    }
}

/**
Reads and unwraps an environment variable.
This should only be used on variables which are *guaranteed* to be defined by Cargo.
*/
macro_rules! env_var {
    ($name:expr) => {
        ::std::env::var($name).expect(concat!($name, " environment variable is not set"))
    };
}

/**
Reads, unwraps, and parses an environment variable.
This should only be used on variables which are *guaranteed* to be defined by Cargo.
*/
macro_rules! parse_env_var {
    (try: $name:expr, $ty_desc:expr) => {{
        ::std::env::var($name).ok().map(|v| {
            v.parse().expect(&format!(
                concat!($name, " {:?} is not a valid ", $ty_desc),
                v
            ))
        })
    }};

    ($name:expr, $ty_desc:expr) => {{
        let v = env_var!($name);
        v.parse().expect(&format!(
            concat!($name, " {:?} is not a valid ", $ty_desc),
            v
        ))
    }};
}

/**
Target platform information.
*/
pub mod target {
    use super::*;

    /**
    Platform endianness.
    **Requires**: Rust 1.14.
    */
    pub fn endian() -> Option<Endianness> {
        parse_env_var!(try: "CARGO_CFG_TARGET_ENDIAN", "endianness")
    }

    /**
    Platform processor features.
    A list of features can be obtained using `rustc --print target-features`.
    **Requires**: Rust nightly.
    */
    #[cfg(feature = "nightly")]
    pub fn features() -> Option<Vec<String>> {
        env::var("CARGO_CFG_TARGET_FEATURE")
            .ok()
            .map(|v| v.split(',').map(Into::into).collect())
    }

    /**
    List of types which are atomic on this platform.
    **Requires**: Rust nightly.
    */
    #[cfg(feature = "nightly")]
    pub fn has_atomic() -> Option<Vec<Atomic>> {
        env::var("CARGO_CFG_TARGET_HAS_ATOMIC").ok().map(|v| {
            v.split(',')
                .map(|s| {
                    s.parse().expect(&format!(
                        "CARGO_CFG_TARGET_HAS_ATOMIC \
                         contained invalid atomic type {:?}",
                        s
                    ))
                })
                .collect()
        })
    }

    /**
    Width, in bits, of a pointer on this platform.
    **Requires**: Rust 1.14.
    */
    pub fn pointer_width() -> Option<u8> {
        parse_env_var!(try: "CARGO_CFG_TARGET_POINTER_WIDTH", "integer")
    }

    /**
    Platform triple.
    A list of target triples can be obtained using `rustc --print target-list`.
    */
    pub fn triple() -> Triple {
        parse_env_var!("TARGET", "triple")
    }
}

/**
Represents the target platform's endianness.
*/
#[derive(Copy, Clone, Eq, PartialEq, Debug)]
pub enum Endianness {
    Big,
    Little,
}

impl Display for Endianness {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            Endianness::Big => "big".fmt(fmt),
            Endianness::Little => "little".fmt(fmt),
        }
    }
}

#[derive(Clone, Eq, PartialEq, Debug)]
pub struct InvalidInput(String);

impl FromStr for Endianness {
    type Err = InvalidInput;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "big" => Ok(Endianness::Big),
            "little" => Ok(Endianness::Little),
            _ => Err(InvalidInput(s.into())),
        }
    }
}

/**
Platform triple.
*/
#[derive(Clone, Eq, PartialEq, Debug)]
pub struct Triple {
    triple: String,
    arch: Range<usize>,
    env: Option<Range<usize>>,
    family: Range<usize>,
    os: Range<usize>,
}

impl Triple {
    /// Create a `Triple` from its string representation.
    pub fn new(triple: String) -> Triple {
        let arch;
        let env;
        let family;
        let os;

        {
            let mut parts = triple.splitn(4, '-').map(|s| {
                let off = subslice_offset(&triple, s);
                off..(off + s.len())
            });

            arch = parts.next().expect(&format!(
                "could not find architecture in triple {:?}",
                triple
            ));
            family = parts
                .next()
                .expect(&format!("could not find family in triple {:?}", triple));
            os = parts
                .next()
                .expect(&format!("could not find os in triple {:?}", triple));
            env = parts.next();
        }

        Triple {
            triple,
            arch,
            env,
            family,
            os,
        }
    }

    /// Get triple as a string.
    pub fn as_str(&self) -> &str {
        &self.triple
    }

    /**
    Platform processor architecture.
    Values include `"i686"`, `"x86_64"`, `"arm"`, *etc.*
    */
    pub fn arch(&self) -> &str {
        &self.triple[self.arch.clone()]
    }

    /**
    Platform toolchain environment.
    Values include `"gnu"`, `"msvc"`, `"musl"`, `"android"` *etc.*  Value is `None` if the platform doesn't specify an environment.
    */
    pub fn env(&self) -> Option<&str> {
        self.env.as_ref().map(|s| &self.triple[s.clone()])
    }

    /**
    Platform machine family.
    Values include `"apple"`, `"pc"`, `"unknown"`, *etc.*
    <!-- Definitive proof that Apples aren't PCs.  *mic drop* -->
    */
    pub fn family(&self) -> &str {
        &self.triple[self.family.clone()]
    }

    /**
    Platform operating system.
    Values include `"linux"`, `"windows"`, `"ios"`, *etc.*
    */
    pub fn os(&self) -> &str {
        &self.triple[self.os.clone()]
    }
}

impl Display for Triple {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        self.triple.fmt(fmt)
    }
}

impl FromStr for Triple {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(Triple::new(s.into()))
    }
}

/// Offset of slice within a base string.
fn subslice_offset(base: &str, inner: &str) -> usize {
    let base_beg = base.as_ptr() as usize;
    let inner = inner.as_ptr() as usize;
    if inner < base_beg || inner > base_beg.wrapping_add(base.len()) {
        panic!("cannot compute subslice offset of disjoint strings")
    } else {
        inner.wrapping_sub(base_beg)
    }
}
