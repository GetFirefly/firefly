use std::convert::AsRef;
use std::ffi::CString;
use std::fs;
use std::io;
use std::os;
use std::path::{Path, PathBuf};

use cfg_if::cfg_if;

cfg_if! {
    if #[cfg(windows)] {
        use std::os::windows::io::AsRawHandle;
    } else {
        use std::os::unix::io::AsRawFd;
    }
}

pub use glob::{GlobError, Pattern, PatternError};

/// Compiles a glob pattern, to a `Pattern`
///
/// The compiled pattern can be used to then match paths or strings.
///
/// Returns `Err(PatternError)` if the input is not a valid glob
pub fn glob<S: AsRef<str>>(pattern: S) -> Result<Pattern, PatternError> {
    Pattern::new(pattern.as_ref())
}

/// Get the underlying raw file descriptor for a File
#[cfg(windows)]
pub fn get_file_descriptor<F: AsRawHandle>(f: &F) -> os::windows::io::RawHandle {
    f.as_raw_handle()
}

/// Get the underlying raw file descriptor for a File
#[cfg(not(windows))]
pub fn get_file_descriptor<F: AsRawFd>(f: &F) -> os::unix::io::RawFd {
    f.as_raw_fd()
}

#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum NativeLibraryKind {
    /// Static library (e.g. `libfoo.a` on Linux or `foo.lib` on Windows/MSVC)
    Static {
        /// Whether to bundle objects from static library into produced lib
        bundle: Option<bool>,
        /// Whether to link static library without throwing any object files away
        whole_archive: Option<bool>,
    },
    /// Dynamic library (e.g. `libfoo.so` on Linux)
    Dylib {
        /// Whether the dynamic library will be linked only if it satisfies some undefined symbol
        as_needed: Option<bool>,
    },
    /// Dynamic library without a corresponding import library.
    RawDylib,
    /// A macOS-specific kind of dynamic library
    Framework {
        /// Whether the framework will be linked only if it satisfies some undefined symbol
        as_needed: Option<bool>,
    },
    /// The library kind wasn't specified, dylibs are assumed by default
    Unspecified,
}
impl NativeLibraryKind {
    pub fn has_modifiers(&self) -> bool {
        match self {
            Self::Static {
                bundle,
                whole_archive,
            } => bundle.is_some() || whole_archive.is_some(),
            Self::Dylib { as_needed } | Self::Framework { as_needed } => as_needed.is_some(),
            Self::RawDylib | Self::Unspecified => false,
        }
    }
}

// Unfortunately, on windows, it looks like msvcrt.dll is silently translating
// verbatim paths under the hood to non-verbatim paths! This manifests itself as
// gcc looking like it cannot accept paths of the form `\\?\C:\...`, but the
// real bug seems to lie in msvcrt.dll.
//
// Verbatim paths are generally pretty rare, but the implementation of
// `fs::canonicalize` currently generates paths of this form, meaning that we're
// going to be passing quite a few of these down to gcc, so we need to deal with
// this case.
//
// For now we just strip the "verbatim prefix" of `\\?\` from the path. This
// will probably lose information in some cases, but there's not a whole lot
// more we can do with a buggy msvcrt...
//
// For some more information, see this comment:
//   https://github.com/rust-lang/rust/issues/25505#issuecomment-102876737
#[cfg(windows)]
pub fn fix_windows_verbatim_for_gcc(p: &Path) -> PathBuf {
    use std::ffi::OsString;
    use std::path;
    let mut components = p.components();
    let prefix = match components.next() {
        Some(path::Component::Prefix(p)) => p,
        _ => return p.to_path_buf(),
    };
    match prefix.kind() {
        path::Prefix::VerbatimDisk(disk) => {
            let mut base = OsString::from(format!("{}:", disk as char));
            base.push(components.as_path());
            PathBuf::from(base)
        }
        path::Prefix::VerbatimUNC(server, share) => {
            let mut base = OsString::from(r"\\");
            base.push(server);
            base.push(r"\");
            base.push(share);
            base.push(components.as_path());
            PathBuf::from(base)
        }
        _ => p.to_path_buf(),
    }
}

#[cfg(not(windows))]
pub fn fix_windows_verbatim_for_gcc(p: &Path) -> PathBuf {
    p.to_path_buf()
}

pub enum LinkOrCopy {
    Link,
    Copy,
}

/// Copies `p` into `q`, preferring to use hard-linking if possible. If
/// `q` already exists, it is removed first.
/// The result indicates which of the two operations has been performed.
pub fn link_or_copy<P: AsRef<Path>, Q: AsRef<Path>>(p: P, q: Q) -> io::Result<LinkOrCopy> {
    let p = p.as_ref();
    let q = q.as_ref();
    if q.exists() {
        fs::remove_file(&q)?;
    }

    match fs::hard_link(p, q) {
        Ok(()) => Ok(LinkOrCopy::Link),
        Err(_) => match fs::copy(p, q) {
            Ok(_) => Ok(LinkOrCopy::Copy),
            Err(e) => Err(e),
        },
    }
}

#[derive(Debug)]
pub enum RenameOrCopyRemove {
    Rename,
    CopyRemove,
}

/// Rename `p` into `q`, preferring to use `rename` if possible.
/// If `rename` fails (rename may fail for reasons such as crossing
/// filesystem), fallback to copy & remove
pub fn rename_or_copy_remove<P: AsRef<Path>, Q: AsRef<Path>>(
    p: P,
    q: Q,
) -> io::Result<RenameOrCopyRemove> {
    let p = p.as_ref();
    let q = q.as_ref();
    match fs::rename(p, q) {
        Ok(()) => Ok(RenameOrCopyRemove::Rename),
        Err(_) => match fs::copy(p, q) {
            Ok(_) => {
                fs::remove_file(p)?;
                Ok(RenameOrCopyRemove::CopyRemove)
            }
            Err(e) => Err(e),
        },
    }
}

#[cfg(all(unix, target_env = "wasi"))]
pub fn path_to_c_string(p: &Path) -> CString {
    use std::ffi::OsStr;
    use std::os::wasi::ffi::OsStrExt;
    let p: &OsStr = p.as_ref();
    CString::new(p.as_bytes()).unwrap()
}

#[cfg(all(unix, not(target_env = "wasi")))]
pub fn path_to_c_string(p: &Path) -> CString {
    use std::ffi::OsStr;
    use std::os::unix::ffi::OsStrExt;
    let p: &OsStr = p.as_ref();
    CString::new(p.as_bytes()).unwrap()
}

#[cfg(windows)]
pub fn path_to_c_string(p: &Path) -> CString {
    CString::new(p.to_str().unwrap()).unwrap()
}
