//! Various source mapping utilities

use std::borrow::Cow;
use std::fmt;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum FileName {
    /// A real file on disk
    Real(PathBuf),
    /// A synthetic file, eg. from the REPL
    Virtual(Cow<'static, str>),
}

impl From<PathBuf> for FileName {
    fn from(name: PathBuf) -> FileName {
        FileName::real(name)
    }
}

impl From<FileName> for PathBuf {
    fn from(name: FileName) -> PathBuf {
        match name {
            FileName::Real(path) => path,
            FileName::Virtual(Cow::Owned(owned)) => PathBuf::from(owned),
            FileName::Virtual(Cow::Borrowed(borrowed)) => PathBuf::from(borrowed),
        }
    }
}

impl<'a> From<&'a FileName> for &'a Path {
    fn from(name: &'a FileName) -> &'a Path {
        match *name {
            FileName::Real(ref path) => path,
            FileName::Virtual(ref cow) => Path::new(cow.as_ref()),
        }
    }
}

impl<'a> From<&'a Path> for FileName {
    fn from(name: &Path) -> FileName {
        FileName::real(name)
    }
}

impl From<String> for FileName {
    fn from(name: String) -> FileName {
        FileName::virtual_(name)
    }
}

impl From<&'static str> for FileName {
    fn from(name: &'static str) -> FileName {
        FileName::virtual_(name)
    }
}

impl AsRef<Path> for FileName {
    fn as_ref(&self) -> &Path {
        match *self {
            FileName::Real(ref path) => path.as_ref(),
            FileName::Virtual(ref cow) => Path::new(cow.as_ref()),
        }
    }
}

impl PartialEq<Path> for FileName {
    fn eq(&self, other: &Path) -> bool {
        self.as_ref() == other
    }
}

impl PartialEq<PathBuf> for FileName {
    fn eq(&self, other: &PathBuf) -> bool {
        self.as_ref() == other.as_path()
    }
}

impl FileName {
    pub fn real<T: Into<PathBuf>>(name: T) -> FileName {
        FileName::Real(name.into())
    }

    pub fn virtual_<T: Into<Cow<'static, str>>>(name: T) -> FileName {
        FileName::Virtual(name.into())
    }

    /// Returns true if this filename represents a real directory on disk
    pub fn is_dir(&self) -> bool {
        match self {
            FileName::Real(ref path) => path.exists() && path.is_dir(),
            _ => false,
        }
    }

    /// Returns true if this filename represents a real file on disk
    pub fn is_file(&self) -> bool {
        match self {
            FileName::Real(ref path) => path.exists() && path.is_file(),
            _ => false,
        }
    }

    /// Tries to return this filename as a `&str`, avoiding any allocations
    ///
    /// This will only return None if the filename is a path which requires lossy conversion to unicode.
    /// See `to_string` if you want an infallible conversion to a Rust string, albeit at the cost of an allocation
    pub fn as_str(&self) -> Option<&str> {
        match self {
            FileName::Real(ref path) => path.to_str(),
            FileName::Virtual(Cow::Borrowed(s)) => Some(s),
            FileName::Virtual(s) => Some(s.as_ref()),
        }
    }

    pub fn to_string(&self) -> String {
        match *self {
            FileName::Real(ref path) => match path.to_str() {
                None => path.to_string_lossy().into_owned(),
                Some(s) => s.to_owned(),
            },
            FileName::Virtual(ref s) => s.clone().into_owned(),
        }
    }
}

impl fmt::Display for FileName {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            FileName::Real(ref path) => write!(fmt, "{}", path.display()),
            FileName::Virtual(ref name) => write!(fmt, "<{}>", name),
        }
    }
}
