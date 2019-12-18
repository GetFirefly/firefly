use std::borrow::Cow;
use std::convert::{From, TryFrom, TryInto};
use std::path::{Path, PathBuf};

use libeir_diagnostics::FileName;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Input {
    /// Load source code from a file.
    File(PathBuf),
    /// Load source code from a string.
    Str {
        /// A string that is shown in place of a filename.
        name: String,
        /// An anonymous string containing the source code.
        input: Cow<'static, str>,
    },
}

impl Input {
    pub fn new<S: Into<String>, I: Into<Cow<'static, str>>>(name: S, input: I) -> Self {
        Self::Str {
            name: name.into(),
            input: input.into(),
        }
    }

    pub fn is_virtual(&self) -> bool {
        if let &Input::Str { .. } = self {
            return true;
        }
        false
    }

    pub fn is_real(&self) -> bool {
        if let &Input::File(_) = self {
            return true;
        }
        false
    }

    pub fn source_name(&self) -> FileName {
        self.into()
    }

    pub fn file_stem(&self) -> PathBuf {
        match self {
            Input::File(ref file) => PathBuf::from(file.file_stem().unwrap().to_str().unwrap()),
            Input::Str { ref name, .. } => match name.as_str() {
                "nofile" => PathBuf::from("out"),
                s => {
                    let path = Path::new(s);
                    PathBuf::from(path.file_stem().unwrap().to_str().unwrap())
                }
            },
        }
    }

    pub fn as_path(&self) -> Result<&Path, ()> {
        self.try_into()
    }

    pub fn get_input(&mut self) -> Option<Cow<'static, str>> {
        match *self {
            Input::File(_) => None,
            Input::Str { ref input, .. } => Some(input.clone()),
        }
    }
}

impl TryFrom<&FileName> for Input {
    type Error = ();

    fn try_from(filename: &FileName) -> Result<Self, Self::Error> {
        match filename {
            &FileName::Real(ref path) => Ok(Input::File(path.clone())),
            &FileName::Virtual(_) => Err(()),
        }
    }
}

impl Into<FileName> for &Input {
    fn into(self) -> FileName {
        match self {
            Input::File(ref file) => FileName::Real(file.clone()),
            Input::Str { ref name, .. } => FileName::Virtual(name.clone().into()),
        }
    }
}

impl From<PathBuf> for Input {
    fn from(path: PathBuf) -> Self {
        Self::File(path)
    }
}
impl From<&Path> for Input {
    fn from(path: &Path) -> Self {
        Self::File(path.to_path_buf())
    }
}
impl<'p> TryInto<&'p Path> for &'p Input {
    type Error = ();
    fn try_into(self) -> Result<&'p Path, Self::Error> {
        match self {
            &Input::File(ref path) => Ok(path.as_path()),
            _ => Err(()),
        }
    }
}
