use std::path::{Path, PathBuf};

use crate::filesearch::make_target_lib_path;

#[derive(Clone, Debug)]
pub struct SearchPath {
    pub kind: PathKind,
    pub dir: PathBuf,
    pub files: Vec<PathBuf>,
}

#[derive(PartialEq, Clone, Copy, Debug, Hash, Eq)]
pub enum PathKind {
    Native,
    Crate,
    Dependency,
    Framework,
    ExternFlag,
    All,
}
impl PathKind {
    pub fn matches(&self, kind: PathKind) -> bool {
        match (self, kind) {
            (PathKind::All, _) | (_, PathKind::All) => true,
            _ => *self == kind,
        }
    }
}

impl SearchPath {
    pub fn from_cli_opt(path: &str) -> Result<Self, &'static str> {
        let (kind, path) = if path.starts_with("native=") {
            (PathKind::Native, &path["native=".len()..])
        } else if path.starts_with("crate=") {
            (PathKind::Crate, &path["crate=".len()..])
        } else if path.starts_with("dependency=") {
            (PathKind::Dependency, &path["dependency=".len()..])
        } else if path.starts_with("framework=") {
            (PathKind::Framework, &path["framework=".len()..])
        } else if path.starts_with("all=") {
            (PathKind::All, &path["all=".len()..])
        } else {
            (PathKind::All, path)
        };
        if path.is_empty() {
            return Err("empty search path");
        }

        let dir = PathBuf::from(path);
        Ok(Self::new(kind, dir))
    }

    pub fn from_sysroot_and_triple(sysroot: &Path, triple: &str) -> Self {
        Self::new(PathKind::All, make_target_lib_path(sysroot, triple))
    }

    fn new(kind: PathKind, dir: PathBuf) -> Self {
        // Get the files within the directory.
        let files = match std::fs::read_dir(&dir) {
            Ok(files) => files
                .filter_map(|p| p.ok().map(|s| s.path()))
                .collect::<Vec<_>>(),
            Err(..) => vec![],
        };

        SearchPath { kind, dir, files }
    }
}
