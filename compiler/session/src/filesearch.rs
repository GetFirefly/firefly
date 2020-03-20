#![allow(non_camel_case_types)]

pub use self::FileMatch::*;

use std::borrow::Cow;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

use log::debug;

use liblumen_util as util;

use crate::search_paths::{PathKind, SearchPath};

#[derive(Copy, Clone)]
pub enum FileMatch {
    FileMatches,
    FileDoesntMatch,
}

// A module for searching for libraries

#[derive(Clone)]
pub struct FileSearch<'a> {
    sysroot: &'a Path,
    triple: &'a str,
    search_paths: &'a [SearchPath],
    tlib_path: &'a SearchPath,
    kind: PathKind,
}

impl<'a> FileSearch<'a> {
    pub fn search_paths(&self) -> impl Iterator<Item = &'a SearchPath> {
        let kind = self.kind;
        self.search_paths
            .iter()
            .filter(move |sp| sp.kind.matches(kind))
            .chain(std::iter::once(self.tlib_path))
    }

    pub fn get_lib_path(&self) -> PathBuf {
        make_target_lib_path(self.sysroot, self.triple)
    }

    pub fn search<F>(&self, mut pick: F)
    where
        F: FnMut(&Path, PathKind) -> FileMatch,
    {
        for search_path in self.search_paths() {
            debug!("searching {}", search_path.dir.display());
            for path in search_path.files.iter() {
                debug!("testing {}", path.display());
                let maybe_picked = pick(path, search_path.kind);
                match maybe_picked {
                    FileMatches => {
                        debug!("picked {}", path.display());
                    }
                    FileDoesntMatch => {
                        debug!("rejected {}", path.display());
                    }
                }
            }
        }
    }

    pub fn new(
        sysroot: &'a Path,
        triple: &'a str,
        search_paths: &'a Vec<SearchPath>,
        tlib_path: &'a SearchPath,
        kind: PathKind,
    ) -> FileSearch<'a> {
        debug!("using sysroot = {}, triple = {}", sysroot.display(), triple);
        FileSearch {
            sysroot,
            triple,
            search_paths,
            tlib_path,
            kind,
        }
    }

    // Returns just the directories within the search paths.
    pub fn search_path_dirs(&self) -> Vec<PathBuf> {
        self.search_paths().map(|sp| sp.dir.to_path_buf()).collect()
    }

    // Returns a list of directories where target-specific tool binaries are located.
    pub fn get_tools_search_paths(&self) -> Vec<PathBuf> {
        let mut p = PathBuf::from(self.sysroot);
        p.push(find_libdir(self.sysroot).as_ref());
        p.push(LUMEN_LIB_DIR);
        p.push(&self.triple);
        p.push("bin");
        vec![p]
    }
}

pub fn relative_target_lib_path(sysroot: &Path, target_triple: &str) -> PathBuf {
    let mut p = PathBuf::from(find_libdir(sysroot).as_ref());
    assert!(p.is_relative());
    p.push(LUMEN_LIB_DIR);
    p.push(target_triple);
    p.push("lib");
    p
}

pub fn make_target_lib_path(sysroot: &Path, target_triple: &str) -> PathBuf {
    sysroot.join(&relative_target_lib_path(sysroot, target_triple))
}

pub fn get_or_default_sysroot() -> PathBuf {
    // Follow symlinks.  If the resolved path is relative, make it absolute.
    fn canonicalize(path: Option<PathBuf>) -> Option<PathBuf> {
        path.and_then(|path| {
            match fs::canonicalize(&path) {
                // See comments on this target function, but the gist is that
                // gcc chokes on verbatim paths which fs::canonicalize generates
                // so we try to avoid those kinds of paths.
                Ok(canon) => Some(util::fs::fix_windows_verbatim_for_gcc(&canon)),
                Err(e) => panic!("failed to get realpath: {}", e),
            }
        })
    }

    match env::current_exe() {
        Ok(exe) => match canonicalize(Some(exe)) {
            Some(mut p) => {
                p.pop();
                p.pop();
                p
            }
            None => panic!("can't determine value for sysroot"),
        },
        Err(ref e) => panic!(format!("failed to get current_exe: {}", e)),
    }
}

// The name of the directory lumen expects libraries to be located.
fn find_libdir(sysroot: &Path) -> Cow<'static, str> {
    // FIXME: This is a quick hack to make the lumen binary able to locate
    // Lumen libraries in Linux environments where libraries might be installed
    // to lib64/lib32. This would be more foolproof by basing the sysroot off
    // of the directory where liblumen is located, rather than where the lumen
    // binary is.
    // If --libdir is set during configuration to the value other than
    // "lib" (i.e., non-default), this value is used (see issue #16552).

    #[cfg(target_pointer_width = "64")]
    const PRIMARY_LIB_DIR: &str = "lib64";

    #[cfg(target_pointer_width = "32")]
    const PRIMARY_LIB_DIR: &str = "lib32";

    const SECONDARY_LIB_DIR: &str = "lib";

    match option_env!("CFG_LIBDIR_RELATIVE") {
        Some(libdir) if libdir != "lib" => libdir.into(),
        _ => {
            if sysroot.join(PRIMARY_LIB_DIR).join(LUMEN_LIB_DIR).exists() {
                PRIMARY_LIB_DIR.into()
            } else {
                SECONDARY_LIB_DIR.into()
            }
        }
    }
}

// The name of lumen's own place to organize libraries.
const LUMEN_LIB_DIR: &str = "lumenlib";
