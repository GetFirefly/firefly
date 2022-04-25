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
    pub fn new(
        sysroot: &'a Path,
        triple: &'a str,
        search_paths: &'a [SearchPath],
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

    pub fn get_self_contained_lib_path(&self) -> PathBuf {
        self.get_lib_path().join("self-contained")
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

    // Returns just the directories within the search paths.
    pub fn search_path_dirs(&self) -> Vec<PathBuf> {
        self.search_paths().map(|sp| sp.dir.to_path_buf()).collect()
    }
}

// Returns a list of directories where target-specific tool binaries are located.
pub fn get_tools_search_paths(sysroot: &Path, self_contained: bool) -> Vec<PathBuf> {
    let lumenlib_path = target_lumenlib_path(sysroot, liblumen_target::host_triple());
    let p = PathBuf::from_iter([
        Path::new(sysroot),
        Path::new(&lumenlib_path),
        Path::new("bin"),
    ]);
    if self_contained {
        vec![p.clone(), p.join("self-contained")]
    } else {
        vec![p]
    }
}

pub fn make_target_lib_path(sysroot: &Path, target_triple: &str) -> PathBuf {
    let lumenlib_path = target_lumenlib_path(sysroot, target_triple);
    PathBuf::from_iter([sysroot, Path::new(&lumenlib_path), Path::new("lib")])
}

pub fn get_or_default_sysroot() -> PathBuf {
    // Follow symlinks.  If the resolved path is relative, make it absolute.
    fn canonicalize(path: PathBuf) -> PathBuf {
        let path = fs::canonicalize(&path).unwrap_or(path);
        // See comments on this target function, but the gist is that
        // gcc chokes on verbatim paths which fs::canonicalize generates
        // so we try to avoid those kinds of paths.
        util::fs::fix_windows_verbatim_for_gcc(&path)
    }

    // Use env::current_exe() to get the path of the executable following
    // symlinks/canonicalizing components.
    fn from_current_exe() -> PathBuf {
        match env::current_exe() {
            Ok(exe) => {
                let mut p = canonicalize(exe);
                p.pop();
                p.pop();
                p
            }
            Err(e) => panic!("failed to get current_exe: {e}"),
        }
    }

    // Use env::args().next() to get the path of the executable without
    // following symlinks/canonicalizing any component. This makes the lumen
    // binary able to locate Lumen libraries in systems using content-addressable
    // storage (CAS).
    fn from_env_args_next() -> Option<PathBuf> {
        match env::args_os().next() {
            Some(first_arg) => {
                let mut p = PathBuf::from(first_arg);

                // Check if sysroot is found using env::args().next() only if the rustc in argv[0]
                // is a symlink (see #79253). We might want to change/remove it to conform with
                // https://www.gnu.org/prep/standards/standards.html#Finding-Program-Files in the
                // future.
                if fs::read_link(&p).is_err() {
                    // Path is not a symbolic link or does not exist.
                    return None;
                }

                // Pop off `bin/rustc`, obtaining the suspected sysroot.
                p.pop();
                p.pop();
                // Look for the target lumenlib directory in the suspected sysroot.
                let mut lumenlib_path = target_lumenlib_path(&p, "dummy");
                lumenlib_path.pop(); // pop off the dummy target.
                if lumenlib_path.exists() {
                    Some(p)
                } else {
                    None
                }
            }
            None => None,
        }
    }

    // Check if sysroot is found using env::args().next(), and if is not found,
    // use env::current_exe() to imply sysroot.
    from_env_args_next().unwrap_or_else(from_current_exe)
}

/// Returns a `lumenlib` path for this particular target, relative to the provided sysroot.
///
/// For example: `target_sysroot_path("/usr", "x86_64-unknown-linux-gnu")` =>
/// `"lib*/lumenlib/x86_64-unknown-linux-gnu"`.
pub fn target_lumenlib_path(sysroot: &Path, target_triple: &str) -> PathBuf {
    let libdir = find_libdir(sysroot);
    PathBuf::from_iter([
        Path::new(libdir.as_ref()),
        Path::new(LUMEN_LIB_DIR),
        Path::new(target_triple),
    ])
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
        None | Some("lib") => {
            if sysroot.join(PRIMARY_LIB_DIR).join(LUMEN_LIB_DIR).exists() {
                PRIMARY_LIB_DIR.into()
            } else {
                SECONDARY_LIB_DIR.into()
            }
        }
        Some(libdir) => libdir.into(),
    }
}

// The name of lumen's own place to organize libraries.
const LUMEN_LIB_DIR: &str = "lumenlib";
