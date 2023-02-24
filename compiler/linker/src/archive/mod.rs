mod builder;

use std::path::{Path, PathBuf};

use anyhow::anyhow;
use log::debug;

use firefly_session::Options;

pub use self::builder::LlvmArchiveBuilder;

pub fn find_library(
    name: &str,
    verbatim: bool,
    search_paths: &[PathBuf],
    options: &Options,
) -> anyhow::Result<PathBuf> {
    // On Windows, static libraries sometimes show up as libfoo.a and other
    // times show up as foo.lib
    let oslibname = if verbatim {
        name.to_string()
    } else {
        format!(
            "{}{}{}",
            options.target.options.staticlib_prefix, name, options.target.options.staticlib_suffix
        )
    };
    let unixlibname = format!("lib{}.a", name);

    for path in search_paths {
        debug!("looking for {} inside {:?}", name, path);
        let test = path.join(&oslibname);
        if test.exists() {
            return Ok(test);
        }
        if oslibname != unixlibname {
            let test = path.join(&unixlibname);
            if test.exists() {
                return Ok(test);
            }
        }
    }
    return Err(anyhow!(
        "could not find native static library `{}`, perhaps an -L flag is missing?",
        name
    ));
}

pub trait ArchiveBuilder<'a> {
    fn new(options: &'a Options, output: &Path, input: Option<&Path>) -> Self;

    fn add_file(&mut self, path: &Path);
    fn remove_file(&mut self, name: &str);
    fn src_files(&mut self) -> Vec<String>;

    fn add_archive<F>(&mut self, archive: &Path, skip: F) -> anyhow::Result<()>
    where
        F: FnMut(&str) -> bool + 'static;

    fn build(self);
}
