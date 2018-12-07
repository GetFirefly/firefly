use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use trackable::track;

use super::{Error, Result};

pub fn substitute_path_variables<P: AsRef<Path>>(path: P) -> Result<PathBuf> {
    let mut new = PathBuf::new();
    for c in path.as_ref().components() {
        if let Some(s) = c.as_os_str().to_str() {
            if s.as_bytes().get(0) == Some(&b'$') {
                let c = track!(std::env::var(s.split_at(1).1).map_err(Error::from))?;
                new.push(c);
                continue;
            }
        }
        new.push(c.as_os_str());
    }
    Ok(new)
}

pub fn read_file<P: AsRef<Path>>(path: P) -> Result<String> {
    let mut buf = String::new();
    let mut file = track!(File::open(&path).map_err(Error::from))?;
    track!(file.read_to_string(&mut buf).map_err(Error::from))?;
    Ok(buf)
}
