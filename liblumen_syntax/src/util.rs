use crate::lexer::SourceError;
use std::env;
use std::path::{Path, PathBuf};

pub fn substitute_path_variables<P: AsRef<Path>>(path: P) -> Result<PathBuf, SourceError> {
    let mut new = PathBuf::new();
    for c in path.as_ref().components() {
        if let Some(s) = c.as_os_str().to_str() {
            if s.as_bytes().get(0) == Some(&b'$') {
                let var = s.split_at(1).1;
                match env::var(var) {
                    Ok(c) => {
                        new.push(c);
                        continue;
                    }
                    Err(e) => {
                        return Err(SourceError::InvalidEnvironmentVariable(e, var.to_owned()));
                    }
                }
            }
        }
        new.push(c.as_os_str());
    }
    Ok(new)
}
