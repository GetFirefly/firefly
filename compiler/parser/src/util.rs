use std::env;
use std::path::{Path, PathBuf};

use firefly_diagnostics::Diagnostic;

#[derive(Debug, thiserror::Error)]
pub enum PathVariableSubstituteError {
    #[error("invalid path substition variable {variable:?}")]
    InvalidPathVariable {
        variable: String,
        source: std::env::VarError,
    },
}
impl PathVariableSubstituteError {
    pub fn to_diagnostic(&self) -> Diagnostic {
        match self {
            PathVariableSubstituteError::InvalidPathVariable {
                source: env::VarError::NotPresent,
                variable,
            } => Diagnostic::error().with_message(format!(
                "invalid environment variable '{}': not defined",
                variable,
            )),
            PathVariableSubstituteError::InvalidPathVariable {
                source: env::VarError::NotUnicode { .. },
                variable,
            } => Diagnostic::error().with_message(format!(
                "invalid environment variable '{}': contains invalid unicode data",
                variable,
            )),
        }
    }
}

pub fn substitute_path_variables<P: AsRef<Path>>(
    path: P,
) -> Result<PathBuf, PathVariableSubstituteError> {
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
                        return Err(PathVariableSubstituteError::InvalidPathVariable {
                            variable: var.to_owned(),
                            source: e,
                        });
                    }
                }
            }
        }
        new.push(c.as_os_str());
    }
    Ok(new)
}
