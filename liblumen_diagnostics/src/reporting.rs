use std::cmp::Ordering;
use std::fmt;
use std::str::FromStr;

pub use termcolor::{Color, ColorChoice, ColorSpec};

mod diagnostic;
pub mod emitter;

pub use self::diagnostic::{Diagnostic, Label, LabelStyle};
pub use self::emitter::{Emitter, NullEmitter, StandardStreamEmitter};

/// A severity level for diagnostic messages
///
/// These are ordered in the following way:
///
/// * Bug
/// * Error
/// * Warning
/// * Note
/// * Help
#[derive(Copy, Clone, PartialEq, Hash, Debug)]
pub enum Severity {
    /// An unexpected bug.
    Bug,
    /// An error.
    Error,
    /// A warning.
    Warning,
    /// A note.
    Note,
    /// A help message.
    Help,
}
impl Severity {
    /// We want bugs to be the maximum severity, errors next, etc...
    fn to_cmp_int(self) -> u8 {
        match self {
            Severity::Bug => 5,
            Severity::Error => 4,
            Severity::Warning => 3,
            Severity::Note => 2,
            Severity::Help => 1,
        }
    }
}
impl PartialOrd for Severity {
    fn partial_cmp(&self, other: &Severity) -> Option<Ordering> {
        u8::partial_cmp(&self.to_cmp_int(), &other.to_cmp_int())
    }
}
impl fmt::Display for Severity {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.to_str().fmt(f)
    }
}
impl Severity {
    /// Return the termcolor to use when rendering messages of this diagnostic severity
    pub fn color(self) -> Color {
        match self {
            Severity::Bug | Severity::Error => Color::Red,
            Severity::Warning => Color::Yellow,
            Severity::Note => Color::Green,
            Severity::Help => Color::Cyan,
        }
    }

    /// A string that explains this diagnostic severity
    pub fn to_str(self) -> &'static str {
        match self {
            Severity::Bug => "error: internal compiler error",
            Severity::Error => "error",
            Severity::Warning => "warning",
            Severity::Note => "note",
            Severity::Help => "help",
        }
    }
}

/// A command line argument that configures the coloring of the output
///
/// This can be used with command line argument parsers like `clap` or `structopt`.
///
/// # Example
///
/// ```rust
/// use liblumen_diagnostics::UseColors;
/// use std::str::FromStr;
///
/// fn main() {
///     let _color = UseColors::from_str("always");
/// }
/// ```
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct UseColors(pub ColorChoice);
impl UseColors {
    /// Allowed string variants to be used on the command line
    pub const VARIANTS: &'static [&'static str] = &["auto", "always", "ansi", "never"];
}
impl FromStr for UseColors {
    type Err = &'static str;

    fn from_str(src: &str) -> Result<UseColors, &'static str> {
        match src {
            _ if src.eq_ignore_ascii_case("auto") => Ok(UseColors(ColorChoice::Auto)),
            _ if src.eq_ignore_ascii_case("always") => Ok(UseColors(ColorChoice::Always)),
            _ if src.eq_ignore_ascii_case("ansi") => Ok(UseColors(ColorChoice::AlwaysAnsi)),
            _ if src.eq_ignore_ascii_case("never") => Ok(UseColors(ColorChoice::Never)),
            _ => Err("valid values: auto, always, ansi, never"),
        }
    }
}
impl Into<ColorChoice> for UseColors {
    fn into(self) -> ColorChoice {
        self.0
    }
}
