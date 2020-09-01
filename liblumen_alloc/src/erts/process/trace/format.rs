use std::borrow::Cow;
use std::convert::TryInto;
use std::fmt;
use std::io::Write;
use std::path::Path;

use termcolor::{Color, ColorSpec, WriteColor};

use crate::erts::exception::ArcError;
use crate::erts::term::prelude::*;

use super::Trace;

pub fn print(
    trace: &Trace,
    kind: Term,
    reason: Term,
    source: Option<ArcError>,
) -> std::io::Result<()> {
    use termcolor::{BufferWriter, ColorChoice};

    let out = BufferWriter::stderr(ColorChoice::Auto);
    let mut buffer = out.buffer();

    format_write(trace, &mut buffer, kind, reason, source)?;
    out.print(&buffer)?;

    Ok(())
}

struct FormatterWrapper<'f> {
    f: &'f mut fmt::Formatter<'static>,
}
impl<'f> FormatterWrapper<'f> {
    unsafe fn new(f: &mut fmt::Formatter<'_>) -> Self {
        use core::mem;

        Self {
            f: mem::transmute::<_, &'f mut fmt::Formatter<'static>>(f),
        }
    }
}
impl<'f> Write for &mut FormatterWrapper<'f> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        use std::io::{Error, ErrorKind};
        // Internally the formatter just writes these bytes
        // to the actual output without treating them as a utf8
        // string, so this is unsafe, but not in practice
        let raw = unsafe { core::str::from_utf8_unchecked(buf) };
        self.f
            .write_str(raw)
            .map_err(|_| Error::new(ErrorKind::Other, "failed to print trace"))?;
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

pub fn format(
    trace: &Trace,
    f: &mut fmt::Formatter,
    kind: Term,
    reason: Term,
    source: Option<ArcError>,
) -> std::io::Result<()> {
    use termcolor::Ansi;

    let mut wrapper = unsafe { FormatterWrapper::new(f) };
    let mut ansi = Ansi::new(&mut wrapper);
    format_write(trace, &mut ansi, kind, reason, source)
}

fn format_write<W>(
    trace: &Trace,
    out: &mut W,
    kind: Term,
    reason: Term,
    source: Option<ArcError>,
) -> std::io::Result<()>
where
    W: WriteColor,
{
    let mut bold = ColorSpec::new();
    bold.set_bold(true);
    let mut underlined = ColorSpec::new();
    underlined.set_underline(true);
    let mut yellow = ColorSpec::new();
    yellow.set_fg(Some(Color::Yellow));
    let mut green = ColorSpec::new();
    green.set_fg(Some(Color::Green));

    out.set_color(&bold)?;
    writeln!(out, "Backtrace (most recent call last):")?;

    for symbol in trace.iter_symbols().rev() {
        let mfa = symbol.module_function_arity();
        if mfa.is_none() {
            continue;
        }
        let mfa = mfa.unwrap();
        let filename = symbol.filename();
        let line = symbol.line();

        out.reset()?;
        write!(out, "  File ")?;

        match filename {
            Some(f) => {
                let filename = trim_filename(f);
                out.set_color(&underlined)?;
                write!(out, "{}", filename)?;
                out.reset()?;
                write!(out, ":")?;
                out.set_color(&yellow)?;
                write!(out, "{}", line.unwrap_or(0))?;
            }
            None => {
                out.set_color(&underlined)?;
                write!(out, "<unknown>")?;
            }
        }

        out.reset()?;
        write!(out, ", in ")?;
        out.set_color(&green)?;
        writeln!(out, "{}", &mfa)?;
    }

    out.set_color(&bold)?;
    let kind: Result<Atom, _> = kind.decode().unwrap().try_into();
    if let Ok(kind) = kind {
        if kind == "error" {
            writeln!(out, "\nProcess raised an error.")?;
        } else if kind == "exit" {
            writeln!(out, "\nProcess exited abnormally.")?;
        } else if kind == "throw" {
            writeln!(out, "\nProcess threw an exception.")?;
        } else {
            writeln!(out, "\nProcess crashed.")?;
        }
    }
    out.set_color(&yellow)?;
    writeln!(out, "  {}\n", reason)?;

    if let Some(source) = source {
        writeln!(out, "  {}\n", source)?;
    }

    out.reset()?;

    Ok(())
}

fn trim_filename(file: &Path) -> Cow<'_, str> {
    let filename = file.to_string_lossy();
    if filename.starts_with("/rustc/") {
        if let Some(filename) = filename.get(48..) {
            Cow::Owned(format!("rust:{}", filename))
        } else {
            filename
        }
    } else if let Some(basename) = file.file_name().and_then(|x| x.to_str()) {
        if basename.starts_with('<') && basename.ends_with('>') {
            Cow::Borrowed(basename)
        } else {
            filename
        }
    } else {
        filename
    }
}
