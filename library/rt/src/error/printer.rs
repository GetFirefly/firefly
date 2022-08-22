use std::io::{self, Write};

use termcolor::{BufferWriter, Color, ColorChoice, ColorSpec, WriteColor};

use crate::error::ErlangException;
use crate::process::Process;
use crate::term::*;

pub fn print(process: &Process, exception: &ErlangException) -> io::Result<()> {
    let stderr = BufferWriter::stderr(ColorChoice::Auto);
    let mut writer = stderr.buffer();

    let mut bold = ColorSpec::new();
    bold.set_bold(true);
    let mut underlined = ColorSpec::new();
    underlined.set_underline(true);

    let mut yellow = ColorSpec::new();
    yellow.set_fg(Some(Color::Yellow));
    let mut green = ColorSpec::new();
    green.set_fg(Some(Color::Green));

    writer.set_color(&bold)?;
    writeln!(writer, "Backtrace (most recent call last):")?;

    let trace = exception.trace();

    for symbol in trace.iter_symbols().rev() {
        let mfa = symbol.mfa();
        if mfa.is_none() {
            continue;
        }
        let mfa = mfa.unwrap();
        let filename = symbol.filename();

        writer.reset()?;
        write!(writer, "  File ")?;

        match filename {
            Some(f) => {
                writer.set_color(&underlined)?;
                write_filename(&mut writer, f)?;
                writer.reset()?;
                if let Some(line) = symbol.line() {
                    write!(writer, ":")?;
                    writer.set_color(&yellow)?;
                    write!(writer, "{}", line)?;
                }
            }
            None => {
                writer.set_color(&underlined)?;
                write!(writer, "<unknown>")?;
            }
        }

        writer.reset()?;
        write!(writer, ", in ")?;
        writer.set_color(&green)?;
        writeln!(writer, "{}", &mfa)?;
    }

    writer.set_color(&bold)?;

    write!(writer, "\nProcess ({}) ", Pid::Local { id: process.pid() })?;

    let kind = exception.kind();
    let kind_suffix = if kind == atoms::Error {
        "raised an error"
    } else if kind == atoms::Exit {
        "exited abnormally."
    } else if kind == atoms::Throw {
        "threw an exception."
    } else {
        "crashed."
    };

    writeln!(writer, "{}", kind_suffix)?;
    writer.set_color(&yellow)?;
    writeln!(writer, "  {}\n", exception.reason())?;

    writer.reset()?;

    stderr.print(&writer)
}

fn write_filename(writer: &mut dyn WriteColor, file: &str) -> io::Result<()> {
    if file.starts_with("/rustc/") {
        if let Some(filename) = file.get(48..) {
            return write!(writer, "rust:{}", filename);
        } else {
            return write!(writer, "{}", file);
        }
    }

    match file.rsplit_once('/') {
        None => write!(writer, "{}", file),
        Some((_, basename)) => {
            if basename.starts_with('<') && basename.ends_with('>') {
                write!(writer, "{}", basename)
            } else {
                write!(writer, "{}", file)
            }
        }
    }
}
