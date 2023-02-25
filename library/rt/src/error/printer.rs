use std::io::{self, Write};

use termcolor::{BufferWriter, Color, ColorChoice, ColorSpec, WriteColor};

use crate::backtrace::Symbol;
use crate::error::ExceptionClass;
use crate::process::ProcessLock;
use crate::term::*;

pub fn print(process: &mut ProcessLock) -> io::Result<()> {
    let stderr = BufferWriter::stderr(ColorChoice::Auto);
    let mut writer = stderr.buffer();

    let mut bold = ColorSpec::new();
    bold.set_bold(true);
    let mut underlined = ColorSpec::new();
    underlined.set_dimmed(true);

    let mut yellow = ColorSpec::new();
    yellow.set_fg(Some(Color::Yellow));
    let mut green = ColorSpec::new();
    green.set_fg(Some(Color::Green));

    if let Some(trace) = process.exception_info.trace.as_deref() {
        writer.set_color(&bold)?;
        writeln!(writer, "Backtrace (most recent call last):")?;

        for frame in trace.frames().iter().rev() {
            let Some(symbol) = frame.symbolicate() else { continue; };
            let line = symbol.line();
            let column = symbol.line();
            let filename = symbol.filename();
            let Some(symbol) = symbol.symbol() else { continue; };

            writer.reset()?;
            write!(writer, "  File ")?;

            match filename {
                Some(f) => {
                    writer.set_color(&underlined)?;
                    write_filename(&mut writer, f)?;
                    writer.reset()?;
                    if let Some(line) = line {
                        write!(writer, ":")?;
                        writer.set_color(&yellow)?;
                        write!(writer, "{}", line)?;
                    }
                    if let Some(col) = column {
                        write!(writer, ":")?;
                        writer.set_color(&yellow)?;
                        write!(writer, "{}", col)?;
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

            if let Some(args) = frame.frame.args() {
                // We have an argument list available, print it
                match symbol {
                    Symbol::Erlang(mfa) => match args.into() {
                        Term::Nil => {
                            writeln!(writer, "{}:{}()", &mfa.module, &mfa.function)?;
                        }
                        Term::Cons(argv) => {
                            write!(writer, "{}:{}(", &mfa.module, &mfa.function)?;
                            for (i, arg) in argv.iter().enumerate() {
                                let arg = arg.unwrap();
                                if i > 0 {
                                    write!(writer, ", {}", &arg)?;
                                } else {
                                    write!(writer, "{}", &arg)?;
                                }
                            }
                            writeln!(writer, ")")?;
                        }
                        _ => unreachable!(),
                    },
                    Symbol::Native(name) => match args.into() {
                        Term::Nil => {
                            writeln!(writer, "{}()", &name)?;
                        }
                        Term::Cons(argv) => {
                            write!(writer, "{}(", &name)?;
                            for (i, arg) in argv.iter().enumerate() {
                                let arg = arg.unwrap();
                                if i > 0 {
                                    write!(writer, ", {}", &arg)?;
                                } else {
                                    write!(writer, "{}", &arg)?;
                                }
                            }
                            writeln!(writer, ")")?;
                        }
                        _ => unreachable!(),
                    },
                }
            } else {
                match symbol {
                    Symbol::Erlang(mfa) => writeln!(writer, "{}", &mfa)?,
                    Symbol::Native(name) => writeln!(writer, "{}", &name)?,
                }
            }
        }
    }

    writer.set_color(&bold)?;

    write!(writer, "\nProcess ({}) ", process.pid())?;

    let kind_suffix = match process.exception_info.class().unwrap() {
        ExceptionClass::Error => "raised an error.",
        ExceptionClass::Exit => "exited abnormally.",
        ExceptionClass::Throw => "threw an exception.",
    };

    let value: Term = process.exception_info.value.into();
    writeln!(writer, "{}", kind_suffix)?;
    writer.set_color(&yellow)?;
    writeln!(writer, "  {}\n", &value)?;

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
