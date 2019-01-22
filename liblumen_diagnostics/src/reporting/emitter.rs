use std::fmt::{self, Display};
use std::io;
use std::sync::{Arc, Mutex};

use failure::Error;
use termcolor::{Color, ColorChoice, ColorSpec, StandardStream, WriteColor};
use unicode_width::UnicodeWidthStr;

use crate::diagnostics::CodeMap;

use super::{Diagnostic, LabelStyle, Severity};

struct Pad<T>(T, usize);
impl<T: fmt::Display> fmt::Display for Pad<T> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for _ in 0..(self.1) {
            self.0.fmt(f)?;
        }
        Ok(())
    }
}

pub trait Emitter {
    fn emit(&self, color: Option<ColorSpec>, message: &str) -> io::Result<()>;
    fn debug(&self, color: Option<ColorSpec>, message: &str) -> io::Result<()>;
    fn warn(&self, color: Option<ColorSpec>, message: &str) -> io::Result<()>;
    fn error(&self, err: Error) -> io::Result<()>;
    fn diagnostic(&self, diagnostic: &Diagnostic) -> io::Result<()>;
}

/// This emitter swallows all output
pub struct NullEmitter(Severity);
impl NullEmitter {
    pub fn new() -> Self {
        NullEmitter(Severity::Help)
    }
}
impl Emitter for NullEmitter {
    fn emit(&self, _color: Option<ColorSpec>, _message: &str) -> io::Result<()> {
        Ok(())
    }

    fn debug(&self, _color: Option<ColorSpec>, _message: &str) -> io::Result<()> {
        Ok(())
    }

    fn warn(&self, _color: Option<ColorSpec>, _message: &str) -> io::Result<()> {
        Ok(())
    }

    fn error(&self, _err: Error) -> io::Result<()> {
        Ok(())
    }

    fn diagnostic(&self, _diagnostic: &Diagnostic) -> io::Result<()> {
        Ok(())
    }
}

/// This emitter writes output to either stdout/stderr
pub struct StandardStreamEmitter {
    severity: Severity,
    stdout: StandardStream,
    stderr: StandardStream,
    codemap: Option<Arc<Mutex<CodeMap>>>,
}
impl StandardStreamEmitter {
    pub fn new(color: ColorChoice) -> Self {
        StandardStreamEmitter {
            severity: Severity::Help,
            stdout: StandardStream::stdout(color),
            stderr: StandardStream::stderr(color),
            codemap: None,
        }
    }

    pub fn set_min_severity(mut self, severity: Severity) -> Self {
        self.severity = severity;
        self
    }

    pub fn set_codemap(mut self, codemap: Arc<Mutex<CodeMap>>) -> Self {
        self.codemap = Some(codemap);
        self
    }
}
impl Emitter for StandardStreamEmitter {
    fn emit(&self, color: Option<ColorSpec>, message: &str) -> io::Result<()> {
        let mut handle = self.stdout.lock();
        write_color(&mut handle, color, message)
    }

    fn debug(&self, color: Option<ColorSpec>, message: &str) -> io::Result<()> {
        let mut handle = self.stderr.lock();
        write_color(&mut handle, color, message)
    }

    fn warn(&self, color: Option<ColorSpec>, message: &str) -> io::Result<()> {
        let mut handle = self.stderr.lock();
        write_color(&mut handle, color, message)
    }

    fn error(&self, err: Error) -> io::Result<()> {
        let mut handle = self.stderr.lock();
        write_error(&mut handle, err)
    }

    fn diagnostic(&self, diagnostic: &Diagnostic) -> io::Result<()> {
        if diagnostic.severity >= self.severity {
            let mut handle = self.stdout.lock();

            match self.codemap {
                None => write_unlabeled_diagnostic(&mut handle, diagnostic)?,
                Some(ref codemap) => write_labeled_diagnostic(&mut handle, codemap, diagnostic)?,
            };
        }
        Ok(())
    }
}

fn write_color<W, M>(mut writer: W, color: Option<ColorSpec>, message: M) -> io::Result<()>
where
    W: WriteColor,
    M: Display,
{
    if let Some(ref color) = color {
        writer.set_color(&color)?;
    }
    write!(writer, "{}", message)?;

    writer.reset()?;

    Ok(())
}

fn write_error<W>(mut writer: W, err: Error) -> io::Result<()>
where
    W: WriteColor,
{
    let severity = Severity::Error;
    let failure = err.as_fail();

    let highlight_color = ColorSpec::new().set_bold(true).set_intense(true).clone();

    writer.set_color(&highlight_color.clone().set_fg(Some(severity.color())))?;
    write!(writer, "{}", severity)?;

    writer.set_color(&highlight_color)?;
    writeln!(writer, ": {}", failure)?;
    writer.reset()?;

    // Print root cause
    let cause = failure.find_root_cause();
    // It is possible that the failure has no root, which returns self,
    // so don't print errors twice in that case
    if std::ptr::eq(cause, failure) {
        write!(writer, "\n")?;
        writer.set_color(&highlight_color.clone().set_fg(Some(severity.color())))?;
        write!(writer, "CAUSE")?;

        writer.set_color(&highlight_color)?;
        writeln!(writer, ": {}", cause)?;
    }

    // Show backtrace only in debug mode
    if cfg!(debug_assertions) {
        if let Some(ref trace) = failure.backtrace() {
            write!(writer, "\n\n{:?}", trace)?;
        }
    }

    Ok(())
}

fn write_unlabeled_diagnostic<W>(mut writer: W, diagnostic: &Diagnostic) -> io::Result<()>
where
    W: WriteColor,
{
    let highlight_color = ColorSpec::new().set_bold(true).set_intense(true).clone();

    writer.set_color(
        &highlight_color
            .clone()
            .set_fg(Some(diagnostic.severity.color())),
    )?;
    write!(writer, "{}", diagnostic.severity)?;

    if let Some(ref code) = diagnostic.code {
        write!(writer, "[{}]", code)?;
    }

    writer.set_color(&highlight_color)?;
    writeln!(writer, ": {}", diagnostic.message)?;
    writer.reset()?;

    Ok(())
}

fn write_labeled_diagnostic<W>(
    mut writer: W,
    codemap: &Arc<Mutex<CodeMap>>,
    diagnostic: &Diagnostic,
) -> io::Result<()>
where
    W: WriteColor,
{
    let supports_color = writer.supports_color();
    let line_location_color = ColorSpec::new()
        // Blue is really difficult to see on the standard windows command line
        .set_fg(Some(if cfg!(windows) {
            Color::Cyan
        } else {
            Color::Blue
        }))
        .clone();
    let diagnostic_color = ColorSpec::new()
        .set_fg(Some(diagnostic.severity.color()))
        .clone();

    let highlight_color = ColorSpec::new().set_bold(true).set_intense(true).clone();

    writer.set_color(
        &highlight_color
            .clone()
            .set_fg(Some(diagnostic.severity.color())),
    )?;
    write!(writer, "{}", diagnostic.severity)?;

    if let Some(ref code) = diagnostic.code {
        write!(writer, "[{}]", code)?;
    }

    writer.set_color(&highlight_color)?;
    writeln!(writer, ": {}", diagnostic.message)?;
    writer.reset()?;

    for label in &diagnostic.labels {
        match codemap.lock().unwrap().find_file(label.span.start()) {
            None => {
                if let Some(ref message) = label.message {
                    writeln!(writer, "- {}", message)?
                }
            }
            Some(file) => {
                let (line, column) = file.location(label.span.start()).expect("location");
                writeln!(
                    writer,
                    "- {file}:{line}:{column}",
                    file = file.name(),
                    line = line.number(),
                    column = column.number(),
                )?;

                let line_span = file.line_span(line).expect("line_span");

                let line_prefix = file
                    .src_slice(line_span.with_end(label.span.start()))
                    .expect("line_prefix");
                let line_marked = file.src_slice(label.span).expect("line_marked");
                let line_suffix = file
                    .src_slice(line_span.with_start(label.span.end()))
                    .expect("line_suffix")
                    .trim_end_matches(|ch: char| ch == '\r' || ch == '\n');

                let mark = match label.style {
                    LabelStyle::Primary => '^',
                    LabelStyle::Secondary => '-',
                };
                let label_color = match label.style {
                    LabelStyle::Primary => diagnostic_color.clone(),
                    LabelStyle::Secondary => ColorSpec::new()
                        .set_fg(Some(if cfg!(windows) {
                            Color::Cyan
                        } else {
                            Color::Blue
                        }))
                        .clone(),
                };

                writer.set_color(&line_location_color)?;
                let line_number_string = line.number().to_string();
                let line_location_prefix = format!("{} | ", Pad(' ', line_number_string.len()));
                write!(writer, "{} | ", line_number_string)?;
                writer.reset()?;

                write!(writer, "{}", line_prefix)?;
                writer.set_color(&label_color)?;
                write!(writer, "{}", line_marked)?;
                writer.reset()?;
                writeln!(writer, "{}", line_suffix)?;

                if !supports_color || label.message.is_some() {
                    writer.set_color(&line_location_color)?;
                    write!(writer, "{}", line_location_prefix)?;
                    writer.reset()?;

                    writer.set_color(&label_color)?;
                    write!(
                        writer,
                        "{}{}",
                        Pad(' ', line_prefix.width()),
                        Pad(mark, line_marked.width()),
                    )?;
                    writer.reset()?;

                    if label.message.is_none() {
                        writeln!(writer)?;
                    }
                }

                match label.message {
                    None => (),
                    Some(ref label) => {
                        writer.set_color(&label_color)?;
                        writeln!(writer, " {}", label)?;
                        writer.reset()?;
                    }
                }
            }
        }
    }
    Ok(())
}

#[inline]
pub fn green() -> ColorSpec {
    color(Color::Green)
}

#[inline]
pub fn green_bold() -> ColorSpec {
    color_bold(Color::Green)
}

#[inline]
pub fn yellow() -> ColorSpec {
    color(Color::Yellow)
}

#[inline]
pub fn yellow_bold() -> ColorSpec {
    color_bold(Color::Yellow)
}

#[inline]
pub fn white() -> ColorSpec {
    color_bold(Color::White)
}

#[inline]
pub fn cyan() -> ColorSpec {
    if cfg!(windows) {
        color(Color::Cyan)
    } else {
        color(Color::Blue)
    }
}

#[inline]
pub fn color(color: Color) -> ColorSpec {
    let mut spec = ColorSpec::new();
    spec.set_fg(Some(color));
    spec
}

#[inline]
pub fn color_bold(color: Color) -> ColorSpec {
    let mut spec = ColorSpec::new();
    spec.set_fg(Some(color)).set_bold(true).set_intense(true);
    spec
}
