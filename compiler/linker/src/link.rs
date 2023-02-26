use std::ascii;
use std::cell::OnceCell;
use std::char;
use std::env;
use std::ffi::OsString;
use std::fmt;
use std::fs;
use std::io;
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::process::{Output, Stdio};
use std::str;

use anyhow::{anyhow, Context};
use cc::windows_registry;
use log::info;
use tempfile::Builder as TempFileBuilder;

use firefly_diagnostics::Severity;
use firefly_session::filesearch;
use firefly_session::search_paths::PathKind;
use firefly_session::{
    CFGuard, DebugInfo, LdImpl, Options, ProjectType, Sanitizer, SplitDwarfKind, Strip,
};
use firefly_target::crt_objects::LinkSelfContainedDefault;
use firefly_target::{
    LinkOutputKind, LinkerFlavor, LldFlavor, PanicStrategy, RelocModel, RelroLevel, SplitDebugInfo,
};
use firefly_util::diagnostics::DiagnosticsHandler;
use firefly_util::fs::{fix_windows_verbatim_for_gcc, NativeLibraryKind};

use crate::archive::{ArchiveBuilder, LlvmArchiveBuilder};
use crate::linker;
use crate::rpath::{self, RPathConfig};
use crate::{AppArtifacts, Command, Linker};

/// Performs the linkage portion of the compilation phase. This will generate all
/// of the requested outputs for this compilation session.
pub fn link_binary(
    options: &Options,
    diagnostics: &DiagnosticsHandler,
    artifacts: &AppArtifacts,
) -> anyhow::Result<()> {
    let project_type = options.project_type;
    if invalid_output_for_target(options) {
        return Err(anyhow!(
            "invalid output type `{:?}` for target os `{}`",
            project_type,
            options.target.triple()
        ));
    }

    if options.codegen_opts.no_codegen {
        return Ok(());
    }

    for obj in artifacts.modules.iter().filter_map(|m| m.object()) {
        check_file_is_writeable(obj)?;
    }

    let tmpdir = TempFileBuilder::new()
        .prefix("firefly")
        .tempdir()
        .map_err(|err| anyhow!("couldn't create a temp dir: {}", err))?;

    let output_dir = options.output_dir();
    let output_file = options
        .output_file
        .as_ref()
        .map(|of| of.clone())
        .unwrap_or_else(|| {
            let ext = match project_type {
                ProjectType::Executable => &options.target.options.exe_suffix,
                ProjectType::Staticlib => &options.target.options.staticlib_suffix,
                _ => &options.target.options.dll_suffix,
            };
            let name = options.app.name.as_str().get();
            let name = match project_type {
                ProjectType::Executable => format!("{}{}", name, ext),
                ProjectType::Staticlib => format!(
                    "{}{}{}",
                    &options.target.options.staticlib_prefix, name, ext
                ),
                _ => format!("{}{}{}", &options.target.options.dll_prefix, name, ext),
            };
            output_dir.as_path().join(&name)
        });

    // `output_dir` is not necessarily the parent of `output_file`, such as with
    // `--output-dir _build --output bin/myapp`
    let output_file_parent = if output_file.is_absolute() {
        output_file.parent().unwrap().to_path_buf()
    } else {
        output_dir.clone()
    };
    fs::create_dir_all(output_file_parent.as_path()).with_context(|| {
        format!(
            "Could not create parent directories ({}) of file ({})",
            output_file_parent.display(),
            output_file.display()
        )
    })?;

    match project_type {
        ProjectType::Staticlib => {
            link_staticlib(
                options,
                diagnostics,
                project_type,
                artifacts,
                output_file.as_path(),
                tmpdir.path(),
            )?;
        }
        _ => {
            link_natively(
                options,
                diagnostics,
                project_type,
                artifacts,
                output_file.as_path(),
                tmpdir.path(),
            )?;
        }
    }

    if options.debugging_opts.print_artifact_sizes {
        let file_size = fs::metadata(&output_file).map(|m| m.len()).unwrap_or(0);
        diagnostics.note(format!(
            "Generated artifact of {} bytes: {}",
            file_size,
            output_file.display()
        ));
    }

    if !options.should_link() {
        return Ok(());
    }

    // Remove the temporary object file and metadata
    let (preserve_objects, preserve_dwarf_objects) = preserve_objects_for_their_debuginfo(options);
    for module in artifacts.modules.iter() {
        if !preserve_objects {
            if let Some(obj) = module.object() {
                if let Err(e) = remove(obj) {
                    diagnostics.error(format!("{}", e));
                }
            }
        }

        if !preserve_dwarf_objects {
            if let Some(ref obj) = module.dwarf_object {
                if let Err(e) = remove(obj) {
                    diagnostics.error(format!("{}", e));
                }
            }
        }
    }

    Ok(())
}

// Create a static archive
//
// There's no way for us to link dynamic libraries, so we warn
// about all dynamic library dependencies that they're not linked in.
fn link_staticlib<'a>(
    options: &'a Options,
    diagnostics: &DiagnosticsHandler,
    project_type: ProjectType,
    artifacts: &AppArtifacts,
    output_file: &Path,
    _tmpdir: &Path,
) -> anyhow::Result<()> {
    info!("preparing {:?} to {:?}", project_type, output_file);

    let mut ab = LlvmArchiveBuilder::new(options, output_file, None);

    for module in artifacts.modules.iter() {
        if let Some(obj) = module.object.as_ref() {
            ab.add_file(obj);
        }

        if let Some(dwarf_obj) = module.dwarf_object.as_ref() {
            ab.add_file(dwarf_obj);
        }
    }

    ab.build();

    diagnostics.success(
        "Linker",
        format!("generated static library to {}", output_file.display()),
    );

    Ok(())
}

fn escape_stdout_stderr_string(s: &[u8]) -> String {
    str::from_utf8(s).map(|s| s.to_owned()).unwrap_or_else(|_| {
        let mut x = "Non-UTF-8 output: ".to_string();
        x.extend(
            s.iter()
                .flat_map(|&b| ascii::escape_default(b))
                .map(char::from),
        );
        x
    })
}

// Create a dynamic library or executable
//
// This will invoke the system linker/cc to create the resulting file. This
// links to all upstream files as well.
fn link_natively(
    options: &Options,
    diagnostics: &DiagnosticsHandler,
    project_type: ProjectType,
    artifacts: &AppArtifacts,
    output_file: &Path,
    tmpdir: &Path,
) -> anyhow::Result<()> {
    info!("preparing {:?} to {:?}", project_type, output_file);
    let (linker_path, flavor) = linker_and_flavor(options);

    let mut cmd = linker_with_args(
        &linker_path,
        flavor,
        options,
        diagnostics,
        project_type,
        tmpdir,
        output_file,
        artifacts,
    );

    cmd.disable_localization();

    for &(ref k, ref v) in &options.target.options.link_env {
        cmd.env(k.as_ref(), v.as_ref());
    }
    for k in &options.target.options.link_env_remove {
        cmd.env_remove(k.as_ref());
    }

    if options.debugging_opts.print_link_args {
        diagnostics.notice("Linker", format!("{:?}", &cmd));
    }

    // May have not found libraries in the right formats.
    diagnostics.abort_if_errors();

    match exec_linker(options, &mut cmd, output_file, tmpdir) {
        Ok(prog) => {
            if !prog.status.success() {
                let mut output = prog.stderr.clone();
                output.extend_from_slice(&prog.stdout);
                let escaped_output = escape_stdout_stderr_string(&output);
                let mut err = diagnostics
                    .diagnostic(Severity::Error)
                    .with_message(format!(
                        "linking with `{}` failed: {}",
                        linker_path.display(),
                        prog.status
                    ))
                    .with_note(format!("{:?}", &cmd));
                if escaped_output.contains("undefined reference to") {
                    err.add_note(escaped_output);
                    err.add_note(
                        "some `extern` functions couldn't be found; some native libraries may \
                        need to be installed or have their path specified",
                    );
                    err.add_note("use the `-l` flag to specify native libraries to link");
                } else {
                    err.add_note(escaped_output);
                }
                err.emit();

                // If MSVC's `link.exe` was expected but the return code
                // is not a Microsoft LNK error then suggest a way to fix or
                // install the Visual Studio build tools.
                if let Some(code) = prog.status.code() {
                    if options.target.options.is_like_msvc
                        && flavor == LinkerFlavor::Msvc
                        // Respect the command line override
                        && options.codegen_opts.linker.is_none()
                        // Match exactly "link.exe"
                        && linker_path.to_str() == Some("link.exe")
                        // All Microsoft `link.exe` linking error codes are
                        // four digit numbers in the range 1000 to 9999 inclusive
                        && (code < 1000 || code > 9999)
                    {
                        let is_vs_installed = windows_registry::find_vs_version().is_ok();
                        let has_linker =
                            windows_registry::find_tool(&options.target.triple(), "link.exe")
                                .is_some();
                        diagnostics.note("`link.exe` returned an unexpected error");
                        if is_vs_installed && has_linker {
                            // the linker is broken
                            diagnostics.note(
                                "the Visual Studio build tools may need to be repaired \
                                                          using the Visual Studio installer",
                            );
                            diagnostics.note(
                                "or a necessary component may be missing from the \
                                                          \"C++ build tools\" workload",
                            );
                        } else if is_vs_installed {
                            // the linker is not installed
                            diagnostics.note(
                                "in the Visual Studio installer, ensure the \
                                                          \"C++ build tools\" workload is selected",
                            );
                        } else {
                            // visual studio is not installed
                            diagnostics.note(
                                "you may need to install Visual Studio build tools with the \
                                                          \"C++ build tools\" workload",
                            );
                        }
                    }
                }
            }
            diagnostics.abort_if_errors();
        }
        Err(e) => {
            let linker_not_found = e.kind() == io::ErrorKind::NotFound;
            let mut linker_error = {
                if linker_not_found {
                    diagnostics
                        .diagnostic(Severity::Error)
                        .with_message(format!("linker `{}` not found", linker_path.display()))
                } else {
                    diagnostics
                        .diagnostic(Severity::Error)
                        .with_message(format!(
                            "could not exec the linker `{}`",
                            linker_path.display()
                        ))
                }
            };
            linker_error.add_note(e.to_string());
            if !linker_not_found {
                linker_error.add_note(format!("{:?}", &cmd));
            }
            linker_error.emit();
            if options.target.options.is_like_msvc && linker_not_found {
                diagnostics.note(
                    "the msvc targets depend on the msvc linker \
                    but `link.exe` was not found",
                );
                diagnostics.note(
                    "please ensure that one of VS 2013-2022 was installed \
                    with the Visual C++ option",
                );
            }
            diagnostics.abort_if_errors();
        }
    }

    match options.split_debuginfo() {
        // If split debug information is disabled or located in individual files
        // there's nothing to do here.
        SplitDebugInfo::Off | SplitDebugInfo::Unpacked => {}

        // If packed split-debuginfo is requested, but the final compilation
        // doesn't actually have any debug information, then we skip this step.
        SplitDebugInfo::Packed if options.debug_info == DebugInfo::None => {}

        // On macOS the external `dsymutil` tool is used to create the packed
        // debug information. Note that this will read debug information from
        // the objects on the filesystem which we'll clean up later.
        SplitDebugInfo::Packed if options.target.options.is_like_osx => {
            let prog = Command::new("dsymutil").arg(output_file).output();
            match prog {
                Ok(prog) => {
                    if !prog.status.success() {
                        let mut output = prog.stderr.clone();
                        output.extend_from_slice(&prog.stdout);
                        diagnostics
                            .diagnostic(Severity::Error)
                            .with_message(format!(
                                "processing debug info with `dsymutil` failed: {}",
                                prog.status
                            ))
                            .with_note(escape_string(&output))
                            .emit();
                    }
                }
                Err(e) => diagnostics
                    .fatal(format!("unable to run `dsymutil`: {}", e))
                    .raise(),
            }
        }

        // On MSVC packed debug information is produced by the linker itself so
        // there's no need to do anything else here.
        SplitDebugInfo::Packed if options.target.options.is_like_windows => {}

        // ... and otherwise we're processing a `*.dwp` packed dwarf file.
        //
        // We cannot rely on the .o paths in the executable because they may have been
        // remapped by --remap-path-prefix and therefore invalid, so we need to provide
        // the .o/.dwo paths explicitly.
        //SplitDebugInfo::Packed => link_dwarf_object(options, artifacts, out_filename),
        _ => (),
    }

    if options.target.options.is_like_osx {
        match (options.codegen_opts.strip, project_type) {
            (Strip::DebugInfo, _) => {
                strip_symbols_in_osx(options, diagnostics, output_file, Some("-S"))
            }
            // Per the manpage, `-x` is the maximum safe strip level for dynamic libraries. (rust-lang/rust#93988)
            (Strip::Symbols, ProjectType::Dylib | ProjectType::Cdylib) => {
                strip_symbols_in_osx(options, diagnostics, output_file, Some("-x"))
            }
            (Strip::Symbols, _) => strip_symbols_in_osx(options, diagnostics, output_file, None),
            (Strip::None, _) => (),
        }
    }

    diagnostics.success(
        "Linker",
        format!("generated executable to {}", output_file.display()),
    );

    Ok(())
}

fn strip_symbols_in_osx(
    _options: &Options,
    diagnostics: &DiagnosticsHandler,
    out_filename: &Path,
    option: Option<&str>,
) {
    let mut cmd = Command::new("strip");
    if let Some(option) = option {
        cmd.arg(option);
    }
    let prog = cmd.arg(out_filename).output();
    match prog {
        Ok(prog) => {
            if !prog.status.success() {
                let mut output = prog.stderr.clone();
                output.extend_from_slice(&prog.stdout);
                diagnostics
                    .diagnostic(Severity::Error)
                    .with_message(format!(
                        "stripping debug info with `strip` failed: {}",
                        prog.status
                    ))
                    .with_note(escape_string(&output))
                    .emit();
            }
        }
        Err(e) => diagnostics
            .fatal(format!("unable to run `strip`: {}", e))
            .raise(),
    }
}

fn escape_string(s: &[u8]) -> String {
    str::from_utf8(s).map(|s| s.to_owned()).unwrap_or_else(|_| {
        let mut x = "Non-UTF-8 output: ".to_string();
        x.extend(
            s.iter()
                .flat_map(|&b| ascii::escape_default(b))
                .map(char::from),
        );
        x
    })
}

fn add_sanitizer_libraries(linker: &mut dyn Linker, options: &Options, project_type: ProjectType) {
    // On macOS the runtimes are distributed as dylibs which should be linked to
    // both executables and dynamic shared objects. Everywhere else the runtimes
    // are currently distributed as static libraries which should be linked to
    // executables only.
    let needs_runtime = match project_type {
        ProjectType::Executable => true,
        ProjectType::Dylib | ProjectType::Cdylib => options.target.options.is_like_osx,
        ProjectType::Staticlib => false,
    };

    if !needs_runtime {
        return;
    }

    let sanitizers = options.debugging_opts.sanitizers.as_slice();
    if sanitizers.contains(&Sanitizer::Address) {
        link_sanitizer_runtime(linker, options, "asan");
    }
    if sanitizers.contains(&Sanitizer::Leak) {
        link_sanitizer_runtime(linker, options, "lsan");
    }
    if sanitizers.contains(&Sanitizer::Memory) {
        link_sanitizer_runtime(linker, options, "msan");
    }
    if sanitizers.contains(&Sanitizer::Thread) {
        link_sanitizer_runtime(linker, options, "thread");
    }
    //if sanitizers.contains(&Sanitizer::HardwareAddress) {
    //link_sanitizer_runtime(linker, options, "hwasan");
    //}
}

fn link_sanitizer_runtime(linker: &mut dyn Linker, options: &Options, name: &str) {
    fn find_sanitizer_runtime(options: &Options, filename: &str) -> PathBuf {
        let tlib = filesearch::make_target_lib_path(&options.sysroot, options.target.triple());
        let path = tlib.join(filename);
        if path.exists() {
            return tlib;
        } else {
            let default_sysroot = filesearch::get_or_default_sysroot();
            let default_tlib =
                filesearch::make_target_lib_path(&default_sysroot, options.target.triple());
            return default_tlib;
        }
    }

    if options.target.options.is_like_osx {
        // On Apple platforms, the sanitizer is always built as a dylib, and
        // LLVM will link to `@rpath/*.dylib`, so we need to specify an
        // rpath to the library as well (the rpath should be absolute, see
        // PR #41352 for details).
        let filename = format!("firefly_rt.{}", name);
        let path = find_sanitizer_runtime(options, &filename);
        let rpath = path.to_str().expect("non-utf8 component in path");
        linker.args(&["-Wl,-rpath", "-Xlinker", rpath]);
        linker.link_dylib(&filename, false, true);
    } else {
        let filename = format!("libfirefly_rt.{}.a", name);
        let path = find_sanitizer_runtime(options, &filename).join(&filename);
        linker.link_whole_rlib(&path);
    }
}

fn linker_and_flavor(options: &Options) -> (PathBuf, LinkerFlavor) {
    fn infer_from(
        options: &Options,
        linker: Option<PathBuf>,
        flavor: Option<LinkerFlavor>,
    ) -> Option<(PathBuf, LinkerFlavor)> {
        match (linker, flavor) {
            // Explicit linker+flavor
            (Some(linker), Some(flavor)) => Some((linker, flavor)),
            // Only the linker flavor is known; use the default linker for the selected flavor
            (None, Some(flavor)) => {
                let prog = match flavor {
                    LinkerFlavor::Gcc => {
                        if cfg!(any(target_os = "solaris", target_os = "illumos")) {
                            // On historical Solaris systems, "cc" may have
                            // been Sun Studio, which is not flag-compatible
                            // with "gcc".  This history casts a long shadow,
                            // and many modern illumos distributions today
                            // ship GCC as "gcc" without also making it
                            // available as "cc".
                            "gcc"
                        } else {
                            "cc"
                        }
                    }
                    LinkerFlavor::Ld => "ld",
                    LinkerFlavor::Lld(_) => "lld",
                    LinkerFlavor::Msvc => "link.exe",
                    LinkerFlavor::EmCc => {
                        if cfg!(windows) {
                            "emcc.bat"
                        } else {
                            "emcc"
                        }
                    }
                    f => {
                        panic!("invalid linker flavor '{}': flavor is unimplemented", f)
                    }
                };
                Some((PathBuf::from(prog), flavor))
            }
            (Some(linker), None) => {
                let stem = linker
                    .file_stem()
                    .and_then(|stem| stem.to_str())
                    .expect("couldn't extract file stem from specified linker");

                let flavor = if stem == "emcc" {
                    LinkerFlavor::EmCc
                } else if stem == "gcc"
                    || stem.ends_with("-gcc")
                    || stem == "clang"
                    || stem.ends_with("-clang")
                {
                    LinkerFlavor::Gcc
                } else if stem == "wasm-ld" || stem.ends_with("-wasm-ld") {
                    LinkerFlavor::Lld(LldFlavor::Wasm)
                } else if stem == "ld" || stem == "ld.lld" || stem.ends_with("-ld") {
                    LinkerFlavor::Ld
                } else if stem == "link" || stem == "lld-link" {
                    LinkerFlavor::Msvc
                } else if stem == "lld" || stem == "firefly-lld" {
                    LinkerFlavor::Lld(options.target.options.lld_flavor)
                } else {
                    // fall back to the value in the target spec
                    options.target.options.linker_flavor
                };

                Some((linker, flavor))
            }
            (None, None) => None,
        }
    }

    // linker and linker flavor specified via command line have precedence over what the target
    // specification specifies
    if let Some(ret) = infer_from(
        options,
        options.codegen_opts.linker.clone(),
        options.codegen_opts.linker_flavor,
    ) {
        return ret;
    }

    if let Some(ret) = infer_from(
        options,
        options.target.options.linker.as_deref().map(PathBuf::from),
        Some(options.target.options.linker_flavor),
    ) {
        return ret;
    }

    panic!("Not enough information provided to determine how to invoke the linker")
}

/// Returns a pair of boolean indicating whether we should preserve the object and
/// dwarf object files on the filesystem for their debug information. This is often
/// useful with split-dwarf like schemes.
fn preserve_objects_for_their_debuginfo(options: &Options) -> (bool, bool) {
    if options.debug_info == DebugInfo::None {
        return (false, false);
    }

    // If we're only producing artifacts that are archives, no need to preserve the objects
    if options.project_type == ProjectType::Staticlib {
        return (false, false);
    }

    match (
        options.split_debuginfo(),
        options.debugging_opts.split_dwarf_kind,
    ) {
        // If there is no split debuginfo then do not preserve objects.
        (SplitDebugInfo::Off, _) => (false, false),
        // If there is packed split debuginfo, then the debuginfo in the objects
        // has been packaged and the objects can be deleted.
        (SplitDebugInfo::Packed, _) => (false, false),
        // If there is unpacked split debuginfo and the current target can not use
        // split dwarf, then keep objects.
        (SplitDebugInfo::Unpacked, _) if !options.target_can_use_split_dwarf() => (true, false),
        // If there is unpacked split debuginfo and the target can use split dwarf, then
        // keep the object containing that debuginfo (whether that is an object file or
        // dwarf object file depends on the split dwarf kind).
        (SplitDebugInfo::Unpacked, SplitDwarfKind::Single) => (true, false),
        (SplitDebugInfo::Unpacked, SplitDwarfKind::Split) => (false, true),
    }
}

fn archive_search_paths(options: &Options) -> Vec<PathBuf> {
    options
        .target_filesearch(PathKind::Native)
        .search_path_dirs()
}

fn get_object_file_path(options: &Options, name: &str, self_contained: bool) -> PathBuf {
    let fs = options.target_filesearch(PathKind::Native);
    let file_path = fs.get_lib_path().join(name);
    if file_path.exists() {
        return file_path;
    }
    // Special directory with objects used only in self-contained linkage mode
    if self_contained {
        let file_path = fs.get_self_contained_lib_path().join(name);
        if file_path.exists() {
            return file_path;
        }
    }
    for search_path in fs.search_paths() {
        let file_path = search_path.dir.join(name);
        if file_path.exists() {
            return file_path;
        }
    }
    PathBuf::from(name)
}

fn exec_linker(
    options: &Options,
    cmd: &mut Command,
    output_file: &Path,
    tmpdir: &Path,
) -> io::Result<Output> {
    // When attempting to spawn the linker we run a risk of blowing out the
    // size limits for spawning a new process with respect to the arguments
    // we pass on the command line.
    //
    // Here we attempt to handle errors from the OS saying "your list of
    // arguments is too big" by reinvoking the linker again with an `@`-file
    // that contains all the arguments. The theory is that this is then
    // accepted on all linkers and the linker will read all its options out of
    // there instead of looking at the command line.
    if !cmd.very_likely_to_exceed_some_spawn_limit() {
        match cmd
            .command()
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
        {
            Ok(child) => {
                let output = child.wait_with_output();
                flush_linked_file(&output, output_file)?;
                return output;
            }
            Err(ref e) if command_line_too_big(e) => {
                info!("command line to linker was too big: {}", e);
            }
            Err(e) => return Err(e),
        }
    }

    info!("falling back to passing arguments to linker via an @-file");
    let mut cmd2 = cmd.clone();
    let mut args = String::new();
    for arg in cmd2.take_args() {
        args.push_str(
            &Escape {
                arg: arg.to_str().unwrap(),
                is_like_msvc: options.target.options.is_like_msvc,
            }
            .to_string(),
        );
        args.push_str("\n");
    }
    let file = tmpdir.join("linker-arguments");
    let bytes = if options.target.options.is_like_msvc {
        let mut out = Vec::with_capacity((1 + args.len()) * 2);
        // start the stream with a UTF-16 BOM
        for c in std::iter::once(0xFEFF).chain(args.encode_utf16()) {
            // encode in little endian
            out.push(c as u8);
            out.push((c >> 8) as u8);
        }
        out
    } else {
        args.into_bytes()
    };
    fs::write(&file, &bytes)?;
    cmd2.arg(format!("@{}", file.display()));
    info!("invoking linker {:?}", cmd2);
    let output = cmd2.output();
    flush_linked_file(&output, output_file)?;
    return output;

    #[cfg(not(windows))]
    fn flush_linked_file(_: &io::Result<Output>, _: &Path) -> io::Result<()> {
        Ok(())
    }

    #[cfg(windows)]
    fn flush_linked_file(
        command_output: &io::Result<Output>,
        output_file: &Path,
    ) -> io::Result<()> {
        // On Windows, under high I/O load, output buffers are sometimes not flushed,
        // even long after process exit, causing nasty, non-reproducible output bugs.
        //
        // File::sync_all() calls FlushFileBuffers() down the line, which solves the problem.
        //
        // А full writeup of the original Chrome bug can be found at
        // randomascii.wordpress.com/2018/02/25/compiler-bug-linker-bug-windows-kernel-bug/amp

        if let &Ok(ref out) = command_output {
            if out.status.success() {
                if let Ok(of) = fs::OpenOptions::new().write(true).open(output_file) {
                    of.sync_all()?;
                }
            }
        }

        Ok(())
    }

    #[cfg(unix)]
    fn command_line_too_big(err: &io::Error) -> bool {
        err.raw_os_error() == Some(::libc::E2BIG)
    }

    #[cfg(windows)]
    fn command_line_too_big(err: &io::Error) -> bool {
        const ERROR_FILENAME_EXCED_RANGE: i32 = 206;
        err.raw_os_error() == Some(ERROR_FILENAME_EXCED_RANGE)
    }

    #[cfg(not(any(unix, windows)))]
    fn command_line_too_big(_: &io::Error) -> bool {
        false
    }

    struct Escape<'a> {
        arg: &'a str,
        is_like_msvc: bool,
    }

    impl<'a> fmt::Display for Escape<'a> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            if self.is_like_msvc {
                // This is "documented" at
                // https://msdn.microsoft.com/en-us/library/4xdcbak7.aspx
                //
                // Unfortunately there's not a great specification of the
                // syntax I could find online (at least) but some local
                // testing showed that this seemed sufficient-ish to catch
                // at least a few edge cases.
                write!(f, "\"")?;
                for c in self.arg.chars() {
                    match c {
                        '"' => write!(f, "\\{}", c)?,
                        c => write!(f, "{}", c)?,
                    }
                }
                write!(f, "\"")?;
            } else {
                // This is documented at https://linux.die.net/man/1/ld, namely:
                //
                // > Options in file are separated by whitespace. A whitespace
                // > character may be included in an option by surrounding the
                // > entire option in either single or double quotes. Any
                // > character (including a backslash) may be included by
                // > prefixing the character to be included with a backslash.
                //
                // We put an argument on each line, so all we need to do is
                // ensure the line is interpreted as one whole argument.
                for c in self.arg.chars() {
                    match c {
                        '\\' | ' ' => write!(f, "\\{}", c)?,
                        c => write!(f, "{}", c)?,
                    }
                }
            }
            Ok(())
        }
    }
}

fn link_output_kind(options: &Options, project_type: ProjectType) -> LinkOutputKind {
    let kind = match (
        project_type,
        options.crt_static(Some(project_type)),
        options.relocation_model(),
    ) {
        (ProjectType::Executable, _, _) if options.is_wasi_reactor() => {
            LinkOutputKind::WasiReactorExe
        }
        (ProjectType::Executable, false, RelocModel::Pic) => LinkOutputKind::DynamicPicExe,
        (ProjectType::Executable, false, _) => LinkOutputKind::DynamicNoPicExe,
        (ProjectType::Executable, true, RelocModel::Pic) => LinkOutputKind::StaticPicExe,
        (ProjectType::Executable, true, _) => LinkOutputKind::StaticNoPicExe,
        (_, true, _) => LinkOutputKind::StaticDylib,
        (_, false, _) => LinkOutputKind::DynamicDylib,
    };

    // Adjust the output kind to target capabilities.
    let opts = &options.target.options;
    let pic_exe_supported = opts.position_independent_executables;
    let static_pic_exe_supported = opts.static_position_independent_executables;
    let static_dylib_supported = opts.crt_static_allows_dylibs;
    match kind {
        LinkOutputKind::DynamicPicExe if !pic_exe_supported => LinkOutputKind::DynamicNoPicExe,
        LinkOutputKind::StaticPicExe if !static_pic_exe_supported => LinkOutputKind::StaticNoPicExe,
        LinkOutputKind::StaticDylib if !static_dylib_supported => LinkOutputKind::DynamicDylib,
        _ => kind,
    }
}

// Returns true if linker is located within sysroot
fn detect_self_contained_mingw(options: &Options) -> bool {
    let (linker, _) = linker_and_flavor(options);
    // Assume `-C linker=firefly-lld` as self-contained mode
    if linker == Path::new("firefly-lld") {
        return true;
    }
    let linker_with_extension = if cfg!(windows) && linker.extension().is_none() {
        linker.with_extension("exe")
    } else {
        linker
    };
    for dir in env::split_paths(&env::var_os("PATH").unwrap_or_default()) {
        let full_path = dir.join(&linker_with_extension);
        // If linker comes from sysroot assume self-contained mode
        if full_path.is_file() && !full_path.starts_with(&options.sysroot) {
            return false;
        }
    }
    true
}

/// Various toolchain components used during linking are used from firefly distribution
/// instead of being found somewhere on the host system.
///
/// We only provide such support for a very limited number of targets.
fn self_contained(options: &Options, project_type: ProjectType) -> bool {
    if let Some(self_contained) = options.codegen_opts.link_self_contained {
        return self_contained;
    }

    match options.target.options.link_self_contained {
        LinkSelfContainedDefault::False => false,
        LinkSelfContainedDefault::True => true,
        // FIXME: Find a better heuristic for "native musl toolchain is available",
        // based on host and linker path, for example.
        // (https://github.com/rust-lang/rust/pull/71769#issuecomment-626330237).
        LinkSelfContainedDefault::Musl => options.crt_static(Some(project_type)),
        LinkSelfContainedDefault::Mingw => {
            options.host == options.target
                && options.target.options.vendor != "uwp"
                && detect_self_contained_mingw(options)
        }
    }
}

/// Add pre-link object files defined by the target spec.
fn add_pre_link_objects(
    cmd: &mut dyn Linker,
    options: &Options,
    link_output_kind: LinkOutputKind,
    self_contained: bool,
) {
    let opts = &options.target.options;
    let objects = if self_contained {
        &opts.pre_link_objects_self_contained
    } else {
        &opts.pre_link_objects
    };
    for obj in objects.get(&link_output_kind).iter().copied().flatten() {
        cmd.add_object(&get_object_file_path(options, obj, self_contained));
    }
}

/// Add post-link object files defined by the target spec.
fn add_post_link_objects(
    cmd: &mut dyn Linker,
    options: &Options,
    link_output_kind: LinkOutputKind,
    self_contained: bool,
) {
    let opts = &options.target.options;
    let objects = if self_contained {
        &opts.post_link_objects_self_contained
    } else {
        &opts.post_link_objects
    };
    for obj in objects.get(&link_output_kind).iter().copied().flatten() {
        cmd.add_object(&get_object_file_path(options, obj, self_contained));
    }
}

/// Add arbitrary "pre-link" args defined by the target spec or from command line.
/// FIXME: Determine where exactly these args need to be inserted.
fn add_pre_link_args(cmd: &mut dyn Linker, options: &Options, flavor: LinkerFlavor) {
    if let Some(args) = options.target.options.pre_link_args.get(&flavor) {
        cmd.args(args.iter().map(Deref::deref));
    }
    if let Some(args) = options.codegen_opts.pre_link_args.as_ref() {
        for arg in args {
            cmd.arg(arg);
        }
    }
}

/// Add a link script embedded in the target, if applicable.
fn add_link_script(
    cmd: &mut dyn Linker,
    options: &Options,
    diagnostics: &DiagnosticsHandler,
    tmpdir: &Path,
    project_type: ProjectType,
) {
    match (project_type, options.target.options.link_script.as_deref()) {
        (ProjectType::Cdylib | ProjectType::Executable, Some(script)) => {
            if !options.target.options.linker_is_gnu {
                diagnostics
                    .fatal("can only use link script when linking with GNU-like linker")
                    .raise();
            }

            let file_name = ["firefly", &options.target.llvm_target, "linkfile.ld"].join("-");

            let path = tmpdir.join(file_name);
            if let Err(e) = fs::write(&path, script) {
                diagnostics
                    .fatal(&format!(
                        "failed to write link script to {}: {}",
                        path.display(),
                        e
                    ))
                    .raise();
            }

            cmd.arg("--script");
            cmd.arg(path);
        }
        _ => {}
    }
}

/// Add arbitrary "user defined" args defined from command line.
/// FIXME: Determine where exactly these args need to be inserted.
fn add_user_defined_link_args(cmd: &mut dyn Linker, options: &Options) {
    if let Some(args) = options.codegen_opts.linker_args.as_ref() {
        for arg in args {
            cmd.arg(arg);
        }
    }
}

/// Add arbitrary "late link" args defined by the target spec.
/// FIXME: Determine where exactly these args need to be inserted.
fn add_late_link_args(
    cmd: &mut dyn Linker,
    options: &Options,
    flavor: LinkerFlavor,
    project_type: ProjectType,
    _artifacts: &AppArtifacts,
) {
    let any_dynamic_crate = project_type == ProjectType::Dylib;
    if any_dynamic_crate {
        if let Some(args) = options.target.options.late_link_args_dynamic.get(&flavor) {
            cmd.args(args.iter().map(Deref::deref));
        }
    } else {
        if let Some(args) = options.target.options.late_link_args_static.get(&flavor) {
            cmd.args(args.iter().map(Deref::deref));
        }
    }
    if let Some(args) = options.target.options.late_link_args.get(&flavor) {
        cmd.args(args.iter().map(Deref::deref));
    }
}

/// Add arbitrary "post-link" args defined by the target spec.
/// FIXME: Determine where exactly these args need to be inserted.
fn add_post_link_args(cmd: &mut dyn Linker, options: &Options, flavor: LinkerFlavor) {
    if let Some(args) = options.target.options.post_link_args.get(&flavor) {
        cmd.args(args.iter().map(Deref::deref));
    }
}

/// Add object files containing code from the current crate.
fn add_local_objects(cmd: &mut dyn Linker, artifacts: &AppArtifacts) {
    for obj in artifacts.modules.iter().filter_map(|m| m.object()) {
        cmd.add_object(obj);
    }
}

/// Add sysroot and other globally set directories to the directory search list.
fn add_library_search_dirs(cmd: &mut dyn Linker, options: &Options, self_contained: bool) {
    // The default library location, we need this to find the runtime.
    // The location of crates will be determined as needed.
    let lib_path = options.target_filesearch(PathKind::All).get_lib_path();
    cmd.include_path(&fix_windows_verbatim_for_gcc(&lib_path));

    // Special directory with libraries used only in self-contained linkage mode
    if self_contained {
        let lib_path = options
            .target_filesearch(PathKind::All)
            .get_self_contained_lib_path();
        cmd.include_path(&fix_windows_verbatim_for_gcc(&lib_path));
    }
}

/// Add options making relocation sections in the produced ELF files read-only
/// and suppressing lazy binding.
fn add_relro_args(cmd: &mut dyn Linker, options: &Options) {
    match options
        .codegen_opts
        .relro_level
        .unwrap_or(options.target.options.relro_level)
    {
        RelroLevel::Full => cmd.full_relro(),
        RelroLevel::Partial => cmd.partial_relro(),
        RelroLevel::Off => cmd.no_relro(),
        RelroLevel::None => {}
    }
}

/// Add library search paths used at runtime by dynamic linkers.
fn add_rpath_args(
    cmd: &mut dyn Linker,
    options: &Options,
    _artifacts: &AppArtifacts,
    out_filename: &Path,
) {
    // FIXME (#2397): At some point we want to rpath our guesses as to
    // where extern libraries might live, based on the
    // add_lib_search_paths
    if options.codegen_opts.rpath {
        let mut rpath_config = RPathConfig {
            libs: &[],
            output_file: out_filename.to_path_buf(),
            has_rpath: options.target.options.has_rpath,
            is_like_osx: options.target.options.is_like_osx,
            linker_is_gnu: options.target.options.linker_is_gnu,
        };
        cmd.args(&rpath::get_rpath_flags(&mut rpath_config));
    }
}

/// Produce the linker command line containing linker path and arguments.
///
/// When comments in the function say "order-(in)dependent" they mean order-dependence between
/// options and libraries/object files. For example `--whole-archive` (order-dependent) applies
/// to specific libraries passed after it, and `-o` (output file, order-independent) applies
/// to the linking process as a whole.
/// Order-independent options may still override each other in order-dependent fashion,
/// e.g `--foo=yes --foo=no` may be equivalent to `--foo=no`.
fn linker_with_args(
    path: &Path,
    flavor: LinkerFlavor,
    options: &Options,
    diagnostics: &DiagnosticsHandler,
    project_type: ProjectType,
    tmpdir: &Path,
    out_filename: &Path,
    artifacts: &AppArtifacts,
) -> Command {
    let self_contained = self_contained(options, project_type);
    let cmd = &mut *linker::get(
        options,
        diagnostics,
        path,
        flavor,
        self_contained,
        artifacts.project_info.target_cpu.as_str(),
    );
    let link_output_kind = link_output_kind(options, project_type);

    // ------------ Early order-dependent options ------------

    // If we're building something like a dynamic library then some platforms
    // need to make sure that all symbols are exported correctly from the
    // dynamic library.
    // Must be passed before any libraries to prevent the symbols to export from being thrown away,
    // at least on some platforms (e.g. windows-gnu).
    cmd.export_symbols(
        tmpdir,
        project_type,
        &artifacts.project_info.exported_symbols.as_slice(),
    );

    // Can be used for adding custom CRT objects or overriding order-dependent options above.
    // FIXME: In practice built-in target specs use this for arbitrary order-independent options,
    // introduce a target spec option for order-independent linker options and migrate built-in
    // specs to it.
    add_pre_link_args(cmd, options, flavor);

    // ------------ Object code and libraries, order-dependent ------------

    // Pre-link CRT objects.
    add_pre_link_objects(cmd, options, link_output_kind, self_contained);

    // Sanitizer libraries.
    add_sanitizer_libraries(cmd, options, project_type);

    // Object code from the current project.
    // Take careful note of the ordering of the arguments we pass to the linker
    // here. Linkers will assume that things on the left depend on things to the
    // right. Things on the right cannot depend on things on the left. This is
    // all formally implemented in terms of resolving symbols (libs on the right
    // resolve unknown symbols of libs on the left, but not vice versa).
    //
    // For this reason, we have organized the arguments we pass to the linker as
    // such:
    //
    // 1. The local object that LLVM just generated
    // 2. Local native libraries
    // 3. Upstream dependency libraries
    // 4. Upstream native libraries
    //
    // The rationale behind this ordering is that those items lower down in the
    // list can't depend on items higher up in the list. For example nothing can
    // depend on what we just generated (e.g., that'd be a circular dependency).
    // Upstream libraries are not supposed to depend on our local native
    // libraries as that would violate the structure of the DAG, in that
    // scenario they are required to link to them as well in a shared fashion.
    // (The current implementation still doesn't prevent it though, see the FIXME below.)
    //
    // Note that upstream libraries may contain native dependencies as
    // well, but they also can't depend on what we just started to add to the
    // link line. And finally upstream native libraries can't depend on anything
    // in this DAG so far because they can only depend on other native libraries
    // and such dependencies are also required to be specified.
    add_local_objects(cmd, artifacts);

    // Avoid linking to dynamic libraries unless they satisfy some undefined symbols
    // at the point at which they are specified on the command line.
    // Must be passed before any (dynamic) libraries to have effect on them.
    // On Solaris-like systems, `-z ignore` acts as both `--as-needed` and `--gc-sections`
    // so it will ignore unreferenced ELF sections from relocatable objects.
    // For that reason, we put this flag after metadata objects as they would otherwise be removed.
    // FIXME: Support more fine-grained dead code removal on Solaris/illumos
    // and move this option back to the top.
    cmd.add_as_needed();

    // FIXME: Move this below to other native libraries
    // (or alternatively link all native libraries after their respective crates).
    // This change is somewhat breaking in practice due to local static libraries being linked
    // as whole-archive (#85144), so removing whole-archive may be a pre-requisite.
    if options.codegen_opts.link_native_libraries {
        add_local_native_libraries(cmd, options, diagnostics, artifacts, project_type, tmpdir);
    }

    // Upstream Erlang libraries and their nobundle static libraries
    add_upstream_erlang_libraries(cmd, options, artifacts, project_type, tmpdir);

    // Upstream dynamic native libraries linked with `#[link]` attributes at and `-l`
    // command line options.
    // If -Zlink-native-libraries=false is set, then the assumption is that an
    // external build system already has the native dependencies defined, and it
    // will provide them to the linker itself.
    if options.codegen_opts.link_native_libraries {
        add_upstream_native_libraries(cmd, options, artifacts);
    }

    // Library linking above uses some global state for things like `-Bstatic`/`-Bdynamic` to make
    // command line shorter, reset it to default here before adding more libraries.
    cmd.reset_per_library_state();

    // FIXME: Built-in target specs occasionally use this for linking system libraries,
    // eliminate all such uses by migrating them to `#[link]` attributes in `lib(std,c,unwind)`
    // and remove the option.
    add_late_link_args(cmd, options, flavor, project_type, artifacts);

    // ------------ Arbitrary order-independent options ------------

    // Add order-independent options determined by rustc from its compiler options,
    // target properties and source code.
    add_order_independent_options(
        cmd,
        options,
        diagnostics,
        link_output_kind,
        self_contained,
        flavor,
        project_type,
        artifacts,
        out_filename,
        tmpdir,
    );

    // Can be used for arbitrary order-independent options.
    // In practice may also be occasionally used for linking native libraries.
    // Passed after compiler-generated options to support manual overriding when necessary.
    add_user_defined_link_args(cmd, options);

    // ------------ Object code and libraries, order-dependent ------------

    // Post-link CRT objects.
    add_post_link_objects(cmd, options, link_output_kind, self_contained);

    // ------------ Late order-dependent options ------------

    // Doesn't really make sense.
    // FIXME: In practice built-in target specs use this for arbitrary order-independent options,
    // introduce a target spec option for order-independent linker options, migrate built-in specs
    // to it and remove the option.
    add_post_link_args(cmd, options, flavor);

    cmd.take_cmd()
}

fn add_order_independent_options(
    cmd: &mut dyn Linker,
    options: &Options,
    diagnostics: &DiagnosticsHandler,
    link_output_kind: LinkOutputKind,
    self_contained: bool,
    flavor: LinkerFlavor,
    project_type: ProjectType,
    artifacts: &AppArtifacts,
    out_filename: &Path,
    tmpdir: &Path,
) {
    add_gcc_ld_path(cmd, options, diagnostics, flavor);

    add_apple_sdk(cmd, options, diagnostics, flavor);

    add_link_script(cmd, options, diagnostics, tmpdir, project_type);

    if options.target.options.os == "fuchsia"
        && project_type == ProjectType::Executable
        && flavor != LinkerFlavor::Gcc
    {
        let prefix = if options
            .debugging_opts
            .sanitizers
            .contains(&Sanitizer::Address)
        {
            "asan/"
        } else {
            ""
        };
        cmd.arg(format!("--dynamic-linker={}ld.so.1", prefix));
    }

    if options.target.options.eh_frame_header {
        cmd.add_eh_frame_header();
    }

    // Make the binary compatible with data execution prevention schemes.
    cmd.add_no_exec();

    if self_contained {
        cmd.no_crt_objects();
    }

    if options.target.options.os == "emscripten" {
        cmd.arg("-s");
        cmd.arg(
            if options.target.options.panic_strategy == PanicStrategy::Abort {
                "DISABLE_EXCEPTION_CATCHING=1"
            } else {
                "DISABLE_EXCEPTION_CATCHING=0"
            },
        );
    }

    if flavor == LinkerFlavor::Ptx {
        // Provide the linker with fallback to internal `target-cpu`.
        cmd.arg("--fallback-arch");
        cmd.arg(&artifacts.project_info.target_cpu);
    } else if flavor == LinkerFlavor::Bpf {
        cmd.arg("--cpu");
        cmd.arg(&artifacts.project_info.target_cpu);
        cmd.arg("--cpu-features");
        let features = options
            .codegen_opts
            .target_features
            .as_ref()
            .map(|f| f.into())
            .unwrap_or_else(|| options.target.options.features.clone());
        cmd.arg(features.as_ref());
    }

    cmd.linker_plugin_lto();

    add_library_search_dirs(cmd, options, self_contained);

    cmd.output_filename(out_filename);

    if project_type == ProjectType::Executable && options.target.options.is_like_windows {
        if let Some(ref s) = artifacts.project_info.windows_subsystem {
            cmd.subsystem(s);
        }
    }

    // Try to strip as much out of the generated object by removing unused
    // sections if possible. See more comments in linker.rs
    //
    // NOTE: The default here is `true`, i.e. we DON'T garbage collect sections
    // that are unreachable from the entry point. This is because apply/3 makes
    // it impossible to see what symbols are actually reachable statically, so we
    // have to be pessimistic and assume everything is reachable.
    //
    // TODO(pauls): We should track the list of atoms/functions that are defined
    // in Erlang code, and then pass a list of just those symbols to the linker
    // to force them to be exported, but permit the linker to clean up anything
    // else that is unreachable. Right now we can't get rid of anything
    if !options.codegen_opts.link_dead_code.unwrap_or(true) {
        // If PGO is enabled sometimes gc_sections will remove the profile data section
        // as it appears to be unused. This can then cause the PGO profile file to lose
        // some functions. If we are generating a profile we shouldn't strip those metadata
        // sections to ensure we have all the data for PGO.
        let keep_metadata = project_type == ProjectType::Dylib; // || options.codegen_opts.profile_generate.enabled();
        if project_type != ProjectType::Executable
            || !options.codegen_opts.export_executable_symbols
        {
            cmd.gc_sections(keep_metadata)
        } else {
            cmd.no_gc_sections();
        }
    }

    cmd.set_output_kind(link_output_kind, out_filename);

    add_relro_args(cmd, options);

    // Pass optimization flags down to the linker.
    cmd.optimize();

    // Pass debuginfo and strip flags down to the linker.
    cmd.debuginfo(options.codegen_opts.strip);

    // We want to prevent the compiler from accidentally leaking in any system libraries,
    // so by default we tell linkers not to link to any default libraries.
    if !options.codegen_opts.default_linker_libraries && options.target.options.no_default_libraries
    {
        cmd.no_default_libraries();
    }

    if options.codegen_opts.control_flow_guard != CFGuard::Disabled {
        cmd.control_flow_guard();
    }

    add_rpath_args(cmd, options, artifacts, out_filename);
}

// A dylib may reexport symbols from the linked rlib or native static library.
// Even if some symbol is reexported it's still not necessarily counted as used and may be
// dropped, at least with `ld`-like ELF linkers. So we have to link some rlibs and static
// libraries as whole-archive to avoid losing reexported symbols.
// FIXME: Find a way to mark reexported symbols as used and avoid this use of whole-archive.
fn default_to_whole_archive(
    cmd: &dyn Linker,
    options: &Options,
    project_type: ProjectType,
) -> bool {
    project_type == ProjectType::Dylib
        && !(options.target.options.limit_rdylib_exports && cmd.exported_symbol_means_used_symbol())
}

// # Native library linking
//
// User-supplied library search paths (-L on the command line). These are
// the same paths used to find Rust crates, so some of them may have been
// added already by the previous crate linking code. This only allows them
// to be found at compile time so it is still entirely up to outside
// forces to make sure that library can be found at runtime.
//
// Also note that the native libraries linked here are only the ones located
// in the current crate. Upstream crates with native library dependencies
// may have their native library pulled in above.
fn add_local_native_libraries(
    cmd: &mut dyn Linker,
    options: &Options,
    diagnostics: &DiagnosticsHandler,
    artifacts: &AppArtifacts,
    project_type: ProjectType,
    _tmpdir: &Path,
) {
    let filesearch = options.target_filesearch(PathKind::All);
    for search_path in filesearch.search_paths() {
        match search_path.kind {
            PathKind::Framework => {
                cmd.framework_path(&search_path.dir);
            }
            _ => {
                cmd.include_path(&fix_windows_verbatim_for_gcc(&search_path.dir));
            }
        }
    }

    let search_path = OnceCell::new();
    let mut last = (None, NativeLibraryKind::Unspecified, None);
    for lib in artifacts.project_info.used_libraries.as_slice() {
        let Some(name) = lib.name.as_ref() else {
            continue;
        };

        // Skip if this library is the same as the last.
        let this = (lib.name.clone(), lib.kind, lib.verbatim);
        last = if this == last {
            continue;
        } else {
            this
        };
        let verbatim = lib.verbatim.unwrap_or(false);

        match lib.kind {
            NativeLibraryKind::Dylib { as_needed } => {
                cmd.link_dylib(name, verbatim, as_needed.unwrap_or(true))
            }
            NativeLibraryKind::Unspecified => cmd.link_dylib(name, verbatim, true),
            NativeLibraryKind::Framework { as_needed } => {
                cmd.link_framework(name, as_needed.unwrap_or(true))
            }
            NativeLibraryKind::Static {
                whole_archive,
                bundle,
                ..
            } => {
                if whole_archive == Some(true)
                    || (whole_archive == None && default_to_whole_archive(cmd, options, project_type))
                    // Backward compatibility case: this can be a rlib (so `+whole-archive` cannot
                    // be added explicitly if necessary, see the error in `fn link_rlib`) compiled
                    // as an executable due to `--test`. Use whole-archive implicitly, like before
                    // the introduction of native lib modifiers.
                    || (bundle != Some(false) && options.test)
                {
                    cmd.link_whole_staticlib(
                        name,
                        verbatim,
                        &search_path.get_or_init(|| archive_search_paths(options)),
                    );
                } else {
                    cmd.link_staticlib(name, verbatim)
                }
            }
            NativeLibraryKind::RawDylib => {
                // FIXME(#58713): Proper handling for raw dylibs.
                diagnostics
                    .fatal("raw_dylib feature not yet implemented")
                    .raise();
            }
        }
    }
}

fn add_upstream_erlang_libraries(
    cmd: &mut dyn Linker,
    options: &Options,
    artifacts: &AppArtifacts,
    project_type: ProjectType,
    _tmpdir: &Path,
) {
    for dependency in &artifacts.project_info.used_deps {
        let Some(path) = dependency.source.as_ref() else { continue; };

        let path = fix_windows_verbatim_for_gcc(path);
        if default_to_whole_archive(cmd, options, project_type) {
            cmd.link_whole_rlib(&path);
        } else {
            cmd.link_rlib(&path);
        }
    }
}

fn add_upstream_native_libraries(
    cmd: &mut dyn Linker,
    _options: &Options,
    artifacts: &AppArtifacts,
) {
    let mut last = (None, NativeLibraryKind::Unspecified, None);
    for lib in &artifacts.project_info.native_libraries {
        let Some(name) = lib.name.as_ref() else { continue; };
        let this = (lib.name.clone(), lib.kind, lib.verbatim);
        last = if this == last {
            continue;
        } else {
            this
        };

        let verbatim = lib.verbatim.unwrap_or(false);
        match lib.kind {
            NativeLibraryKind::Dylib { as_needed } => {
                cmd.link_dylib(name, verbatim, as_needed.unwrap_or(true))
            }
            NativeLibraryKind::Unspecified => cmd.link_dylib(name, verbatim, true),
            NativeLibraryKind::Framework { as_needed } => {
                cmd.link_framework(name, as_needed.unwrap_or(true))
            }
            // ignore static native libraries here as we've
            // already included them in add_local_native_libraries and
            // add_upstream_erlang_libraries
            NativeLibraryKind::Static { .. } => {}
            NativeLibraryKind::RawDylib => {}
        }
    }
}

fn add_apple_sdk(
    cmd: &mut dyn Linker,
    options: &Options,
    diagnostics: &DiagnosticsHandler,
    flavor: LinkerFlavor,
) {
    let arch = &options.target.arch;
    let os = &options.target.options.os;
    let llvm_target = &options.target.llvm_target;
    if options.target.options.vendor != "apple"
        || !matches!(os.as_ref(), "ios" | "tvos" | "watchos" | "macos")
        || (flavor != LinkerFlavor::Gcc && flavor != LinkerFlavor::Lld(LldFlavor::Ld64))
    {
        return;
    }

    if os == "macos" && flavor != LinkerFlavor::Lld(LldFlavor::Ld64) {
        return;
    }

    let sdk_name = match (arch.as_ref(), os.as_ref()) {
        ("aarch64", "tvos") => "appletvos",
        ("x86_64", "tvos") => "appletvsimulator",
        ("arm", "ios") => "iphoneos",
        ("aarch64", "ios") if llvm_target.contains("macabi") => "macosx",
        ("aarch64", "ios") if llvm_target.ends_with("-simulator") => "iphonesimulator",
        ("aarch64", "ios") => "iphoneos",
        ("x86", "ios") => "iphonesimulator",
        ("x86_64", "ios") if llvm_target.contains("macabi") => "macosx",
        ("x86_64", "ios") => "iphonesimulator",
        ("x86_64", "watchos") => "watchsimulator",
        ("arm64_32", "watchos") => "watchos",
        ("aarch64", "watchos") if llvm_target.ends_with("-simulator") => "watchsimulator",
        ("aarch64", "watchos") => "watchos",
        ("arm", "watchos") => "watchos",
        (_, "macos") => "macosx",
        _ => {
            diagnostics.error(format!("unsupported arch `{}` for os `{}`", arch, os));
            return;
        }
    };
    let sdk_root = match get_apple_sdk_root(sdk_name) {
        Ok(s) => s,
        Err(e) => {
            diagnostics.error(e.to_string());
            return;
        }
    };
    match flavor {
        LinkerFlavor::Gcc => {
            cmd.args(&["-isysroot", &sdk_root, "-Wl,-syslibroot", &sdk_root]);
        }
        LinkerFlavor::Lld(LldFlavor::Ld64) => {
            cmd.args(&["-syslibroot", &sdk_root]);
        }
        _ => unreachable!(),
    }
}

fn get_apple_sdk_root(sdk_name: &str) -> Result<String, String> {
    // Following what clang does
    // (https://github.com/llvm/llvm-project/blob/
    // 296a80102a9b72c3eda80558fb78a3ed8849b341/clang/lib/Driver/ToolChains/Darwin.cpp#L1661-L1678)
    // to allow the SDK path to be set. (For clang, xcrun sets
    // SDKROOT; for rustc, the user or build system can set it, or we
    // can fall back to checking for xcrun on PATH.)
    if let Ok(sdkroot) = env::var("SDKROOT") {
        let p = Path::new(&sdkroot);
        match sdk_name {
            // Ignore `SDKROOT` if it's clearly set for the wrong platform.
            "appletvos"
                if sdkroot.contains("TVSimulator.platform")
                    || sdkroot.contains("MacOSX.platform") => {}
            "appletvsimulator"
                if sdkroot.contains("TVOS.platform") || sdkroot.contains("MacOSX.platform") => {}
            "iphoneos"
                if sdkroot.contains("iPhoneSimulator.platform")
                    || sdkroot.contains("MacOSX.platform") => {}
            "iphonesimulator"
                if sdkroot.contains("iPhoneOS.platform") || sdkroot.contains("MacOSX.platform") => {
            }
            "macosx10.15"
                if sdkroot.contains("iPhoneOS.platform")
                    || sdkroot.contains("iPhoneSimulator.platform") => {}
            "watchos"
                if sdkroot.contains("WatchSimulator.platform")
                    || sdkroot.contains("MacOSX.platform") => {}
            "watchsimulator"
                if sdkroot.contains("WatchOS.platform") || sdkroot.contains("MacOSX.platform") => {}
            // Ignore `SDKROOT` if it's not a valid path.
            _ if !p.is_absolute() || p == Path::new("/") || !p.exists() => {}
            _ => return Ok(sdkroot),
        }
    }
    let res = Command::new("xcrun")
        .arg("--show-sdk-path")
        .arg("-sdk")
        .arg(sdk_name)
        .output()
        .and_then(|output| {
            if output.status.success() {
                Ok(String::from_utf8(output.stdout).unwrap())
            } else {
                let error = String::from_utf8(output.stderr);
                let error = format!("process exit with error: {}", error.unwrap());
                Err(io::Error::new(io::ErrorKind::Other, &error[..]))
            }
        });

    match res {
        Ok(output) => Ok(output.trim().to_string()),
        Err(e) => Err(format!("failed to get {} SDK path: {}", sdk_name, e)),
    }
}

fn add_gcc_ld_path(
    cmd: &mut dyn Linker,
    options: &Options,
    diagnostics: &DiagnosticsHandler,
    flavor: LinkerFlavor,
) {
    if let Some(ld_impl) = options.codegen_opts.gcc_ld {
        if let LinkerFlavor::Gcc = flavor {
            match ld_impl {
                LdImpl::Lld => {
                    // Implement the "self-contained" part of -Zgcc-ld
                    // by adding the firefly distribution directories to the tool search path.
                    for path in options.get_tools_search_paths(false) {
                        cmd.cmd().arg({
                            let mut arg = OsString::from("-B");
                            arg.push(path.join("gcc-ld"));
                            arg
                        });
                    }
                    // Implement the "linker flavor" part of -Zgcc-ld
                    // by asking cc to use some kind of lld.
                    cmd.arg("-fuse-ld=lld");
                    if options.target.options.lld_flavor != LldFlavor::Ld {
                        // Tell clang to use a non-default LLD flavor.
                        // Gcc doesn't understand the target option, but we currently assume
                        // that gcc is not used for Apple and Wasm targets (rust-lang/rust#97402).
                        cmd.arg(format!("--target={}", options.target.llvm_target));
                    }
                }
            }
        } else {
            diagnostics
                .fatal("option `-Z gcc-ld` is used even though linker flavor is not gcc")
                .raise();
        }
    }
}

/// Checks if target supports project_type as output
fn invalid_output_for_target(options: &Options) -> bool {
    let project_type = options.project_type;
    match project_type {
        ProjectType::Cdylib | ProjectType::Dylib => {
            if !options.target.options.dynamic_linking {
                return true;
            }
            if options.crt_static(Some(project_type))
                && !options.target.options.crt_static_allows_dylibs
            {
                return true;
            }
        }
        _ => {}
    }
    if options.target.options.only_cdylib {
        match project_type {
            ProjectType::Dylib => return true,
            _ => {}
        }
    }
    if !options.target.options.executables {
        if project_type == ProjectType::Executable {
            return true;
        }
    }

    false
}

fn remove(path: &Path) -> anyhow::Result<()> {
    if let Err(err) = fs::remove_file(path) {
        return Err(anyhow!("failed to remove {}: {}", path.display(), err));
    }
    Ok(())
}

// Make sure files are writeable.  Mac, FreeBSD, and Windows system linkers
// check this already -- however, the Linux linker will happily overwrite a
// read-only file.  We should be consistent.
fn check_file_is_writeable(file: &Path) -> anyhow::Result<()> {
    if !is_writeable(file) {
        return Err(anyhow!(format!(
            "output file {} is not writeable -- check its \
             permissions",
            file.display()
        )));
    }
    Ok(())
}

fn is_writeable(p: &Path) -> bool {
    match p.metadata() {
        Err(..) => true,
        Ok(m) => !m.permissions().readonly(),
    }
}
