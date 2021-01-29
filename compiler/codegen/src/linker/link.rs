use std::ascii;
use std::char;
use std::env;
use std::ffi::OsString;
use std::fmt;
use std::fs;

use std::io;
use std::path::{Path, PathBuf};
use std::process::{ExitStatus, Output, Stdio};
use std::str;

use anyhow::*;
use cc::windows_registry;
use log::{info, warn};
use tempfile::Builder as TempFileBuilder;
use thiserror::private::PathAsDisplay;

use liblumen_core::util::thread_local::ThreadLocalCell;
use liblumen_session::filesearch;
use liblumen_session::search_paths::PathKind;
use liblumen_session::{CFGuard, DebugInfo, Options, ProjectType};
use liblumen_target::crt_objects::CrtObjectsFallback;
use liblumen_target::{
    LinkOutputKind, LinkerFlavor, LldFlavor, PanicStrategy, RelocModel, RelroLevel,
};
use liblumen_util::diagnostics::DiagnosticsHandler;
use liblumen_util::fs::{fix_windows_verbatim_for_gcc, NativeLibraryKind};
use liblumen_util::time::time;

use crate::linker::command::Command;
use crate::linker::rpath::{self, RPathConfig};
use crate::linker::Linker;
use crate::meta::{CodegenResults, LibSource};

use super::archive::{ArchiveBuilder, LlvmArchiveBuilder};

enum RlibFlavor {
    #[allow(dead_code)]
    Normal,
    StaticlibBase,
}

/// Performs the linkage portion of the compilation phase. This will generate all
/// of the requested outputs for this compilation session.
pub fn link_binary(
    options: &Options,
    diagnostics: &DiagnosticsHandler,
    codegen_results: &CodegenResults,
) -> anyhow::Result<()> {
    let project_type = options.project_type;
    if invalid_output_for_target(options) {
        return Err(anyhow!(
            "invalid output type `{:?}` for target os `{}`",
            project_type,
            options.target.triple()
        ));
    }

    for obj in codegen_results.modules.iter().filter_map(|m| m.object()) {
        check_file_is_writeable(obj)?;
    }

    let tmpdir = TempFileBuilder::new()
        .prefix("lumen")
        .tempdir()
        .map_err(|err| anyhow!("couldn't create a temp dir: {}", err))?;

    let output_dir = options.output_dir();
    let output_file = options
        .output_file
        .as_ref()
        .map(|of| of.clone())
        .unwrap_or_else(|| {
            let name = PathBuf::from(options.project_name.as_str());
            let ext = match project_type {
                ProjectType::Executable if options.target.options.is_like_windows => "exe",
                ProjectType::Executable => "out",
                ProjectType::Staticlib => "a",
                _ => "o",
            };
            let mut p = output_dir.as_path().join(name);
            p.set_extension(ext);
            p
        });

    // `output_dir` is not necessarily the parent of `output_file`, such as with
    // `--output-dir _build --output bin/myapp`
    let output_file_parent = output_file.parent().with_context(|| {
        format!(
            "{} does not have a parent directory",
            output_file.as_display()
        )
    })?;
    fs::create_dir_all(output_file_parent).with_context(|| {
        format!(
            "Could not create parent directories ({}) of file ({})",
            output_file_parent.as_display(),
            output_file.as_display()
        )
    })?;

    match project_type {
        ProjectType::Staticlib => {
            link_staticlib(
                options,
                diagnostics,
                project_type,
                codegen_results,
                output_file.as_path(),
                tmpdir.path(),
            )?;
        }
        _ => {
            link_natively(
                options,
                diagnostics,
                project_type,
                output_dir.as_path(),
                output_file.as_path(),
                codegen_results,
                tmpdir.path(),
            )?;
        }
    }

    // Remove the temporary object file and metadata if we aren't saving temps
    for obj in codegen_results.modules.iter().filter_map(|m| m.object()) {
        if let Err(e) = remove(obj) {
            diagnostics.error(format!("{}", e));
        }
    }

    Ok(())
}

// The third parameter is for env vars, used on windows to set up the
// path for MSVC to find its DLLs, and gcc to find its bundled
// toolchain
pub fn get_linker(
    options: &Options,
    linker: &Path,
    flavor: LinkerFlavor,
    self_contained: bool,
) -> Command {
    let msvc_tool = windows_registry::find_tool(&options.target.triple(), "link.exe");

    // If our linker looks like a batch script on Windows then to execute this
    // we'll need to spawn `cmd` explicitly. This is primarily done to handle
    // emscripten where the linker is `emcc.bat` and needs to be spawned as
    // `cmd /c emcc.bat ...`.
    //
    // This worked historically but is needed manually since #42436 (regression
    // was tagged as #42791) and some more info can be found on #44443 for
    // emscripten itself.
    let mut cmd = match linker.to_str() {
        Some(linker) if cfg!(windows) && linker.ends_with(".bat") => Command::bat_script(linker),
        _ => match flavor {
            LinkerFlavor::Lld(f) => Command::lld(linker, f),
            LinkerFlavor::Msvc
                if options.codegen_opts.linker.is_none()
                    && options.target.options.linker.is_none() =>
            {
                Command::new(msvc_tool.as_ref().map(|t| t.path()).unwrap_or(linker))
            }
            _ => Command::new(linker),
        },
    };

    // UWP apps have API restrictions enforced during Store submissions.
    // To comply with the Windows App Certification Kit,
    // MSVC needs to link with the Store versions of the runtime libraries (vcruntime, msvcrt, etc).
    let t = &options.target;
    if (flavor == LinkerFlavor::Msvc || flavor == LinkerFlavor::Lld(LldFlavor::Link))
        && t.target_vendor == "uwp"
    {
        if let Some(ref tool) = msvc_tool {
            let original_path = tool.path();
            if let Some(ref root_lib_path) = original_path.ancestors().skip(4).next() {
                let arch = match t.arch.as_str() {
                    "x86_64" => Some("x64".to_string()),
                    "x86" => Some("x86".to_string()),
                    "aarch64" => Some("arm64".to_string()),
                    "arm" => Some("arm".to_string()),
                    _ => None,
                };
                if let Some(ref a) = arch {
                    let mut arg = OsString::from("/LIBPATH:");
                    arg.push(format!(
                        "{}\\lib\\{}\\store",
                        root_lib_path.display(),
                        a.to_string()
                    ));
                    cmd.arg(&arg);
                } else {
                    warn!("arch is not supported");
                }
            } else {
                warn!("MSVC root path lib location not found");
            }
        } else {
            warn!("link.exe not found");
        }
    }

    // The compiler's sysroot often has some bundled tools, so add it to the
    // PATH for the child.
    let mut new_path = options
        .host_filesearch(PathKind::All)
        .get_tools_search_paths(self_contained);
    let mut msvc_changed_path = false;
    if options.target.options.is_like_msvc {
        if let Some(ref tool) = msvc_tool {
            cmd.args(tool.args());
            for &(ref k, ref v) in tool.env() {
                if k == "PATH" {
                    new_path.extend(env::split_paths(v));
                    msvc_changed_path = true;
                } else {
                    cmd.env(k, v);
                }
            }
        }
    }

    if !msvc_changed_path {
        if let Some(path) = env::var_os("PATH") {
            new_path.extend(env::split_paths(&path));
        }
    }
    cmd.env("PATH", env::join_paths(new_path).unwrap());

    cmd
}

// Create a static archive
//
// There's no way for us to link dynamic libraries, so we warn
// about all dynamic library dependencies that they're not linked in.
fn link_staticlib(
    options: &Options,
    _diagnostics: &DiagnosticsHandler,
    project_type: ProjectType,
    codegen_results: &CodegenResults,
    output_file: &Path,
    tmpdir: &Path,
) -> anyhow::Result<()> {
    info!("preparing {:?} to {:?}", project_type, output_file);

    let mut ab = create_rlib(
        options,
        codegen_results,
        RlibFlavor::StaticlibBase,
        output_file,
        tmpdir,
    );

    for (name, source) in codegen_results.project_info.used_deps_static.iter() {
        match source {
            LibSource::Some(path) => {
                ab.add_rlib(
                    path,
                    name.as_str(),
                    /* lto= */ false,
                    /* skip_objects= */ false,
                )
                .unwrap();
            }
            LibSource::None => {
                return Err(anyhow!("could not find rlib for: `{}`", name));
            }
        }
    }

    ab.update_symbols();
    ab.build();

    Ok(())
}

// Create a dynamic library or executable
//
// This will invoke the system linker/cc to create the resulting file. This
// links to all upstream files as well.
fn link_natively(
    options: &Options,
    diagnostics: &DiagnosticsHandler,
    project_type: ProjectType,
    _output_dir: &Path,
    output_file: &Path,
    codegen_results: &CodegenResults,
    tmpdir: &Path,
) -> anyhow::Result<()> {
    info!("preparing {:?} to {:?}", project_type, output_file);
    let (linker_path, flavor) = linker_and_flavor(options)?;

    let mut cmd = linker_with_args(
        &linker_path,
        flavor,
        options,
        diagnostics,
        project_type,
        tmpdir,
        output_file,
        codegen_results,
    );

    //linker::disable_localization(&mut cmd);

    for &(ref k, ref v) in &options.target.options.link_env {
        cmd.env(k, v);
    }
    for k in &options.target.options.link_env_remove {
        cmd.env_remove(k);
    }

    if options.debugging_opts.print_link_args {
        println!("{:?}", &cmd);
    }

    // May have not found libraries in the right formats.
    diagnostics.abort_if_errors();

    use_system_linker(options, diagnostics, cmd, flavor, output_file, tmpdir)
}

fn use_system_linker(
    options: &Options,
    diagnostics: &DiagnosticsHandler,
    mut cmd: Command,
    flavor: LinkerFlavor,
    output_file: &Path,
    tmpdir: &Path,
) -> anyhow::Result<()> {
    // Invoke the system linker
    info!("invoking system linker: {:?}", &cmd);
    let retry_on_segfault = env::var("LUMEN_RETRY_LINKER_ON_SEGFAULT").is_ok();
    let mut prog;
    let mut i = 0;
    loop {
        i += 1;
        prog = time(options.debugging_opts.time_passes, "running linker", || {
            exec_linker(options, &mut cmd, output_file, tmpdir)
        });
        let output = match prog {
            Ok(ref output) => output,
            Err(_) => break,
        };
        if output.status.success() {
            break;
        }
        let mut out = output.stderr.clone();
        out.extend(&output.stdout);
        let out = String::from_utf8_lossy(&out);

        // Check to see if the link failed with "unrecognized command line option:
        // '-no-pie'" for gcc or "unknown argument: '-no-pie'" for clang. If so,
        // reperform the link step without the -no-pie option. This is safe because
        // if the linker doesn't support -no-pie then it should not default to
        // linking executables as pie. Different versions of gcc seem to use
        // different quotes in the error message so don't check for them.
        if options.target.options.linker_is_gnu
            && flavor != LinkerFlavor::Ld
            && (out.contains("unrecognized command line option")
                || out.contains("unknown argument"))
            && out.contains("-no-pie")
            && cmd
                .get_args()
                .iter()
                .any(|e| e.to_string_lossy() == "-no-pie")
        {
            info!("linker output: {:?}", out);
            warn!("Linker does not support -no-pie command line option. Retrying without.");
            for arg in cmd.take_args() {
                if arg.to_string_lossy() != "-no-pie" {
                    cmd.arg(arg);
                }
            }
            info!("{:?}", &cmd);
            continue;
        }

        // Here's a terribly awful hack that really shouldn't be present in any
        // compiler. Here an environment variable is supported to automatically
        // retry the linker invocation if the linker looks like it segfaulted.
        //
        // Gee that seems odd, normally segfaults are things we want to know
        // about!  Unfortunately though in rust-lang/rust#38878 we're
        // experiencing the linker segfaulting on Travis quite a bit which is
        // causing quite a bit of pain to land PRs when they spuriously fail
        // due to a segfault.
        //
        // The issue #38878 has some more debugging information on it as well,
        // but this unfortunately looks like it's just a race condition in
        // macOS's linker with some thread pool working in the background. It
        // seems that no one currently knows a fix for this so in the meantime
        // we're left with this...
        if !retry_on_segfault || i > 3 {
            break;
        }
        let msg_segv = "clang: error: unable to execute command: Segmentation fault: 11";
        let msg_bus = "clang: error: unable to execute command: Bus error: 10";
        if out.contains(msg_segv) || out.contains(msg_bus) {
            warn!(
                "looks like the linker segfaulted when we tried to call it, \
                 automatically retrying again. cmd = {:?}, out = {}.",
                cmd, out,
            );
            continue;
        }

        if is_illegal_instruction(&output.status) {
            warn!(
                "looks like the linker hit an illegal instruction when we \
                 tried to call it, automatically retrying again. cmd = {:?}, ]\
                 out = {}, status = {}.",
                cmd, out, output.status,
            );
            continue;
        }

        #[cfg(unix)]
        fn is_illegal_instruction(status: &ExitStatus) -> bool {
            use std::os::unix::prelude::*;
            status.signal() == Some(libc::SIGILL)
        }

        #[cfg(windows)]
        fn is_illegal_instruction(_status: &ExitStatus) -> bool {
            false
        }
    }

    match prog {
        Ok(prog) => {
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
            if !prog.status.success() {
                let mut output = prog.stderr.clone();
                output.extend_from_slice(&prog.stdout);
                diagnostics.error(format!(
                    "linking failed: {}\n\
                     \n\
                     {:?}\n\
                     \n\
                     {}",
                    prog.status,
                    &cmd,
                    &escape_string(&output)
                ));
                diagnostics.abort_if_errors();
            }
        }
        Err(e) => {
            let linker_not_found = e.kind() == io::ErrorKind::NotFound;

            let mut linker_error = {
                if linker_not_found {
                    format!("linker not found")
                } else {
                    format!("could not exec the linker")
                }
            };

            linker_error = format!("{}\n\n{}", linker_error, &e.to_string());

            if !linker_not_found {
                linker_error = format!("{}\n\n{:?}", linker_error, &cmd);
            }

            diagnostics.error(linker_error);

            if options.target.options.is_like_msvc && linker_not_found {
                warn!(
                    "the msvc targets depend on the msvc linker \
                     but `link.exe` was not found",
                );
                warn!(
                    "please ensure that VS 2013, VS 2015, VS 2017 or VS 2019 \
                     was installed with the Visual C++ option",
                );
            }
            diagnostics.abort_if_errors();
        }
    }

    // On macOS, debuggers need this utility to get run to do some munging of
    // the symbols. Note, though, that if the object files are being preserved
    // for their debug information there's no need for us to run dsymutil.
    if options.target.options.is_like_osx
        && options.debug_info != DebugInfo::None
        && !preserve_objects_for_their_debuginfo(options)
    {
        if let Err(e) = Command::new("dsymutil").arg(output_file).output() {
            diagnostics
                .fatal(format!("failed to run dsymutil: {}", e))
                .raise();
        }
    }

    Ok(())
}

pub fn linker_and_flavor(options: &Options) -> anyhow::Result<(PathBuf, LinkerFlavor)> {
    fn infer_from(
        options: &Options,
        linker: Option<PathBuf>,
        flavor: Option<LinkerFlavor>,
    ) -> anyhow::Result<Option<(PathBuf, LinkerFlavor)>> {
        match (linker, flavor) {
            // Explicit linker+flavor
            (Some(linker), Some(flavor)) => Ok(Some((linker, flavor))),
            // Only the linker flavor is known; use the default linker for the selected flavor
            (None, Some(flavor)) => {
                let prog = match flavor {
                    LinkerFlavor::Em => {
                        if cfg!(windows) {
                            "emcc.bat"
                        } else {
                            "emcc"
                        }
                    }
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
                    LinkerFlavor::Msvc => "link.exe",
                    LinkerFlavor::Lld(_) => "lld",
                    f => {
                        return Err(anyhow!(
                            "invalid linker flavor '{}': flavor is unimplemented",
                            f
                        ))
                    }
                };
                Ok(Some((PathBuf::from(prog), flavor)))
            }
            (Some(linker), None) => {
                let stem = linker
                    .file_stem()
                    .and_then(|stem| stem.to_str())
                    .ok_or_else(|| anyhow!("couldn't extract file stem from specified linker"))?;

                let flavor = if stem == "emcc" {
                    LinkerFlavor::Em
                } else if stem == "gcc"
                    || stem.ends_with("-gcc")
                    || stem == "clang"
                    || stem.ends_with("-clang")
                {
                    LinkerFlavor::Gcc
                } else if stem == "ld" || stem == "ld.lld" || stem.ends_with("-ld") {
                    LinkerFlavor::Ld
                } else if stem == "link" || stem == "lld-link" {
                    LinkerFlavor::Msvc
                } else if stem == "lld" || stem == "lumen-lld" {
                    LinkerFlavor::Lld(options.target.options.lld_flavor)
                } else {
                    // fall back to the value in the target spec
                    options.target.linker_flavor
                };

                Ok(Some((linker, flavor)))
            }
            (None, None) => Ok(None),
        }
    }

    // linker and linker flavor specified via command line have precedence over what the target
    // specification specifies
    if let Some(ret) = infer_from(
        options,
        options.codegen_opts.linker.clone(),
        options.codegen_opts.linker_flavor,
    )? {
        return Ok(ret);
    }

    if let Some(ret) = infer_from(
        options,
        options.target.options.linker.clone().map(PathBuf::from),
        Some(options.target.linker_flavor),
    )? {
        return Ok(ret);
    }

    Err(anyhow!(
        "Not enough information provided to determine how to invoke the linker"
    ))
}

/// Returns a boolean indicating whether we should preserve the object files on
/// the filesystem for their debug information. This is often useful with
/// split-dwarf like schemes.
pub fn preserve_objects_for_their_debuginfo(options: &Options) -> bool {
    // If the objects don't have debuginfo there's nothing to preserve.
    if options.debug_info == DebugInfo::None {
        return false;
    }

    // If we're only producing artifacts that are archives, no need to preserve
    // the objects as they're losslessly contained inside the archives.
    if options.project_type != ProjectType::Staticlib {
        return false;
    }

    // If we're on OSX then the equivalent of split dwarf is turned on by
    // default. The final executable won't actually have any debug information
    // except it'll have pointers to elsewhere. Historically we've always run
    // `dsymutil` to "link all the dwarf together" but this is actually sort of
    // a bummer for incremental compilation! (the whole point of split dwarf is
    // that you don't do this sort of dwarf link).
    //
    // Basically as a result this just means that if we're on OSX and we're
    // *not* running dsymutil then the object files are the only source of truth
    // for debug information, so we must preserve them.
    if options.target.options.is_like_osx {
        match options.codegen_opts.run_dsymutil {
            // dsymutil is not being run, preserve objects
            Some(false) => return true,

            // dsymutil is being run, no need to preserve the objects
            Some(true) => return false,

            // The default historical behavior was to always run dsymutil, so
            // we're preserving that temporarily, but we're likely to switch the
            // default soon.
            None => return false,
        }
    }

    false
}

pub fn archive_search_paths(options: &Options) -> Vec<PathBuf> {
    options
        .target_filesearch(PathKind::Native)
        .search_path_dirs()
}

// Path for libraries that will take preference over libraries shipped by Rust.
// Used by windows-gnu targets to priortize system mingw-w64 libraries.
thread_local!(static SYSTEM_LIB_PATH: ThreadLocalCell<Option<Option<PathBuf>>> = ThreadLocalCell::new(None));

// Because windows-gnu target is meant to be self-contained for pure Rust code it bundles
// own mingw-w64 libraries. These libraries are usually not compatible with mingw-w64
// installed in the system. This breaks many cases where Rust is mixed with other languages
// (e.g. *-sys crates).
// We prefer system mingw-w64 libraries if they are available to avoid this issue.
fn get_crt_libs_path(options: &Options) -> Option<PathBuf> {
    fn find_exe_in_path<P>(exe_name: P) -> Option<PathBuf>
    where
        P: AsRef<Path>,
    {
        for dir in env::split_paths(&env::var_os("PATH")?) {
            let full_path = dir.join(&exe_name);
            if full_path.is_file() {
                return Some(fix_windows_verbatim_for_gcc(&full_path));
            }
        }
        None
    }

    fn probe(options: &Options) -> Option<PathBuf> {
        if let Ok((linker, LinkerFlavor::Gcc)) = linker_and_flavor(&options) {
            let linker_path = if cfg!(windows) && linker.extension().is_none() {
                linker.with_extension("exe")
            } else {
                linker
            };
            if let Some(linker_path) = find_exe_in_path(linker_path) {
                let mingw_arch = match &options.target.arch {
                    x if x == "x86" => "i686",
                    x => x,
                };
                let mingw_bits = &options.target.target_pointer_width;
                let mingw_dir = format!("{}-w64-mingw32", mingw_arch);
                // Here we have path/bin/gcc but we need path/
                let mut path = linker_path;
                path.pop();
                path.pop();
                // Loosely based on Clang MinGW driver
                let probe_paths = vec![
                    path.join(&mingw_dir).join("lib"),                // Typical path
                    path.join(&mingw_dir).join("sys-root/mingw/lib"), // Rare path
                    path.join(format!(
                        "lib/mingw/tools/install/mingw{}/{}/lib",
                        &mingw_bits, &mingw_dir
                    )), // Chocolatey is creative
                ];
                for probe_path in probe_paths {
                    if probe_path.join("crt2.o").exists() {
                        return Some(probe_path);
                    };
                }
            };
        };
        None
    }

    SYSTEM_LIB_PATH.with(|slp| match slp.as_ref() {
        Some(Some(compiler_libs_path)) => Some(compiler_libs_path.clone()),
        Some(None) => None,
        None => {
            let path = probe(options);
            unsafe {
                slp.set(Some(path.clone()));
            }
            path
        }
    })
}

fn get_object_file_path(options: &Options, name: &str, self_contained: bool) -> PathBuf {
    // prefer system {,dll}crt2.o libs, see get_crt_libs_path comment for more details
    if options.debugging_opts.link_self_contained.is_none()
        && options.target.llvm_target.contains("windows-gnu")
    {
        if let Some(compiler_libs_path) = get_crt_libs_path(options) {
            let file_path = compiler_libs_path.join(name);
            if file_path.exists() {
                return file_path;
            }
        }
    }
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

pub fn exec_linker(
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

    #[cfg(unix)]
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
pub fn add_local_native_libraries(
    cmd: &mut dyn Linker,
    options: &Options,
    codegen_results: &CodegenResults,
    tmpdir: &Path,
) -> anyhow::Result<()> {
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

    let search_path = archive_search_paths(options);

    // Add runtime libs we depend on
    let no_std = options.codegen_opts.no_std.unwrap_or(false);
    let libstd_libs = match options.target.arch.as_str() {
        "x86_64" if !no_std => vec![
            "libpanic_unwind.rlib",
            "lumen_rt_minimal",
            "libliblumen_otp.rlib",
        ],
        "wasm32" if !no_std => vec!["libpanic_abort.rlib", "lumen_web"],
        _ => vec!["libpanic_unwind.rlib"],
    };
    let rlib_dir = filesearch.get_lib_path();
    for lib in libstd_libs {
        if lib.ends_with(".rlib") {
            link_rlib(cmd, options, tmpdir, &rlib_dir.join(lib));
        } else {
            let search_path = archive_search_paths(options);
            cmd.link_whole_staticlib(lib, &search_path);
        }
    }

    for lib in codegen_results.project_info.native_libraries.iter() {
        let name = match lib.name {
            Some(ref l) => l,
            None => continue,
        };
        match lib.kind {
            NativeLibraryKind::NativeUnknown => cmd.link_dylib(name.as_str()),
            NativeLibraryKind::NativeFramework => cmd.link_framework(name.as_str()),
            NativeLibraryKind::NativeStaticNobundle => cmd.link_staticlib(name.as_str()),
            NativeLibraryKind::NativeStatic => {
                cmd.link_whole_staticlib(name.as_str(), &search_path)
            }
            NativeLibraryKind::NativeRawDylib => {
                // FIXME(#58713): Proper handling for raw dylibs.
                return Err(anyhow!("raw_dylib feature not yet implemented"));
            }
        }
    }

    let relevant_libs = &codegen_results.project_info.used_libraries;

    for lib in relevant_libs.iter() {
        let name = match lib.name {
            Some(ref l) => l,
            None => continue,
        };
        match lib.kind {
            NativeLibraryKind::NativeUnknown => cmd.link_dylib(name.as_str()),
            NativeLibraryKind::NativeFramework => cmd.link_framework(name.as_str()),
            NativeLibraryKind::NativeStaticNobundle => cmd.link_staticlib(name.as_str()),
            NativeLibraryKind::NativeStatic => {
                cmd.link_whole_staticlib(name.as_str(), &search_path)
            }
            NativeLibraryKind::NativeRawDylib => {
                // FIXME(#58713): Proper handling for raw dylibs.
                return Err(anyhow!("raw_dylib feature not yet implemented"));
            }
        }
    }

    Ok(())
}

fn create_rlib<'a>(
    options: &'a Options,
    codegen_results: &CodegenResults,
    flavor: RlibFlavor,
    output_file: &Path,
    _tmpdir: &Path,
) -> LlvmArchiveBuilder<'a> {
    info!("preparing rlib to {}", output_file.display());
    let mut ab = LlvmArchiveBuilder::new(options, output_file, None);

    for obj in codegen_results.modules.iter().filter_map(|m| m.object()) {
        ab.add_file(obj);
    }

    // Note that in this loop we are ignoring the value of `lib.cfg`. That is,
    // we may not be configured to actually include a static library if we're
    // adding it here. That's because later when we consume this rlib we'll
    // decide whether we actually needed the static library or not.
    //
    // To do this "correctly" we'd need to keep track of which libraries added
    // which object files to the archive. We don't do that here, however. The
    // #[link(cfg(..))] feature is unstable, though, and only intended to get
    // liblibc working. In that sense the check below just indicates that if
    // there are any libraries we want to omit object files for at link time we
    // just exclude all custom object files.
    //
    // Eventually if we want to stabilize or flesh out the #[link(cfg(..))]
    // feature then we'll need to figure out how to record what objects were
    // loaded from the libraries found here and then encode that into the
    // metadata of the rlib we're generating somehow.
    for lib in codegen_results.project_info.used_libraries.iter() {
        match lib.kind {
            NativeLibraryKind::NativeStatic => {}
            NativeLibraryKind::NativeStaticNobundle
            | NativeLibraryKind::NativeFramework
            | NativeLibraryKind::NativeRawDylib
            | NativeLibraryKind::NativeUnknown => continue,
        }
        if let Some(ref name) = lib.name {
            ab.add_native_library(name.as_str());
        }
    }

    // After adding all files to the archive, we need to update the
    // symbol table of the archive.
    ab.update_symbols();

    // Note that it is important that we add all of our non-object "magical
    // files" *after* all of the object files in the archive. The reason for
    // this is as follows:
    //
    // * When performing LTO, this archive will be modified to remove objects from above. The reason
    //   for this is described below.
    //
    // * When the system linker looks at an archive, it will attempt to determine the architecture
    //   of the archive in order to see whether its linkable.
    //
    //   The algorithm for this detection is: iterate over the files in the
    //   archive. Skip magical SYMDEF names. Interpret the first file as an
    //   object file. Read architecture from the object file.
    //
    // * As one can probably see, if "metadata" and "foo.bc" were placed before all of the objects,
    //   then the architecture of this archive would not be correctly inferred once 'foo.o' is
    //   removed.
    //
    // Basically, all this means is that this code should not move above the
    // code above.
    match flavor {
        RlibFlavor::Normal => {
            // In the future we may emit metadata like Rust does
            // ab.add_file(&emit_metadata(options, &codegen_results.metadata, tmpdir));

            // For LTO purposes, the bytecode of this library is also inserted
            // into the archive.
            for bytecode in codegen_results
                .modules
                .iter()
                .filter_map(|m| m.bytecode_compressed())
            {
                ab.add_file(bytecode);
            }

            // After adding all files to the archive, we need to update the
            // symbol table of the archive. This currently dies on macOS (see
            // #11162), and isn't necessary there anyway
            if !options.target.options.is_like_osx {
                ab.update_symbols();
            }
        }

        RlibFlavor::StaticlibBase => { /* nothing to do here for now */ }
    }

    ab
}

fn link_rlib(cmd: &mut dyn Linker, options: &Options, tmpdir: &Path, rlib_path: &Path) {
    use super::archive::builder::{METADATA_FILENAME, RLIB_BYTECODE_EXTENSION};

    debug_assert!(
        rlib_path.exists(),
        "rlib path not found {}",
        rlib_path.display()
    );
    let dst = tmpdir.join(rlib_path.file_name().unwrap());
    let mut archive = LlvmArchiveBuilder::new(options, &dst, Some(rlib_path));
    archive.update_symbols();

    for f in archive.src_files() {
        if f.ends_with(RLIB_BYTECODE_EXTENSION) || f == METADATA_FILENAME {
            archive.remove_file(&f);
        }
    }

    archive.build();

    cmd.link_whole_rlib(&dst);
}

fn link_output_kind(options: &Options, project_type: ProjectType) -> LinkOutputKind {
    let kind = match (
        project_type,
        options.crt_static(Some(project_type)),
        options.relocation_model(),
    ) {
        (ProjectType::Executable, false, RelocModel::PIC) => LinkOutputKind::DynamicPicExe,
        (ProjectType::Executable, false, _) => LinkOutputKind::DynamicNoPicExe,
        (ProjectType::Executable, true, RelocModel::PIC) => LinkOutputKind::StaticPicExe,
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

/// Whether we link to our own CRT objects instead of relying on gcc to pull them.
/// We only provide such support for a very limited number of targets.
fn crt_objects_fallback(options: &Options, project_type: ProjectType) -> bool {
    if let Some(self_contained) = options.debugging_opts.link_self_contained {
        return self_contained;
    }

    match options.target.options.crt_objects_fallback {
        // FIXME: Find a better heuristic for "native musl toolchain is available",
        // based on host and linker path, for example.
        // (https://github.com/rust-lang/rust/pull/71769#issuecomment-626330237).
        Some(CrtObjectsFallback::Musl) => options.crt_static(Some(project_type)),
        // FIXME: Find some heuristic for "native mingw toolchain is available",
        // likely based on `get_crt_libs_path` (https://github.com/rust-lang/rust/pull/67429).
        Some(CrtObjectsFallback::Mingw) => options.target.target_vendor != "uwp",
        // FIXME: Figure out cases in which WASM needs to link with a native toolchain.
        Some(CrtObjectsFallback::Wasm) => true,
        None => false,
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
        &opts.pre_link_objects_fallback
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
        &opts.post_link_objects_fallback
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
        cmd.args(args);
    }
    if let Some(args) = options.debugging_opts.pre_link_args.as_ref() {
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
    match (project_type, &options.target.options.link_script) {
        (ProjectType::Cdylib | ProjectType::Executable, Some(script)) => {
            if !options.target.options.linker_is_gnu {
                diagnostics.fatal("can only use link script when linking with GNU-like linker").raise();
            }

            let file_name = ["lumen", &options.target.llvm_target, "linkfile.ld"].join("-");

            let path = tmpdir.join(file_name);
            if let Err(e) = fs::write(&path, script) {
                diagnostics.fatal(&format!(
                    "failed to write link script to {}: {}",
                    path.display(),
                    e
                )).raise();
            }

            cmd.arg("--script");
            cmd.arg(path);
        }
        _ => {}
    }
}

/// Add arbitrary "user defined" args defined from command line and by `#[link_args]` attributes.
/// FIXME: Determine where exactly these args need to be inserted.
fn add_user_defined_link_args(
    cmd: &mut dyn Linker,
    options: &Options,
    codegen_results: &CodegenResults,
) {
    if let Some(args) = options.codegen_opts.linker_args.as_ref() {
        for arg in args {
            cmd.arg(arg);
        }
    }
    cmd.args(&*codegen_results.project_info.link_args);
}

/// Add arbitrary "late link" args defined by the target spec.
/// FIXME: Determine where exactly these args need to be inserted.
fn add_late_link_args(
    cmd: &mut dyn Linker,
    options: &Options,
    flavor: LinkerFlavor,
    project_type: ProjectType,
    codegen_results: &CodegenResults,
) {
    if let Some(args) = options.target.options.late_link_args.get(&flavor) {
        cmd.args(args);
    }
    let any_dynamic_crate = project_type == ProjectType::Dylib;
    if any_dynamic_crate {
        if let Some(args) = options.target.options.late_link_args_dynamic.get(&flavor) {
            cmd.args(args);
        }
    } else {
        if let Some(args) = options.target.options.late_link_args_static.get(&flavor) {
            cmd.args(args);
        }
    }
}

/// Add arbitrary "post-link" args defined by the target spec.
/// FIXME: Determine where exactly these args need to be inserted.
fn add_post_link_args(cmd: &mut dyn Linker, options: &Options, flavor: LinkerFlavor) {
    if let Some(args) = options.target.options.post_link_args.get(&flavor) {
        cmd.args(args);
    }
}

/// Add object files containing code from the current crate.
fn add_local_crate_regular_objects(cmd: &mut dyn Linker, codegen_results: &CodegenResults) {
    for obj in codegen_results.modules.iter().filter_map(|m| m.object()) {
        cmd.add_object(obj);
    }
}

/// Link native libraries corresponding to the current crate and all libraries corresponding to
/// all its dependency crates.
/// FIXME: Consider combining this with the functions above adding object files for the local crate.
fn link_local_crate_native_libs_and_dependent_crate_libs<'a>(
    cmd: &mut dyn Linker,
    options: &'a Options,
    project_type: ProjectType,
    codegen_results: &CodegenResults,
    tmpdir: &Path,
) {
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
    // 3. Upstream rust libraries
    // 4. Upstream native libraries
    //
    // The rationale behind this ordering is that those items lower down in the
    // list can't depend on items higher up in the list. For example nothing can
    // depend on what we just generated (e.g., that'd be a circular dependency).
    // Upstream rust libraries are not allowed to depend on our local native
    // libraries as that would violate the structure of the DAG, in that
    // scenario they are required to link to them as well in a shared fashion.
    //
    // Note that upstream rust libraries may contain native dependencies as
    // well, but they also can't depend on what we just started to add to the
    // link line. And finally upstream native libraries can't depend on anything
    // in this DAG so far because they're only dylibs and dylibs can only depend
    // on other dylibs (e.g., other native deps).
    //
    // If -Zlink-native-libraries=false is set, then the assumption is that an
    // external build system already has the native dependencies defined, and it
    // will provide them to the linker itself.
    if options.debugging_opts.link_native_libraries {
        add_local_native_libraries(cmd, options, codegen_results, tmpdir).unwrap();
    }
    //if options.debugging_opts.link_native_libraries {
    //    add_upstream_native_libraries(cmd, options, codegen_results, project_type);
    //}
}

/// Add sysroot and other globally set directories to the directory search list.
fn add_library_search_dirs(cmd: &mut dyn Linker, options: &Options, self_contained: bool) {
    // Prefer system mingw-w64 libs, see get_crt_libs_path comment for more details.
    if options.debugging_opts.link_self_contained.is_none()
        && cfg!(windows)
        && options.target.llvm_target.contains("windows-gnu")
    {
        if let Some(compiler_libs_path) = get_crt_libs_path(options) {
            cmd.include_path(&compiler_libs_path);
        }
    }

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
        .debugging_opts
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
    codegen_results: &CodegenResults,
    out_filename: &Path,
) {
    // FIXME (#2397): At some point we want to rpath our guesses as to
    // where extern libraries might live, based on the
    // addl_lib_search_paths
    if options.codegen_opts.rpath {
        let target_triple = options.target.triple();
        let mut get_install_prefix_lib_path = || {
            let install_prefix = option_env!("CFG_PREFIX").expect("CFG_PREFIX");
            let tlib = filesearch::relative_target_lib_path(&options.sysroot, target_triple);
            let mut path = PathBuf::from(install_prefix);
            path.push(&tlib);

            path
        };
        let mut rpath_config = RPathConfig {
            used_libs: codegen_results.project_info.used_deps_dynamic.as_slice(),
            output_file: out_filename.to_path_buf(),
            has_rpath: options.target.options.has_rpath,
            is_like_osx: options.target.options.is_like_osx,
            linker_is_gnu: options.target.options.linker_is_gnu,
            get_install_prefix_lib_path: &mut get_install_prefix_lib_path,
        };
        cmd.args(&rpath::get_rpath_flags(&mut rpath_config));
    }
}

/// Produce the linker command line containing linker path and arguments.
/// `NO-OPT-OUT` marks the arguments that cannot be removed from the command line
/// by the user without creating a custom target specification.
/// `OBJECT-FILES` specify whether the arguments can add object files.
/// `CUSTOMIZATION-POINT` means that arbitrary arguments defined by the user
/// or by the target spec can be inserted here.
/// `AUDIT-ORDER` - need to figure out whether the option is order-dependent or not.
fn linker_with_args(
    path: &Path,
    flavor: LinkerFlavor,
    options: &Options,
    diagnostics: &DiagnosticsHandler,
    project_type: ProjectType,
    tmpdir: &Path,
    out_filename: &Path,
    codegen_results: &CodegenResults,
) -> Command {
    let crt_objects_fallback = crt_objects_fallback(options, project_type);
    let base_cmd = get_linker(options, path, flavor, crt_objects_fallback);
    // FIXME: Move `/LIBPATH` addition for uwp targets from the linker construction
    // to the linker args construction.
    assert!(base_cmd.get_args().is_empty() || options.target.target_vendor == "uwp");
    let cmd = &mut *codegen_results
        .linker_info
        .to_linker(base_cmd, options, diagnostics, flavor);
    let link_output_kind = link_output_kind(options, project_type);

    // NO-OPT-OUT, OBJECT-FILES-MAYBE, CUSTOMIZATION-POINT
    add_pre_link_args(cmd, options, flavor);

    // NO-OPT-OUT
    add_link_script(cmd, options, diagnostics, tmpdir, project_type);

    // NO-OPT-OUT, OBJECT-FILES-NO, AUDIT-ORDER
    if options.target.options.eh_frame_header {
        cmd.add_eh_frame_header();
    }

    // NO-OPT-OUT, OBJECT-FILES-NO
    if crt_objects_fallback {
        cmd.no_crt_objects();
    }

    // NO-OPT-OUT, OBJECT-FILES-YES
    add_pre_link_objects(cmd, options, link_output_kind, crt_objects_fallback);

    // NO-OPT-OUT, OBJECT-FILES-NO, AUDIT-ORDER
    if options.target.options.is_like_emscripten {
        cmd.arg("-s");
        cmd.arg(if options.panic_strategy() == PanicStrategy::Abort {
            "DISABLE_EXCEPTION_CATCHING=1"
        } else {
            "DISABLE_EXCEPTION_CATCHING=0"
        });
    }

    // OBJECT-FILES-YES, AUDIT-ORDER
    // link_sanitizers(options, project_type, cmd);

    // OBJECT-FILES-NO, AUDIT-ORDER
    // Linker plugins should be specified early in the list of arguments
    // FIXME: How "early" exactly?
    cmd.linker_plugin_lto();

    // NO-OPT-OUT, OBJECT-FILES-NO, AUDIT-ORDER
    // FIXME: Order-dependent, at least relatively to other args adding searh directories.
    add_library_search_dirs(cmd, options, crt_objects_fallback);

    // OBJECT-FILES-YES
    add_local_crate_regular_objects(cmd, codegen_results);

    // NO-OPT-OUT, OBJECT-FILES-NO, AUDIT-ORDER
    cmd.output_filename(out_filename);

    // OBJECT-FILES-NO, AUDIT-ORDER
    if project_type == ProjectType::Executable && options.target.options.is_like_windows {
        if let Some(ref s) = codegen_results.windows_subsystem {
            cmd.subsystem(s);
        }
    }

    // NO-OPT-OUT, OBJECT-FILES-NO, AUDIT-ORDER
    // If we're building something like a dynamic library then some platforms
    // need to make sure that all symbols are exported correctly from the
    // dynamic library.
    cmd.export_symbols(tmpdir, project_type);

    // OBJECT-FILES-NO, AUDIT-ORDER
    // FIXME: Order dependent, applies to the following objects. Where should it be placed?
    // Try to strip as much out of the generated object by removing unused
    // sections if possible. See more comments in linker.rs
    if options.codegen_opts.link_dead_code != Some(true) {
        let keep_metadata = project_type == ProjectType::Dylib;
        cmd.gc_sections(keep_metadata);
    }

    // NO-OPT-OUT, OBJECT-FILES-NO, AUDIT-ORDER
    cmd.set_output_kind(link_output_kind, out_filename);

    // OBJECT-FILES-NO, AUDIT-ORDER
    add_relro_args(cmd, options);

    // OBJECT-FILES-NO, AUDIT-ORDER
    // Pass optimization flags down to the linker.
    cmd.optimize();

    // OBJECT-FILES-NO, AUDIT-ORDER
    // Pass debuginfo and strip flags down to the linker.
    cmd.debuginfo(options.debugging_opts.strip);

    // OBJECT-FILES-NO, AUDIT-ORDER
    // We want to prevent the compiler from accidentally leaking in any system libraries,
    // so by default we tell linkers not to link to any default libraries.
    if !options.codegen_opts.default_linker_libraries && options.target.options.no_default_libraries
    {
        cmd.no_default_libraries();
    }

    // OBJECT-FILES-YES
    link_local_crate_native_libs_and_dependent_crate_libs(
        cmd,
        options,
        project_type,
        codegen_results,
        tmpdir,
    );

    // OBJECT-FILES-NO, AUDIT-ORDER
    if options.codegen_opts.control_flow_guard != CFGuard::Disabled {
        cmd.control_flow_guard();
    }

    // OBJECT-FILES-NO, AUDIT-ORDER
    add_rpath_args(cmd, options, codegen_results, out_filename);

    // OBJECT-FILES-MAYBE, CUSTOMIZATION-POINT
    add_user_defined_link_args(cmd, options, codegen_results);

    // NO-OPT-OUT, OBJECT-FILES-NO, AUDIT-ORDER
    cmd.finalize();

    // NO-OPT-OUT, OBJECT-FILES-MAYBE, CUSTOMIZATION-POINT
    add_late_link_args(cmd, options, flavor, project_type, codegen_results);

    // NO-OPT-OUT, OBJECT-FILES-YES
    add_post_link_objects(cmd, options, link_output_kind, crt_objects_fallback);

    // NO-OPT-OUT, OBJECT-FILES-MAYBE, CUSTOMIZATION-POINT
    add_post_link_args(cmd, options, flavor);

    cmd.take_cmd()
}

/// Checks if target supports project_type as output
pub fn invalid_output_for_target(options: &Options) -> bool {
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

pub fn remove(path: &Path) -> anyhow::Result<()> {
    if let Err(err) = fs::remove_file(path) {
        return Err(anyhow!("failed to remove {}: {}", path.display(), err));
    }
    Ok(())
}

// Make sure files are writeable.  Mac, FreeBSD, and Windows system linkers
// check this already -- however, the Linux linker will happily overwrite a
// read-only file.  We should be consistent.
pub fn check_file_is_writeable(file: &Path) -> anyhow::Result<()> {
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
