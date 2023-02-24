mod em;
mod gcc;
mod msvc;
mod wasm;

pub use self::em::EmLinker;
pub use self::gcc::GccLinker;
pub use self::msvc::MsvcLinker;
pub use self::wasm::WasmLinker;

use std::borrow::Borrow;
use std::env;
use std::ffi::OsString;
use std::path::Path;

use firefly_session::Options;
use firefly_target::{LinkerFlavor, LldFlavor};
use firefly_util::diagnostics::DiagnosticsHandler;

use log::warn;

use crate::{Command, Linker};

// The third parameter is for env vars, used on windows to set up the
// path for MSVC to find its DLLs, and gcc to find its bundled
// toolchain
pub fn get<'a>(
    options: &'a Options,
    diagnostics: &'a DiagnosticsHandler,
    linker: &Path,
    flavor: LinkerFlavor,
    self_contained: bool,
    target_cpu: &'a str,
) -> Box<dyn Linker + 'a> {
    let msvc_tool = cc::windows_registry::find_tool(&options.target.triple(), "link.exe");

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
                Command::new(msvc_tool.as_ref().map_or(linker, |t| t.path()))
            }
            _ => Command::new(linker),
        },
    };

    // UWP apps have API restrictions enforced during Store submissions.
    // To comply with the Windows App Certification Kit,
    // MSVC needs to link with the Store versions of the runtime libraries (vcruntime, msvcrt, etc).
    let t = &options.target;
    if (flavor == LinkerFlavor::Msvc || flavor == LinkerFlavor::Lld(LldFlavor::Link))
        && t.options.vendor == "uwp"
    {
        if let Some(ref tool) = msvc_tool {
            let original_path = tool.path();
            if let Some(ref root_lib_path) = original_path.ancestors().skip(4).next() {
                let arch = match t.arch.borrow() {
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
    let mut new_path = options.get_tools_search_paths(self_contained);
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

    assert!(cmd.get_args().is_empty() || options.target.options.vendor == "uwp");
    match flavor {
        LinkerFlavor::Gcc => Box::new(GccLinker {
            cmd,
            options,
            diagnostics,
            target_cpu,
            hinted_static: false,
            is_ld: false,
        }) as Box<dyn Linker>,

        LinkerFlavor::Lld(LldFlavor::Ld)
        | LinkerFlavor::Lld(LldFlavor::Ld64)
        | LinkerFlavor::Ld => Box::new(GccLinker {
            cmd,
            options,
            diagnostics,
            target_cpu,
            hinted_static: false,
            is_ld: true,
        }) as Box<dyn Linker>,

        LinkerFlavor::Lld(LldFlavor::Link) | LinkerFlavor::Msvc => Box::new(MsvcLinker {
            cmd,
            options,
            diagnostics,
        }) as Box<dyn Linker>,

        LinkerFlavor::Lld(LldFlavor::Wasm) => {
            Box::new(WasmLinker::new(cmd, options)) as Box<dyn Linker>
        }

        LinkerFlavor::EmCc => Box::new(EmLinker { cmd, options }) as Box<dyn Linker>,

        other => panic!("unsupported linker flavor: {}", other),
    }
}
