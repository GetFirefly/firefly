//! A thin wrapper around `Command` in the standard library which allows us to
//! read the arguments that are built up.

use std::ffi::{OsStr, OsString};
use std::fmt;
use std::io;
use std::mem;
use std::process::{self, Output};

use liblumen_target::LldFlavor;

#[derive(Clone)]
pub struct Command {
    program: Program,
    args: Vec<OsString>,
    env: Vec<(OsString, OsString)>,
    env_remove: Vec<OsString>,
}

#[derive(Clone)]
enum Program {
    Normal(OsString),
    CmdBatScript(OsString),
    Lld(OsString, LldFlavor),
}

impl Command {
    pub fn new<P: AsRef<OsStr>>(program: P) -> Command {
        Command::_new(Program::Normal(program.as_ref().to_owned()))
    }

    pub fn bat_script<P: AsRef<OsStr>>(program: P) -> Command {
        Command::_new(Program::CmdBatScript(program.as_ref().to_owned()))
    }

    pub fn lld<P: AsRef<OsStr>>(program: P, flavor: LldFlavor) -> Command {
        Command::_new(Program::Lld(program.as_ref().to_owned(), flavor))
    }

    fn _new(program: Program) -> Command {
        Command {
            program,
            args: Vec::new(),
            env: Vec::new(),
            env_remove: Vec::new(),
        }
    }

    pub fn arg<P: AsRef<OsStr>>(&mut self, arg: P) -> &mut Command {
        self._arg(arg.as_ref());
        self
    }

    pub fn args<I>(&mut self, args: I) -> &mut Command
    where
        I: IntoIterator<Item: AsRef<OsStr>>,
    {
        for arg in args {
            self._arg(arg.as_ref());
        }
        self
    }

    fn _arg(&mut self, arg: &OsStr) {
        self.args.push(arg.to_owned());
    }

    pub fn env<K, V>(&mut self, key: K, value: V) -> &mut Command
    where
        K: AsRef<OsStr>,
        V: AsRef<OsStr>,
    {
        self._env(key.as_ref(), value.as_ref());
        self
    }

    fn _env(&mut self, key: &OsStr, value: &OsStr) {
        self.env.push((key.to_owned(), value.to_owned()));
    }

    pub fn env_remove<K>(&mut self, key: K) -> &mut Command
    where
        K: AsRef<OsStr>,
    {
        self._env_remove(key.as_ref());
        self
    }

    fn _env_remove(&mut self, key: &OsStr) {
        self.env_remove.push(key.to_owned());
    }

    pub fn output(&mut self) -> io::Result<Output> {
        self.command().output()
    }

    pub fn command(&self) -> process::Command {
        let mut ret = match self.program {
            Program::Normal(ref p) => process::Command::new(p),
            Program::CmdBatScript(ref p) => {
                let mut c = process::Command::new("cmd");
                c.arg("/c").arg(p);
                c
            }
            Program::Lld(ref p, _flavor) => process::Command::new(p),
        };
        ret.args(&self.args);
        ret.envs(self.env.clone());
        for k in &self.env_remove {
            ret.env_remove(k);
        }
        return ret;
    }

    #[allow(unused)]
    pub fn command_with_flavor(&self) -> process::Command {
        let mut ret = match self.program {
            Program::Normal(ref p) => process::Command::new(p),
            Program::CmdBatScript(ref p) => {
                let mut c = process::Command::new("cmd");
                c.arg("/c").arg(p);
                c
            }
            Program::Lld(ref p, flavor) => {
                let mut c = process::Command::new(p);
                c.arg("-flavor").arg(match flavor {
                    LldFlavor::Wasm => "wasm",
                    LldFlavor::Ld => "gnu",
                    LldFlavor::Link => "link",
                    LldFlavor::Ld64 => "darwin",
                });
                c
            }
        };
        ret.args(&self.args);
        ret.envs(self.env.clone());
        for k in &self.env_remove {
            ret.env_remove(k);
        }
        return ret;
    }

    // extensions
    pub fn as_vec(&self) -> Vec<OsString> {
        let mut vec = Vec::new();
        match self.program {
            Program::Normal(ref p) => vec.push(OsString::from(p.as_os_str())),
            Program::CmdBatScript(ref p) => {
                vec.push(OsString::from("cmd"));
                vec.push(OsString::from("/c"));
                vec.push(OsString::from(p.as_os_str()));
            }
            Program::Lld(ref p, flavor) => {
                vec.push(OsString::from(p));
                vec.push(OsString::from("-flavor"));
                vec.push(OsString::from(match flavor {
                    LldFlavor::Wasm => "wasm",
                    LldFlavor::Ld => "gnu",
                    LldFlavor::Link => "link",
                    LldFlavor::Ld64 => "darwin",
                }));
            }
        };
        for arg in &self.args {
            vec.push(arg.clone());
        }
        vec
    }

    pub fn get_args(&self) -> &[OsString] {
        &self.args
    }

    pub fn take_args(&mut self) -> Vec<OsString> {
        mem::take(&mut self.args)
    }

    /// Returns a `true` if we're pretty sure that this'll blow OS spawn limits,
    /// or `false` if we should attempt to spawn and see what the OS says.
    pub fn very_likely_to_exceed_some_spawn_limit(&self) -> bool {
        // We mostly only care about Windows in this method, on Unix the limits
        // can be gargantuan anyway so we're pretty unlikely to hit them
        if cfg!(unix) {
            return false;
        }

        // Right now LLD doesn't support the `@` syntax of passing an argument
        // through files, so regardless of the platform we try to go to the OS
        // on this one.
        if let Program::Lld(..) = self.program {
            return false;
        }

        // Ok so on Windows to spawn a process is 32,768 characters in its
        // command line [1]. Unfortunately we don't actually have access to that
        // as it's calculated just before spawning. Instead we perform a
        // poor-man's guess as to how long our command line will be. We're
        // assuming here that we don't have to escape every character...
        //
        // Turns out though that `cmd.exe` has even smaller limits, 8192
        // characters [2]. Linkers can often be batch scripts (for example
        // Emscripten, Gecko's current build system) which means that we're
        // running through batch scripts. These linkers often just forward
        // arguments elsewhere (and maybe tack on more), so if we blow 8192
        // bytes we'll typically cause them to blow as well.
        //
        // Basically as a result just perform an inflated estimate of what our
        // command line will look like and test if it's > 8192 (we actually
        // test against 6k to artificially inflate our estimate). If all else
        // fails we'll fall back to the normal unix logic of testing the OS
        // error code if we fail to spawn and automatically re-spawning the
        // linker with smaller arguments.
        //
        // [1]: https://msdn.microsoft.com/en-us/library/windows/desktop/ms682425(v=vs.85).aspx
        // [2]: https://blogs.msdn.microsoft.com/oldnewthing/20031210-00/?p=41553

        let estimated_command_line_len = self.args.iter().map(|a| a.len()).sum::<usize>();
        estimated_command_line_len > 1024 * 6
    }
}

impl fmt::Debug for Command {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.command().fmt(f)
    }
}
