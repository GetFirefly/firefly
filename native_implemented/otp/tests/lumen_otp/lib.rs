//! https://github.com/lumen/otp/tree/lumen/lib

#[path = "lib/asn1.rs"]
mod asn1;
#[path = "lib/common_test.rs"]
mod common_test;
#[path = "lib/compiler.rs"]
mod compiler;
#[path = "lib/crypto.rs"]
mod crypto;
#[path = "lib/debugger.rs"]
mod debugger;
#[path = "lib/dialyzer.rs"]
mod dialyzer;
#[path = "lib/diameter.rs"]
mod diameter;
#[path = "lib/edoc.rs"]
mod edoc;
#[path = "lib/eldap.rs"]
mod eldap;
#[path = "lib/erl_docgen.rs"]
mod erl_docgen;
#[path = "lib/et.rs"]
mod et;
#[path = "lib/eunit.rs"]
mod eunit;
#[path = "lib/ftp.rs"]
mod ftp;
#[path = "lib/inets.rs"]
mod inets;
#[path = "lib/kernel.rs"]
mod kernel;
#[path = "lib/megaco.rs"]
mod megaco;
#[path = "lib/mnesia.rs"]
mod mnesia;
#[path = "lib/observer.rs"]
mod observer;
#[path = "lib/odbc.rs"]
mod odbc;
#[path = "lib/os_mon.rs"]
mod os_mon;
#[path = "lib/parsetools.rs"]
mod parsetools;
#[path = "lib/public_key.rs"]
mod public_key;
#[path = "lib/reltool.rs"]
mod reltool;
#[path = "lib/runtime_tools.rs"]
mod runtime_tools;
#[path = "lib/sasl.rs"]
mod sasl;
#[path = "lib/snmp.rs"]
mod snmp;
#[path = "lib/ssh.rs"]
mod ssh;
#[path = "lib/ssl.rs"]
mod ssl;
#[path = "lib/stdlib.rs"]
mod stdlib;
#[path = "lib/syntax_tools.rs"]
mod syntax_tools;
#[path = "lib/tftp.rs"]
mod tftp;
#[path = "lib/tools.rs"]
mod tools;
#[path = "lib/wx.rs"]
mod wx;
#[path = "lib/xmerl.rs"]
mod xmerl;

use super::*;

fn includes() -> Vec<&'static str> {
    vec!["lib"]
}

fn relative_directory_path() -> PathBuf {
    "lib".into()
}
