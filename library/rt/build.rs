extern crate inflector;
extern crate toml;

use std::cmp::Ordering;
use std::collections::BTreeSet;
use std::env;
use std::fs::{self, File};
use std::io::prelude::*;
use std::path::PathBuf;

use inflector::Inflector;
use toml::Value;

#[derive(Debug, Default, Clone)]
struct Symbol {
    key: String,
    value: String,
}
impl Eq for Symbol {}
impl PartialEq for Symbol {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}
impl PartialOrd for Symbol {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Symbol {
    fn cmp(&self, other: &Self) -> Ordering {
        self.value.cmp(&other.value)
    }
}
impl Symbol {
    fn from_value<S: Into<String>>(name: S, value: &Value) -> Self {
        let name = name.into();
        let table = value.as_table().unwrap();
        let value = match table
            .get("value")
            .map(|v| v.as_str().expect("value must be a string"))
        {
            None => name.clone(),
            Some(value) => value.to_string(),
        };
        // When the name is, e.g. MODULE_INFO, keep it that way rather than transforming it
        // as the casing is intentional
        let key = if name.is_screaming_snake_case() {
            name
        } else {
            name.to_pascal_case()
        };
        Self { key, value }
    }
}

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=src/term/atom/atoms.toml");

    let contents = fs::read_to_string("src/term/atom/atoms.toml").unwrap();
    let root = contents.parse::<Value>().unwrap();
    let root = root.as_table().unwrap();
    let mut seen: BTreeSet<Symbol> = BTreeSet::new();
    let mut symbols = vec![];
    for (_, section_value) in root.iter() {
        let table = section_value.as_table().unwrap();
        for (name, value) in table.iter() {
            let sym = Symbol::from_value(name, value);
            assert!(seen.insert(sym.clone()), "duplicate symbol {}", name);
            symbols.push(sym);
        }
    }

    generate_symbols_rs(symbols).unwrap();
}

fn generate_symbols_rs(symbols: Vec<Symbol>) -> std::io::Result<()> {
    let out = PathBuf::from(env::var("OUT_DIR").unwrap()).join("atoms.rs");
    let mut file = File::create(&out)?;
    file.write_all(b"use core::ptr::NonNull;\n")?;
    file.write_all(b"use super::{Atom, AtomData};\n")?;

    // All other symbol data declarations
    for symbol in symbols.iter() {
        write!(
            &mut file,
            r#"
pub const {0}_VALUE: &'static [u8] = b"{1}";

#[cfg_attr(target_os = "macos", link_section = "__DATA,__atoms")]
#[cfg_attr(all(linux, not(target_os = "macos")), link_section = "__atoms")]
#[export_name = "atom_{1}"]
#[linkage = "linkonce_odr"]
pub static {0}_ATOM: AtomData = AtomData {{
    size: {0}_VALUE.len(),
    ptr: {0}_VALUE.as_ptr(),
}};

"#,
            &symbol.key, &symbol.value,
        )?;
    }

    // Symbol shorthand declarations
    file.write_all(b"\n\n")?;
    for symbol in symbols.iter() {
        writeln!(
            &mut file,
            r#"
pub static {0}: Atom = Atom(unsafe {{ NonNull::new_unchecked(&{0}_ATOM as *const AtomData as *mut AtomData) }});"#,
            &symbol.key,
        )?;
    }

    file.sync_data()?;

    Ok(())
}
