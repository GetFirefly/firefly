use std::env;
use std::path::Path;

fn main() {
    let cwd = env::current_dir().expect("unable to access current directory");

    let root_dir = cwd.parent().unwrap();
    let liblumen_llvm_dir = root_dir.join("compiler/llvm");
    let liblumen_term_dir = root_dir.join("compiler/term");
    let lumen_rt_minimal_dir = root_dir.join("runtimes").join("minimal");

    rerun_if_changed_anything_in_dir(&liblumen_llvm_dir);
    rerun_if_changed_anything_in_dir(&liblumen_term_dir);
    rerun_if_changed_anything_in_dir(&lumen_rt_minimal_dir);
}

pub fn rerun_if_changed_anything_in_dir(dir: &Path) {
    let mut stack = dir
        .read_dir()
        .unwrap()
        .map(|e| e.unwrap())
        .filter(|e| !ignore_changes(Path::new(&*e.file_name())))
        .collect::<Vec<_>>();
    while let Some(entry) = stack.pop() {
        let path = entry.path();
        if entry.file_type().unwrap().is_dir() {
            stack.extend(path.read_dir().unwrap().map(|e| e.unwrap()));
        } else {
            println!("cargo:rerun-if-changed={}", path.display());
        }
    }
}

fn ignore_changes(name: &Path) -> bool {
    return name
        .file_name()
        .map(|f| {
            let name = f.to_string_lossy();
            if name.starts_with(".") {
                return true;
            }
            if name.ends_with(".rs") || name == "Cargo.toml" {
                return false;
            }
            true
        })
        .unwrap_or(true);
}

#[allow(unused)]
fn warn(s: &str) {
    println!("cargo:warning={}", s);
}

#[allow(unused)]
fn fail(s: &str) -> ! {
    panic!("\n{}\n\nbuild script failed, must exit now", s)
}
