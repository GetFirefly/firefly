use std::alloc::Layout;
use std::borrow::Borrow;
use std::env::ArgsOs;
use std::mem;
use std::path::Path;
use std::ptr;
use std::sync::OnceLock;

use anyhow::anyhow;

use liblumen_arena::DroplessArena;
use liblumen_binary::{BinaryFlags, Encoding};
use liblumen_rt::term::BinaryData;

static ARGV: OnceLock<EnvTable> = OnceLock::new();

/// Returns all arguments this executable was invoked with
pub fn argv() -> &'static [&'static BinaryData] {
    ARGV.get().unwrap().argv.as_slice()
}

/// Performs one-time initialization of the environment for the current executable.
/// This is used to cache the arguments vector as constant binary values.
pub fn init(mut argv: ArgsOs) -> anyhow::Result<()> {
    let mut table = EnvTable::with_capacity(argv.len());

    let arg0 = argv.next().unwrap();
    let arg0 = arg0.to_string_lossy();

    // Allocate a single shared "empty" value for optionals below
    let empty = unsafe { table.alloc(&[]) };

    unsafe {
        table.insert(arg0.as_bytes());
    }

    // Register `root` flag
    let current_exe = std::env::current_exe()?;
    let root = current_exe.parent().unwrap().to_string_lossy();
    unsafe {
        table.insert("-root".as_bytes());
        table.insert(root.as_bytes());
    }

    // Register 'progname' flag
    let arg0 = {
        let arg0: &str = arg0.borrow();
        Path::new(arg0)
    };
    let progname = arg0.file_name().unwrap().to_string_lossy();
    unsafe {
        table.insert("-progname".as_bytes());
        table.insert(progname.as_bytes());
    }

    // Register `home` flag
    if let Some(home) = dirs::home_dir() {
        unsafe {
            table.insert("-home".as_bytes());
            let home = home.to_string_lossy();
            table.insert(home.as_bytes());
        }
    } else {
        unsafe {
            table.insert("-home".as_bytes());
            table.argv.push(empty);
        }
    }

    for arg in argv {
        let arg = arg.to_string_lossy();
        unsafe {
            table.insert(arg.as_bytes());
        }
    }

    ARGV.set(table)
        .map_err(|_| anyhow!("arguments were already initialized"))
        .unwrap();

    Ok(())
}

#[derive(Default)]
struct EnvTable {
    argv: Vec<&'static BinaryData>,
    arena: DroplessArena,
}
impl EnvTable {
    fn with_capacity(size: usize) -> Self {
        Self {
            argv: Vec::with_capacity(size),
            arena: Default::default(),
        }
    }

    unsafe fn insert(&mut self, bytes: &[u8]) {
        let data = self.alloc(bytes);
        self.argv.push(data);
    }

    unsafe fn alloc(&mut self, bytes: &[u8]) -> &'static BinaryData {
        // Allocate memory for binary metadata and value
        let size = bytes.len();
        let (layout, value_offset) = Layout::new::<BinaryFlags>()
            .extend(Layout::from_size_align_unchecked(
                size,
                mem::align_of::<u8>(),
            ))
            .unwrap();
        let layout = layout.pad_to_align();
        let ptr = self.arena.alloc_raw(layout);

        // Write flags
        let flags_ptr: *mut BinaryFlags = ptr.cast();
        flags_ptr.write(BinaryFlags::new_literal(size, Encoding::detect(bytes)));

        // Write data
        let bytes_ptr: *mut u8 = (flags_ptr as *mut u8).add(value_offset);
        ptr::copy_nonoverlapping(bytes.as_ptr(), bytes_ptr, size);

        // Reify as static reference
        let data_ptr: *const BinaryData = ptr::from_raw_parts(ptr.cast(), size);
        &*data_ptr
    }
}
unsafe impl Send for EnvTable {}
unsafe impl Sync for EnvTable {}
