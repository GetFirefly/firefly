use std::alloc::Layout;
use std::env::ArgsOs;
use std::lazy::SyncOnceCell;
use std::mem;
use std::ptr;
use std::slice;

use anyhow::anyhow;

use liblumen_arena::DroplessArena;
use liblumen_rt::term::{BinaryData, BinaryFlags, Encoding};

static ARGV: SyncOnceCell<EnvTable> = SyncOnceCell::new();

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
        // Allocate memory for atom metadata and value
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
        let data: &'static BinaryData = &*data_ptr;

        // Register in atom table
        self.argv.push(data);
    }

    fn argv(&self) -> &[&'static BinaryData] {
        self.argv.as_slice()
    }
}
unsafe impl Send for EnvTable {}
unsafe impl Sync for EnvTable {}

pub(crate) fn init_argv_from_slice(argv: ArgsOs) -> anyhow::Result<()> {
    let mut table = EnvTable::with_capacity(argv.len());
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

#[allow(unused)]
pub(crate) fn init_argv(argv: *const *const std::os::raw::c_char, argc: u32) -> anyhow::Result<()> {
    use std::ffi::CStr;

    let argc = argc as usize;
    if argc == 0 {
        let _ = ARGV.set(Default::default());
        return Ok(());
    }

    let argv = unsafe { slice::from_raw_parts::<'static>(argv, argc) };

    let mut table = EnvTable::with_capacity(argc);
    for arg_ptr in argv.iter().copied() {
        let cs = unsafe { CStr::from_ptr::<'static>(arg_ptr) };
        let arg = cs.to_string_lossy();
        unsafe {
            table.insert(arg.as_bytes());
        }
    }

    ARGV.set(table)
        .map_err(|_| anyhow!("arguments were already initialized"))
        .unwrap();
    Ok(())
}

pub fn get_argv() -> &'static [&'static BinaryData] {
    ARGV.get().unwrap().argv()
}
