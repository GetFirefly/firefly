use std::alloc::Layout;
use std::borrow::Borrow;
use std::env::ArgsOs;
use std::fmt;
use std::path::Path;
use std::ptr;
use std::sync::OnceLock;

use firefly_arena::DroplessArena;
use firefly_binary::{BinaryFlags, Encoding};
use firefly_rt::term::{BinaryData, OpaqueTerm};

static ARGV: OnceLock<EnvTable> = OnceLock::new();

/// Returns all arguments this executable was invoked with
pub fn argv() -> &'static [OpaqueTerm] {
    ARGV.get().unwrap().argv.as_slice()
}

pub struct AlreadyInitializedError;
impl fmt::Debug for AlreadyInitializedError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}
impl fmt::Display for AlreadyInitializedError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "arguments were already initialized")
    }
}

/// Performs one-time initialization of the environment for the current executable.
/// This is used to cache the arguments vector as constant binary values.
pub fn init(mut argv: ArgsOs) -> Result<(), AlreadyInitializedError> {
    let mut table = EnvTable::with_capacity(argv.len());

    let arg0 = argv.next().unwrap();
    let arg0 = arg0.to_string_lossy();

    // Allocate a single shared "empty" value for optionals below
    let empty = unsafe { table.alloc(&[]) };

    unsafe {
        table.insert(arg0.as_bytes());
    }

    // Register `root` flag
    let current_exe = std::env::current_exe().unwrap();
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
            table.argv.push(empty.into());
        }
    }

    for arg in argv {
        let arg = arg.to_string_lossy();
        unsafe {
            table.insert(arg.as_bytes());
        }
    }

    ARGV.set(table)
        .map_err(|_| AlreadyInitializedError)
        .unwrap();

    Ok(())
}

#[derive(Default)]
struct EnvTable {
    argv: Vec<OpaqueTerm>,
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
        self.argv.push(data.into());
    }

    unsafe fn alloc(&mut self, bytes: &[u8]) -> &'static BinaryData {
        use firefly_rt::term::{Header, Tag};

        // Allocate memory for binary metadata and value
        let size = bytes.len();
        let (layout, value_offset) = Layout::new::<Header>()
            .extend(Layout::array::<u8>(size).unwrap())
            .unwrap();
        let layout = layout.pad_to_align();
        let ptr = self.arena.alloc_raw(layout);

        // Write flags
        let header_ptr: *mut Header = ptr.cast();
        let flags = BinaryFlags::new(size, Encoding::detect(bytes));
        header_ptr.write(Header::new(Tag::Binary, flags.into_raw()));

        // Write data
        let bytes_ptr: *mut u8 = ptr.add(value_offset);
        ptr::copy_nonoverlapping(bytes.as_ptr(), bytes_ptr, size);

        // Reify as static reference
        let data_ptr: *const BinaryData = ptr::from_raw_parts(ptr.cast(), size);
        &*data_ptr
    }
}
unsafe impl Send for EnvTable {}
unsafe impl Sync for EnvTable {}
