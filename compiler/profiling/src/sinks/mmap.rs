use std::fs::File;
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use memmap::MmapMut;

use crate::serialization::{Addr, SerializationSink};

pub struct MmapSerializationSink {
    mapped_file: MmapMut,
    current_pos: AtomicUsize,
    path: PathBuf,
}

impl SerializationSink for MmapSerializationSink {
    fn from_path(path: &Path) -> anyhow::Result<Self> {
        // Lazily allocate 1 GB :O
        let file_size = 1 << 30;

        let mapped_file = MmapMut::map_anon(file_size)?;

        Ok(MmapSerializationSink {
            mapped_file,
            current_pos: AtomicUsize::new(0),
            path: path.to_path_buf(),
        })
    }

    #[inline]
    fn write_atomic<W>(&self, num_bytes: usize, write: W) -> Addr
    where
        W: FnOnce(&mut [u8]),
    {
        // Reserve the range of bytes we'll copy to
        let pos = self.current_pos.fetch_add(num_bytes, Ordering::SeqCst);

        // Bounds checks
        assert!(pos.checked_add(num_bytes).unwrap() <= self.mapped_file.len());

        // We don't have `&mut self.mapped_file` available, so we have to go
        // through raw pointers instead of `MmapMut::get_mut()`. This is OK
        // because our runtime checks guarantee that we have exclusive access
        // to the byte range in question.
        let bytes: &mut [u8] = unsafe {
            let start: *mut u8 = self.mapped_file.as_ptr().offset(pos as isize) as *mut u8;
            std::slice::from_raw_parts_mut(start, num_bytes)
        };

        write(bytes);

        Addr(pos as u32)
    }
}

impl Drop for MmapSerializationSink {
    fn drop(&mut self) {
        let actual_size = *self.current_pos.get_mut();

        let file = match File::create(&self.path) {
            Ok(file) => file,
            Err(e) => {
                eprintln!("Error opening file for writing: {:?}", e);
                return;
            }
        };

        let mut file = BufWriter::new(file);

        if let Err(e) = file.write_all(&self.mapped_file[0..actual_size]) {
            eprintln!("Error writing file: {:?}", e);
        }
    }
}
