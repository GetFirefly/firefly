use parking_lot::Mutex;
use std::path::Path;

#[derive(Clone, Copy, Eq, PartialEq, Debug)]
pub struct Addr(pub u32);

impl Addr {
    pub fn as_usize(self) -> usize {
        self.0 as usize
    }
}

pub trait SerializationSink: Sized + Send + Sync + 'static {
    fn from_path(path: &Path) -> anyhow::Result<Self>;

    /// Atomically write `num_bytes` to the sink. The implementation must ensure
    /// that concurrent invocations of `write_atomic` do not conflict with each
    /// other.
    ///
    /// The `write` argument is a function that must fill the output buffer
    /// passed to it. The output buffer is guaranteed to be exactly `num_bytes`
    /// large.
    fn write_atomic<W>(&self, num_bytes: usize, write: W) -> Addr
    where
        W: FnOnce(&mut [u8]);

    /// Same as write_atomic() but might be faster in cases where bytes to be
    /// written are already present in a buffer (as opposed to when it is
    /// benefical to directly serialize into the output buffer).
    fn write_bytes_atomic(&self, bytes: &[u8]) -> Addr {
        self.write_atomic(bytes.len(), |sink| sink.copy_from_slice(bytes))
    }
}

/// A `SerializationSink` that writes to an internal `Vec<u8>` and can be
/// converted into this raw `Vec<u8>`. This implementation is only meant to be
/// used for testing and is not very efficient.
pub struct ByteVecSink {
    data: Mutex<Vec<u8>>,
}

impl ByteVecSink {
    pub fn new() -> ByteVecSink {
        ByteVecSink {
            data: Mutex::new(Vec::new()),
        }
    }

    pub fn into_bytes(self) -> Vec<u8> {
        self.data.into_inner()
    }
}

impl SerializationSink for ByteVecSink {
    fn from_path(_path: &Path) -> anyhow::Result<Self> {
        unimplemented!()
    }

    fn write_atomic<W>(&self, num_bytes: usize, write: W) -> Addr
    where
        W: FnOnce(&mut [u8]),
    {
        let mut data = self.data.lock();

        let start = data.len();

        data.resize(start + num_bytes, 0);

        write(&mut data[start..]);

        Addr(start as u32)
    }
}

impl std::fmt::Debug for ByteVecSink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ByteVecSink")
    }
}
