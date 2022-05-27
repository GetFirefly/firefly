use core::fmt;

/// This struct facilitates writing strings/chars into an already allocated
/// slice of bytes (specifically, the bytes backing a binary).
///
/// It implements `core::fmt::Write`, and so can be used as a target for Rust
/// formatting facilities, e.g. `write!`.
pub struct BinaryWriter<'a> {
    data: &'a mut [u8],
    len: usize,
    pos: usize,
}
impl<'a> BinaryWriter<'a> {
    /// Create a new BinaryWriter over the given slice
    ///
    /// To write via the binary, you must use the Rust formatting facilities:
    ///
    /// ```ignore
    /// let mut w = BinaryWriter(&mut binary.data);
    /// write!(w, "Hello, {}!", name)?;
    /// ```
    pub fn new(data: &'a mut [u8]) -> Self {
        let len = data.len();
        Self { data, len, pos: 0 }
    }

    /// Write a `str` via this writer
    #[inline]
    pub fn push_str(&mut self, s: &str) {
        self.push_bytes(s.as_bytes())
    }

    /// Write a single `char` via this writer
    #[inline]
    pub fn push_char(&mut self, c: char) {
        self.push_str(c.encode_utf8(&mut [0; 4]));
    }

    /// Write a single byte via this writer
    #[inline]
    pub fn push_byte(&mut self, byte: u8) {
        if self.pos >= self.len {
            return;
        }

        self.data[self.pos] = byte;
        self.pos += 1;
    }

    /// Write a slice of bytes via this writer
    #[inline]
    pub fn push_bytes(&mut self, bytes: &[u8]) {
        if self.pos >= self.len {
            return;
        }

        // Calculate the number of bytes we can write from the string into the binary that was allocated,
        // taking only the number of bytes remaining if the entire string is too large.
        let bytes_size = bytes.len();
        let writable = self.len - self.pos;
        let len = if bytes_size > writable {
            writable
        } else {
            bytes_size
        };

        // Copy the bytes from `s` into `self.data`
        let buf = &mut self.data[self.pos..(self.pos + len)];
        buf.copy_from_slice(bytes);

        // Shift the position of the writer to the next byte after the last one written
        self.pos += len;
    }
}
impl<'a> fmt::Write for BinaryWriter<'a> {
    #[inline]
    fn write_str(&mut self, s: &str) -> fmt::Result {
        self.push_str(s);
        Ok(())
    }

    #[inline]
    fn write_char(&mut self, c: char) -> fmt::Result {
        self.push_char(c);
        Ok(())
    }
}
