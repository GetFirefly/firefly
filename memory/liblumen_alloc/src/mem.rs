const BITS_PER_BYTE: usize = 8;

pub(crate) const fn bit_size_of<T>() -> usize {
    core::mem::size_of::<T>() * BITS_PER_BYTE
}
