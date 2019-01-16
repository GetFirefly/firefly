use std::mem;

#[repr(transparent)]
pub struct Address(usize);

impl Address {
    #[inline(always)]
    pub fn plus(&self, bytes: usize) -> Self {
        Address(self.0 + bytes)
    }

    #[inline(always)]
    pub fn minus(&self, bytes: usize) -> Self {
        Address(self.0 - bytes)
    }

    #[inline(always)]
    pub fn offset<T>(&self, offset: isize) -> Self {
        Address((self.0 as isize + mem::size_of::<T>() as isize * offset) as usize)
    }

    #[inline(always)]
    pub fn diff(&self, another: Address) -> usize {
        debug_assert!(self.0 >= another.0, "for a.diff(b), a needs to be larger than b");
        self.0 - another.0
    }

    #[inline(always)]
    pub unsafe fn load<T: Copy> (&self) -> T {
        *(self.0 as *mut T)
    }

    #[inline(always)]
    pub unsafe fn store<T> (&self, value: T) {
        *(self.0 as *mut T) = value;
    }

    #[inline(always)]
    pub fn is_zero(&self) -> bool {
        self.0 == 0
    }

    #[inline(always)]
    pub fn align_up(&self, align: usize) -> Address {
        Address((self.0 + align - 1) & !(align - 1))
    }

    pub fn is_aligned_to(&self, align: usize) -> bool {
        self.0 % align == 0
    }

    pub fn memset(&self, char: u8, length: usize) {
        let mut cur : *mut u8 = self.0 as *mut u8;
        for _ in 0..length {
            unsafe {
                *cur = char;
                cur = cur.offset(1);
            }
        }
    }

    #[inline(always)]
    pub unsafe fn to_object_reference(&self) -> ObjectReference {
        mem::transmute(self.0)
    }

    #[inline(always)]
    pub fn from_ptr<T> (ptr: *const T) -> Address {
        unsafe {mem::transmute(ptr)}
    }

    #[inline(always)]
    pub fn to_ptr<T> (&self) -> *const T {
        unsafe {mem::transmute(self.0)}
    }

    #[inline(always)]
    pub fn to_ptr_mut<T> (&self) -> *mut T {
        unsafe {mem::transmute(self.0)}
    }

    #[inline(always)]
    pub fn as_usize(&self) -> usize {
        self.0
    }

    #[inline(always)]
    pub unsafe fn zero() -> Address {
        Address(0)
    }
}

impl PartialOrd for Address {
    #[inline(always)]
    fn partial_cmp(&self, other: &Address) -> Option<std::cmp::Ordering> {
        Some(self.0.cmp(& other.0))
    }
}

impl PartialEq for Address {
    #[inline(always)]
    fn eq(&self, other: &Address) -> bool {
        self.0 == other.0
    }
    #[inline(always)]
    fn ne(&self, other: &Address) -> bool {
        self.0 != other.0
    }
}

impl std::fmt::UpperHex for Address {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:X}", self.0)
    }
}

impl std::fmt::Display for Address {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "0x{:X}", self.0)
    }
}

impl std::fmt::Debug for Address {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "0x{:X}", self.0)
    }
}

#[repr(transparent)]
#[derive(Copy, Clone, Eq, Hash)]
pub struct ObjectReference(usize);

impl ObjectReference {
    #[inline(always)]
    pub fn to_address(&self) -> Address {
        unsafe {mem::transmute(self.0)}
    }

    #[inline(always)]
    pub fn is_null(&self) -> bool {
        self.0 != 0
    }

    pub fn value(&self) -> usize {
        self.0
    }
}

impl PartialOrd for ObjectReference {
    #[inline(always)]
    fn partial_cmp(&self, other: &ObjectReference) -> Option<std::cmp::Ordering> {
        Some(self.0.cmp(& other.0))
    }
}

impl PartialEq for ObjectReference {
    #[inline(always)]
    fn eq(&self, other: &ObjectReference) -> bool {
        self.0 == other.0
    }
    #[inline(always)]
    fn ne(&self, other: &ObjectReference) -> bool {
        self.0 != other.0
    }
}

impl std::fmt::UpperHex for ObjectReference {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{:X}", self.0)
    }
}

impl std::fmt::Display for ObjectReference {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "0x{:X}", self.0)
    }
}

impl std::fmt::Debug for ObjectReference {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "0x{:X}", self.0)
    }
}

#[inline(always)]
pub fn test_nth_bit(value: u8, index: usize) -> bool {
    value & (1 << index) != 0
}

#[inline(always)]
pub fn lower_bits(value: u8, len: usize) -> u8 {
    value & ((1 << len) - 1)
}
