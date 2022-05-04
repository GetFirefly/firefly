use core::mem;

pub unsafe fn inherit_lifetime<'a, 'b: 'a>(s: &'a [u8]) -> &'b [u8] {
    mem::transmute::<&'a [u8], &'b [u8]>(s)
}
