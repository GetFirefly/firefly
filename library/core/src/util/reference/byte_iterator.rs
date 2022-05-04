use core::mem;

pub unsafe fn inherit_lifetime<'a, 'b: 'a, A, B>(s: &'a A) -> &'b B
where
    A: Iterator<Item = u8>,
    B: Iterator<Item = u8>,
{
    mem::transmute::<&'a A, &'b B>(s)
}
