use core::mem;

#[inline]
pub unsafe fn as_static<'a>(s: &'a str) -> &'static str {
    inherit_lifetime(s)
}

#[inline]
pub unsafe fn inherit_lifetime<'a, 'b: 'a>(s: &'a str) -> &'b str {
    mem::transmute::<&'a str, &'b str>(s)
}
