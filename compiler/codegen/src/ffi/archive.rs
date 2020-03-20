use std::marker::PhantomData;
use std::str::FromStr;

use libc::{c_char, size_t};

#[derive(Copy, Clone, PartialEq)]
#[repr(C)]
#[allow(dead_code)] // Variants constructed by C++.
pub enum LLVMLumenResult {
    Success,
    Failure,
}
impl LLVMLumenResult {
    pub fn into_result(self) -> Result<(), ()> {
        match self {
            LLVMLumenResult::Success => Ok(()),
            LLVMLumenResult::Failure => Err(()),
        }
    }
}

extern "C" {
    type Opaque;
}
#[repr(C)]
struct InvariantOpaque<'a> {
    _marker: PhantomData<&'a mut &'a ()>,
    _opaque: Opaque,
}

/// LLVMLumenArchiveKind
#[allow(non_camel_case_types)]
#[derive(Copy, Clone)]
#[repr(C)]
pub enum ArchiveKind {
    // FIXME: figure out if this variant is needed at all.
    #[allow(dead_code)]
    Other,
    K_GNU,
    K_BSD,
    K_COFF,
}
impl FromStr for ArchiveKind {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "gnu" => Ok(ArchiveKind::K_GNU),
            "bsd" => Ok(ArchiveKind::K_BSD),
            "coff" => Ok(ArchiveKind::K_COFF),
            _ => Err(()),
        }
    }
}

extern "C" {
    pub type Archive;
}

#[repr(C)]
pub struct LumenArchiveMember<'a>(InvariantOpaque<'a>);
#[repr(C)]
pub struct ArchiveIterator<'a>(InvariantOpaque<'a>);
#[repr(C)]
pub struct ArchiveChild<'a>(InvariantOpaque<'a>);

extern "C" {
    pub fn LLVMLumenOpenArchive(path: *const c_char) -> Option<&'static mut Archive>;
    pub fn LLVMLumenArchiveIteratorNew<'a>(AR: &'a Archive) -> &'a mut ArchiveIterator<'a>;
    pub fn LLVMLumenArchiveIteratorNext<'a>(
        AIR: &ArchiveIterator<'a>,
    ) -> Option<&'a mut ArchiveChild<'a>>;
    pub fn LLVMLumenArchiveChildName(ACR: &ArchiveChild<'_>, size: &mut size_t) -> *const c_char;
    pub fn LLVMLumenArchiveChildData(ACR: &ArchiveChild<'_>, size: &mut size_t) -> *const c_char;
    pub fn LLVMLumenArchiveChildFree<'a>(ACR: &'a mut ArchiveChild<'a>);
    pub fn LLVMLumenArchiveIteratorFree<'a>(AIR: &'a mut ArchiveIterator<'a>);
    pub fn LLVMLumenDestroyArchive(AR: &'static mut Archive);

    pub fn LLVMLumenWriteArchive(
        Dst: *const c_char,
        NumMembers: size_t,
        Members: *const &LumenArchiveMember<'_>,
        WriteSymbtab: bool,
        Kind: ArchiveKind,
    ) -> LLVMLumenResult;

    pub fn LLVMLumenArchiveMemberNew<'a>(
        Filename: *const c_char,
        Name: *const c_char,
        Child: Option<&ArchiveChild<'a>>,
    ) -> &'a mut LumenArchiveMember<'a>;
    pub fn LLVMLumenArchiveMemberFree<'a>(Member: &'a mut LumenArchiveMember<'a>);
}
