///! A wrapper around LLVM's archive (.a) code
use std::marker::PhantomData;
use std::path::Path;
use std::slice;
use std::str;
use std::str::FromStr;

use libc::{c_char, size_t};

use liblumen_util::fs;

use crate::diagnostics;
use crate::LLVMResult;

extern "C" {
    pub type Archive;
    type Opaque;
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

#[repr(C)]
struct InvariantOpaque<'a> {
    _marker: PhantomData<&'a mut &'a ()>,
    _opaque: Opaque,
}
#[repr(C)]
pub struct LumenArchiveMember<'a>(InvariantOpaque<'a>);
#[repr(C)]
pub struct ArchiveIterator<'a>(InvariantOpaque<'a>);
#[repr(C)]
pub struct ArchiveChild<'a>(InvariantOpaque<'a>);

pub struct ArchiveRO {
    pub raw: &'static mut Archive,
}

unsafe impl Send for ArchiveRO {}

pub struct Iter<'a> {
    raw: &'a mut ArchiveIterator<'a>,
}

pub struct Child<'a> {
    pub raw: &'a mut ArchiveChild<'a>,
}

impl ArchiveRO {
    /// Opens a static archive for read-only purposes. This is more optimized
    /// than the `open` method because it uses LLVM's internal `Archive` class
    /// rather than shelling out to `ar` for everything.
    ///
    /// If this archive is used with a mutable method, then an error will be
    /// raised.
    pub fn open(dst: &Path) -> Result<ArchiveRO, String> {
        return unsafe {
            let s = fs::path_to_c_string(dst);
            let ar = LLVMLumenOpenArchive(s.as_ptr()).ok_or_else(|| {
                diagnostics::last_error().unwrap_or_else(|| "failed to open archive".to_owned())
            })?;
            Ok(ArchiveRO { raw: ar })
        };
    }

    pub fn iter(&self) -> Iter<'_> {
        unsafe {
            Iter {
                raw: LLVMLumenArchiveIteratorNew(self.raw),
            }
        }
    }
}

impl Drop for ArchiveRO {
    fn drop(&mut self) {
        unsafe {
            LLVMLumenDestroyArchive(&mut *(self.raw as *mut _));
        }
    }
}

impl<'a> Iterator for Iter<'a> {
    type Item = Result<Child<'a>, String>;

    fn next(&mut self) -> Option<Result<Child<'a>, String>> {
        unsafe {
            match LLVMLumenArchiveIteratorNext(self.raw) {
                Some(raw) => Some(Ok(Child { raw })),
                None => diagnostics::last_error().map(Err),
            }
        }
    }
}

impl<'a> Drop for Iter<'a> {
    fn drop(&mut self) {
        unsafe {
            LLVMLumenArchiveIteratorFree(&mut *(self.raw as *mut _));
        }
    }
}

impl<'a> Child<'a> {
    pub fn name(&self) -> Option<&'a str> {
        unsafe {
            let mut name_len = 0;
            let name_ptr = LLVMLumenArchiveChildName(self.raw, &mut name_len);
            if name_ptr.is_null() {
                None
            } else {
                let name = slice::from_raw_parts(name_ptr as *const u8, name_len as usize);
                str::from_utf8(name).ok().map(|s| s.trim())
            }
        }
    }

    #[allow(dead_code)]
    pub fn data(&self) -> &'a [u8] {
        unsafe {
            let mut data_len = 0;
            let data_ptr = LLVMLumenArchiveChildData(self.raw, &mut data_len);
            if data_ptr.is_null() {
                panic!("failed to read data from archive child");
            }
            slice::from_raw_parts(data_ptr as *const u8, data_len as usize)
        }
    }
}

impl<'a> Drop for Child<'a> {
    fn drop(&mut self) {
        unsafe {
            LLVMLumenArchiveChildFree(&mut *(self.raw as *mut _));
        }
    }
}

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
    ) -> LLVMResult;

    pub fn LLVMLumenArchiveMemberNew<'a>(
        Filename: *const c_char,
        Name: *const c_char,
        Child: Option<&ArchiveChild<'a>>,
    ) -> &'a mut LumenArchiveMember<'a>;
    pub fn LLVMLumenArchiveMemberFree<'a>(Member: &'a mut LumenArchiveMember<'a>);
}
