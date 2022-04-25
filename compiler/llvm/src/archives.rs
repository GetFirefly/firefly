use std::convert::AsRef;
///! A wrapper around LLVM's archive (.a) code
use std::mem::MaybeUninit;
use std::ops::Deref;
use std::path::Path;
use std::str::FromStr;

use anyhow::anyhow;

use crate::support::*;

extern "C" {
    type LlvmArchive;
    type LlvmArchiveMember;
    type LlvmArchiveIterator;
    type LlvmNewArchiveMember;
}

/// Corresponds to llvm::Archive::Kind
#[derive(Copy, Clone)]
#[repr(C)]
pub enum ArchiveKind {
    Gnu = 0,
    Gnu64,
    Bsd,
    Darwin,
    Darwin64,
    Coff,
    AixBig,
}
impl FromStr for ArchiveKind {
    type Err = ();

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s {
            "gnu" => Ok(Self::Gnu),
            "gnu64" => Ok(Self::Gnu64),
            "bsd" => Ok(Self::Bsd),
            "darwin" => Ok(Self::Darwin),
            "darwin64" => Ok(Self::Darwin64),
            "coff" => Ok(Self::Coff),
            "aix-big" => Ok(Self::AixBig),
            _ => Err(()),
        }
    }
}

/// Represents a borrowed reference to an archive
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct Archive(*const LlvmArchive);
impl Archive {
    #[inline]
    pub fn is_null(&self) -> bool {
        self.0.is_null()
    }

    /// Opens a static archive for read-only purposes. This is more optimized
    /// than the `open` method because it uses LLVM's internal `Archive` class
    /// rather than shelling out to `ar` for everything.
    ///
    /// If this archive is used with a mutable method, then an error will be
    /// raised.
    pub fn open(dst: &Path) -> anyhow::Result<OwnedArchive> {
        extern "C" {
            fn LLVMLumenOpenArchive(
                path: StringRef,
                error: *mut *mut std::os::raw::c_char,
            ) -> Archive;
        }
        let mut error = MaybeUninit::zeroed();
        let dst = StringRef::from(dst);
        let archive = unsafe { LLVMLumenOpenArchive(dst, error.as_mut_ptr()) };
        if archive.is_null() {
            unsafe {
                let error = error.assume_init();
                assert!(!error.is_null());
                let reason = OwnedStringRef::from_ptr(error);
                Err(anyhow!("{}", &reason))
            }
        } else {
            Ok(OwnedArchive(archive))
        }
    }

    /// Writes an archive of the given kind to `dst`, with the given members.
    ///
    /// If `write_symtab` is true, a symbol table is generated and written
    ///
    /// You can use `open` after creation to open the archive if desired
    pub fn create(
        dst: &Path,
        members: &[NewArchiveMember<'_>],
        write_symtab: bool,
        kind: ArchiveKind,
    ) -> anyhow::Result<()> {
        extern "C" {
            fn LLVMLumenWriteArchive(
                dst: StringRef,
                num_members: usize,
                members: *const *const LlvmNewArchiveMember,
                write_symbtab: bool,
                kind: ArchiveKind,
                error: *mut *mut std::os::raw::c_char,
            ) -> bool;
        }
        let dst = StringRef::from(dst);
        let mut error = MaybeUninit::zeroed();
        let success = unsafe {
            LLVMLumenWriteArchive(
                dst,
                members.len(),
                members.as_ptr().cast(),
                write_symtab,
                kind,
                error.as_mut_ptr(),
            )
        };
        if success {
            Ok(())
        } else {
            let error = unsafe { error.assume_init() };
            assert!(!error.is_null());
            let error = unsafe { OwnedStringRef::from_ptr(error) };
            Err(anyhow!("{}", &error))
        }
    }

    pub fn iter(&self) -> ArchiveIter<'_> {
        ArchiveIter::new(self).unwrap()
    }
}

/// Represents an owning reference to an archive
pub struct OwnedArchive(Archive);
impl AsRef<Archive> for OwnedArchive {
    #[inline]
    fn as_ref(&self) -> &Archive {
        &self.0
    }
}
impl Deref for OwnedArchive {
    type Target = Archive;
    #[inline]
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl Drop for OwnedArchive {
    fn drop(&mut self) {
        extern "C" {
            fn LLVMLumenDestroyArchive(archive: Archive);
        }
        unsafe {
            LLVMLumenDestroyArchive(self.0);
        }
    }
}

/// This struct corresponds to llvm::Archive::Child, i.e. a reference to a
/// member of an opened Archive, and as such, its lifetime is tied to that
/// of its parent archive.
///
/// You can obtain the members of an archive member by calling `Archive::iter`
#[derive(Debug)]
pub struct ArchiveMember<'a> {
    ptr: *mut LlvmArchiveMember,
    _marker: core::marker::PhantomData<&'a Archive>,
}
impl<'a> ArchiveMember<'a> {
    /// Returns the name of this archive member, which may or may not be set
    pub fn name(&self) -> Option<StringRef> {
        extern "C" {
            fn LLVMLumenArchiveChildName(
                child: *mut LlvmArchiveMember,
                error: *mut *mut std::os::raw::c_char,
            ) -> StringRef;
        }

        let mut error = MaybeUninit::zeroed();
        let name = unsafe { LLVMLumenArchiveChildName(self.ptr, error.as_mut_ptr()) };
        if name.is_null() {
            let error = unsafe { error.assume_init() };
            if error.is_null() {
                None
            } else {
                let error = unsafe { OwnedStringRef::from_ptr(error) };
                panic!("{}", &error)
            }
        } else {
            Some(name)
        }
    }

    /// Returns the data contained in this archive member
    pub fn data(&self) -> &[u8] {
        extern "C" {
            fn LLVMLumenArchiveChildData(
                child: *mut LlvmArchiveMember,
                error: *mut *mut std::os::raw::c_char,
            ) -> StringRef;
        }

        let mut error = MaybeUninit::zeroed();
        let data = unsafe { LLVMLumenArchiveChildData(self.ptr, error.as_mut_ptr()) };
        if data.is_null() {
            let error = unsafe { error.assume_init() };
            if error.is_null() {
                panic!("failed to read data from archive child");
            } else {
                let error = unsafe { OwnedStringRef::from_ptr(error) };
                panic!("{}", &error);
            }
        } else {
            unsafe { std::slice::from_raw_parts(data.data, data.len) }
        }
    }
}
impl<'a> Drop for ArchiveMember<'a> {
    fn drop(&mut self) {
        extern "C" {
            fn LLVMLumenArchiveChildFree(child: *mut LlvmArchiveMember);
        }
        unsafe {
            LLVMLumenArchiveChildFree(self.ptr);
        }
    }
}

/// This struct corresponds to an instance of llvm::NewArchiveMember,
/// and is used when constructing a new archive, either from an existing
/// archive or from scratch.
///
/// An archive member is a named object, whose contents are either:
///
/// * The contents of a file
/// * A member of another archive
///
/// The name is required and must not be empty.
#[repr(transparent)]
pub struct NewArchiveMember<'a> {
    ptr: *const LlvmNewArchiveMember,
    _marker: core::marker::PhantomData<&'a ArchiveMember<'a>>,
}
impl<'a> NewArchiveMember<'a> {
    /// Creates an archive member that references an existing archive member
    pub fn from_child<S: Into<StringRef>>(name: S, child: ArchiveMember<'a>) -> Self {
        extern "C" {
            fn LLVMLumenNewArchiveMemberFromChild(
                name: StringRef,
                child: *mut LlvmArchiveMember,
            ) -> *const LlvmNewArchiveMember;
        }
        let name = name.into();
        assert!(
            !name.empty(),
            "an archive member must be named with a non-empty string"
        );
        let ptr = unsafe { LLVMLumenNewArchiveMemberFromChild(name, child.ptr) };
        Self {
            ptr,
            _marker: core::marker::PhantomData,
        }
    }

    /// Creates an archive member that references the given file
    pub fn from_path<S: Into<StringRef>>(name: S, path: &Path) -> Self {
        extern "C" {
            fn LLVMLumenNewArchiveMemberFromFile(
                name: StringRef,
                filename: StringRef,
            ) -> *const LlvmNewArchiveMember;
        }
        let name = name.into();
        assert!(
            !name.empty(),
            "an archive member must be named with a non-empty string"
        );
        let path = StringRef::from(path);
        let ptr = unsafe { LLVMLumenNewArchiveMemberFromFile(name, path) };
        Self {
            ptr,
            _marker: core::marker::PhantomData,
        }
    }
}
impl<'a> Drop for NewArchiveMember<'a> {
    fn drop(&mut self) {
        extern "C" {
            fn LLVMLumenNewArchiveMemberFree(member: *const LlvmNewArchiveMember);
        }
        unsafe { LLVMLumenNewArchiveMemberFree(self.ptr) }
    }
}

/// An iterator over members of an archive
pub struct ArchiveIter<'a> {
    iterator: *mut LlvmArchiveIterator,
    done: bool,
    _marker: core::marker::PhantomData<&'a Archive>,
}
impl<'a> ArchiveIter<'a> {
    fn new(archive: &'a Archive) -> anyhow::Result<Self> {
        extern "C" {
            fn LLVMLumenArchiveIteratorNew(
                archive: *const LlvmArchive,
                error: *mut *mut std::os::raw::c_char,
            ) -> *mut LlvmArchiveIterator;
        }
        let mut error = MaybeUninit::zeroed();
        let iterator = unsafe { LLVMLumenArchiveIteratorNew(archive.0, error.as_mut_ptr()) };
        if iterator.is_null() {
            let error = unsafe { error.assume_init() };
            assert!(!error.is_null());
            let error = unsafe { OwnedStringRef::from_ptr(error) };
            Err(anyhow!("{}", &error))
        } else {
            Ok(Self {
                iterator,
                done: false,
                _marker: core::marker::PhantomData,
            })
        }
    }
}
impl<'a> Drop for ArchiveIter<'a> {
    fn drop(&mut self) {
        extern "C" {
            fn LLVMLumenArchiveIteratorFree(i: *mut LlvmArchiveIterator);
        }
        unsafe { LLVMLumenArchiveIteratorFree(self.iterator) }
    }
}
impl<'a> Iterator for ArchiveIter<'a> {
    type Item = anyhow::Result<ArchiveMember<'a>>;

    fn next(&mut self) -> Option<Self::Item> {
        extern "C" {
            fn LLVMLumenArchiveIteratorNext(
                iter: *mut LlvmArchiveIterator,
                error: *mut *mut std::os::raw::c_char,
            ) -> *mut LlvmArchiveMember;
        }

        if self.done {
            return None;
        }

        let mut error = MaybeUninit::zeroed();
        let ptr = unsafe { LLVMLumenArchiveIteratorNext(self.iterator, error.as_mut_ptr()) };
        if ptr.is_null() {
            let error = unsafe { error.assume_init() };
            self.done = true;
            if error.is_null() {
                None
            } else {
                let error = unsafe { OwnedStringRef::from_ptr(error) };
                Some(Err(anyhow!("{}", &error)))
            }
        } else {
            Some(Ok(ArchiveMember {
                ptr,
                _marker: core::marker::PhantomData,
            }))
        }
    }
}
impl<'a> std::iter::FusedIterator for ArchiveIter<'a> {}
