use alloc::collections::BTreeMap;
use alloc::rc::Rc;
use alloc::vec::Vec;

use super::{Atom, HashMap, ModuleFunctionArity};

pub type FileId = u32;
pub type LocationId = u32;

/// This corresponds to `firefly_rt::backtrace::Symbol` and `firefly_rt::backtrace::Symbolication`
///
/// This is used to provide a nice API for symbolication without having to depend on `firefly_rt`
/// directly
#[derive(Debug, Clone)]
pub enum Symbol<A: Atom> {
    /// This is a normal Erlang function implemented in bytecode
    Erlang {
        mfa: ModuleFunctionArity<A>,
        loc: Option<SourceLocation>,
    },
    /// This is an Erlang built-in, implemented natively
    Bif(ModuleFunctionArity<A>),
    /// This is a natively-implemented function, using the C calling convention
    Native(A),
}

/// This is used to represent source locations for debug info generically
#[derive(Debug, Clone)]
pub struct SourceLocation {
    pub file: Rc<str>,
    pub line: u32,
    pub column: u32,
}

/// A compressed version of [`SourceLocation`] used by [`DebugInfoTable`]
#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub struct Location {
    pub file: FileId,
    pub line: u32,
    pub column: u32,
}

/// This table tracks source files and locations and their relationship to instruction offsets.
///
/// We attempt to balance compression of the data with performance of lookups.
#[derive(Default)]
pub struct DebugInfoTable {
    // Only store filenames once
    pub(crate) files: Vec<Rc<str>>,
    pub(crate) files_to_id: BTreeMap<Rc<str>, FileId>,
    // Only store unique source locations once
    pub(crate) locations: Vec<Location>,
    pub(crate) locations_to_id: HashMap<Location, LocationId>,
    // Maps ranges of instruction offsets to source locations
    pub(crate) offsets: BTreeMap<usize, LocationId>,
}
impl DebugInfoTable {
    pub fn append(&mut self, other: &Self, ix_offset: usize) {
        // Add all new files from `other` to `self`
        for (file, _) in other.files_to_id.iter() {
            use alloc::collections::btree_map::Entry;

            if let Entry::Vacant(entry) = self.files_to_id.entry(file.clone()) {
                assert!(self.files.len() < FileId::MAX as usize);
                let id = self.files.len() as FileId;
                self.files.push(file.clone());
                entry.insert(id);
            }
        }

        // For every location in `other`, add it to `self` if it doesn't already exist
        for (location, _) in other.locations_to_id.iter() {
            use hashbrown::hash_map::Entry;

            let file_id = self
                .files_to_id
                .get(&other.files[location.file as usize])
                .unwrap();
            let mut location = location.clone();
            location.file = *file_id;

            if let Entry::Vacant(entry) = self.locations_to_id.entry(location.clone()) {
                assert!(self.locations.len() < LocationId::MAX as usize);
                let location_id = self.locations.len() as LocationId;
                self.locations.push(location.clone());
                entry.insert(location_id);
            }
        }

        // For each offset, add the current instruction offset and insert the mapping
        // to its corresponding source location in `self`
        for (offset, id) in other.offsets.iter() {
            let new_offset = ix_offset + *offset;
            let new_id = self
                .locations_to_id
                .get(&other.locations[*id as usize])
                .unwrap();
            assert_eq!(self.offsets.insert(new_offset, *new_id), None);
        }
    }

    pub fn get_or_insert_file(&mut self, file: &str) -> FileId {
        match self.files_to_id.get(file) {
            None => {
                assert!(self.files.len() < FileId::MAX as usize);
                let id = self.files.len() as FileId;
                let file: Rc<str> = file.into();
                self.files.push(file.clone());
                self.files_to_id.insert(file, id);
                id
            }
            Some(id) => *id,
        }
    }

    pub fn get_or_insert_location(&mut self, location: Location) -> LocationId {
        assert!((location.file as usize) < self.files.len());
        match self.locations_to_id.get(&location) {
            None => {
                assert!(self.locations.len() < LocationId::MAX as usize);
                let id = self.locations.len() as LocationId;
                self.locations.push(location);
                self.locations_to_id.insert(location, id);
                id
            }
            Some(id) => *id,
        }
    }

    pub fn register_offset(&mut self, offset: usize, loc: LocationId) {
        self.offsets.insert(offset, loc);
    }

    /// Try to locate a source location for the current instruction pointer, looking no
    /// further back than the current function pointer for info.
    pub(super) fn offset_to_source_location(&self, fp: usize, ip: usize) -> Option<SourceLocation> {
        let mut iter = self.offsets.range(fp..=ip);
        let (_, id) = iter.next_back()?;
        self.location_to_source_location(*id)
    }

    #[inline]
    pub(super) fn function_pointer_to_source_location(&self, fp: usize) -> Option<SourceLocation> {
        self.location_to_source_location(self.offsets.get(&fp).copied()?)
    }

    pub(super) fn location_to_source_location(&self, loc: LocationId) -> Option<SourceLocation> {
        let loc = self.locations.get(loc as usize)?;
        let file = self.files[loc.file as usize].clone();
        Some(SourceLocation {
            file,
            line: loc.line,
            column: loc.column,
        })
    }
}
