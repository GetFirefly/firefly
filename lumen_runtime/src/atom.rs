#![cfg_attr(not(test), allow(dead_code))]

use crate::exception::Exception;
use std::collections::HashMap;
use std::sync::{Arc, RwLock};

pub enum Encoding {
    Latin1,
    Unicode,
    Utf8,
}

#[derive(Clone, Copy)]
pub enum Existence {
    DoNotCare,
    Exists,
}

use self::Existence::*;

#[derive(Clone, Copy)]
#[repr(transparent)]
pub struct Index(pub usize);

pub fn index_to_string(Index(index): Index) -> Result<Arc<String>, Exception> {
    let table_arc_rw_lock = TABLE_ARC_RW_LOCK.clone();
    let readable_table = table_arc_rw_lock.read().unwrap();

    match readable_table.atoms.get(index) {
        Some(Atom { name, .. }) => Ok(name.clone()),
        None => Err(bad_argument!()),
    }
}

pub fn str_to_index(name: &str, existence: Existence) -> Option<Index> {
    let table_arc_rw_lock = TABLE_ARC_RW_LOCK.clone();

    match existence {
        // Can only read, so we can take the cheaper read lock
        Exists => {
            let readable_table = table_arc_rw_lock.read().unwrap();

            // `index_by_name` never erases entries, so it is safe to take the index out of the
            // read-locked region.
            readable_table
                .index_by_name
                .get(&name.to_string())
                .map(|ref_index| ref_index.clone())
        }
        // May write, so take write lock even when reading so that we don't get duplicate names in
        // `atoms` after checking `index_by_str`.
        DoNotCare => {
            let mut writable_table = table_arc_rw_lock.write().unwrap();
            let name_string = name.to_string();

            match writable_table.index_by_name.get(&name_string) {
                Some(index) => Some(index.clone()),
                None => {
                    let name_arc: Arc<String> = Arc::new(name_string);
                    let atom = Atom::new(name_arc.clone());
                    let atoms = &mut writable_table.atoms;
                    atoms.push(atom);
                    let index = Index(atoms.len() - 1);
                    writable_table.index_by_name.insert(name_arc.clone(), index);

                    Some(index)
                }
            }
        }
    }
}

// Private

struct Atom {
    /// TODO ordinal value of first 3 bytes + 7 bits
    #[allow(dead_code)]
    ordinal: u32,
    name: Arc<String>,
}

impl Atom {
    fn new(name: Arc<String>) -> Self {
        Atom {
            ordinal: 0,
            name: name.clone(),
        }
    }
}

struct Table {
    atoms: Vec<Atom>,
    index_by_name: HashMap<Arc<String>, Index>,
}

impl Default for Table {
    fn default() -> Table {
        Table {
            atoms: Default::default(),
            index_by_name: Default::default(),
        }
    }
}

lazy_static! {
    static ref TABLE_ARC_RW_LOCK: Arc<RwLock<Table>> = Default::default();
}

#[cfg(test)]
mod tests {
    use super::*;

    mod str_to_index {
        use super::*;

        #[test]
        fn without_same_string_have_different_index() {
            assert_ne!(
                str_to_index("true", DoNotCare).unwrap().0,
                str_to_index("false", DoNotCare).unwrap().0
            )
        }

        #[test]
        fn with_same_string_have_same_index() {
            assert_eq!(
                str_to_index("atom", DoNotCare).unwrap().0,
                str_to_index("atom", DoNotCare).unwrap().0
            )
        }
    }
}
