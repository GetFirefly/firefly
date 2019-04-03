#![cfg_attr(not(test), allow(dead_code))]

use std::cmp::Ordering::{self, *};
use std::collections::HashMap;
use std::fmt::{self, Debug};
use std::mem::size_of;
use std::sync::{Arc, RwLock};

use crate::exception::Exception;

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

impl Eq for Index {}

impl Ord for Index {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl PartialEq for Index {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl PartialOrd for Index {
    fn partial_cmp(&self, other: &Index) -> Option<Ordering> {
        let self_inner = self.0;
        let other_inner = other.0;

        if self_inner == other_inner {
            Some(Equal)
        } else {
            let table_arc_rw_lock = TABLE_ARC_RW_LOCK.clone();
            let readable_table = table_arc_rw_lock.read().unwrap();
            let atoms = &readable_table.atoms;

            match (atoms.get(self.0), atoms.get(other.0)) {
                (Some(self_atom), Some(other_atom)) => self_atom.partial_cmp(other_atom),
                _ => None,
            }
        }
    }
}

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
    ///
    /// Precomputed ordinal value of first 3 bytes + 7 bits.
    ///
    /// This is used by [crate::atom::Atom::Eq].
    /// We cannot use the full 32 bits of the first 4 bytes,
    /// since we use the sign of the difference between two
    /// ordinal values to represent their relative order.
    ordinal: usize,
    name: Arc<String>,
}

const PREFIX_BYTE_COUNT: usize = size_of::<usize>();
// It has one fewer complete bytes because it is shifted (`>> 1`) to leave the sign bit clear
const ORDINAL_BYTE_COUNT: usize = PREFIX_BYTE_COUNT - 1;

impl Atom {
    fn new(name: Arc<String>) -> Self {
        let name = name.clone();
        let ordinal = Self::ordinal(name.as_bytes());

        // See https://github.com/erlang/otp/blob/be44d6827e2374a43068b35de85ed16441c771be/erts/emulator/beam/atom.c#L175-L192
        Atom { ordinal, name }
    }

    /// See  https://github.com/erlang/otp/blob/be44d6827e2374a43068b35de85ed16441c771be/erts/emulator/beam/atom.c#L175-L192
    fn ordinal(bytes: &[u8]) -> usize {
        let bytes_length = bytes.len();
        let packed_prefix: usize = (0..PREFIX_BYTE_COUNT).fold(0, |acc, i| {
            let byte: u8 = if i < bytes_length { bytes[i] } else { 0 };

            (acc << 8) | (byte as usize)
        });

        // We cannot use the full bits of the first bytes, since we use the sign of the
        // difference between two ordinal values to represent their relative order.
        packed_prefix >> 1
    }
}

impl Debug for Atom {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Atom::new({:?})", self.name)
    }
}

impl Eq for Atom {}

impl Ord for Atom {
    fn cmp(&self, other: &Self) -> Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl PartialEq for Atom {
    /// See https://github.com/erlang/otp/blob/be44d6827e2374a43068b35de85ed16441c771be/erts/emulator/beam/erl_utils.h#L159-L186
    fn eq(&self, other: &Atom) -> bool {
        (self.ordinal == other.ordinal) & {
            let length = self.name.len();

            // for equality, can check len before bytes because it is faster
            (length == other.name.len())
                & if ORDINAL_BYTE_COUNT < length {
                    let range = ORDINAL_BYTE_COUNT..length;

                    self.name.as_bytes()[range.clone()] == other.name.as_bytes()[range.clone()]
                } else {
                    true
                }
        }
    }
}

impl PartialOrd for Atom {
    /// See https://github.com/erlang/otp/blob/be44d6827e2374a43068b35de85ed16441c771be/erts/emulator/beam/erl_utils.h#L159-L186
    fn partial_cmp(&self, other: &Atom) -> Option<Ordering> {
        match self.ordinal.partial_cmp(&other.ordinal) {
            Some(Equal) => {
                let self_length = self.name.len();
                let other_length = other.name.len();

                let bytes_partial_ordering =
                    if (ORDINAL_BYTE_COUNT < self_length) & (ORDINAL_BYTE_COUNT < other_length) {
                        let range = ORDINAL_BYTE_COUNT..self_length.min(other_length);

                        self.name.as_bytes()[range.clone()]
                            .partial_cmp(&other.name.as_bytes()[range.clone()])
                    } else {
                        Some(Equal)
                    };

                match bytes_partial_ordering {
                    Some(Equal) => self_length.partial_cmp(&other_length),
                    partial_ordering => partial_ordering,
                }
            }
            partial_ordering => partial_ordering,
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

    mod atom {
        use super::*;

        mod cmp {
            use super::*;

            #[test]
            fn without_bytes_is_equal() {
                assert_eq!(
                    Atom::new(Arc::new("".to_string())).cmp(&Atom::new(Arc::new("".to_string()))),
                    Equal
                )
            }

            #[test]
            fn with_1_byte() {
                let first = Atom::new(Arc::new("b".to_string()));

                assert_eq!(first.cmp(&Atom::new(Arc::new("b".to_string()))), Equal);

                // content
                assert_eq!(first.cmp(&Atom::new(Arc::new("c".to_string()))), Less);
                assert_eq!(first.cmp(&Atom::new(Arc::new("a".to_string()))), Greater);

                // length
                assert_eq!(first.cmp(&Atom::new(Arc::new("bb".to_string()))), Less);
                assert_eq!(first.cmp(&Atom::new(Arc::new("".to_string()))), Greater);
            }

            #[test]
            fn with_2_bytes() {
                let first = Atom::new(Arc::new("bb".to_string()));

                assert_eq!(first.cmp(&Atom::new(Arc::new("bb".to_string()))), Equal);

                // content

                assert_eq!(first.cmp(&Atom::new(Arc::new("bc".to_string()))), Less);
                assert_eq!(first.cmp(&Atom::new(Arc::new("cb".to_string()))), Less);

                assert_eq!(first.cmp(&Atom::new(Arc::new("ab".to_string()))), Greater);
                assert_eq!(first.cmp(&Atom::new(Arc::new("ba".to_string()))), Greater);

                // length

                assert_eq!(first.cmp(&Atom::new(Arc::new("bbb".to_string()))), Less);

                assert_eq!(first.cmp(&Atom::new(Arc::new("b".to_string()))), Greater);
                assert_eq!(first.cmp(&Atom::new(Arc::new("".to_string()))), Greater);
            }

            #[test]
            fn with_3_bytes() {
                let first = Atom::new(Arc::new("bbb".to_string()));

                assert_eq!(first.cmp(&Atom::new(Arc::new("bbb".to_string()))), Equal);

                // content

                assert_eq!(first.cmp(&Atom::new(Arc::new("bbc".to_string()))), Less);
                assert_eq!(first.cmp(&Atom::new(Arc::new("bcb".to_string()))), Less);
                assert_eq!(first.cmp(&Atom::new(Arc::new("cbb".to_string()))), Less);

                assert_eq!(first.cmp(&Atom::new(Arc::new("abb".to_string()))), Greater);
                assert_eq!(first.cmp(&Atom::new(Arc::new("bab".to_string()))), Greater);
                assert_eq!(first.cmp(&Atom::new(Arc::new("bba".to_string()))), Greater);

                // length

                assert_eq!(first.cmp(&Atom::new(Arc::new("bbbb".to_string()))), Less);

                assert_eq!(first.cmp(&Atom::new(Arc::new("bb".to_string()))), Greater);
                assert_eq!(first.cmp(&Atom::new(Arc::new("b".to_string()))), Greater);
                assert_eq!(first.cmp(&Atom::new(Arc::new("".to_string()))), Greater);
            }

            #[test]
            fn with_4_bytes() {
                let first = Atom::new(Arc::new("bbbb".to_string()));

                assert_eq!(first.cmp(&Atom::new(Arc::new("bbbb".to_string()))), Equal);

                // content

                assert_eq!(first.cmp(&Atom::new(Arc::new("bbbc".to_string()))), Less);
                assert_eq!(first.cmp(&Atom::new(Arc::new("bbcb".to_string()))), Less);
                assert_eq!(first.cmp(&Atom::new(Arc::new("bcbb".to_string()))), Less);
                assert_eq!(first.cmp(&Atom::new(Arc::new("cbbb".to_string()))), Less);

                assert_eq!(first.cmp(&Atom::new(Arc::new("abbb".to_string()))), Greater);
                assert_eq!(first.cmp(&Atom::new(Arc::new("babb".to_string()))), Greater);
                assert_eq!(first.cmp(&Atom::new(Arc::new("bbab".to_string()))), Greater);
                assert_eq!(first.cmp(&Atom::new(Arc::new("bbba".to_string()))), Greater);

                // length

                assert_eq!(first.cmp(&Atom::new(Arc::new("bbbbb".to_string()))), Less);

                assert_eq!(first.cmp(&Atom::new(Arc::new("bbb".to_string()))), Greater);
                assert_eq!(first.cmp(&Atom::new(Arc::new("bb".to_string()))), Greater);
                assert_eq!(first.cmp(&Atom::new(Arc::new("b".to_string()))), Greater);
                assert_eq!(first.cmp(&Atom::new(Arc::new("".to_string()))), Greater);
            }

            #[test]
            fn with_5_bytes() {
                let first = Atom::new(Arc::new("bbbbb".to_string()));

                assert_eq!(first.cmp(&Atom::new(Arc::new("bbbbb".to_string()))), Equal);

                // content

                assert_eq!(first.cmp(&Atom::new(Arc::new("bbbbc".to_string()))), Less);
                assert_eq!(first.cmp(&Atom::new(Arc::new("bbbcb".to_string()))), Less);
                assert_eq!(first.cmp(&Atom::new(Arc::new("bbcbb".to_string()))), Less);
                assert_eq!(first.cmp(&Atom::new(Arc::new("bcbbb".to_string()))), Less);
                assert_eq!(first.cmp(&Atom::new(Arc::new("cbbbb".to_string()))), Less);

                assert_eq!(
                    first.cmp(&Atom::new(Arc::new("abbbb".to_string()))),
                    Greater
                );
                assert_eq!(
                    first.cmp(&Atom::new(Arc::new("babbb".to_string()))),
                    Greater
                );
                assert_eq!(
                    first.cmp(&Atom::new(Arc::new("bbabb".to_string()))),
                    Greater
                );
                assert_eq!(
                    first.cmp(&Atom::new(Arc::new("bbbab".to_string()))),
                    Greater
                );
                assert_eq!(
                    first.cmp(&Atom::new(Arc::new("bbbba".to_string()))),
                    Greater
                );

                // length

                assert_eq!(first.cmp(&Atom::new(Arc::new("bbbbbb".to_string()))), Less);

                assert_eq!(first.cmp(&Atom::new(Arc::new("bbbb".to_string()))), Greater);
                assert_eq!(first.cmp(&Atom::new(Arc::new("bbb".to_string()))), Greater);
                assert_eq!(first.cmp(&Atom::new(Arc::new("bb".to_string()))), Greater);
                assert_eq!(first.cmp(&Atom::new(Arc::new("b".to_string()))), Greater);
                assert_eq!(first.cmp(&Atom::new(Arc::new("".to_string()))), Greater);
            }

            #[test]
            fn with_6_bytes() {
                let first = Atom::new(Arc::new("bbbbbb".to_string()));

                assert_eq!(first.cmp(&Atom::new(Arc::new("bbbbbb".to_string()))), Equal);

                // content

                assert_eq!(first.cmp(&Atom::new(Arc::new("bbbbbc".to_string()))), Less);
                assert_eq!(first.cmp(&Atom::new(Arc::new("bbbbcb".to_string()))), Less);
                assert_eq!(first.cmp(&Atom::new(Arc::new("bbbcbb".to_string()))), Less);
                assert_eq!(first.cmp(&Atom::new(Arc::new("bbcbbb".to_string()))), Less);
                assert_eq!(first.cmp(&Atom::new(Arc::new("bcbbbb".to_string()))), Less);
                assert_eq!(first.cmp(&Atom::new(Arc::new("cbbbbb".to_string()))), Less);

                assert_eq!(
                    first.cmp(&Atom::new(Arc::new("abbbbb".to_string()))),
                    Greater
                );
                assert_eq!(
                    first.cmp(&Atom::new(Arc::new("babbbb".to_string()))),
                    Greater
                );
                assert_eq!(
                    first.cmp(&Atom::new(Arc::new("bbabbb".to_string()))),
                    Greater
                );
                assert_eq!(
                    first.cmp(&Atom::new(Arc::new("bbbabb".to_string()))),
                    Greater
                );
                assert_eq!(
                    first.cmp(&Atom::new(Arc::new("bbbbab".to_string()))),
                    Greater
                );
                assert_eq!(
                    first.cmp(&Atom::new(Arc::new("bbbbba".to_string()))),
                    Greater
                );

                // length

                assert_eq!(first.cmp(&Atom::new(Arc::new("bbbbbbb".to_string()))), Less);

                assert_eq!(
                    first.cmp(&Atom::new(Arc::new("bbbbb".to_string()))),
                    Greater
                );
                assert_eq!(first.cmp(&Atom::new(Arc::new("bbbb".to_string()))), Greater);
                assert_eq!(first.cmp(&Atom::new(Arc::new("bbb".to_string()))), Greater);
                assert_eq!(first.cmp(&Atom::new(Arc::new("bb".to_string()))), Greater);
                assert_eq!(first.cmp(&Atom::new(Arc::new("b".to_string()))), Greater);
                assert_eq!(first.cmp(&Atom::new(Arc::new("".to_string()))), Greater);
            }

            #[test]
            fn with_7_bytes() {
                let first = Atom::new(Arc::new("bbbbbbb".to_string()));

                assert_eq!(
                    first.cmp(&Atom::new(Arc::new("bbbbbbb".to_string()))),
                    Equal
                );

                // content

                assert_eq!(first.cmp(&Atom::new(Arc::new("bbbbbbc".to_string()))), Less);
                assert_eq!(first.cmp(&Atom::new(Arc::new("bbbbbcb".to_string()))), Less);
                assert_eq!(first.cmp(&Atom::new(Arc::new("bbbbcbb".to_string()))), Less);
                assert_eq!(first.cmp(&Atom::new(Arc::new("bbbcbbb".to_string()))), Less);
                assert_eq!(first.cmp(&Atom::new(Arc::new("bbcbbbb".to_string()))), Less);
                assert_eq!(first.cmp(&Atom::new(Arc::new("bcbbbbb".to_string()))), Less);
                assert_eq!(first.cmp(&Atom::new(Arc::new("cbbbbbb".to_string()))), Less);

                assert_eq!(
                    first.cmp(&Atom::new(Arc::new("abbbbbb".to_string()))),
                    Greater
                );
                assert_eq!(
                    first.cmp(&Atom::new(Arc::new("babbbbb".to_string()))),
                    Greater
                );
                assert_eq!(
                    first.cmp(&Atom::new(Arc::new("bbabbbb".to_string()))),
                    Greater
                );
                assert_eq!(
                    first.cmp(&Atom::new(Arc::new("bbbabbb".to_string()))),
                    Greater
                );
                assert_eq!(
                    first.cmp(&Atom::new(Arc::new("bbbbabb".to_string()))),
                    Greater
                );
                assert_eq!(
                    first.cmp(&Atom::new(Arc::new("bbbbbab".to_string()))),
                    Greater
                );
                assert_eq!(
                    first.cmp(&Atom::new(Arc::new("bbbbbba".to_string()))),
                    Greater
                );

                // length

                assert_eq!(
                    first.cmp(&Atom::new(Arc::new("bbbbbbbb".to_string()))),
                    Less
                );

                assert_eq!(
                    first.cmp(&Atom::new(Arc::new("bbbbbb".to_string()))),
                    Greater
                );
                assert_eq!(
                    first.cmp(&Atom::new(Arc::new("bbbbb".to_string()))),
                    Greater
                );
                assert_eq!(first.cmp(&Atom::new(Arc::new("bbbb".to_string()))), Greater);
                assert_eq!(first.cmp(&Atom::new(Arc::new("bbb".to_string()))), Greater);
                assert_eq!(first.cmp(&Atom::new(Arc::new("bb".to_string()))), Greater);
                assert_eq!(first.cmp(&Atom::new(Arc::new("b".to_string()))), Greater);
                assert_eq!(first.cmp(&Atom::new(Arc::new("".to_string()))), Greater);
            }

            #[test]
            fn with_8_bytes() {
                let first = Atom::new(Arc::new("bbbbbbbb".to_string()));

                assert_eq!(
                    first.cmp(&Atom::new(Arc::new("bbbbbbbb".to_string()))),
                    Equal
                );

                // content

                assert_eq!(
                    first.cmp(&Atom::new(Arc::new("bbbbbbbc".to_string()))),
                    Less
                );
                assert_eq!(
                    first.cmp(&Atom::new(Arc::new("bbbbbbcb".to_string()))),
                    Less
                );
                assert_eq!(
                    first.cmp(&Atom::new(Arc::new("bbbbbcbb".to_string()))),
                    Less
                );
                assert_eq!(
                    first.cmp(&Atom::new(Arc::new("bbbbcbbb".to_string()))),
                    Less
                );
                assert_eq!(
                    first.cmp(&Atom::new(Arc::new("bbbcbbbb".to_string()))),
                    Less
                );
                assert_eq!(
                    first.cmp(&Atom::new(Arc::new("bbcbbbbb".to_string()))),
                    Less
                );
                assert_eq!(
                    first.cmp(&Atom::new(Arc::new("bcbbbbbb".to_string()))),
                    Less
                );
                assert_eq!(
                    first.cmp(&Atom::new(Arc::new("cbbbbbbb".to_string()))),
                    Less
                );

                assert_eq!(
                    first.cmp(&Atom::new(Arc::new("abbbbbbb".to_string()))),
                    Greater
                );
                assert_eq!(
                    first.cmp(&Atom::new(Arc::new("babbbbbb".to_string()))),
                    Greater
                );
                assert_eq!(
                    first.cmp(&Atom::new(Arc::new("bbabbbbb".to_string()))),
                    Greater
                );
                assert_eq!(
                    first.cmp(&Atom::new(Arc::new("bbbabbbb".to_string()))),
                    Greater
                );
                assert_eq!(
                    first.cmp(&Atom::new(Arc::new("bbbbabbb".to_string()))),
                    Greater
                );
                assert_eq!(
                    first.cmp(&Atom::new(Arc::new("bbbbbabb".to_string()))),
                    Greater
                );
                assert_eq!(
                    first.cmp(&Atom::new(Arc::new("bbbbbbab".to_string()))),
                    Greater
                );
                assert_eq!(
                    first.cmp(&Atom::new(Arc::new("bbbbbbba".to_string()))),
                    Greater
                );

                // length

                assert_eq!(
                    first.cmp(&Atom::new(Arc::new("bbbbbbbbb".to_string()))),
                    Less
                );

                assert_eq!(
                    first.cmp(&Atom::new(Arc::new("bbbbbbb".to_string()))),
                    Greater
                );
                assert_eq!(
                    first.cmp(&Atom::new(Arc::new("bbbbbb".to_string()))),
                    Greater
                );
                assert_eq!(
                    first.cmp(&Atom::new(Arc::new("bbbbb".to_string()))),
                    Greater
                );
                assert_eq!(first.cmp(&Atom::new(Arc::new("bbbb".to_string()))), Greater);
                assert_eq!(first.cmp(&Atom::new(Arc::new("bbb".to_string()))), Greater);
                assert_eq!(first.cmp(&Atom::new(Arc::new("bb".to_string()))), Greater);
                assert_eq!(first.cmp(&Atom::new(Arc::new("b".to_string()))), Greater);
                assert_eq!(first.cmp(&Atom::new(Arc::new("".to_string()))), Greater);
            }
        }

        mod eq {
            use super::*;

            #[test]
            fn without_bytes_is_eq() {
                let first = Atom::new(Arc::new("".to_string()));
                let second = Atom::new(Arc::new("".to_string()));

                assert_eq!(first, second)
            }

            #[test]
            fn with_1_byte() {
                let first = Atom::new(Arc::new("a".to_string()));
                let equal = Atom::new(Arc::new("a".to_string()));

                assert_eq!(first, equal);

                let unequal = Atom::new(Arc::new("b".to_string()));

                assert_ne!(first, unequal);
            }

            #[test]
            fn with_2_bytes() {
                let first = Atom::new(Arc::new("aa".to_string()));
                let equal = Atom::new(Arc::new("aa".to_string()));

                assert_eq!(first, equal);

                assert_ne!(first, Atom::new(Arc::new("ba".to_string())));
                assert_ne!(first, Atom::new(Arc::new("ab".to_string())));
            }

            #[test]
            fn with_3_bytes() {
                let first = Atom::new(Arc::new("aaa".to_string()));
                let equal = Atom::new(Arc::new("aaa".to_string()));

                assert_eq!(first, equal);

                assert_ne!(first, Atom::new(Arc::new("baa".to_string())));
                assert_ne!(first, Atom::new(Arc::new("aba".to_string())));
                assert_ne!(first, Atom::new(Arc::new("aab".to_string())));
            }

            #[test]
            fn with_4_bytes() {
                let first = Atom::new(Arc::new("aaaa".to_string()));
                let equal = Atom::new(Arc::new("aaaa".to_string()));

                assert_eq!(first, equal);

                assert_ne!(first, Atom::new(Arc::new("baaa".to_string())));
                assert_ne!(first, Atom::new(Arc::new("abaa".to_string())));
                assert_ne!(first, Atom::new(Arc::new("aaba".to_string())));
                assert_ne!(first, Atom::new(Arc::new("aaab".to_string())));
            }

            #[test]
            fn with_5_bytes() {
                let first = Atom::new(Arc::new("aaaaa".to_string()));
                let equal = Atom::new(Arc::new("aaaaa".to_string()));

                assert_eq!(first, equal);

                assert_ne!(first, Atom::new(Arc::new("baaaa".to_string())));
                assert_ne!(first, Atom::new(Arc::new("abaaa".to_string())));
                assert_ne!(first, Atom::new(Arc::new("aabaa".to_string())));
                assert_ne!(first, Atom::new(Arc::new("aaaba".to_string())));
                assert_ne!(first, Atom::new(Arc::new("aaaab".to_string())));
            }

            #[test]
            fn with_6_bytes() {
                let first = Atom::new(Arc::new("aaaaaa".to_string()));
                let equal = Atom::new(Arc::new("aaaaaa".to_string()));

                assert_eq!(first, equal);

                assert_ne!(first, Atom::new(Arc::new("baaaaa".to_string())));
                assert_ne!(first, Atom::new(Arc::new("abaaaa".to_string())));
                assert_ne!(first, Atom::new(Arc::new("aabaaa".to_string())));
                assert_ne!(first, Atom::new(Arc::new("aaabaa".to_string())));
                assert_ne!(first, Atom::new(Arc::new("aaaaba".to_string())));
                assert_ne!(first, Atom::new(Arc::new("aaaaab".to_string())));
            }

            #[test]
            fn with_7_bytes() {
                let first = Atom::new(Arc::new("aaaaaaa".to_string()));
                let equal = Atom::new(Arc::new("aaaaaaa".to_string()));

                assert_eq!(first, equal);

                assert_ne!(first, Atom::new(Arc::new("baaaaaa".to_string())));
                assert_ne!(first, Atom::new(Arc::new("abaaaaa".to_string())));
                assert_ne!(first, Atom::new(Arc::new("aabaaaa".to_string())));
                assert_ne!(first, Atom::new(Arc::new("aaabaaa".to_string())));
                assert_ne!(first, Atom::new(Arc::new("aaaabaa".to_string())));
                assert_ne!(first, Atom::new(Arc::new("aaaaaba".to_string())));
                assert_ne!(first, Atom::new(Arc::new("aaaaaab".to_string())));
            }

            #[test]
            fn with_8_bytes() {
                let first = Atom::new(Arc::new("aaaaaaaa".to_string()));
                let equal = Atom::new(Arc::new("aaaaaaaa".to_string()));

                assert_eq!(first, equal);

                assert_ne!(first, Atom::new(Arc::new("baaaaaaa".to_string())));
                assert_ne!(first, Atom::new(Arc::new("abaaaaaa".to_string())));
                assert_ne!(first, Atom::new(Arc::new("aabaaaaa".to_string())));
                assert_ne!(first, Atom::new(Arc::new("aaabaaaa".to_string())));
                assert_ne!(first, Atom::new(Arc::new("aaaabaaa".to_string())));
                assert_ne!(first, Atom::new(Arc::new("aaaaabaa".to_string())));
                assert_ne!(first, Atom::new(Arc::new("aaaaaaba".to_string())));
                assert_ne!(first, Atom::new(Arc::new("aaaaaaab".to_string())));
            }

            #[test]
            fn with_9_bytes() {
                let first = Atom::new(Arc::new("aaaaaaaaa".to_string()));
                let equal = Atom::new(Arc::new("aaaaaaaaa".to_string()));

                assert_eq!(first, equal);

                assert_ne!(first, Atom::new(Arc::new("baaaaaaaa".to_string())));
                assert_ne!(first, Atom::new(Arc::new("abaaaaaaa".to_string())));
                assert_ne!(first, Atom::new(Arc::new("aabaaaaaa".to_string())));
                assert_ne!(first, Atom::new(Arc::new("aaabaaaaa".to_string())));
                assert_ne!(first, Atom::new(Arc::new("aaaabaaaa".to_string())));
                assert_ne!(first, Atom::new(Arc::new("aaaaabaaa".to_string())));
                assert_ne!(first, Atom::new(Arc::new("aaaaaabaa".to_string())));
                assert_ne!(first, Atom::new(Arc::new("aaaaaaaba".to_string())));
                assert_ne!(first, Atom::new(Arc::new("aaaaaaaab".to_string())));
            }

            #[test]
            fn with_different_lengths() {
                assert_ne!(
                    Atom::new(Arc::new("a".to_string())),
                    Atom::new(Arc::new("aa".to_string()))
                );
                assert_ne!(
                    Atom::new(Arc::new("aa".to_string())),
                    Atom::new(Arc::new("aaa".to_string()))
                );
                assert_ne!(
                    Atom::new(Arc::new("aaa".to_string())),
                    Atom::new(Arc::new("aaaa".to_string()))
                );
                assert_ne!(
                    Atom::new(Arc::new("aaaa".to_string())),
                    Atom::new(Arc::new("aaaaa".to_string()))
                );
                assert_ne!(
                    Atom::new(Arc::new("aaaaa".to_string())),
                    Atom::new(Arc::new("aaaaaa".to_string()))
                );
                assert_ne!(
                    Atom::new(Arc::new("aaaaaa".to_string())),
                    Atom::new(Arc::new("aaaaaaa".to_string()))
                );
                assert_ne!(
                    Atom::new(Arc::new("aaaaaaa".to_string())),
                    Atom::new(Arc::new("aaaaaaaa".to_string()))
                );
                assert_ne!(
                    Atom::new(Arc::new("aaaaaaaa".to_string())),
                    Atom::new(Arc::new("aaaaaaaaa".to_string()))
                );
                assert_ne!(
                    Atom::new(Arc::new("aaaaaaaaa".to_string())),
                    Atom::new(Arc::new("aaaaaaaaaa".to_string()))
                );
            }
        }
    }

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
