use core::cmp;
use core::convert::{TryFrom, TryInto};
use core::fmt::{self, Debug, Display};
use core::hash::{Hash, Hasher};
use core::mem;
use core::ptr;

use alloc::vec::Vec;

use hashbrown::HashMap;

use crate::erts::exception::system::Alloc;
use crate::erts::process::HeapAlloc;
use crate::erts::term::{AsTerm, Boxed, Term, TypeError, TypedTerm};
use crate::erts::to_word_size;

#[derive(Clone)]
#[repr(C)]
pub struct Map {
    header: Term,
    value: HashMap<Term, Term>,
}

impl Map {
    pub(in crate::erts) fn from_hash_map(value: HashMap<Term, Term>) -> Self {
        let arity = to_word_size(mem::size_of_val(&value));
        let header = Term::make_header(arity, Term::FLAG_MAP);

        Self { header, value }
    }

    pub(in crate::erts) fn from_slice(slice: &[(Term, Term)]) -> Self {
        let mut value: HashMap<Term, Term> = HashMap::with_capacity(slice.len());

        for (entry_key, entry_value) in slice {
            value.insert(*entry_key, *entry_value);
        }

        Self::from_hash_map(value)
    }

    pub fn get(&self, key: Term) -> Option<Term> {
        self.value.get(&key).copied()
    }

    pub fn is_key(&self, key: Term) -> bool {
        self.value.contains_key(&key)
    }

    pub fn keys(&self) -> Vec<Term> {
        self.value.keys().into_iter().copied().collect()
    }

    pub fn values(&self) -> Vec<Term> {
        self.value.values().into_iter().copied().collect()
    }

    pub fn len(&self) -> usize {
        self.value.len()
    }

    // Private

    fn sorted_keys(&self) -> Vec<Term> {
        let mut key_vec: Vec<Term> = Vec::new();
        key_vec.extend(self.value.keys());
        key_vec.sort_unstable_by(|key1, key2| key1.cmp(&key2));

        key_vec
    }
}

impl AsRef<HashMap<Term, Term>> for Boxed<Map> {
    fn as_ref(&self) -> &HashMap<Term, Term> {
        &self.value
    }
}

impl AsRef<HashMap<Term, Term>> for Map {
    fn as_ref(&self) -> &HashMap<Term, Term> {
        &self.value
    }
}

unsafe impl AsTerm for Map {
    unsafe fn as_term(&self) -> Term {
        Term::make_boxed(self)
    }
}

impl crate::borrow::CloneToProcess for Map {
    fn clone_to_heap<A: HeapAlloc>(&self, heap: &mut A) -> Result<Term, Alloc> {
        let size = mem::size_of_val(self);
        let size_in_words = to_word_size(size);
        let ptr = unsafe { heap.alloc(size_in_words)?.as_ptr() };

        let self_value = &self.value;
        let mut heap_value = HashMap::with_capacity(self_value.len());

        for (entry_key, entry_value) in self_value {
            let heap_entry_key = entry_key.clone_to_heap(heap)?;
            let heap_entry_value = entry_value.clone_to_heap(heap)?;
            heap_value.insert(heap_entry_key, heap_entry_value);
        }

        // Clone to ensure `value` remains valid if caller is dropped
        let heap_self = Self {
            header: self.header,
            value: heap_value,
        };

        unsafe {
            ptr::copy_nonoverlapping(&heap_self as *const _ as *const u8, ptr as *mut u8, size);
        }

        mem::forget(heap_self);

        Ok(Term::make_boxed(ptr as *mut Self))
    }
}

impl Debug for Map {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("Map")
            .field("header", &format_args!("{:#b}", &self.header.as_usize()))
            .field("value", &self.value)
            .finish()
    }
}

impl Display for Map {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self.value)
    }
}

impl Eq for Map {}

impl Hash for Map {
    fn hash<H: Hasher>(&self, state: &mut H) {
        for key in self.sorted_keys() {
            let value = self.value[&key];

            key.hash(state);
            value.hash(state);
        }
    }
}

impl PartialEq for Map {
    fn eq(&self, other: &Map) -> bool {
        self.value.eq(&other.value)
    }
}

impl PartialOrd for Map {
    fn partial_cmp(&self, other: &Map) -> Option<cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Map {
    /// > * Maps are compared by size, then by keys in ascending term order,
    /// >   then by values in key order.   In the specific case of maps' key
    /// >   ordering, integers are always considered to be less than floats.
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        match self.len().cmp(&other.len()) {
            cmp::Ordering::Equal => {
                let self_key_vec = self.sorted_keys();
                let other_key_vec = other.sorted_keys();

                match self_key_vec.cmp(&other_key_vec) {
                    cmp::Ordering::Equal => {
                        let self_value = &self.value;
                        let other_value = &other.value;
                        let mut final_ordering = cmp::Ordering::Equal;

                        for key in self_key_vec {
                            match self_value
                                .get(&key)
                                .unwrap()
                                .cmp(other_value.get(&key).unwrap())
                            {
                                cmp::Ordering::Equal => continue,
                                ordering => {
                                    final_ordering = ordering;

                                    break;
                                }
                            }
                        }

                        final_ordering
                    }
                    ordering => ordering,
                }
            }
            ordering => ordering,
        }
    }
}

impl TryFrom<Term> for Boxed<Map> {
    type Error = TypeError;

    fn try_from(term: Term) -> Result<Boxed<Map>, Self::Error> {
        term.to_typed_term().unwrap().try_into()
    }
}

impl TryFrom<TypedTerm> for Boxed<Map> {
    type Error = TypeError;

    fn try_from(typed_term: TypedTerm) -> Result<Boxed<Map>, Self::Error> {
        match typed_term {
            TypedTerm::Boxed(boxed_map) => boxed_map.to_typed_term().unwrap().try_into(),
            TypedTerm::Map(map) => Ok(map),
            _ => Err(TypeError),
        }
    }
}
