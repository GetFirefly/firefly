use std::cmp::Ordering::{self, *};
use std::hash::{Hash, Hasher};

use im::hashmap::{HashMap, Iter};

use crate::heap::{CloneIntoHeap, Heap};
use crate::integer::Integer;
use crate::term::{Tag, Term};

pub struct Map {
    #[allow(dead_code)]
    header: Term,
    inner: HashMap<Term, Term>,
}

impl Map {
    pub fn new(inner: HashMap<Term, Term>) -> Self {
        Map {
            header: Term {
                tagged: Tag::Map as usize,
            },
            inner,
        }
    }

    pub fn get(&self, key: Term) -> Option<Term> {
        self.inner.get(&key).map(|ref_value| ref_value.clone())
    }

    pub fn iter(&self) -> Iter<Term, Term> {
        self.inner.iter()
    }

    pub fn is_key(&self, key: Term) -> bool {
        self.inner.contains_key(&key)
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn size(&self) -> Integer {
        self.len().into()
    }

    // Private

    fn sorted_keys(&self) -> Vec<Term> {
        let mut key_vec: Vec<Term> = Vec::new();
        key_vec.extend(self.inner.keys());
        key_vec.sort_unstable_by(|key1, key2| key1.cmp(&key2));

        key_vec
    }
}

impl CloneIntoHeap for &'static Map {
    fn clone_into_heap(&self, heap: &Heap) -> &'static Map {
        let mut heap_inner: HashMap<Term, Term> = HashMap::new();

        for (key, value) in &self.inner {
            heap_inner.insert(key.clone_into_heap(heap), value.clone_into_heap(heap));
        }

        heap.im_hash_map_to_map(heap_inner)
    }
}

impl Eq for Map {}

impl Hash for Map {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.inner.hash(state)
    }
}

impl Ord for Map {
    /// > * Maps are compared by size, then by keys in ascending term order,
    /// >   then by values in key order.   In the specific case of maps' key
    /// >   ordering, integers are always considered to be less than floats.
    fn cmp(&self, other: &Self) -> Ordering {
        match self.len().cmp(&other.len()) {
            Equal => {
                let self_key_vec = self.sorted_keys();
                let other_key_vec = other.sorted_keys();

                match self_key_vec.cmp(&other_key_vec) {
                    Equal => {
                        let self_inner = &self.inner;
                        let other_inner = &other.inner;
                        let mut final_ordering = Equal;

                        for key in self_key_vec {
                            match self_inner
                                .get(&key)
                                .unwrap()
                                .cmp(other_inner.get(&key).unwrap())
                            {
                                Equal => continue,
                                partial_ordering => {
                                    final_ordering = partial_ordering;

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

impl PartialEq for Map {
    fn eq(&self, other: &Map) -> bool {
        self.cmp(other) == Equal
    }
}

impl PartialOrd for Map {
    fn partial_cmp(&self, other: &Map) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
