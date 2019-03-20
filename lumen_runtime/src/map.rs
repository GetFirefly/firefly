use std::cmp::Ordering;

use im_rc::hashmap::HashMap;

use crate::process::{OrderInProcess, Process};
use crate::term::{Tag, Term};

pub struct Map {
    #[allow(dead_code)]
    header: Term,
    #[allow(dead_code)]
    inner: HashMap<Term, Term>,
}

impl Map {
    pub fn from_slice(slice: &[(Term, Term)], process: &mut Process) -> &'static Self {
        let mut inner: HashMap<Term, Term> = HashMap::new();

        for (key, value) in slice {
            inner.insert(key.clone(), value.clone());
        }

        let pointer = process.map_arena.alloc(Self::new(inner)) as *const Self;

        unsafe { &*pointer }
    }

    fn new(inner: HashMap<Term, Term>) -> Self {
        Map {
            header: Term {
                tagged: Tag::Map as usize,
            },
            inner,
        }
    }
}

impl OrderInProcess for Map {
    /// > * Maps are compared by size, then by keys in ascending term order,
    /// >   then by values in key order.   In the specific case of maps' key
    /// >   ordering, integers are always considered to be less than floats.
    fn cmp_in_process(&self, other: &Map, process: &Process) -> Ordering {
        let self_inner = &self.inner;
        let other_inner = &other.inner;

        match self_inner.len().cmp(&other_inner.len()) {
            Ordering::Equal => {
                let mut self_key_vec: Vec<Term> = Vec::new();
                self_key_vec.extend(self_inner.keys());
                self_key_vec.sort_unstable_by(|key1, key2| key1.cmp_in_process(key2, process));

                let mut other_key_vec: Vec<Term> = Vec::new();
                other_key_vec.extend(other_inner.keys());
                other_key_vec.sort_unstable_by(|key1, key2| key1.cmp_in_process(key2, process));

                match self_key_vec.cmp_in_process(&other_key_vec, process) {
                    Ordering::Equal => {
                        let mut final_ordering = Ordering::Equal;

                        for key in self_key_vec {
                            match self_inner
                                .get(&key)
                                .unwrap()
                                .cmp_in_process(other_inner.get(&key).unwrap(), &process)
                            {
                                Ordering::Equal => continue,
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
