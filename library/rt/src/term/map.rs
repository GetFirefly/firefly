use alloc::alloc::{AllocError, Allocator};
use alloc::format;
use alloc::vec::Vec;
use core::any::TypeId;
use core::fmt;
use core::hash::{Hash, Hasher};

use anyhow::anyhow;
use firefly_alloc::gc::GcBox;
use rpds::HashTrieMap;

pub use rpds::map::hash_trie_map::{Iter, IterKeys, IterValues};

use crate::cmp::ExactEq;

use super::{Cons, Term};

/// This enforces strict equality for map keys
#[derive(Copy, Clone, PartialOrd, Ord, Hash)]
struct MapKey(Term);
impl fmt::Debug for MapKey {
    #[inline]
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        self.0.fmt(f)
    }
}
impl ExactEq for MapKey {
    #[inline]
    fn exact_eq(&self, other: &Self) -> bool {
        self.0.exact_eq(&other.0)
    }

    #[inline]
    fn exact_ne(&self, other: &Self) -> bool {
        self.0.exact_ne(&other.0)
    }
}
impl PartialEq for MapKey {
    fn eq(&self, other: &Self) -> bool {
        self.0.exact_eq(&other.0)
    }

    fn ne(&self, other: &Self) -> bool {
        self.0.exact_ne(&other.0)
    }
}
impl Eq for MapKey {}

#[repr(C)]
#[derive(Clone)]
pub struct Map {
    map: HashTrieMap<MapKey, Term>,
}
impl Map {
    pub const TYPE_ID: TypeId = TypeId::of::<Map>();

    /// Create a new, empty map
    pub fn new() -> Self {
        Self {
            map: HashTrieMap::new(),
        }
    }

    /// Create a new, empty map in the given allocator
    pub fn new_in<A: Allocator>(alloc: A) -> Result<GcBox<Self>, AllocError> {
        GcBox::new_in(Self::new(), alloc)
    }

    /// Create a map, initialized with key/value pairs from the given iterator
    pub fn new_from_iter<I: Iterator<Item = (Term, Term)>>(items: I) -> Self {
        let mut map = HashTrieMap::new();
        for (k, v) in items {
            map.insert_mut(MapKey(k), v);
        }
        Self { map }
    }

    /// Create a map with the given allocator, initialized with key/value pairs from the given
    /// iterator
    pub fn new_from_iter_in<A: Allocator, I: Iterator<Item = (Term, Term)>>(
        items: I,
        alloc: A,
    ) -> Result<GcBox<Self>, AllocError> {
        GcBox::new_in(Self::new_from_iter(items), alloc)
    }

    /// Create a map, initialized with key/value pairs from the given list term
    pub fn from_keyword_list(list: &Cons) -> anyhow::Result<Self> {
        let mut map = Self::new();
        for result in list.iter() {
            match result {
                Ok(term) => match term {
                    Term::Tuple(ptr) => {
                        let pair = unsafe { ptr.as_ref() };
                        match pair.len() {
                            2 => {
                                let key = unsafe { pair.get_unchecked(0) };
                                let value = unsafe { pair.get_unchecked(1) };
                                map.insert_mut(key, value);
                            }
                            len => {
                                return Err(anyhow!("expected tuple of arity 2, but got {}", len))
                            }
                        }
                    }
                    other => return Err(anyhow!("expected tuple, but got {}", &other)),
                },
                Err(_improper) => return Err(anyhow!("list is improper")),
            }
        }

        Ok(map)
    }

    /// Create a map with the given allocator, initialized with key/value pairs from the given list
    /// term
    pub fn from_keyword_list_in<A: Allocator>(
        list: &Cons,
        alloc: A,
    ) -> anyhow::Result<GcBox<Self>> {
        Ok(GcBox::new_in(Self::from_keyword_list(list)?, alloc)?)
    }

    /// Returns the number of keys in this map
    #[inline]
    pub fn size(&self) -> usize {
        self.map.size()
    }

    /// Returns true if this map is empty
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Returns true if this map contains `key`, otherwise false.
    #[inline]
    pub fn contains_key<K: Into<Term>>(&self, key: K) -> bool {
        let key = MapKey(key.into());
        self.map.contains_key(&key)
    }

    /// Returns the value associated with `key` in this map
    #[inline]
    pub fn get<K: Into<Term>>(&self, key: K) -> Option<Term> {
        let key = MapKey(key.into());
        self.map.get(&key).copied()
    }

    /// Takes a key out of the map, returning both the key and the new map if the key was present.
    pub fn take<K: Into<Term>>(&self, key: K) -> Option<(Term, Self)> {
        let key = MapKey(key.into());
        match self.map.get(&key).copied() {
            None => None,
            Some(value) => Some((
                value,
                Self {
                    map: self.map.remove(&key),
                },
            )),
        }
    }

    /// Inserts the `value` in the map under `key`, replacing any existing value
    /// and returning a new map
    pub fn insert(&self, key: Term, value: Term) -> Self {
        Self {
            map: self.map.insert(MapKey(key), value),
        }
    }

    /// Like `insert`, except if `key` already exists in the map, `None` is returned.
    pub fn insert_new(&self, key: Term, value: Term) -> Option<Self> {
        let mk = MapKey(key);
        if self.map.contains_key(&mk) {
            None
        } else {
            Some(self.insert(mk.0, value))
        }
    }

    /// Like `insert`, but mutates the map directly.
    ///
    /// This should be used when it is known that the map is not referenced
    /// anywhere else (e.g. such as constructing a map from a list of values)
    pub fn insert_mut(&mut self, key: Term, value: Term) {
        self.map.insert_mut(MapKey(key), value)
    }

    /// Removes `key` from this map, returning the modified map
    #[inline]
    pub fn remove<K: Into<Term>>(&self, key: K) -> Self {
        let key = MapKey(key.into());
        Self {
            map: self.map.remove(&key),
        }
    }

    /// Like `remove`, but mutates the map directly.
    ///
    /// Returns true if the key was present and removed, otherwise false.
    #[inline]
    pub fn remove_mut<K: Into<Term>>(&mut self, key: K) -> bool {
        let key = MapKey(key.into());
        self.map.remove_mut(&key)
    }

    pub fn iter(&self) -> impl Iterator<Item = (&Term, &Term)> {
        self.map.iter().map(|(k, v)| (&k.0, v))
    }

    pub fn keys(&self) -> impl Iterator<Item = &Term> {
        self.map.keys().map(|k| &k.0)
    }

    pub fn values(&self) -> impl Iterator<Item = &Term> {
        self.map.values()
    }

    fn sorted_map_keys(&self) -> Vec<MapKey> {
        let mut keys = self.map.keys().copied().collect::<Vec<_>>();
        keys.sort_unstable();
        keys
    }
}
impl fmt::Debug for Map {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("#{")?;
        for (i, (key, value)) in self.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{:?} => {:?}", key, value)?;
        }
        f.write_str("}")
    }
}
impl fmt::Display for Map {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.write_str("#{")?;
        for (i, (key, value)) in self.iter().enumerate() {
            if i > 0 {
                f.write_str(", ")?;
            }
            write!(f, "{} => {}", key, value)?;
        }
        f.write_str("}")
    }
}
impl Eq for Map {}
impl PartialEq for Map {
    fn eq(&self, other: &Self) -> bool {
        self.map.eq(&other.map)
    }
}
impl ExactEq for Map {
    fn exact_eq(&self, other: &Self) -> bool {
        if self.size() != other.size() {
            return false;
        }
        for (k, v) in self.map.iter() {
            match other.map.get(k) {
                None => return false,
                Some(v2) => {
                    if v.exact_eq(v2) {
                        continue;
                    } else {
                        return false;
                    }
                }
            }
        }
        true
    }
}
impl Hash for Map {
    fn hash<H: Hasher>(&self, state: &mut H) {
        // Hash in sort order to keep the hash stable
        let mut sorted = self.sorted_map_keys();
        for key in sorted.drain(..) {
            let value = self.map.get(&key);
            key.hash(state);
            value.hash(state);
        }
    }
}
impl PartialOrd for Map {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Map {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        use std::cmp::Ordering;

        // Maps are ordered as follows:
        //
        // * First by size, with smaller maps being "less" than larger maps
        // * If the same size, then by keys in term order
        // * If the keys are the same, then by values in key order

        // While comparing vecs will properly order two sets of sorted keys correctly,
        // it incurs an allocation when we do so. To avoid that allocation unless necessary,
        // we first compare the map sizes directly, which is redundant but much more efficient
        // when the maps are not the same size
        match self.size().cmp(&other.size()) {
            Ordering::Equal => {
                let m1 = self.sorted_map_keys();
                let m2 = other.sorted_map_keys();

                match m1.cmp(&m2) {
                    Ordering::Equal => {
                        for k in &m1 {
                            match self.map.get(k).unwrap().cmp(other.map.get(k).unwrap()) {
                                Ordering::Equal => continue,
                                other => return other,
                            }
                        }
                        Ordering::Equal
                    }
                    other => other,
                }
            }
            other => other,
        }
    }
}
