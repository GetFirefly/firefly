use std::collections::HashSet;

use rpds::RedBlackTreeMap;

use liblumen_intern::Symbol;

use super::Var;

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Name {
    Var(Symbol),
    Fun(Symbol, usize),
}
impl From<&Var> for Name {
    fn from(v: &Var) -> Self {
        match v.arity {
            None => Self::Var(v.name.name),
            Some(i) => Self::Fun(v.name.name, i),
        }
    }
}
impl Name {
    pub fn symbol(&self) -> Symbol {
        match self {
            Self::Var(s) | Self::Fun(s, _) => *s,
        }
    }
}

/// A many-to-one bi-directional map, used to allow us
/// to rename all references to a variable without having
/// to scan through all of them, potentially causing compile
/// times to explode
#[derive(Clone, Default)]
pub struct BiMap {
    // key => value
    map: RedBlackTreeMap<Name, Name>,
    // value => keys
    inv: RedBlackTreeMap<Name, HashSet<Name>>,
}
impl BiMap {
    pub fn get_vsub(&self, name: Symbol) -> Symbol {
        let key = Name::Var(name);
        self.get(key).map(|v| v.symbol()).unwrap_or(name)
    }

    pub fn get_fsub(&self, name: Symbol, arity: usize) -> Symbol {
        let key = Name::Fun(name, arity);
        self.get(key).map(|v| v.symbol()).unwrap_or(name)
    }

    pub fn get(&self, key: Name) -> Option<Name> {
        self.map.get(&key).copied()
    }

    pub fn set_vsub(&self, var: Symbol, sub: Symbol) -> Self {
        if var == sub {
            return self.clone();
        }
        let name = Name::Var(var);
        let value = Name::Var(sub);
        self.set(name, value)
    }

    pub fn set_fsub(&self, name: Symbol, arity: usize, value: Name) -> Self {
        let name = Name::Fun(name, arity);
        if name == value {
            return self.clone();
        }
        self.set(name, value)
    }

    pub fn subst_vsub(&self, key: Name, value: Name) -> Self {
        self.rename(key, value)
    }

    /// Maps key to value without touching existing references to `key`
    pub fn set(&self, key: Name, value: Name) -> Self {
        let inv = self.update_inv_lookup(key, value);
        let map = self.map.insert(key, value);
        Self { map, inv }
    }

    fn update_inv_lookup(&self, key: Name, value: Name) -> RedBlackTreeMap<Name, HashSet<Name>> {
        let mut inv = self.cleanup_inv_lookup(key);
        match inv.get_mut(&value) {
            None => {
                let mut set = HashSet::new();
                set.insert(key);
                inv.insert_mut(value, set);
            }
            Some(set) => {
                set.insert(key);
            }
        }
        inv
    }

    fn cleanup_inv_lookup(&self, key: Name) -> RedBlackTreeMap<Name, HashSet<Name>> {
        let mut inv = self.inv.clone();
        match self.map.get(&key) {
            None => inv,
            Some(old) => {
                let remove = match inv.get_mut(old) {
                    None => false,
                    Some(oinv) => {
                        oinv.remove(&key);
                        oinv.is_empty()
                    }
                };
                if remove {
                    inv.remove_mut(old);
                }
                inv
            }
        }
    }

    /// Maps `key` to `value`, and replaces all existing references to `key` with `value`
    fn rename(&self, key: Name, value: Name) -> Self {
        let keys = self.inv.get(&key).map(|keys| keys.clone());
        match keys {
            None => self.set(key, value),
            Some(mut keys) => {
                let mut inv = self.inv.clone();
                let mut map = self.map.clone();
                map.insert_mut(key, value);
                for key in keys.iter().copied() {
                    map.insert_mut(key, value);
                }
                keys.insert(key);
                inv.remove_mut(&key);
                inv.insert_mut(value, keys);
                Self { inv, map }
            }
        }
    }
}
