use im_rc::hashmap::HashMap;

use crate::process::Process;
use crate::term::{Tag, Term};

pub struct Map {
    #[allow(dead_code)]
    header: Term,
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
