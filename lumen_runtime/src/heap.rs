use std::fmt::{self, Debug};
use std::sync::Arc;

use im::hashmap::HashMap;
use num_bigint::BigInt;

use liblumen_arena::TypedArena;

use crate::binary;
use crate::code::Code;
use crate::float::Float;
use crate::function::Function;
use crate::integer::big;
use crate::list::Cons;
use crate::map::Map;
use crate::process::{identifier, ModuleFunctionArity};
use crate::reference;
use crate::scheduler;
use crate::term::Term;
use crate::tuple::Tuple;

pub struct Heap {
    big_integer_arena: TypedArena<big::Integer>,
    byte_arena: TypedArena<u8>,
    cons_arena: TypedArena<Cons>,
    external_pid_arena: TypedArena<identifier::External>,
    float_arena: TypedArena<Float>,
    function_arena: TypedArena<Function>,
    heap_binary_arena: TypedArena<binary::heap::Binary>,
    map_arena: TypedArena<Map>,
    local_reference_arena: TypedArena<reference::local::Reference>,
    subbinary_arena: TypedArena<binary::sub::Binary>,
    term_arena: TypedArena<Term>,
}

impl Heap {
    pub fn alloc_term_slice(&self, slice: &[Term]) -> *const Term {
        self.term_arena.alloc_slice(slice).as_ptr()
    }

    /// Combines the two `Term`s into a list `Term`.  The list is only a proper list if the `tail`
    /// is a list `Term` (`Term.tag` is `List`) or empty list (`Term.tag` is `EmptyList`).
    pub fn cons(&self, head: Term, tail: Term) -> &'static Cons {
        let pointer = self.cons_arena.alloc(Cons::new(head, tail)) as *const Cons;

        unsafe { &*pointer }
    }

    pub fn external_pid(
        &self,
        node: usize,
        number: usize,
        serial: usize,
    ) -> &'static identifier::External {
        let pointer = self
            .external_pid_arena
            .alloc(identifier::External::new(node, number, serial))
            as *const identifier::External;

        unsafe { &*pointer }
    }

    pub fn f64_to_float(&self, f: f64) -> &'static Float {
        let pointer = self.float_arena.alloc(Float::new(f)) as *const Float;

        unsafe { &*pointer }
    }

    pub fn function(
        &self,
        module_function_arity: Arc<ModuleFunctionArity>,
        code: Code,
    ) -> &'static Function {
        let pointer =
            self.function_arena
                .alloc(Function::new(module_function_arity, code)) as *const Function;

        unsafe { &*pointer }
    }

    pub fn im_hash_map_to_map(&self, hash_map: HashMap<Term, Term>) -> &'static Map {
        let pointer = self.map_arena.alloc(Map::new(hash_map)) as *const Map;

        unsafe { &*pointer }
    }

    pub fn local_reference(
        &self,
        scheduler_id: &scheduler::ID,
        number: reference::local::Number,
    ) -> &'static reference::local::Reference {
        let pointer = self
            .local_reference_arena
            .alloc(reference::local::Reference::new(scheduler_id, number))
            as *const reference::local::Reference;

        unsafe { &*pointer }
    }

    pub fn num_bigint_big_to_big_integer(&self, big_int: BigInt) -> &'static big::Integer {
        let pointer =
            self.big_integer_arena.alloc(big::Integer::new(big_int)) as *const big::Integer;

        unsafe { &*pointer }
    }

    pub fn subbinary(
        &self,
        original: Term,
        byte_offset: usize,
        bit_offset: u8,
        byte_count: usize,
        bit_count: u8,
    ) -> &'static binary::sub::Binary {
        let pointer = self.subbinary_arena.alloc(binary::sub::Binary::new(
            original,
            byte_offset,
            bit_offset,
            byte_count,
            bit_count,
        )) as *const binary::sub::Binary;

        unsafe { &*pointer }
    }

    pub fn slice_to_binary(&self, slice: &[u8]) -> binary::Binary<'static> {
        // TODO use reference counted binaries for bytes.len() > 64
        let heap_binary = self.slice_to_heap_binary(slice);

        binary::Binary::Heap(heap_binary)
    }

    pub fn slice_to_heap_binary(&self, bytes: &[u8]) -> &'static binary::heap::Binary {
        let arena_bytes: &[u8] = if bytes.len() != 0 {
            self.byte_arena.alloc_slice(bytes)
        } else {
            &[]
        };

        let pointer = self
            .heap_binary_arena
            .alloc(binary::heap::Binary::new(arena_bytes))
            as *const binary::heap::Binary;

        unsafe { &*pointer }
    }

    pub fn slice_to_map(&self, slice: &[(Term, Term)]) -> &'static Map {
        let mut inner: HashMap<Term, Term> = HashMap::new();

        for (key, value) in slice {
            inner.insert(key.clone(), value.clone());
        }

        self.im_hash_map_to_map(inner)
    }

    pub fn slice_to_tuple(&self, slice: &[Term]) -> &'static Tuple {
        Tuple::from_slice(slice, &self)
    }
}

pub trait CloneIntoHeap {
    fn clone_into_heap(&self, heap: &Heap) -> Self;
}

impl Debug for Heap {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Heap {{ ... }}")
    }
}

impl Default for Heap {
    fn default() -> Heap {
        Heap {
            big_integer_arena: Default::default(),
            byte_arena: Default::default(),
            cons_arena: Default::default(),
            external_pid_arena: Default::default(),
            float_arena: Default::default(),
            function_arena: Default::default(),
            heap_binary_arena: Default::default(),
            map_arena: Default::default(),
            local_reference_arena: Default::default(),
            subbinary_arena: Default::default(),
            term_arena: Default::default(),
        }
    }
}
