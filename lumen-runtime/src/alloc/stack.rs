/// Stack instructions needed:
///
/// allocate(need_stack, live) { $AH($need_stack, 0, $live) }
/// allocate_heap(need_stack, need_heap, live) { $AH($need_stack, 0, $live); make_blank($Y) }
/// allocate_init(need_stack, live, y) { $AH($need_stack, 0, $live); make_blank($Y) }
/// allocate_zero(need_stack, live) {
///   Eterm* ptr;
///   int i = $need_stack;
///   $AH(i, 0, $live);
///   for (ptr = E + i; ptr > E; ptr--) {
///     make_blank(*ptr)
///   }
/// }
/// allocate_heap_zero(need_stack, need_heap, live) { same as above, but $AH(i, $need_heap, $live) }
///
/// deallocate(deallocate) {
///   SET_CP(c_p, (BeamInstr *) cp_val(*E));
///   E = ADD_BYTE_OFFSET(E, $deallocate)
/// }
const DEFAULT_STACK_SIZE = 16;

pub enum AllocType {
    Default
}

pub struct Stack {
    start: *const Eterm,
    sp: *const Eterm,
    end: *const Eterm,
    edefault: *const Eterm,
    alloc_type: AllocType,
    stack: Vec<Eterm>,
}
impl Stack {
    pub fn new() -> Stack {
        let stack: Vec<Eterm> = Vec::with_capacity(DEFAULT_STACK_SIZE);
        Stack {
            start: &start as *const Eterm,
            sp: &start as *const Eterm,
            end: (&start as *const Eterm) + 16,
            default: &start as *const Eterm,
            alloc_type: AllocType::Default,
            stack
        }
    }

    pub fn grow(&mut self, need: usize) {

    }

    pub fn is_empty(&self) {

    }

    // Do not free the stack after this, it may have pointers into what was saved in 'dst'
    pub fn save(&mut self, dst: &Stack) {
        if self.start == default {
            wsz: UWord = self.count();
            dst.start = erts_alloc(s.alloc_type, DEFAULT_STACK_SIZE * sizeof(Eterm));
            memcpy(dst.start, s.start, wsz * sizeof(Eterm));
            dst.sp = dst.start + wsz;
            dst.end = dst.start + DEFAULT_STACK_SIZE;
            dst.alloc_type = s.alloc_type;
        } else {
            *dst = s;
        }
    }

    pub fn clear_saved(&mut self) {
        self.start = std::ptr::null;
    }

    // Use on empty stack, only the allocator can be changed before this, the src stack is reset to null
    pub fn restore(&mut self, src: &Stack) {
        assert(self.start == DEFAULT_STACK);
        self = *src;
        src.start = std::ptr::null;
        assert(self.sp >= self.start);
        assert(self.sp <= self.end);
    }

    pub fn push(&mut self, t: Eterm) {
        if self.sp == sself.end {
            self.grow(1);
        }
        self.sp++ = t;
    }

    pub fn push2(&mut self, a: Eterm, b: Eterm) {
        if self.sp > self.end - 2 {
            self.grow(2);
        }
        *self.sp++ = x;
        *self.sp++ = y;
    }

    pub fn reserve(&mut self, count: usize) {
        if self.sp > self.end - count {
            self.grow(count);
        }
    }

    pub fn fast_push(&mut self, t: Eterm) {
        assert(self.sp < self.end):
        *self.sp++ = x;
    }

    pub fn count(&self) {
        self.sp - self.start;
    }

    pub fn is_empty(&self) {
        self.sp == self.start
    }

    pub fn pop(&mut self) {
        *(--self.sp)
    }

    pub fn destroy(&mut self) {
        super::alloc::free(self.alloc_type, self.start);
    }

    pub fn destroy_saved(&mut self) {
        if self.start {
            erts_free(self.alloc_type, self.start);
            self.start = std::ptr::null;
        }
    }
}
impl Drop for Stack {
    fn
}
