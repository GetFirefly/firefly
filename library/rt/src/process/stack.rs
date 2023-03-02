use alloc::{vec, vec::Vec};
use core::assert_matches::assert_matches;
use core::cmp;

use crate::term::OpaqueTerm;

/// We require at least 256 words of stack space in a minimal stack
const MIN_STACK_SIZE: usize = 256;
/// The number of reserved registers in each frame on the stack
///
/// The two registers are used for the return value, and the continuation pointer/return address
pub(super) const RESERVED_REGISTERS: usize = 2;

/// The reserved register which stores the return value for the current function
pub const RETURN_REG: Register = 0;
/// The reserved register which stores the continuation pointer/return address for a call frame on
/// the stack
pub const CP_REG: Register = 1;
/// The register holding the first argument value for the current function, only valid to use
/// when the current function has arguments. All additional function arguments follow after this
/// register consecutively
pub const ARG0_REG: Register = 2;

/// Represents a single call frame in the call stack
#[derive(Copy, Clone)]
pub struct StackFrame {
    /// The offset or pointer of the instruction to return control to on exit
    pub ret: usize,
    /// The frame pointer for this frame, i.e. points to the bottom of the stack space reserved for
    /// this frame
    pub fp: usize,
    /// The current stack pointer value in this frame
    pub sp: usize,
}

/// A register is a frame-relative offset to a stack slot in the current frame
///
/// Relative addressing of the stack uses registers, rather than absolute addresses, due to how code
/// is generated.
pub type Register = firefly_bytecode::Register;

/// A mark captures the stack of the stack and frame pointers at a specific point in time,
/// as well as the type of event which produced the mark.
///
/// Currently, there are two occasions where marks are used:
///
/// * When a new call frame is created for a function call
/// * When a catch handler is installed
///
/// When a mark is applied, it restores the captured pointer values.
///
/// To keep this structure compact, we make the stack pointer an offset relative to the frame
/// pointer, which allows the Rust compiler to keep the size of this struct to the equivalent
/// of two `usize` values.
///
/// NOTE: The maximum stack frame size is defined by `u16::MAX` as a result of this structure
#[derive(Debug, Copy, Clone)]
#[repr(u8)]
pub enum Mark {
    /// For a call, we simply mark the frame pointer and offset of the stack pointer
    Call { fp: usize, sp: u16 },
    /// For a catch, we also record the offset of the stack slot where the catch pointer is stored
    Catch { fp: usize, sp: u16, cp: usize },
}
impl Mark {
    #[inline]
    pub fn call(fp: usize, sp: usize) -> Self {
        let sp = (sp - fp).try_into().unwrap();
        Self::Call { fp, sp }
    }

    #[inline]
    pub fn catch(fp: usize, sp: usize, cp: usize) -> Self {
        let sp = (sp - fp).try_into().unwrap();
        Self::Catch { fp, sp, cp }
    }

    #[inline]
    pub fn is_catch(&self) -> bool {
        match self {
            Self::Catch { .. } => true,
            _ => false,
        }
    }
}
impl Default for Mark {
    #[inline]
    fn default() -> Self {
        Self::Call {
            fp: 0,
            // There are reserved slots at the beginning of each stack frame,
            // and as a result the stack pointer always begins after those slots
            sp: RESERVED_REGISTERS as u16,
        }
    }
}

#[derive(Copy, Clone)]
pub struct StackOverflowError;

/// Represents the stack memory of a [`Process`].
///
/// When a process is initialized, a default stack of [`MIN_STACK_SIZE`] words is allocated, and
/// this grows dynamically as needed behind the scenes. Reallocations due to stack growth are
/// performed in such a way that many small pushes near the end of the stack will not result in
/// repeated reallocation. Instead, the stack grows in size rapidly to keep reallocations to a
/// minimum, at the expense of some wasted memory.
///
/// The stack pointer starts at 0, and grows upwards - unlike traditional stack memory whose
/// addresses normally grow downwards. The stack pointer points to the next free slot on the stack,
/// _not_ the most recently pushed element.
///
/// # Features
///
/// * The stack can be manipulated granularly using [`push`]/[`pop`] or in bulk using
///   [`alloca`]/[`dealloc`]
/// * The stack can be marked to indicate a point in the stack to which it should be reset when the
///   mark is consumed
///
/// # Safety
///
/// The stack may only contain immediates and pointers to heap memory, it is not permitted to
/// allocate on the stack as if it was the heap.
///
/// # Layout
///
/// Call frames are laid out as follows, where the address on the left is relative to the frame
/// pointer (fp):
///
/// ```text,ignore
///     N+1 | NONE           <- sp
///     N   | ARGN
///     .   |
///     2   | ARG0
///     1   | RETURN_ADDRESS
///     0   | RETURN_VALUE   <- fp
/// ```
///
/// The first call frame in the call stack has a null return address, which is what signals that
/// control has reached the bottom of the call stack, and that the currently executing process
/// should exit normally.
///
/// When a non-tail call is being made, the caller will have allocated a stack slot for the return
/// value in its call frame. The caller will ensure that immediately following that slot is room for
/// the return address and all of the callee arguments, and will set the frame pointer to the slot
/// containing the return value, and the stack pointer to the first slot following the last
/// argument. At this point control will be transferred to the callee.
///
/// The callee, when returning to the caller, will move its return value into the first slot at the
/// base of the frame, get the return address from the second slot of the frame, and then restore
/// the stack mark that was created by the caller, restoring the stack and frame pointers to their
/// prior positions.
///
/// Tail calls reuse the caller's frame, so arguments to the callee must be moved to their
/// appropriate position on the stack so that they are in the expected slots when control is
/// transferred to the callee. This may require additional stack slots in order to non-destructively
/// shuffle values around.
///
/// Exception handlers also use stack marks to record the state of the stack which should be
/// restored before transfering control to the handler. The stack slot containing the continuation
/// pointer for the handler is stored as part of the mark, and is used to recover the pointer
/// quickly when unwinding in the presence of an exception.
pub struct ProcessStack {
    /// The actual stack memory
    ///
    /// The capacity of the stack is the total allocated size (in words) of the stack,
    /// and its length defines the total live size (in words).
    pub(super) stack: Vec<OpaqueTerm>,
    /// A stack of stack marks (see [`Mark`]).
    ///
    /// When used as a call frame mark, conceptually each mark represents the beginning of a new
    /// call frame, but the actual state captured in the mark is actually the state of the
    /// caller's call frame. Thus, restoring the mark restores the caller's frame.
    ///
    /// When used as a catch handler mark, each mark represents the state of the stack which must
    /// be restored for the handler, as well as the location of the continuation pointer on the
    /// stack.
    marks: Vec<Mark>,
    /// The current stack pointer
    pub(super) sp: usize,
    /// The current frame pointer
    fp: usize,
}
impl Default for ProcessStack {
    fn default() -> ProcessStack {
        Self {
            stack: vec![OpaqueTerm::NONE; MIN_STACK_SIZE],
            marks: vec![],
            sp: RESERVED_REGISTERS,
            fp: 0,
        }
    }
}
impl ProcessStack {
    /// Sets the `nocatch` effect on this stack by clearing all of the stack marks
    /// which catch an exception. This leaves the stack intact, but without any catch
    /// handlers installed
    pub fn nocatch(&mut self) {
        self.marks.retain(|mark| match mark {
            Mark::Catch { .. } => false,
            _ => true,
        });
    }

    /// Returns true if there is at least one catch handler on the stack
    pub fn catches(&self) -> bool {
        self.marks.iter().any(|mark| mark.is_catch())
    }

    /// The size (in words) of the stack which is in use
    ///
    /// This does not reflect the total allocated size of the stack.
    #[inline]
    pub fn size(&self) -> usize {
        self.sp
    }

    /// The size (in words) of the stack which is not in use
    #[inline]
    pub fn available(&self) -> usize {
        self.capacity() - self.size()
    }

    /// The total allocated size (in words) of the stack
    #[inline]
    pub fn capacity(&self) -> usize {
        self.stack.len()
    }

    /// The current position of the top of the stack
    ///
    /// This points to the next free slot on the stack, _not_ the most recently pushed value.
    #[inline(always)]
    pub fn stack_pointer(&self) -> usize {
        self.sp
    }

    /// Get the current frame pointer
    #[inline(always)]
    pub fn frame_pointer(&self) -> usize {
        self.fp
    }

    /// Mark the start of a new frame on the call stack.
    ///
    /// * The new frame will begin at `ret`, which is also the return value register
    /// * The stack pointer of the new frame will begin two slots after `ret`, reserving
    /// space for the return address register.
    ///
    /// # SAFETY
    ///
    /// It is expected that callees will allocate space necessary for its frame upon entry
    pub fn push_frame(&mut self, ret: Register) {
        // Calculate the absolute offset of `ret` from the bottom of the stack
        let ret = self.fp + ret as usize;
        // Calculate the new stack pointer, which must begin after the return address slot
        let sp = ret + RESERVED_REGISTERS;

        // If the new stack pointer would overflow the stack, allocate more memory
        if sp >= self.stack.len() {
            unsafe { self.alloca(1) };
        }

        // Save the current stack state for when this frame is popped
        self.marks.push(Mark::call(self.fp, self.sp));

        // Set the stack and frame pointers to their appropriate starting locations
        self.fp = ret;
        self.sp = sp;
    }

    /// Resets the stack pointer of the current frame to the base of the frame, and
    /// clears all catch marks in the current frame.
    pub fn reset_frame(&mut self) {
        self.sp = self.fp + RESERVED_REGISTERS;
        loop {
            match self.marks.last() {
                None => break,
                Some(Mark::Call { .. }) => break,
                Some(Mark::Catch { .. }) => {
                    self.marks.pop();
                }
            }
        }
    }

    /// Pop the most recent frame from the call stack.
    ///
    /// Returns the continuation pointer/return address, if one is present.
    ///
    /// After this function returns, the stack and frame pointers will be in their previous
    /// locations, and the return value will be available in the slot originally requested by
    /// `push_frame`.
    pub fn pop_frame(&mut self) -> Option<usize> {
        // We may have catch marks on the stack, so skip over them to the nearest call frame
        while let Some(mark) = self.marks.pop() {
            match mark {
                Mark::Call { fp, sp } => {
                    let return_addr = self.load(CP_REG).as_code();
                    self.fp = fp;
                    self.sp = fp + (sp as usize);
                    // The zero pointer is used to signal an invalid address,
                    // so if we pop a frame and encounter it, we should return
                    // None even if there are more frames available - it is up
                    // to the caller to decide how to handle this.
                    if return_addr == 0 {
                        return None;
                    } else {
                        return Some(return_addr);
                    }
                }
                _ => continue,
            }
        }

        self.sp = RESERVED_REGISTERS;
        self.fp = 0;
        None
    }

    /// Pushes a new mark which will be restored if an exception is raised
    ///
    /// The given instruction pointer is where control will be transferred on an exception.
    #[inline]
    pub fn enter_catch(&mut self, cp: usize) {
        self.marks.push(Mark::catch(self.fp, self.sp, cp));
    }

    /// Pops a mark pushed by a previous call to `enter_catch`
    ///
    /// This is intended to be called during non-exceptional control flow.
    ///
    /// When an exception is raised, the stack is automatically unwound to the nearest catch
    /// handler, restoring the mark for that handler in the process. Calling `exit_catch` is
    /// only necessary when explicitly exiting the protected region of the catch handler.
    #[inline]
    pub fn exit_catch(&mut self) {
        assert_matches!(self.marks.pop(), Some(Mark::Catch { .. }));
    }

    /// Copy a value from register `src` to register `dest`
    #[inline]
    pub fn copy(&mut self, src: Register, dest: Register) {
        self.stack[self.fp + dest as usize] = self.stack[self.fp + src as usize];
    }

    /// Load the value from the given register
    #[inline(always)]
    pub fn load(&self, reg: Register) -> OpaqueTerm {
        self.stack[self.fp + reg as usize]
    }

    /// Store a value in the given register
    #[inline(always)]
    pub fn store(&mut self, reg: Register, value: OpaqueTerm) {
        self.stack[self.fp + reg as usize] = value;
    }

    /// Select a slice of `n` registers, beginning with register `start`
    ///
    /// NOTE: This function will panic if the range is out of bounds
    #[inline]
    pub fn select_registers(&self, start: Register, n: usize) -> &[OpaqueTerm] {
        let start = self.fp + start as usize;
        &self.stack[start..(start + n)]
    }

    /// Writes `NONE` to all of the stack slots in the current frame starting from `start` to the
    /// current top of the stack.
    ///
    /// If `start` begins past the top of the stack, this function will panic.
    pub fn zero(&mut self, start: Register) {
        let start = self.fp + (start as usize);
        assert!(start <= self.sp);
        let to_fill = &mut self.stack[start..self.sp];
        to_fill.fill(OpaqueTerm::NONE);
    }

    /// Dynamically allocate stack space in the current frame
    ///
    /// # SAFETY
    ///
    /// This function does not reallocate if there is sufficient capacity available, and
    /// as such, if the slots allocated were previously allocated/deallocated, they may contain
    /// terms which are no longer valid (i.e. a boxed value whose pointee was moved due to GC, but
    /// the box was not updated because it was not live at the time).
    ///
    /// Callers should prefer `alloca_zeroed` if they can't guarantee that the newly allocated slots
    /// will be filled with valid data prior to the next GC.
    pub unsafe fn alloca(&mut self, size: usize) {
        if self.available() < size {
            self.alloca_slow(size);
        }

        self.sp += size;
    }

    /// Dynamically allocate stack space in the current frame, initializing all the new slots as
    /// `OpaqueTerm::NONE`
    pub fn alloca_zeroed(&mut self, size: usize) {
        // Save the stack pointer prior to extension
        let start = self.sp;

        unsafe {
            self.alloca(size);
        }

        // Explicitly initialize all of the new stack slots to NONE
        let alloced = &mut self.stack[start..self.sp];
        alloced.fill(OpaqueTerm::NONE);
    }

    #[inline(never)]
    fn alloca_slow(&mut self, size: usize) {
        // Reserve additional capacity at least twice the size of the current capacity
        let capacity = self.stack.capacity();
        self.stack.reserve(cmp::max(capacity * 2, capacity + size));
        unsafe {
            self.stack.set_len(self.stack.capacity());
        }
    }

    /// Frees stack slots allocated by `alloca`
    ///
    /// # SAFETY
    ///
    /// The caller must ensure that a call to `dealloc` is paired with a previous
    /// call to `alloca`, or this function may panic due to underflow of the stack
    /// pointer.
    #[inline]
    pub unsafe fn dealloc(&mut self, size: usize) {
        self.sp -= size;
    }

    /// Unwinds the stack to the nearest catch handler, if one can be found
    ///
    /// This function returns `None` if no handlers are found, otherwise it
    /// returns `Some(ip)`, where `ip` is the instruction pointer of the handler.
    pub fn unwind(&mut self) -> Option<usize> {
        while let Some(mark) = self.marks.last() {
            match mark {
                Mark::Catch { fp, sp, cp } => {
                    let fp = *fp;
                    let sp = *sp;
                    self.fp = fp;
                    self.sp = fp + (sp as usize);
                    return Some(*cp);
                }
                _ => {
                    self.marks.pop();
                }
            }
        }

        self.fp = 0;
        self.sp = RESERVED_REGISTERS;
        None
    }

    /// Returns an iterator which emits a [`StackFrame`] for each frame on the process stack,
    ///
    /// If `max_frames` is `None`, the trace will be as deep as the call stack.
    /// If `max_frames` is set to `Some`, then only `max_frames` frames will be emitted.
    ///
    /// The frames are emitted in stack order, meaning the most recently called function appears
    /// first
    pub fn trace<'a, 'b: 'a>(
        &'b self,
        max_frames: Option<usize>,
    ) -> impl Iterator<Item = StackFrame> + 'a {
        Tracer::new(self, max_frames.unwrap_or(usize::MAX))
    }
}

/// An iterator over call frames on the process stack, call stack order (i.e. the most recent call
/// is emitted first)
struct Tracer<'a> {
    stack: &'a [OpaqueTerm],
    marks: &'a [Mark],
    max_frames: usize,
}
impl<'a> Tracer<'a> {
    fn new(stack: &'a ProcessStack, max_frames: usize) -> Self {
        Self {
            stack: stack.stack.as_slice(),
            marks: stack.marks.as_slice(),
            max_frames,
        }
    }
}
impl<'a> core::iter::FusedIterator for Tracer<'a> {}
impl<'a> Iterator for Tracer<'a> {
    type Item = StackFrame;

    fn next(&mut self) -> Option<Self::Item> {
        if self.max_frames == 0 {
            return None;
        }

        while let Some((mark, marks)) = self.marks.split_last() {
            self.marks = marks;
            if let Mark::Call { fp, sp } = *mark {
                self.max_frames -= 1;
                let cp = self.stack[fp + CP_REG as usize];
                let ret = if cp.is_none() { 0 } else { cp.as_code() };
                return Some(StackFrame {
                    ret,
                    fp,
                    sp: fp + sp as usize,
                });
            }
        }

        self.max_frames = 0;
        None
    }
}
