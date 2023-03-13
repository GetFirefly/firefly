use core::mem;
use core::num::NonZeroU32;

use firefly_binary::{BinaryEntrySpecifier, Encoding, Endianness};
use firefly_number::BigInt;

use smallvec::SmallVec;

use super::debuginfo::SourceLocation;
use super::ops::*;
use super::*;

pub struct Builder<A: Atom, T: AtomTable<Atom = A>> {
    code: ByteCode<A, T>,
}
impl<A: Atom, T: AtomTable<Atom = A>> Builder<A, T> {
    pub fn new(mut code: ByteCode<A, T>) -> Self {
        // Every bytecode module starts with the following instruction sequence:
        //
        // * Nop
        // * NormalExit
        // * ContinueExit
        // * Await
        // * Trap
        // * ...
        //
        // The `Nop` is used as an invalid instruction pointer/address, which will
        // fall through to a normal exit when it is executed. This is used when a
        // process runs out of code to trigger a normal exit, as its instruction pointer
        // will be set to 0. Once the `NormalExit` is executed, its next instruction
        // will be the `ContinueExit`, which will never be changed once hit. Other
        // non-normal exits will set the instruction pointer to the ContinueExit
        // instruction for the same reason. `Await` and `Trap` are placed after these
        // so that control only reaches them when we set the instruction pointer to them
        // explicitly.
        //
        if code.code.is_empty() {
            code.code.push(Opcode::Nop(Nop));
            code.code.push(Opcode::NormalExit(NormalExit));
            code.code.push(Opcode::ContinueExit(ContinueExit));
            code.code.push(Opcode::Await(Await));
            code.code.push(Opcode::Trap(Trap));
        }
        Self { code }
    }

    pub fn build_function<'f, 'a: 'f>(
        &'a mut self,
        mfa: ModuleFunctionArity<A>,
        loc: Option<Location>,
    ) -> Result<FunctionBuilder<'f, A, T>, InvalidBytecodeError<A>> {
        let offset = self.code.next_instruction();
        let function = self.code.functions.define(mfa, offset)?;
        if let Some(loc) = loc {
            self.code.set_function_location(function, loc)
        } else {
            None
        };
        Ok(FunctionBuilder {
            builder: self,
            blocks: vec![],
            current_block: 0 as BlockId,
            function,
            arity: mfa.arity,
            // The first two registers of each function are reserved for the return
            // value and return address, respectively. As a result, function arguments
            // begin at register 2. The `alloc_register` function ensures that register
            // allocations take this into account, leaving the first two registers unused.
            registers: 1,
        })
    }

    /// Finish building this bytecode module, returning it
    pub fn finish(self) -> ByteCode<A, T> {
        // We must visit all of the instructions once more to resolve any unresolved
        // static calls. We do not raise an error if we can't resolve them at this time,
        // as we may link in more functions later, but we try to resolve any calls that
        // we can now.
        let mut module = self.code;
        let code = module.code.as_mut_slice();
        for op in code {
            match op {
                Opcode::CallStatic(CallStatic { dest, callee, .. }) => {
                    let dest = *dest;
                    match module.functions.get(*callee) {
                        Function::Bytecode {
                            offset,
                            is_nif: false,
                            ..
                        } if *offset > 0 => {
                            drop(mem::replace(
                                op,
                                Opcode::Call(Call {
                                    dest,
                                    offset: *offset,
                                }),
                            ));
                        }
                        _ => continue,
                    }
                }
                Opcode::EnterStatic(EnterStatic { callee, .. }) => {
                    match module.functions.get(*callee) {
                        Function::Bytecode {
                            offset,
                            is_nif: false,
                            ..
                        } if *offset > 0 => {
                            drop(mem::replace(op, Opcode::Enter(Enter { offset: *offset })));
                        }
                        _ => continue,
                    }
                }
                _ => continue,
            }
        }
        module
    }

    pub fn function_by_mfa(&self, mfa: &ModuleFunctionArity<A>) -> Option<&Function<A>> {
        self.code.function_by_mfa(&mfa)
    }

    pub fn function_location(&self, id: FunId) -> Option<SourceLocation> {
        match self.code.functions.get(id) {
            Function::Bytecode { offset, .. } if *offset > 0 => self
                .code
                .debug_info
                .function_pointer_to_source_location(*offset),
            _ => None,
        }
    }

    #[inline]
    pub fn get_or_define_function(&mut self, mfa: ModuleFunctionArity<A>) -> FunId {
        self.code.get_or_define_function(mfa)
    }

    #[inline]
    pub fn get_or_define_bif(&mut self, mfa: ModuleFunctionArity<A>) -> FunId {
        self.code.get_or_define_bif(mfa)
    }

    #[inline]
    pub fn get_or_define_nif<S: AsRef<str>>(&mut self, name: S, arity: u8) -> FunId {
        self.code.get_or_define_nif(name.as_ref(), arity)
    }

    #[inline]
    pub fn get_or_insert_file(&mut self, file: &str) -> FileId {
        self.code.get_or_insert_file(file)
    }

    #[inline]
    pub fn get_or_insert_location(&mut self, loc: Location) -> LocationId {
        self.code.get_or_insert_location(loc)
    }

    #[inline]
    pub fn set_function_location(&mut self, id: FunId, loc: Location) -> Option<LocationId> {
        self.code.set_function_location(id, loc)
    }

    #[inline(always)]
    pub fn next_instruction(&self) -> usize {
        self.code.next_instruction()
    }

    #[inline(always)]
    pub fn function_offset(&self, id: FunId) -> usize {
        unsafe { self.code.function_offset(id) }
    }

    #[inline]
    pub fn insert_atom(&mut self, name: &str) -> A {
        self.code.insert_atom(name).unwrap()
    }

    #[inline]
    pub fn insert_binary(&mut self, bytes: &[u8], encoding: Encoding) -> *const BinaryData {
        self.code.insert_binary(bytes, encoding)
    }

    #[inline]
    pub fn insert_bitstring(&mut self, bytes: &[u8], trailing_bits: u8) -> *const BinaryData {
        self.code.insert_bitstring(bytes, trailing_bits as usize)
    }
}

pub type BlockId = u16;

pub struct Block<A: Atom> {
    id: BlockId,
    args: Vec<Register>,
    offset: usize,
    code: Vec<Opcode<A>>,
    locations: Vec<Option<LocationId>>,
}
impl<A: Atom> Block<A> {
    #[inline]
    pub fn new(id: BlockId, args: Vec<Register>) -> Self {
        Self {
            id,
            args,
            offset: 0,
            code: vec![],
            locations: vec![],
        }
    }

    #[inline]
    pub fn arguments(&self) -> &[Register] {
        self.args.as_slice()
    }

    #[inline]
    pub fn push(&mut self, op: Opcode<A>, loc: Option<LocationId>) {
        self.code.push(op);
        self.locations.push(loc);
    }

    #[inline(always)]
    pub fn id(&self) -> BlockId {
        self.id
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.code.is_empty()
    }
}

pub struct FunctionBuilder<'a, A: Atom, T: AtomTable<Atom = A>> {
    pub builder: &'a mut Builder<A, T>,
    blocks: Vec<Block<A>>,
    current_block: BlockId,
    function: FunId,
    arity: Arity,
    registers: Register,
}
impl<'a, A, T> FunctionBuilder<'a, A, T>
where
    A: Atom,
    T: AtomTable<Atom = A>,
{
    pub fn create_block(&mut self, arity: Arity) -> BlockId {
        let id = self.blocks.len() as BlockId;
        let args = if arity == 0 {
            vec![]
        } else {
            (0..arity)
                .map(|_| self.alloc_register())
                .collect::<Vec<_>>()
        };
        self.blocks.push(Block::new(id, args));
        id
    }

    pub fn block(&self, block: BlockId) -> &Block<A> {
        &self.blocks[block as usize]
    }

    pub fn block_args(&self, block: BlockId) -> &[Register] {
        self.block(block).arguments()
    }

    pub fn switch_to_block(&mut self, block: BlockId) {
        self.current_block = block;
    }

    #[inline]
    pub fn get_or_define_function(&mut self, mfa: ModuleFunctionArity<A>) -> FunId {
        self.builder.get_or_define_function(mfa)
    }

    #[inline]
    pub fn get_or_define_bif(&mut self, mfa: ModuleFunctionArity<A>) -> FunId {
        self.builder.get_or_define_bif(mfa)
    }

    #[inline]
    pub fn get_or_define_nif<S: AsRef<str>>(&mut self, name: S, arity: u8) -> FunId {
        self.builder.get_or_define_nif(name.as_ref(), arity)
    }

    #[inline]
    pub fn get_or_insert_file(&mut self, file: &str) -> FileId {
        self.builder.get_or_insert_file(file)
    }

    #[inline]
    pub fn get_or_insert_location(&mut self, loc: Location) -> LocationId {
        self.builder.get_or_insert_location(loc)
    }

    #[inline]
    pub fn function_location(&self, id: FunId) -> Option<SourceLocation> {
        self.builder.function_location(id)
    }

    #[inline]
    pub fn set_function_location(&mut self, id: FunId, loc: Location) -> Option<LocationId> {
        self.builder.set_function_location(id, loc)
    }

    pub fn mark_as_nif(&mut self) {
        match self.builder.code.function_mut(self.function) {
            Function::Bytecode { ref mut is_nif, .. } => {
                *is_nif = true;
            }
            _ => unreachable!(),
        }
    }

    #[inline]
    fn alloc_register(&mut self) -> Register {
        let next = self
            .registers
            .checked_add(1)
            .expect("no more registers available, only 65,536 are available per-function");
        self.registers = next;
        next
    }

    fn push(&mut self, op: Opcode<A>, loc: Option<LocationId>) {
        self.current_block_mut().push(op, loc);
    }

    #[inline]
    fn current_block_mut(&mut self) -> &mut Block<A> {
        &mut self.blocks[self.current_block as usize]
    }

    #[inline(always)]
    fn function_offset(&self, id: FunId) -> usize {
        self.builder.function_offset(id)
    }

    #[inline]
    pub fn insert_atom(&mut self, name: &str) -> A {
        self.builder.insert_atom(name)
    }

    #[inline]
    pub fn insert_binary(&mut self, bytes: &[u8], encoding: Encoding) -> *const BinaryData {
        self.builder.insert_binary(bytes, encoding)
    }

    #[inline]
    pub fn insert_bitstring(&mut self, bytes: &[u8], trailing_bits: u8) -> *const BinaryData {
        self.builder.insert_bitstring(bytes, trailing_bits)
    }

    /// Consume the builder, writing its blocks to the underlying bytecode module
    pub fn finish(self) {}
}

pub struct JumpTableBuilder<'a, 'b, A, T>
where
    A: Atom,
    T: AtomTable<Atom = A>,
{
    builder: &'b mut FunctionBuilder<'a, A, T>,
    arg: Register,
    loc: Option<LocationId>,
    entries: Vec<(u32, BlockId)>,
}
impl<'a, 'b, A, T> JumpTableBuilder<'a, 'b, A, T>
where
    A: Atom,
    T: AtomTable<Atom = A>,
{
    pub fn entry(&mut self, imm: u32, dest: BlockId) {
        self.entries.push((imm, dest));
    }

    pub fn finish(mut self, default: BlockId, args: &[Register]) {
        self.builder.push(
            Opcode::JumpTable(JumpTable {
                reg: self.arg,
                len: self.entries.len() as u8,
            }),
            self.loc,
        );
        for (imm, dest) in self.entries.drain(..) {
            assert_ne!(dest, self.builder.current_block);
            self.builder.push(
                Opcode::JumpTableEntry(JumpTableEntry {
                    imm,
                    offset: dest as JumpOffset,
                }),
                self.loc,
            )
        }
        self.builder.build_br(default, args, self.loc);
    }
}

// Builders
impl<'a, A, T> FunctionBuilder<'a, A, T>
where
    A: Atom,
    T: AtomTable<Atom = A>,
{
    pub fn build_nop(&mut self) {
        self.push(Opcode::Nop(Nop), None);
    }

    pub fn build_ret(&mut self, reg: Register, loc: Option<LocationId>) {
        self.push(Opcode::Ret(Ret { reg }), loc);
    }

    pub fn build_br(&mut self, dest: BlockId, args: &[Register], loc: Option<LocationId>) {
        self.prepare_br_args(dest, args, None, loc);

        // NOTE: While building bytecode for a function, we store the target block id
        // as the offset, but then when finalizing the function we update the instruction
        // with the real offset in the bytecode where the target block is located
        self.push(
            Opcode::Br(Br {
                offset: dest as JumpOffset,
            }),
            loc,
        );
    }

    pub fn build_br_unless(
        &mut self,
        cond: Register,
        dest: BlockId,
        args: &[Register],
        loc: Option<LocationId>,
    ) {
        if !args.is_empty() {
            // We need to conditionally perform register moves based on `cond`, but this branch
            // will only be taken if the condition is _false_, so we need to invert the value
            // of `cond` so that our cmov instructions see `true` when the branch will be taken
            let is_false = self.alloc_register();
            self.push(
                Opcode::Not(Not {
                    dest: is_false,
                    cond,
                }),
                loc,
            );
            self.prepare_br_args(dest, args, Some(is_false), loc);
        }

        // NOTE: While building bytecode for a function, we store the target block id
        // as the offset, but then when finalizing the function we update the instruction
        // with the real offset in the bytecode where the target block is located
        self.push(
            Opcode::Brz(Brz {
                reg: cond,
                offset: dest as JumpOffset,
            }),
            loc,
        );
    }

    pub fn build_br_if(
        &mut self,
        cond: Register,
        dest: BlockId,
        args: &[Register],
        loc: Option<LocationId>,
    ) {
        self.prepare_br_args(dest, args, Some(cond), loc);

        // NOTE: While building bytecode for a function, we store the target block id
        // as the offset, but then when finalizing the function we update the instruction
        // with the real offset in the bytecode where the target block is located
        self.push(
            Opcode::Brnz(Brnz {
                reg: cond,
                offset: dest as JumpOffset,
            }),
            loc,
        );
    }

    pub fn build_jump_table<'b, 'c: 'b>(
        &'c mut self,
        arg: Register,
        loc: Option<LocationId>,
    ) -> JumpTableBuilder<'a, 'b, A, T> {
        JumpTableBuilder {
            builder: self,
            arg,
            loc,
            entries: vec![],
        }
    }

    fn prepare_br_args(
        &mut self,
        dest: BlockId,
        mut args: &[Register],
        cond: Option<Register>,
        loc: Option<LocationId>,
    ) {
        if args.is_empty() {
            return;
        }

        // When branching to the target block, we need to issue register moves
        // to set the block argument registers with their appropriate source values.
        //
        // This is a little tricky, as we must account for register interference, i.e.
        // a move might clobber the value of a register that hasn't been moved yet if
        // we're branching to a predecessor block or back to the head of the current block.
        //
        // To handle this, for each move we check if the move will clobber the value of
        // a subsequent move, and if so, we allocate a temporary register, move into that,
        // and then after all the non-temp moves have been executed, we move the temps
        let dsts = self
            .block_args(dest)
            .iter()
            .copied()
            .collect::<SmallVec<[_; 4]>>();
        assert_eq!(
            dsts.len(),
            args.len(),
            "incorrect number of arguments for target block"
        );

        let mov = |src, dst| match cond {
            Some(cond) => Opcode::Cmov(Cmov {
                dest: dst,
                src,
                cond,
            }),
            None => Opcode::Mov(Mov { dest: dst, src }),
        };

        let mut deferred_moves = SmallVec::<[Opcode<A>; 4]>::new();
        let mut index = 0;
        while let Some((src, rest)) = args.split_first() {
            let src = *src;
            // Slide our view of the source registers forward
            args = rest;
            let dst = dsts[index];
            index += 1;

            // The destination register is clobbered if a subsequent move relies on it as the source
            // register.
            let is_clobbered = rest.contains(&dst);

            if is_clobbered {
                // If the destination register is used as a source later, this move
                // would result in that value being clobbered, so we must introduce
                // a temporary.
                let tmp = self.alloc_register();
                self.push(mov(src, tmp), loc);
                deferred_moves.push(mov(tmp, dst));
                continue;
            }

            // If the src and dst registers are the same, no move is needed
            if src == dst {
                continue;
            }

            self.push(mov(src, dst), loc);
        }

        // Perform all the deferred moves now
        for mv in deferred_moves.drain(..) {
            self.push(mv, loc);
        }
    }

    pub fn build_call_nif(
        &mut self,
        callee: FunId,
        args: &[Register],
        loc: Option<LocationId>,
    ) -> Register {
        let dest = self.alloc_register();
        // We need to reserve a register for the return address
        self.alloc_register();
        // Then reserve registers for the callee arguments and move their values into place
        for arg in args {
            let dest = self.alloc_register();
            self.push(Opcode::Mov(Mov { dest, src: *arg }), loc);
        }
        self.push(
            Opcode::CallNative(CallNative {
                dest,
                callee: callee as usize as *const (),
                arity: args.len() as Arity,
            }),
            loc,
        );
        dest
    }

    pub fn build_call(
        &mut self,
        callee: FunId,
        args: &[Register],
        loc: Option<LocationId>,
    ) -> Register {
        match self.builder.code.function_by_id(callee) {
            Function::Native { .. } => self.build_call_nif(callee, args, loc),
            Function::Bytecode { mfa, .. } | Function::Bif { mfa, .. } => {
                assert_eq!(
                    mfa.arity as usize,
                    args.len(),
                    "incorrect number of arguments for callee"
                );
                let dest = self.alloc_register();
                // We need to reserve a register for the return address
                self.alloc_register();
                // Then reserve registers for the callee arguments and move their values into place
                for arg in args {
                    let dest = self.alloc_register();
                    self.push(Opcode::Mov(Mov { dest, src: *arg }), loc);
                }
                self.push(Opcode::CallStatic(CallStatic { dest, callee }), loc);
                dest
            }
        }
    }

    pub fn build_call_apply2(
        &mut self,
        callee: Register,
        argv: Register,
        loc: Option<LocationId>,
    ) -> Register {
        // Reserve space for the return value
        let dest = self.alloc_register();
        // We need to reserve a register for the return address
        self.alloc_register();
        // The remaining stack slots needed for the call are allocated at runtime
        self.push(Opcode::CallApply2(CallApply2 { dest, callee, argv }), loc);
        dest
    }

    pub fn build_call_apply3(
        &mut self,
        module: Register,
        function: Register,
        argv: Register,
        loc: Option<LocationId>,
    ) -> Register {
        // Reserve space for the return value
        let dest = self.alloc_register();
        // We need to reserve a register for the return address
        self.alloc_register();
        // The remaining stack slots needed for the call are allocated at runtime
        self.push(
            Opcode::CallApply3(CallApply3 {
                dest,
                module,
                function,
                argv,
            }),
            loc,
        );
        dest
    }

    pub fn build_call_indirect(
        &mut self,
        callee: Register,
        args: &[Register],
        loc: Option<LocationId>,
    ) -> Register {
        let dest = self.alloc_register();
        // We need to reserve a register for the return address
        self.alloc_register();
        // Then reserve registers for the callee arguments and move their values into place
        for arg in args {
            let dest = self.alloc_register();
            self.push(Opcode::Mov(Mov { dest, src: *arg }), loc);
        }
        self.push(
            Opcode::CallIndirect(CallIndirect {
                dest,
                callee,
                arity: args.len() as Arity,
            }),
            loc,
        );
        dest
    }

    pub fn build_enter(&mut self, callee: FunId, args: &[Register], loc: Option<LocationId>) {
        match self.builder.code.function_by_id(callee) {
            Function::Native { .. } => self.build_enter_nif(callee, args, loc),
            Function::Bytecode { mfa, .. } | Function::Bif { mfa, .. } => {
                assert_eq!(
                    mfa.arity as usize,
                    args.len(),
                    "incorrect number of arguments for callee"
                );
                self.prepare_tail_call_args(args, loc);
                self.push(Opcode::EnterStatic(EnterStatic { callee }), loc);
            }
        }
    }

    pub fn build_enter_nif(&mut self, callee: FunId, args: &[Register], loc: Option<LocationId>) {
        self.prepare_tail_call_args(args, loc);
        self.push(
            Opcode::EnterNative(EnterNative {
                callee: callee as usize as *const (),
                arity: args.len() as Arity,
            }),
            loc,
        );
    }

    pub fn build_enter_apply2(
        &mut self,
        callee: Register,
        argv: Register,
        loc: Option<LocationId>,
    ) {
        // The remaining stack slots needed for the call are allocated at runtime
        self.push(Opcode::EnterApply2(EnterApply2 { callee, argv }), loc);
    }

    pub fn build_enter_apply3(
        &mut self,
        module: Register,
        function: Register,
        argv: Register,
        loc: Option<LocationId>,
    ) {
        // The remaining stack slots needed for the call are allocated at runtime
        self.push(
            Opcode::EnterApply3(EnterApply3 {
                module,
                function,
                argv,
            }),
            loc,
        );
    }

    pub fn build_enter_indirect(
        &mut self,
        callee: Register,
        args: &[Register],
        loc: Option<LocationId>,
    ) {
        self.prepare_tail_call_args(args, loc);
        self.push(
            Opcode::EnterIndirect(EnterIndirect {
                callee,
                arity: args.len() as Arity,
            }),
            loc,
        );
    }

    fn prepare_tail_call_args(&mut self, mut args: &[Register], loc: Option<LocationId>) {
        if args.is_empty() {
            return;
        }

        // We need to move `args` into the argument registers of the callee,
        // but since we might be tail calling the current function, some of the
        // argument registers might be the same registers we need to write to.
        //
        // In order to prevent this accidental clobbering, for each move we check
        // if the move will clobber the source of a subsequent move, and if so, we
        // allocate a temporary register, move into that, and then after all the
        // non-temp moves have been executed, we move the temps to their final destination
        let dsts = ((2 as Register)..(args.len() as Register + 2))
            .into_iter()
            .collect::<SmallVec<[_; 4]>>();

        let mut deferred_moves = SmallVec::<[Opcode<A>; 4]>::new();
        let mut index = 0;
        while let Some((src, rest)) = args.split_first() {
            let src = *src;
            // Slide our view of the source registers forward
            args = rest;
            let dst = dsts[index];
            index += 1;

            // The destination register is clobbered if a subsequent move relies on it as the source
            // register.
            let is_clobbered = rest.contains(&dst);

            if is_clobbered {
                // If the destination register is used as a source later, this move
                // would result in that value being clobbered, so we must introduce
                // a temporary.
                let tmp = self.alloc_register();
                self.push(Opcode::Mov(Mov { dest: tmp, src }), loc);
                deferred_moves.push(Opcode::Mov(Mov {
                    dest: dst,
                    src: tmp,
                }));
                continue;
            }

            // If the src and dst registers are the same, no move is needed
            if src == dst {
                continue;
            }

            self.push(Opcode::Mov(Mov { dest: dst, src }), loc);
        }

        // Perform all the deferred moves now
        for mv in deferred_moves.drain(..) {
            self.push(mv, loc);
        }
    }

    pub fn build_is_atom(&mut self, value: Register, loc: Option<LocationId>) -> Register {
        let dest = self.alloc_register();
        self.push(Opcode::IsAtom(IsAtom { dest, value }), loc);
        dest
    }

    pub fn build_is_bool(&mut self, value: Register, loc: Option<LocationId>) -> Register {
        let dest = self.alloc_register();
        self.push(Opcode::IsBool(IsBool { dest, value }), loc);
        dest
    }

    pub fn build_is_nil(&mut self, value: Register, loc: Option<LocationId>) -> Register {
        let dest = self.alloc_register();
        self.push(Opcode::IsNil(IsNil { dest, value }), loc);
        dest
    }

    pub fn build_is_tuple(
        &mut self,
        value: Register,
        arity: Option<NonZeroU32>,
        loc: Option<LocationId>,
    ) -> Register {
        let dest = self.alloc_register();
        self.push(Opcode::IsTuple(IsTuple { dest, value, arity }), loc);
        dest
    }

    pub fn build_is_tuple_fetch_arity(
        &mut self,
        value: Register,
        loc: Option<LocationId>,
    ) -> (Register, Register) {
        let dest = self.alloc_register();
        let arity = self.alloc_register();
        self.push(
            Opcode::IsTupleFetchArity(IsTupleFetchArity { dest, arity, value }),
            loc,
        );
        (dest, arity)
    }

    pub fn build_is_map(&mut self, value: Register, loc: Option<LocationId>) -> Register {
        let dest = self.alloc_register();
        self.push(Opcode::IsMap(IsMap { dest, value }), loc);
        dest
    }

    pub fn build_is_cons(&mut self, value: Register, loc: Option<LocationId>) -> Register {
        let dest = self.alloc_register();
        self.push(Opcode::IsCons(IsCons { dest, value }), loc);
        dest
    }

    pub fn build_is_list(&mut self, value: Register, loc: Option<LocationId>) -> Register {
        let dest = self.alloc_register();
        self.push(Opcode::IsList(IsList { dest, value }), loc);
        dest
    }

    pub fn build_is_int(&mut self, value: Register, loc: Option<LocationId>) -> Register {
        let dest = self.alloc_register();
        self.push(Opcode::IsInt(IsInt { dest, value }), loc);
        dest
    }

    pub fn build_is_float(&mut self, value: Register, loc: Option<LocationId>) -> Register {
        let dest = self.alloc_register();
        self.push(Opcode::IsFloat(IsFloat { dest, value }), loc);
        dest
    }

    pub fn build_is_number(&mut self, value: Register, loc: Option<LocationId>) -> Register {
        let dest = self.alloc_register();
        self.push(Opcode::IsNumber(IsNumber { dest, value }), loc);
        dest
    }

    pub fn build_is_pid(&mut self, value: Register, loc: Option<LocationId>) -> Register {
        let dest = self.alloc_register();
        self.push(Opcode::IsPid(IsPid { dest, value }), loc);
        dest
    }

    pub fn build_is_port(&mut self, value: Register, loc: Option<LocationId>) -> Register {
        let dest = self.alloc_register();
        self.push(Opcode::IsPort(IsPort { dest, value }), loc);
        dest
    }

    pub fn build_is_reference(&mut self, value: Register, loc: Option<LocationId>) -> Register {
        let dest = self.alloc_register();
        self.push(Opcode::IsRef(IsRef { dest, value }), loc);
        dest
    }

    pub fn build_is_binary(&mut self, value: Register, loc: Option<LocationId>) -> Register {
        let dest = self.alloc_register();
        self.push(
            Opcode::IsBinary(IsBinary {
                dest,
                value,
                unit: 8,
            }),
            loc,
        );
        dest
    }

    pub fn build_is_bitstring(&mut self, value: Register, loc: Option<LocationId>) -> Register {
        let dest = self.alloc_register();
        self.push(
            Opcode::IsBinary(IsBinary {
                dest,
                value,
                unit: 1,
            }),
            loc,
        );
        dest
    }

    pub fn build_is_function(
        &mut self,
        value: Register,
        arity: Option<Register>,
        loc: Option<LocationId>,
    ) -> Register {
        let dest = self.alloc_register();
        self.push(Opcode::IsFunction(IsFunction { dest, value, arity }), loc);
        dest
    }

    pub fn build_nil(&mut self, loc: Option<LocationId>) -> Register {
        let dest = self.alloc_register();
        self.push(Opcode::LoadNil(LoadNil { dest }), loc);
        dest
    }

    pub fn build_bool(&mut self, value: bool, loc: Option<LocationId>) -> Register {
        let dest = self.alloc_register();
        self.push(Opcode::LoadBool(LoadBool { dest, value }), loc);
        dest
    }

    pub fn build_atom(&mut self, value: &str, loc: Option<LocationId>) -> Register {
        let dest = self.alloc_register();
        let atom = self.insert_atom(value);
        self.push(Opcode::LoadAtom(LoadAtom { dest, value: atom }), loc);
        dest
    }

    pub fn build_int(&mut self, value: i64, loc: Option<LocationId>) -> Register {
        let dest = self.alloc_register();
        self.push(Opcode::LoadInt(LoadInt { dest, value }), loc);
        dest
    }

    pub fn build_bigint(&mut self, value: BigInt, loc: Option<LocationId>) -> Register {
        let dest = self.alloc_register();
        self.push(Opcode::LoadBig(LoadBig { dest, value }), loc);
        dest
    }

    pub fn build_float(&mut self, value: f64, loc: Option<LocationId>) -> Register {
        let dest = self.alloc_register();
        self.push(Opcode::LoadFloat(LoadFloat { dest, value }), loc);
        dest
    }

    pub fn build_utf8_binary(&mut self, value: &str, loc: Option<LocationId>) -> Register {
        let dest = self.alloc_register();
        let bytes = value.as_bytes();
        let bin = self.insert_binary(bytes, Encoding::Utf8);
        self.push(Opcode::LoadBinary(LoadBinary { dest, value: bin }), loc);
        dest
    }

    pub fn build_raw_binary(&mut self, value: &[u8], loc: Option<LocationId>) -> Register {
        let dest = self.alloc_register();
        let bin = self.insert_binary(value, Encoding::Raw);
        self.push(Opcode::LoadBinary(LoadBinary { dest, value: bin }), loc);
        dest
    }

    pub fn build_bitstring(
        &mut self,
        value: &[u8],
        trailing_bits: u8,
        loc: Option<LocationId>,
    ) -> Register {
        let dest = self.alloc_register();
        let bin = self.insert_bitstring(value, trailing_bits);
        self.push(
            Opcode::LoadBitstring(LoadBitstring { dest, value: bin }),
            loc,
        );
        dest
    }

    pub fn build_not(&mut self, value: Register, loc: Option<LocationId>) -> Register {
        let dest = self.alloc_register();
        self.push(Opcode::Not(Not { dest, cond: value }), loc);
        dest
    }

    pub fn build_and(&mut self, lhs: Register, rhs: Register, loc: Option<LocationId>) -> Register {
        let dest = self.alloc_register();
        self.push(Opcode::And(And { dest, lhs, rhs }), loc);
        dest
    }

    pub fn build_andalso(
        &mut self,
        lhs: Register,
        rhs: Register,
        loc: Option<LocationId>,
    ) -> Register {
        let dest = self.alloc_register();
        self.push(Opcode::AndAlso(AndAlso { dest, lhs, rhs }), loc);
        dest
    }

    pub fn build_or(&mut self, lhs: Register, rhs: Register, loc: Option<LocationId>) -> Register {
        let dest = self.alloc_register();
        self.push(Opcode::Or(Or { dest, lhs, rhs }), loc);
        dest
    }

    pub fn build_orelse(
        &mut self,
        lhs: Register,
        rhs: Register,
        loc: Option<LocationId>,
    ) -> Register {
        let dest = self.alloc_register();
        self.push(Opcode::OrElse(OrElse { dest, lhs, rhs }), loc);
        dest
    }

    pub fn build_bnot(&mut self, rhs: Register, loc: Option<LocationId>) -> Register {
        let dest = self.alloc_register();
        self.push(Opcode::Bnot(Bnot { dest, rhs }), loc);
        dest
    }

    pub fn build_xor(&mut self, lhs: Register, rhs: Register, loc: Option<LocationId>) -> Register {
        let dest = self.alloc_register();
        self.push(Opcode::Xor(Xor { dest, lhs, rhs }), loc);
        dest
    }

    pub fn build_band(
        &mut self,
        lhs: Register,
        rhs: Register,
        loc: Option<LocationId>,
    ) -> Register {
        let dest = self.alloc_register();
        self.push(Opcode::Band(Band { dest, lhs, rhs }), loc);
        dest
    }

    pub fn build_bor(&mut self, lhs: Register, rhs: Register, loc: Option<LocationId>) -> Register {
        let dest = self.alloc_register();
        self.push(Opcode::Bor(Bor { dest, lhs, rhs }), loc);
        dest
    }

    pub fn build_bxor(
        &mut self,
        lhs: Register,
        rhs: Register,
        loc: Option<LocationId>,
    ) -> Register {
        let dest = self.alloc_register();
        self.push(Opcode::Bxor(Bxor { dest, lhs, rhs }), loc);
        dest
    }

    pub fn build_bsl(
        &mut self,
        value: Register,
        shift: Register,
        loc: Option<LocationId>,
    ) -> Register {
        let dest = self.alloc_register();
        self.push(Opcode::Bsl(Bsl { dest, value, shift }), loc);
        dest
    }

    pub fn build_bsr(
        &mut self,
        value: Register,
        shift: Register,
        loc: Option<LocationId>,
    ) -> Register {
        let dest = self.alloc_register();
        self.push(Opcode::Bsr(Bsr { dest, value, shift }), loc);
        dest
    }

    pub fn build_div(
        &mut self,
        value: Register,
        divisor: Register,
        loc: Option<LocationId>,
    ) -> Register {
        let dest = self.alloc_register();
        self.push(
            Opcode::Div(Div {
                dest,
                value,
                divisor,
            }),
            loc,
        );
        dest
    }

    pub fn build_rem(
        &mut self,
        value: Register,
        divisor: Register,
        loc: Option<LocationId>,
    ) -> Register {
        let dest = self.alloc_register();
        self.push(
            Opcode::Rem(Rem {
                dest,
                value,
                divisor,
            }),
            loc,
        );
        dest
    }

    pub fn build_neg(&mut self, value: Register, loc: Option<LocationId>) -> Register {
        let dest = self.alloc_register();
        self.push(Opcode::Neg(Neg { dest, rhs: value }), loc);
        dest
    }

    pub fn build_add(&mut self, lhs: Register, rhs: Register, loc: Option<LocationId>) -> Register {
        let dest = self.alloc_register();
        self.push(Opcode::Add(Add { dest, lhs, rhs }), loc);
        dest
    }

    pub fn build_sub(&mut self, lhs: Register, rhs: Register, loc: Option<LocationId>) -> Register {
        let dest = self.alloc_register();
        self.push(Opcode::Sub(Sub { dest, lhs, rhs }), loc);
        dest
    }

    pub fn build_mul(&mut self, lhs: Register, rhs: Register, loc: Option<LocationId>) -> Register {
        let dest = self.alloc_register();
        self.push(Opcode::Mul(Mul { dest, lhs, rhs }), loc);
        dest
    }

    pub fn build_divide(
        &mut self,
        lhs: Register,
        rhs: Register,
        loc: Option<LocationId>,
    ) -> Register {
        let dest = self.alloc_register();
        self.push(Opcode::Divide(Divide { dest, lhs, rhs }), loc);
        dest
    }

    pub fn build_list_append(
        &mut self,
        list: Register,
        rhs: Register,
        loc: Option<LocationId>,
    ) -> Register {
        let dest = self.alloc_register();
        self.push(Opcode::ListAppend(ListAppend { dest, list, rhs }), loc);
        dest
    }

    pub fn build_list_remove(
        &mut self,
        list: Register,
        rhs: Register,
        loc: Option<LocationId>,
    ) -> Register {
        let dest = self.alloc_register();
        self.push(Opcode::ListRemove(ListRemove { dest, list, rhs }), loc);
        dest
    }

    pub fn build_eq(
        &mut self,
        lhs: Register,
        rhs: Register,
        strict: bool,
        loc: Option<LocationId>,
    ) -> Register {
        let dest = self.alloc_register();
        self.push(
            Opcode::Eq(IsEq {
                dest,
                lhs,
                rhs,
                strict,
            }),
            loc,
        );
        dest
    }

    pub fn build_neq(
        &mut self,
        lhs: Register,
        rhs: Register,
        strict: bool,
        loc: Option<LocationId>,
    ) -> Register {
        let dest = self.alloc_register();
        self.push(
            Opcode::Neq(IsNeq {
                dest,
                lhs,
                rhs,
                strict,
            }),
            loc,
        );
        dest
    }

    pub fn build_gt(&mut self, lhs: Register, rhs: Register, loc: Option<LocationId>) -> Register {
        let dest = self.alloc_register();
        self.push(Opcode::Gt(IsGt { dest, lhs, rhs }), loc);
        dest
    }

    pub fn build_gte(&mut self, lhs: Register, rhs: Register, loc: Option<LocationId>) -> Register {
        let dest = self.alloc_register();
        self.push(Opcode::Gte(IsGte { dest, lhs, rhs }), loc);
        dest
    }

    pub fn build_lt(&mut self, lhs: Register, rhs: Register, loc: Option<LocationId>) -> Register {
        let dest = self.alloc_register();
        self.push(Opcode::Lt(IsLt { dest, lhs, rhs }), loc);
        dest
    }

    pub fn build_lte(&mut self, lhs: Register, rhs: Register, loc: Option<LocationId>) -> Register {
        let dest = self.alloc_register();
        self.push(Opcode::Lte(IsLte { dest, lhs, rhs }), loc);
        dest
    }

    pub fn build_cons(
        &mut self,
        head: Register,
        tail: Register,
        loc: Option<LocationId>,
    ) -> Register {
        let dest = self.alloc_register();
        self.push(Opcode::Cons(Cons { dest, head, tail }), loc);
        dest
    }

    // load the head and tail of a cons cell into `hd` and `tl` respectively
    pub fn build_split(&mut self, list: Register, loc: Option<LocationId>) -> (Register, Register) {
        let hd = self.alloc_register();
        let tl = self.alloc_register();
        self.push(Opcode::Split(Split { hd, tl, list }), loc);
        (hd, tl)
    }

    pub fn build_head(&mut self, list: Register, loc: Option<LocationId>) -> Register {
        let dest = self.alloc_register();
        self.push(Opcode::Head(Head { dest, list }), loc);
        dest
    }

    pub fn build_tail(&mut self, list: Register, loc: Option<LocationId>) -> Register {
        let dest = self.alloc_register();
        self.push(Opcode::Tail(Tail { dest, list }), loc);
        dest
    }

    pub fn build_closure(
        &mut self,
        function: FunId,
        env: &[Register],
        loc: Option<LocationId>,
    ) -> Register {
        let dest = self.alloc_register();
        for value in env {
            let dest = self.alloc_register();
            self.push(Opcode::Mov(Mov { dest, src: *value }), loc);
        }
        self.push(
            Opcode::Closure(Closure {
                dest,
                function,
                arity: env.len() as Arity,
            }),
            loc,
        );
        dest
    }

    pub fn build_unpack_env(
        &mut self,
        fun: Register,
        index: usize,
        loc: Option<LocationId>,
    ) -> Register {
        let dest = self.alloc_register();
        self.push(
            Opcode::UnpackEnv(UnpackEnv {
                dest,
                fun,
                index: index.try_into().unwrap(),
            }),
            loc,
        );
        dest
    }

    pub fn build_tuple(&mut self, elements: &[Register], loc: Option<LocationId>) -> Register {
        let dest = self.alloc_register();
        for element in elements {
            let dest = self.alloc_register();
            self.push(
                Opcode::Mov(Mov {
                    dest,
                    src: *element,
                }),
                loc,
            );
        }
        self.push(
            Opcode::Tuple(Tuple {
                dest,
                arity: elements.len() as Arity,
            }),
            loc,
        );
        dest
    }

    pub fn build_tuple_with_capacity(&mut self, arity: usize, loc: Option<LocationId>) -> Register {
        let dest = self.alloc_register();
        self.push(
            Opcode::TupleWithCapacity(TupleWithCapacity {
                dest,
                arity: arity as Arity,
            }),
            loc,
        );
        dest
    }

    pub fn build_tuple_arity(&mut self, tuple: Register, loc: Option<LocationId>) -> Register {
        let dest = self.alloc_register();
        self.push(Opcode::TupleArity(TupleArity { dest, tuple }), loc);
        dest
    }

    pub fn build_get_element(
        &mut self,
        tuple: Register,
        index: usize,
        loc: Option<LocationId>,
    ) -> Register {
        let dest = self.alloc_register();
        self.push(
            Opcode::GetElement(GetElement {
                dest,
                tuple,
                index: index.try_into().unwrap(),
            }),
            loc,
        );
        dest
    }

    pub fn build_set_element(
        &mut self,
        tuple: Register,
        index: usize,
        value: Register,
        loc: Option<LocationId>,
    ) -> Register {
        let dest = self.alloc_register();
        self.push(
            Opcode::SetElement(SetElement {
                dest,
                tuple,
                index: index.try_into().unwrap(),
                value,
            }),
            loc,
        );
        dest
    }

    pub fn build_set_element_mut(
        &mut self,
        tuple: Register,
        index: usize,
        value: Register,
        loc: Option<LocationId>,
    ) {
        self.push(
            Opcode::SetElementMut(SetElementMut {
                tuple,
                index: index.try_into().unwrap(),
                value,
            }),
            loc,
        );
    }

    pub fn build_map(&mut self, capacity: usize, loc: Option<LocationId>) -> Register {
        let dest = self.alloc_register();
        self.push(Opcode::Map(Map { dest, capacity }), loc);
        dest
    }

    pub fn build_map_insert(
        &mut self,
        map: Register,
        key: Register,
        value: Register,
        loc: Option<LocationId>,
    ) -> Register {
        let dest = self.alloc_register();
        self.push(
            Opcode::MapPut(MapPut {
                dest,
                map,
                key,
                value,
            }),
            loc,
        );
        dest
    }

    pub fn build_map_insert_mut(
        &mut self,
        map: Register,
        key: Register,
        value: Register,
        loc: Option<LocationId>,
    ) {
        self.push(Opcode::MapPutMut(MapPutMut { map, key, value }), loc);
    }

    pub fn build_map_update(
        &mut self,
        map: Register,
        key: Register,
        value: Register,
        loc: Option<LocationId>,
    ) -> Register {
        let dest = self.alloc_register();
        self.push(
            Opcode::MapUpdate(MapUpdate {
                dest,
                map,
                key,
                value,
            }),
            loc,
        );
        dest
    }

    pub fn build_map_update_mut(
        &mut self,
        map: Register,
        key: Register,
        value: Register,
        loc: Option<LocationId>,
    ) {
        self.push(Opcode::MapUpdateMut(MapUpdateMut { map, key, value }), loc);
    }

    pub fn build_map_extend_insert(
        &mut self,
        map: Register,
        pairs: &[Register],
        loc: Option<LocationId>,
    ) -> Register {
        let dest = self.alloc_register();
        self.push(
            Opcode::MapExtendPut(MapExtendPut {
                dest,
                map,
                pairs: pairs.to_vec(),
            }),
            loc,
        );
        dest
    }

    pub fn build_map_extend_update(
        &mut self,
        map: Register,
        pairs: &[Register],
        loc: Option<LocationId>,
    ) -> Register {
        let dest = self.alloc_register();
        self.push(
            Opcode::MapExtendUpdate(MapExtendUpdate {
                dest,
                map,
                pairs: pairs.to_vec(),
            }),
            loc,
        );
        dest
    }

    pub fn build_map_try_get(
        &mut self,
        map: Register,
        key: Register,
        loc: Option<LocationId>,
    ) -> (Register, Register) {
        let is_err = self.alloc_register();
        let value = self.alloc_register();
        self.push(
            Opcode::MapTryGet(MapTryGet {
                is_err,
                value,
                map,
                key,
            }),
            loc,
        );
        (is_err, value)
    }

    pub fn build_mov(&mut self, dest: Register, src: Register, loc: Option<LocationId>) {
        self.push(Opcode::Mov(Mov { dest, src }), loc);
    }

    pub fn build_catch(&mut self, dest: BlockId, loc: Option<LocationId>) {
        let cp = self.alloc_register();
        let handler_args = self.block_args(dest);
        let kind;
        let reason;
        let trace;
        if handler_args.is_empty() {
            kind = self.alloc_register();
            reason = self.alloc_register();
            trace = self.alloc_register();
        } else {
            assert_eq!(handler_args.len(), 3);
            kind = handler_args[0];
            reason = handler_args[1];
            trace = handler_args[2];
        }
        self.push(Opcode::Catch(Catch { cp }), loc);
        self.push(
            Opcode::LandingPad(LandingPad {
                kind,
                reason,
                trace,
                offset: dest as JumpOffset,
            }),
            loc,
        );
    }

    pub fn build_end_catch(&mut self, loc: Option<LocationId>) {
        self.push(Opcode::EndCatch(EndCatch), loc);
    }

    pub fn build_stacktrace(&mut self, loc: Option<LocationId>) -> Register {
        let dest = self.alloc_register();
        self.push(Opcode::StackTrace(StackTrace { dest }), loc);
        dest
    }

    pub fn build_send(&mut self, recipient: Register, message: Register, loc: Option<LocationId>) {
        self.push(Opcode::Send(SendOp { recipient, message }), loc);
    }

    // Returns `(available, message)`
    pub fn build_recv_peek(&mut self, loc: Option<LocationId>) -> (Register, Register) {
        let available = self.alloc_register();
        let message = self.alloc_register();
        self.push(Opcode::RecvPeek(RecvPeek { available, message }), loc);
        (available, message)
    }

    pub fn build_recv_next(&mut self, loc: Option<LocationId>) {
        self.push(Opcode::RecvNext(RecvNext), loc);
    }

    pub fn build_recv_wait_timeout(
        &mut self,
        timeout: Register,
        loc: Option<LocationId>,
    ) -> Register {
        let dest = self.alloc_register();
        self.push(Opcode::RecvWait(RecvWait { dest, timeout }), loc);
        self.push(Opcode::RecvTimeout(RecvTimeout { dest }), loc);
        dest
    }

    pub fn build_recv_pop(&mut self, loc: Option<LocationId>) {
        self.push(Opcode::RecvPop(RecvPop), loc);
    }

    pub fn build_bs_init(&mut self, loc: Option<LocationId>) -> Register {
        let dest = self.alloc_register();
        self.push(Opcode::BsInit(BsInit { dest }), loc);
        dest
    }

    pub fn build_bs_finish(&mut self, builder: Register, loc: Option<LocationId>) -> Register {
        let dest = self.alloc_register();
        self.push(Opcode::BsFinish(BsFinish { dest, builder }), loc);
        dest
    }

    pub fn build_bs_push(
        &mut self,
        builder: Register,
        value: Register,
        size: Option<Register>,
        spec: BinaryEntrySpecifier,
        loc: Option<LocationId>,
    ) -> Register {
        let dest = self.alloc_register();
        self.push(
            Opcode::BsPush(BsPush {
                dest,
                builder,
                value,
                size,
                spec,
            }),
            loc,
        );
        dest
    }

    pub fn build_bs_match_start(
        &mut self,
        bin: Register,
        loc: Option<LocationId>,
    ) -> (Register, Register) {
        let is_err = self.alloc_register();
        let context = self.alloc_register();
        self.push(
            Opcode::BsMatchStart(BsMatchStart {
                is_err,
                context,
                bin,
            }),
            loc,
        );
        (is_err, context)
    }

    pub fn build_bs_match(
        &mut self,
        context: Register,
        size: Option<Register>,
        spec: BinaryEntrySpecifier,
        loc: Option<LocationId>,
    ) -> (Register, Register, Register) {
        let is_err = self.alloc_register();
        let value = self.alloc_register();
        let next = self.alloc_register();
        self.push(
            Opcode::BsMatch(BsMatch {
                is_err,
                value,
                next,
                context,
                size,
                spec,
            }),
            loc,
        );
        (is_err, value, next)
    }

    pub fn build_bs_match_skip(
        &mut self,
        context: Register,
        size: Register,
        spec: BinaryEntrySpecifier,
        value: Register,
        loc: Option<LocationId>,
    ) -> (Register, Register) {
        let is_err = self.alloc_register();
        let next = self.alloc_register();
        let (ty, unit) = match spec {
            BinaryEntrySpecifier::Integer {
                signed: true,
                endianness,
                unit,
            } => {
                let ty = match endianness {
                    Endianness::Big => BsMatchSkipType::BigSigned,
                    Endianness::Little => BsMatchSkipType::LittleSigned,
                    _ => BsMatchSkipType::NativeSigned,
                };
                (ty, unit)
            }
            BinaryEntrySpecifier::Integer {
                endianness, unit, ..
            } => {
                let ty = match endianness {
                    Endianness::Big => BsMatchSkipType::BigUnsigned,
                    Endianness::Little => BsMatchSkipType::LittleUnsigned,
                    _ => BsMatchSkipType::NativeUnsigned,
                };
                (ty, unit)
            }
            spec => panic!(
                "invalid binary spec for bs_match_skip operation: {:?}",
                spec
            ),
        };
        self.push(
            Opcode::BsMatchSkip(BsMatchSkip {
                is_err,
                next,
                context,
                ty,
                size,
                unit,
                value,
            }),
            loc,
        );
        (is_err, next)
    }

    pub fn build_bs_test_tail(
        &mut self,
        context: Register,
        size: usize,
        loc: Option<LocationId>,
    ) -> Register {
        let dest = self.alloc_register();
        self.push(
            Opcode::BsTestTail(BsTestTail {
                dest,
                context,
                size,
            }),
            loc,
        );
        dest
    }

    pub fn build_yield(&mut self, loc: Option<LocationId>) {
        self.push(Opcode::Yield(Yield), loc);
    }

    pub fn build_garbage_collect(&mut self, loc: Option<LocationId>) {
        self.push(
            Opcode::GarbageCollect(GarbageCollect { fullsweep: true }),
            loc,
        );
    }

    pub fn build_raise(
        &mut self,
        kind: Register,
        reason: Register,
        trace: Option<Register>,
        opts: Option<Register>,
        loc: Option<LocationId>,
    ) -> Register {
        let dest = self.alloc_register();
        self.push(
            Opcode::Raise(Raise {
                dest,
                kind,
                reason,
                trace,
                opts,
            }),
            loc,
        );
        dest
    }

    pub fn build_throw(&mut self, reason: Register, loc: Option<LocationId>) {
        self.push(Opcode::Throw1(Throw1 { reason }), loc);
    }

    pub fn build_error(&mut self, reason: Register, loc: Option<LocationId>) {
        self.push(Opcode::Error1(Error1 { reason }), loc);
    }

    pub fn build_exit1(&mut self, reason: Register, loc: Option<LocationId>) {
        self.push(Opcode::Exit1(Exit1 { reason }), loc);
    }

    pub fn build_exit2(
        &mut self,
        pid: Register,
        reason: Register,
        loc: Option<LocationId>,
    ) -> Register {
        let dest = self.alloc_register();
        self.push(Opcode::Exit2(Exit2 { dest, pid, reason }), loc);
        dest
    }

    pub fn build_halt(
        &mut self,
        status: Option<Register>,
        options: Option<Register>,
        loc: Option<LocationId>,
    ) {
        let status = match status {
            Some(reg) => reg,
            None => {
                let dest = self.alloc_register();
                self.push(Opcode::LoadInt(LoadInt { dest, value: 0 }), loc);
                dest
            }
        };
        let options = match options {
            Some(reg) => reg,
            None => {
                let dest = self.alloc_register();
                self.push(Opcode::LoadNil(LoadNil { dest }), loc);
                dest
            }
        };
        self.push(Opcode::Halt(Halt { status, options }), loc);
    }

    pub fn build_self(&mut self, loc: Option<LocationId>) -> Register {
        let dest = self.alloc_register();
        self.push(Opcode::Identity(Identity { dest }), loc);
        dest
    }

    pub fn build_spawn2(
        &mut self,
        fun: Register,
        opts: SpawnOpts,
        loc: Option<LocationId>,
    ) -> Register {
        let dest = self.alloc_register();
        self.push(Opcode::Spawn2(Spawn2 { dest, fun, opts }), loc);
        dest
    }

    pub fn build_spawn3(
        &mut self,
        fun: FunId,
        args: Register,
        opts: SpawnOpts,
        loc: Option<LocationId>,
    ) -> Register {
        let dest = self.alloc_register();
        self.push(
            Opcode::Spawn3(Spawn3 {
                dest,
                fun,
                args,
                opts,
            }),
            loc,
        );
        dest
    }

    pub fn build_spawn3_indirect(
        &mut self,
        module: Register,
        function: Register,
        args: Register,
        opts: SpawnOpts,
        loc: Option<LocationId>,
    ) -> Register {
        let dest = self.alloc_register();
        self.push(
            Opcode::Spawn3Indirect(Spawn3Indirect {
                dest,
                module,
                function,
                args,
                opts,
            }),
            loc,
        );
        dest
    }
}

impl<'a, A, T> Drop for FunctionBuilder<'a, A, T>
where
    A: Atom,
    T: AtomTable<Atom = A>,
{
    fn drop(&mut self) {
        let function_id = self.function;
        let function_offset = self.function_offset(function_id);
        let frame_size = (self.registers + 1) - 2;
        self.builder
            .code
            .set_function_frame_size(function_id, frame_size as usize);

        // Reserve space on the stack for the call frame
        self.builder.code.code.push(Opcode::FuncInfo(FuncInfo {
            id: function_id,
            arity: self.arity,
            frame_size,
        }));

        // Write all blocks, saving their offsets in the code
        let mut block_offset = function_offset + 1;
        for block in self.blocks.iter_mut() {
            block.offset = block_offset;
            // Append code
            self.builder.code.code.append(&mut block.code);
            // Insert corresponding debug info by calculating ranges of instructions covered by the
            // same location
            let mut range_loc = None;
            let mut range_start = block_offset;
            for (i, loc) in block.locations.iter().copied().enumerate() {
                match loc {
                    None => {
                        if let Some(prev_id) = range_loc {
                            self.builder
                                .code
                                .set_instruction_location(range_start, prev_id);
                        }
                        range_start = range_start + i;
                        range_loc = None;
                    }
                    Some(id) => {
                        if let Some(prev_id) = range_loc {
                            if prev_id == id {
                                continue;
                            }
                            self.builder
                                .code
                                .set_instruction_location(range_start, prev_id);
                        }
                        range_start = range_start + i;
                        range_loc = Some(id);
                    }
                }
            }
            block_offset = self.builder.next_instruction();
        }

        // Fix up all block references in the code with their actual offsets, and
        // try to promote static calls to direct jumps when the callee is bytecoded
        let code = &mut self.builder.code.code[(function_offset + 1)..];
        for (i, op) in code.iter_mut().enumerate() {
            let ip = function_offset + 1 + i;
            match op {
                Opcode::Br(Br { ref mut offset })
                | Opcode::Brz(Brz { ref mut offset, .. })
                | Opcode::Brnz(Brnz { ref mut offset, .. })
                | Opcode::JumpTableEntry(JumpTableEntry { ref mut offset, .. })
                | Opcode::LandingPad(LandingPad { ref mut offset, .. }) => {
                    // Locate the offset of the block this instruction occurs in,
                    // and determine the relative offset to the first instruction in
                    // the target block.
                    let block_id = (*offset) as BlockId;
                    assert!(block_id > 0, "cannot branch to entry block");
                    let target_block_offset = self.blocks[block_id as usize].offset;
                    if ip >= target_block_offset {
                        // The current block occurs after target block, so this is a backwards jump
                        let relative: JumpOffset = (ip as isize - target_block_offset as isize)
                            .try_into()
                            .unwrap();
                        *offset = -relative;
                    } else {
                        // The current block occurs before the target block, so this is a forwards
                        // jump
                        let relative = target_block_offset as isize - ip as isize;
                        *offset = relative.try_into().unwrap()
                    }
                }
                Opcode::CallStatic(CallStatic { dest, callee, .. }) => {
                    let dest = *dest;
                    match self.builder.code.functions.get(*callee) {
                        Function::Bytecode {
                            offset,
                            is_nif: false,
                            ..
                        } if *offset > 0 => {
                            drop(mem::replace(
                                op,
                                Opcode::Call(Call {
                                    dest,
                                    offset: *offset,
                                }),
                            ));
                        }
                        _ => continue,
                    }
                }
                Opcode::EnterStatic(EnterStatic { callee, .. }) => {
                    match self.builder.code.functions.get(*callee) {
                        Function::Bytecode {
                            offset,
                            is_nif: false,
                            ..
                        } if *offset > 0 => {
                            drop(mem::replace(op, Opcode::Enter(Enter { offset: *offset })));
                        }
                        _ => continue,
                    }
                }
                _ => continue,
            }
        }
    }
}
