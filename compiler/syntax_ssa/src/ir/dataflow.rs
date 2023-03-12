use std::cell::{Ref, RefCell};
use std::collections::BTreeMap;
use std::ops::{Index, IndexMut};
use std::rc::Rc;

use cranelift_entity::{EntityRef, PrimaryMap, SecondaryMap};
use intrusive_collections::UnsafeRef;

use firefly_diagnostics::{SourceSpan, Span};
use firefly_intern::Symbol;
use firefly_syntax_base::*;

use super::*;

#[derive(Clone)]
pub struct DataFlowGraph {
    pub signatures: Rc<RefCell<PrimaryMap<FuncRef, Signature>>>,
    pub callees: Rc<RefCell<BTreeMap<FunctionName, FuncRef>>>,
    pub constants: Rc<RefCell<ConstantPool>>,
    pub blocks: OrderedArenaMap<Block, BlockData>,
    pub insts: ArenaMap<Inst, InstNode>,
    pub inst_annotations: SecondaryMap<Inst, Annotations>,
    pub results: SecondaryMap<Inst, ValueList>,
    pub values: PrimaryMap<Value, ValueData>,
    pub value_lists: ValueListPool,
}
impl DataFlowGraph {
    pub fn new(
        signatures: Rc<RefCell<PrimaryMap<FuncRef, Signature>>>,
        callees: Rc<RefCell<BTreeMap<FunctionName, FuncRef>>>,
        constants: Rc<RefCell<ConstantPool>>,
    ) -> Self {
        Self {
            signatures,
            callees,
            constants,
            insts: ArenaMap::new(),
            inst_annotations: SecondaryMap::new(),
            results: SecondaryMap::new(),
            blocks: OrderedArenaMap::new(),
            values: PrimaryMap::new(),
            value_lists: ValueListPool::new(),
        }
    }

    /// Returns the signature of the given function reference
    pub fn callee_signature(&self, callee: FuncRef) -> Ref<'_, Signature> {
        Ref::map(self.signatures.borrow(), |sigs| sigs.get(callee).unwrap())
    }

    /// Returns the calling convention of `callee`
    pub fn callee_convention(&self, callee: FuncRef) -> CallConv {
        self.signatures
            .borrow()
            .get(callee)
            .map(|sig| sig.calling_convention())
            .unwrap()
    }

    /// Looks up the concrete function for the given MFA (module of None indicates that it is a
    /// local or imported function)
    pub fn get_callee(&self, mfa: FunctionName) -> Option<FuncRef> {
        self.callees.borrow().get(&mfa).copied()
    }

    /// Registers an MFA as a callable function with a default signature
    pub fn register_callee(&self, mfa: FunctionName) -> FuncRef {
        let mut callees = self.callees.borrow_mut();
        // Don't register duplicates
        if let Some(func) = callees.get(&mfa).copied() {
            return func;
        }
        let mut signatures = self.signatures.borrow_mut();
        let func = signatures.push(Signature::generate(&mfa));
        callees.insert(mfa, func);
        func
    }

    pub fn make_constant(&mut self, data: ConstantItem) -> Constant {
        let mut constants = self.constants.borrow_mut();
        constants.insert(data)
    }

    pub fn constant(&self, handle: Constant) -> Ref<'_, ConstantItem> {
        Ref::map(self.constants.borrow(), |pool| pool.get(handle))
    }

    pub fn constant_type(&self, handle: Constant) -> Type {
        let constants = self.constants.borrow();
        constants.get(handle).ty()
    }

    pub fn make_value(&mut self, data: ValueData) -> Value {
        self.values.push(data)
    }

    pub fn values<'a>(&'a self) -> Values {
        Values {
            inner: self.values.iter(),
        }
    }

    pub fn value_is_valid(&self, v: Value) -> bool {
        self.values.is_valid(v)
    }

    pub fn value_type(&self, v: Value) -> Type {
        self.values[v].ty()
    }

    pub fn set_value_type(&mut self, v: Value, ty: Type) {
        self.values[v].set_type(ty)
    }

    pub fn get_value(&self, v: Value) -> ValueData {
        self.values[v].clone()
    }

    pub fn push_inst(&mut self, block: Block, data: InstData, span: SourceSpan) -> Inst {
        let inst = self.insts.alloc_key();
        let node = InstNode::new(inst, block, Span::new(span, data));
        self.insts.append(inst, node);
        self.results.resize(inst.index() + 1);
        let item = unsafe { UnsafeRef::from_raw(&self.insts[inst]) };
        unsafe {
            self.block_data_mut(block).append(item);
        }
        inst
    }

    pub fn inst_args(&self, inst: Inst) -> &[Value] {
        self.insts[inst].arguments(&self.value_lists)
    }

    pub fn inst_args_mut(&mut self, inst: Inst) -> &mut [Value] {
        self.insts[inst].arguments_mut(&mut self.value_lists)
    }

    pub fn append_inst_args(&mut self, inst: Inst, args: &[Value]) {
        let vlist = self.insts[inst]
            .arguments_list()
            .expect("cannot append arguments to instruction with no valuelist");
        vlist.extend(args.iter().copied(), &mut self.value_lists);
    }

    pub fn annotate_inst<A: Into<Annotation>>(&mut self, inst: Inst, key: Symbol, data: A) {
        self.inst_annotations[inst].insert_mut(key, data);
    }

    pub fn make_inst_results(&mut self, inst: Inst, ty: Type) -> usize {
        self.results[inst].clear(&mut self.value_lists);
        let opcode = self.insts[inst].opcode();
        if let Some(fdata) = self.call_signature(inst) {
            // Tail calls are equivalent to return, they don't have results that are materialized as
            // values
            if opcode == Opcode::Enter || opcode == Opcode::EnterIndirect {
                return 0;
            }

            let mut num_results = 0;
            for ty in fdata.results() {
                self.append_result(inst, ty.clone());
                num_results += 1;
            }
            num_results
        } else {
            // Create result values corresponding to the opcode's constraints.
            match self.insts[inst].opcode() {
                // Tail calls have no materialized results
                Opcode::EnterIndirect => 0,
                // An indirect call has no signature, but we know it must be Erlang convention with
                // a single return value
                Opcode::CallIndirect => {
                    self.append_result(inst, ty);
                    1
                }
                // Initializing a binary match is a type check which produces a match context when
                // successful
                Opcode::BitsMatchStart => {
                    self.append_result(inst, Type::Term(TermType::Bool));
                    self.append_result(inst, Type::Term(TermType::Any));
                    2
                }
                // Binary matches produce three results, an error flag, the matched value, and the
                // rest of the binary
                Opcode::BitsMatch => {
                    self.append_result(inst, Type::Term(TermType::Bool));
                    self.append_result(inst, ty);
                    self.append_result(inst, Type::Term(TermType::Any));
                    3
                }
                // This is an optimized form of BitsMatch that skips extraction of the term to be
                // matched and just advances the position in the underlying match
                // context
                Opcode::BitsMatchSkip => {
                    self.append_result(inst, Type::Term(TermType::Bool));
                    self.append_result(inst, Type::Term(TermType::Any));
                    2
                }
                Opcode::BitsInit => {
                    self.append_result(inst, Type::BinaryBuilder);
                    1
                }
                // Binary construction produces two results, an error flag and the new binary value
                Opcode::BitsPush => {
                    self.append_result(inst, Type::BinaryBuilder);
                    1
                }
                Opcode::BitsTestTail => {
                    self.append_result(inst, Type::Term(TermType::Bool));
                    1
                }
                Opcode::BitsFinish => {
                    self.append_result(inst, Type::Term(TermType::Bitstring));
                    1
                }
                // Constants/immediates have known types
                Opcode::ImmInt
                | Opcode::ImmFloat
                | Opcode::ImmBool
                | Opcode::ImmAtom
                | Opcode::ImmNil
                | Opcode::ImmNone
                | Opcode::ImmNull
                | Opcode::ConstBigInt
                | Opcode::ConstBinary => {
                    self.append_result(inst, ty);
                    1
                }
                Opcode::IsNull => {
                    self.append_result(inst, Type::Primitive(PrimitiveType::I1));
                    1
                }
                // These arithmetic operators always return integers
                Opcode::Bnot
                | Opcode::Band
                | Opcode::Bor
                | Opcode::Bsl
                | Opcode::Bsr
                | Opcode::Div
                | Opcode::Rem => {
                    self.append_result(inst, Type::Term(TermType::Integer));
                    1
                }
                // These arithmetic operators always return floats
                Opcode::Fdiv => {
                    self.append_result(inst, Type::Term(TermType::Float));
                    1
                }
                // These binary arithmetic operators are polymorphic on their argument types
                Opcode::Add | Opcode::Sub | Opcode::Mul => {
                    let (lhs, rhs) = {
                        let args = self.inst_args(inst);
                        (args[0], args[1])
                    };
                    let lhs_ty = self.value_type(lhs);
                    let rhs_ty = self.value_type(rhs);
                    let ty = lhs_ty
                        .as_term()
                        .unwrap()
                        .coerce_to_numeric_with(rhs_ty.as_term().unwrap_or(TermType::Any));
                    self.append_result(inst, Type::Term(ty));
                    1
                }
                // These unary arithmetic operators are polymorphic on their argument type
                Opcode::Neg => {
                    let arg = self.inst_args(inst)[0];
                    let ty = self.value_type(arg).as_term().unwrap().coerce_to_numeric();
                    self.append_result(inst, Type::Term(ty));
                    1
                }
                // Casts produce a single output from a single input
                Opcode::Cast => {
                    self.append_result(inst, ty);
                    1
                }
                // These unary integer operators always produce primitive type outputs
                Opcode::Trunc | Opcode::Zext => {
                    self.append_result(inst, ty);
                    1
                }
                // This type check is a fused operation that also produces an arity value
                Opcode::IsTupleFetchArity => {
                    self.append_result(inst, Type::Term(TermType::Bool));
                    self.append_result(inst, Type::Term(TermType::Any));
                    2
                }
                // These boolean operators always produce primitive boolean outputs
                Opcode::IcmpEq
                | Opcode::IcmpNeq
                | Opcode::IcmpGt
                | Opcode::IcmpGte
                | Opcode::IcmpLt
                | Opcode::IcmpLte
                | Opcode::IsType
                | Opcode::IsFunctionWithArity
                | Opcode::IsTaggedTuple => {
                    self.append_result(inst, Type::Term(TermType::Bool));
                    1
                }
                // These boolean operators always produce boolean term outputs
                Opcode::Eq
                | Opcode::EqExact
                | Opcode::Neq
                | Opcode::NeqExact
                | Opcode::Gt
                | Opcode::Gte
                | Opcode::Lt
                | Opcode::Lte
                | Opcode::And
                | Opcode::AndAlso
                | Opcode::Or
                | Opcode::OrElse
                | Opcode::Not => {
                    self.append_result(inst, Type::Term(TermType::Bool));
                    1
                }
                // These ops have specific types they produce
                Opcode::Cons => {
                    self.append_result(inst, Type::Term(TermType::Cons));
                    1
                }
                Opcode::ListConcat | Opcode::ListSubtract => {
                    self.append_result(inst, Type::Term(TermType::List(None)));
                    1
                }
                Opcode::Head | Opcode::GetElement => {
                    self.append_result(inst, Type::Term(TermType::Any));
                    1
                }
                Opcode::Tail => {
                    self.append_result(inst, Type::Term(TermType::MaybeImproperList));
                    1
                }
                Opcode::Split => {
                    self.append_result(inst, Type::Term(TermType::Any));
                    self.append_result(inst, Type::Term(TermType::MaybeImproperList));
                    2
                }
                Opcode::Tuple | Opcode::SetElement | Opcode::SetElementMut => {
                    self.append_result(inst, Type::Term(TermType::Tuple(None)));
                    1
                }
                Opcode::Map
                | Opcode::MapPut
                | Opcode::MapUpdate
                | Opcode::MapExtendPut
                | Opcode::MapExtendUpdate => {
                    self.append_result(inst, Type::Term(TermType::Map));
                    1
                }
                Opcode::MapTryGet => {
                    self.append_result(inst, Type::Term(TermType::Bool));
                    self.append_result(inst, Type::Term(TermType::Any));
                    2
                }
                Opcode::BuildStacktrace => {
                    self.append_result(inst, ty);
                    1
                }
                Opcode::MakeFun => {
                    self.append_result(inst, Type::Term(TermType::Fun(None)));
                    1
                }
                Opcode::UnpackEnv => {
                    self.append_result(inst, ty);
                    1
                }
                Opcode::RecvPeek => {
                    // This primop returns a boolean indicating whether a message was available,
                    // and the message itself if available
                    self.append_result(inst, Type::Term(TermType::Bool));
                    self.append_result(inst, Type::Term(TermType::Any));
                    2
                }
                Opcode::RecvWaitTimeout => {
                    // This primop returns a boolean indicating whether the wait timed out
                    self.append_result(inst, Type::Term(TermType::Bool));
                    1
                }
                Opcode::Send => {
                    self.append_result(inst, Type::Term(TermType::Any));
                    1
                }
                Opcode::Exit2 => {
                    self.append_result(inst, Type::Term(TermType::Bool));
                    1
                }
                Opcode::Raise => {
                    self.append_result(inst, Type::Term(TermType::Atom));
                    1
                }
                Opcode::Yield => {
                    self.append_result(inst, ty);
                    1
                }
                _ => 0,
            }
        }
    }

    pub fn append_result(&mut self, inst: Inst, ty: Type) -> Value {
        let res = self.values.next_key();
        let num = self.results[inst].push(res, &mut self.value_lists);
        debug_assert!(num <= u16::MAX as usize, "too many result values");
        self.make_value(ValueData::Inst {
            ty,
            inst,
            num: num as u16,
        })
    }

    pub fn first_result(&self, inst: Inst) -> Value {
        self.results[inst]
            .first(&self.value_lists)
            .expect("instruction has no results")
    }

    pub fn has_results(&self, inst: Inst) -> bool {
        !self.results[inst].is_empty()
    }

    pub fn inst_results(&self, inst: Inst) -> &[Value] {
        self.results[inst].as_slice(&self.value_lists)
    }

    pub fn call_signature(&self, inst: Inst) -> Option<Signature> {
        match self.insts[inst].analyze_call(&self.value_lists) {
            CallInfo::NotACall => None,
            CallInfo::Indirect(_, _, _) => None,
            CallInfo::Direct(f, _, _) => Some(self.callee_signature(f).clone()),
        }
    }

    pub fn analyze_call(&self, inst: Inst) -> CallInfo<'_> {
        self.insts[inst].analyze_call(&self.value_lists)
    }

    pub fn analyze_branch(&self, inst: Inst) -> BranchInfo {
        self.insts[inst].analyze_branch(&self.value_lists)
    }

    pub fn blocks<'f>(&'f self) -> impl Iterator<Item = (Block, &'f BlockData)> {
        Blocks {
            cursor: self.blocks.cursor(),
        }
    }

    pub fn block_insts<'f>(&'f self, block: Block) -> impl Iterator<Item = Inst> + 'f {
        self.blocks[block].insts()
    }

    pub fn block_data(&self, block: Block) -> &BlockData {
        &self.blocks[block]
    }

    pub fn block_data_mut(&mut self, block: Block) -> &mut BlockData {
        &mut self.blocks[block]
    }

    pub fn last_inst(&self, block: Block) -> Option<Inst> {
        self.blocks[block].last()
    }

    pub fn is_block_inserted(&self, block: Block) -> bool {
        self.blocks.contains(block)
    }

    pub fn is_block_empty(&self, block: Block) -> bool {
        self.blocks[block].is_empty()
    }

    pub fn make_block(&mut self) -> Block {
        self.blocks.push(BlockData::new())
    }

    pub fn remove_block(&mut self, block: Block) {
        self.blocks.remove(block);
    }

    pub fn num_block_params(&self, block: Block) -> usize {
        self.blocks[block].params.len(&self.value_lists)
    }

    pub fn block_params(&self, block: Block) -> &[Value] {
        self.blocks[block].params.as_slice(&self.value_lists)
    }

    pub fn block_param_types(&self, block: Block) -> Vec<Type> {
        self.block_params(block)
            .iter()
            .map(|&v| self.value_type(v))
            .collect()
    }

    pub fn append_block_param(&mut self, block: Block, ty: Type, span: SourceSpan) -> Value {
        let param = self.values.next_key();
        let num = self.blocks[block].params.push(param, &mut self.value_lists);
        debug_assert!(num <= u16::MAX as usize, "too many parameters on block");
        self.make_value(ValueData::Param {
            ty,
            num: num as u16,
            block,
            span,
        })
    }
}
impl Index<Inst> for DataFlowGraph {
    type Output = Span<InstData>;

    fn index(&self, inst: Inst) -> &Span<InstData> {
        &self.insts[inst]
    }
}
impl IndexMut<Inst> for DataFlowGraph {
    fn index_mut(&mut self, inst: Inst) -> &mut Span<InstData> {
        &mut self.insts[inst]
    }
}

struct Blocks<'f> {
    cursor: intrusive_collections::linked_list::Cursor<'f, LayoutAdapter<Block, BlockData>>,
}
impl<'f> Iterator for Blocks<'f> {
    type Item = (Block, &'f BlockData);

    fn next(&mut self) -> Option<Self::Item> {
        if self.cursor.is_null() {
            return None;
        }
        let next = self.cursor.get().map(|data| (data.key(), data.value()));
        self.cursor.move_next();
        next
    }
}
