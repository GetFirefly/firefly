use std::cell::{Ref, RefCell};
use std::collections::BTreeMap;
use std::io::{self, Write};
use std::ops::{Index, IndexMut};
use std::rc::Rc;

use cranelift_entity::{EntityRef, PrimaryMap, SecondaryMap};
use intrusive_collections::UnsafeRef;

use liblumen_diagnostics::{SourceSpan, Span};
use liblumen_intern::Symbol;

use super::*;

#[derive(Clone)]
pub struct DataFlowGraph {
    pub annotations: Rc<RefCell<PrimaryMap<Annotation, AnnotationData>>>,
    pub signatures: Rc<RefCell<PrimaryMap<FuncRef, Signature>>>,
    pub callees: Rc<RefCell<BTreeMap<FunctionName, FuncRef>>>,
    pub constants: Rc<RefCell<ConstantPool>>,
    pub blocks: OrderedArenaMap<Block, BlockData>,
    pub insts: ArenaMap<Inst, InstNode>,
    pub inst_annotations: SecondaryMap<Inst, AnnotationList>,
    pub results: SecondaryMap<Inst, ValueList>,
    pub values: PrimaryMap<Value, ValueData>,
    pub scopes: SecondaryMap<Block, Rc<RefCell<Scope>>>,
    pub annotation_lists: AnnotationListPool,
    pub value_lists: ValueListPool,
}
impl DataFlowGraph {
    pub fn new(
        annotations: Rc<RefCell<PrimaryMap<Annotation, AnnotationData>>>,
        signatures: Rc<RefCell<PrimaryMap<FuncRef, Signature>>>,
        callees: Rc<RefCell<BTreeMap<FunctionName, FuncRef>>>,
        constants: Rc<RefCell<ConstantPool>>,
    ) -> Self {
        Self {
            annotations,
            signatures,
            callees,
            constants,
            insts: ArenaMap::new(),
            inst_annotations: SecondaryMap::new(),
            results: SecondaryMap::new(),
            blocks: OrderedArenaMap::new(),
            values: PrimaryMap::new(),
            scopes: SecondaryMap::new(),
            annotation_lists: AnnotationListPool::new(),
            value_lists: ValueListPool::new(),
        }
    }

    /// Returns the signature of the given function reference
    pub fn callee_signature(&self, callee: FuncRef) -> Ref<'_, Signature> {
        Ref::map(self.signatures.borrow(), |sigs| sigs.get(callee).unwrap())
    }

    /// Looks up the concrete function for the given MFA (module of None indicates that it is a local or imported function)
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

    pub fn scope(&self, block: Block) -> Rc<RefCell<Scope>> {
        Rc::clone(self.scopes.get(block).unwrap())
    }

    pub fn get_var(&self, block: Block, name: Symbol) -> Option<Value> {
        let scope = self.scopes.get(block).unwrap();
        let cell = scope.as_ref();
        let scope_ref = cell.borrow();
        scope_ref.var(name)
    }

    pub fn define_var(&mut self, block: Block, name: Symbol, value: Value) {
        let scope = self.scopes.get(block).unwrap();
        let cell = scope.as_ref();
        let mut scope_mut = cell.borrow_mut();
        scope_mut.define_var(name, value);
    }

    pub fn get_func(&self, block: Block, name: Symbol) -> Option<FuncRef> {
        let scope = self.scopes.get(block).unwrap();
        let cell = scope.as_ref();
        let scope_ref = cell.borrow();
        scope_ref.function(name)
    }

    pub fn define_func(&mut self, block: Block, name: Symbol, value: FuncRef) {
        let scope = self.scopes.get(block).unwrap();
        let cell = scope.as_ref();
        let mut scope_mut = cell.borrow_mut();
        scope_mut.define_function(name, value);
    }

    pub fn set_scope(&mut self, block: Block, scope: Rc<RefCell<Scope>>) {
        self.scopes[block] = scope;
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

    pub fn display_constant(&self, f: &mut dyn Write, handle: Constant) -> io::Result<()> {
        let constants = self.constants.borrow();
        constants.display(f, handle)
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

    /// Create an annotation to be associated to entities in the module
    pub fn make_annotation(&mut self, data: AnnotationData) -> Annotation {
        self.annotations.borrow_mut().push(data)
    }

    pub fn annotate_inst(&mut self, inst: Inst, data: AnnotationData) -> Annotation {
        // Since this is so common, we use the first
        if data == AnnotationData::CompilerGenerated {
            let anno = Annotation::COMPILER_GENERATED;
            self.inst_annotations[inst].push(anno, &mut self.annotation_lists);
            anno
        } else {
            let mut annos = self.annotations.borrow_mut();
            let anno = annos.next_key();
            let num = self.inst_annotations[inst].push(anno, &mut self.annotation_lists);
            debug_assert!(num <= u16::MAX as usize, "too many annotations");
            annos.push(data)
        }
    }

    pub fn make_inst_results(&mut self, inst: Inst, ty: Type) -> usize {
        self.results[inst].clear(&mut self.value_lists);
        if let Some(fdata) = self.call_signature(inst) {
            // Erlang functions use a multi-value return calling convention
            let mut num_results = 0;
            for ty in fdata.results() {
                self.append_result(inst, ty.clone());
                num_results += 1;
            }
            num_results
        } else {
            // Create result values corresponding to the opcode's constraints.
            match self.insts[inst].opcode() {
                // An indirect call has no signature, but we know it must be Erlang
                // convention, and thus multi-value return
                Opcode::CallIndirect => {
                    self.append_result(inst, Type::Primitive(PrimitiveType::I1));
                    self.append_result(inst, ty);
                    2
                }
                // Binary matches produce three results, a success flag, the matched value, and the rest of the binary
                Opcode::BitsMatch => {
                    self.append_result(inst, Type::Primitive(PrimitiveType::I1));
                    self.append_result(inst, ty);
                    self.append_result(inst, Type::Term(TermType::Bitstring));
                    3
                }
                // Binary construction produces two results, a success flag and the new binary value
                Opcode::BitsPush => {
                    self.append_result(inst, Type::Primitive(PrimitiveType::I1));
                    // This value is either the none term or an exception, depending on the is_err flag
                    self.append_result(inst, Type::Term(TermType::Any));
                    2
                }
                // When a binary is constructed, a single result is returned, the constructed binary
                Opcode::BitsCloseWritable => {
                    self.append_result(inst, Type::Primitive(PrimitiveType::I1));
                    // This value is always a bitstring when the is_err flag is not set
                    self.append_result(inst, Type::Term(TermType::Any));
                    2
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
                | Opcode::ConstBinary
                | Opcode::ConstTuple
                | Opcode::ConstList
                | Opcode::ConstMap => {
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
                        .coerce_to_numeric_with(rhs_ty.as_term().unwrap());
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
                // These boolean operators always produce primitive boolean outputs
                Opcode::IcmpEq
                | Opcode::IcmpNeq
                | Opcode::IcmpGt
                | Opcode::IcmpGte
                | Opcode::IcmpLt
                | Opcode::IcmpLte
                | Opcode::IsType
                | Opcode::IsTaggedTuple => {
                    self.append_result(inst, Type::Primitive(PrimitiveType::I1));
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
                Opcode::Cons | Opcode::ListConcat | Opcode::ListSubtract => {
                    self.append_result(inst, Type::Term(TermType::List(None)));
                    1
                }
                Opcode::Head | Opcode::GetElement | Opcode::BuildStacktrace => {
                    self.append_result(inst, Type::Term(TermType::Any));
                    1
                }
                Opcode::Tail => {
                    self.append_result(inst, Type::Term(TermType::MaybeImproperList));
                    1
                }
                Opcode::Tuple | Opcode::SetElement | Opcode::SetElementMut => {
                    self.append_result(inst, Type::Term(TermType::Tuple(None)));
                    1
                }
                Opcode::MapGet => {
                    self.append_result(inst, Type::Term(TermType::Any));
                    1
                }
                Opcode::Map
                | Opcode::MapPut
                | Opcode::MapPutMut
                | Opcode::MapUpdate
                | Opcode::MapUpdateMut => {
                    self.append_result(inst, Type::Term(TermType::Map));
                    1
                }
                Opcode::MakeFun | Opcode::CaptureFun => {
                    self.append_result(inst, Type::Term(TermType::Fun(None)));
                    1
                }
                Opcode::RecvStart => {
                    // This primop returns a receive context
                    self.append_result(inst, Type::RecvContext);
                    1
                }
                Opcode::RecvNext => {
                    // This opcode returns the receive state machine state
                    self.append_result(inst, Type::RecvState);
                    1
                }
                Opcode::RecvPeek => {
                    // This primop returns the current message which the receive is inspecting
                    self.append_result(inst, Type::Term(TermType::Any));
                    1
                }
                Opcode::BitsInitWritable => {
                    self.append_result(inst, Type::BinaryBuilder);
                    1
                }
                Opcode::ExceptionClass => {
                    self.append_result(inst, Type::Term(TermType::Atom));
                    1
                }
                Opcode::ExceptionReason => {
                    self.append_result(inst, Type::Term(TermType::Any));
                    1
                }
                Opcode::ExceptionTrace => {
                    self.append_result(inst, Type::Term(TermType::List(None)));
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
            CallInfo::Indirect(_, _) => None,
            CallInfo::Direct(f, _) => Some(self.callee_signature(f).clone()),
        }
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

    pub fn make_block(&mut self, scope: Rc<RefCell<Scope>>) -> Block {
        let block = self.blocks.push(BlockData::new());
        self.scopes[block] = scope;
        block
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
