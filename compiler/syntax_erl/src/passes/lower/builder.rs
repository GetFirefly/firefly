use std::cell::RefCell;
use std::rc::Rc;

use cranelift_entity::packed_option::PackedOption;

use liblumen_diagnostics::SourceSpan;
use liblumen_intern::Symbol;
use liblumen_syntax_core::*;
use log::debug;

pub struct IrBuilder<'a> {
    pub func: &'a mut Function,
    pub entry: Block,
    position: PackedOption<Block>,
}
impl<'a> IrBuilder<'a> {
    pub fn new(func: &'a mut Function) -> Self {
        let root_scope = Rc::new(RefCell::new(Scope::default()));
        let entry = func.dfg.make_block(root_scope);
        let position = Some(entry);

        for param_ty in func.signature.params() {
            func.dfg
                .append_block_param(entry, param_ty.clone(), SourceSpan::default());
        }

        Self {
            func,
            entry,
            position: position.into(),
        }
    }

    pub fn current_block(&self) -> Block {
        self.position.expand().unwrap()
    }

    pub fn current_scope(&self) -> Rc<RefCell<Scope>> {
        self.func.dfg.scope(self.current_block())
    }

    pub fn switch_to_block(&mut self, block: Block) {
        debug!("switching to block {:?}", block);
        self.position = PackedOption::from(block);
    }

    pub fn create_block(&mut self) -> Block {
        let scope = Rc::new(RefCell::new(Scope::with_parent(
            self.func.dfg.scope(self.current_block()),
        )));
        let block = self.func.dfg.make_block(scope);
        debug!("created new block {:?}", block);
        block
    }

    pub fn block_params(&self, block: Block) -> &[Value] {
        self.func.dfg.block_params(block)
    }

    pub fn append_block_param(&mut self, block: Block, ty: Type, span: SourceSpan) -> Value {
        self.func.dfg.append_block_param(block, ty, span)
    }

    pub fn get_scope(&self, block: Block) -> Rc<RefCell<Scope>> {
        self.func.dfg.scope(block)
    }

    pub fn set_scope(&mut self, block: Block, scope: Rc<RefCell<Scope>>) {
        self.func.dfg.set_scope(block, scope)
    }

    pub fn define_var(&mut self, name: Symbol, value: Value) {
        let current_block = self.current_block();
        debug!(
            "defined var {} with value {:?} in {:?}",
            name, value, current_block
        );
        self.func.dfg.define_var(current_block, name, value);
    }

    pub fn is_var_defined(&mut self, name: Symbol) -> bool {
        self.get_var(name).is_some()
    }

    pub fn get_var(&self, name: Symbol) -> Option<Value> {
        let current_block = self.current_block();
        let found = self.func.dfg.get_var(current_block, name);
        debug!(
            "looking up var {} in block {:?} found {:?}",
            name, current_block, found
        );
        found
    }

    pub fn define_func(&mut self, name: Symbol, value: FuncRef) {
        let current_block = self.current_block();
        debug!(
            "defining fun {} as {:?} in block {:?}",
            name, value, current_block
        );
        self.func.dfg.define_func(current_block, name, value);
    }

    pub fn get_func(&self, name: Symbol) -> Option<FuncRef> {
        let current_block = self.current_block();
        let found = self.func.dfg.get_func(current_block, name);
        debug!(
            "looking up fun {} in block {:?} found {:?}",
            name, current_block, found
        );
        found
    }

    pub fn get_func_name(&self, name: Symbol) -> Option<FunctionName> {
        if let Some(f) = self.get_func(name) {
            Some(self.func.dfg.callee_signature(f).mfa())
        } else {
            None
        }
    }

    pub fn value_type(&self, value: Value) -> Type {
        self.func.dfg.value_type(value)
    }

    pub fn first_result(&self, inst: Inst) -> Value {
        self.func.dfg.first_result(inst)
    }

    pub fn inst_results(&self, inst: Inst) -> &[Value] {
        self.func.dfg.inst_results(inst)
    }

    pub fn append_inst_args(&mut self, inst: Inst, args: &[Value]) {
        self.func.dfg.append_inst_args(inst, args)
    }

    pub fn get_callee(&self, callee: FunctionName) -> Option<FuncRef> {
        self.func.dfg.get_callee(callee)
    }

    pub fn get_or_register_callee(&self, mfa: FunctionName) -> FuncRef {
        self.func.dfg.register_callee(mfa)
    }

    pub fn make_constant(&mut self, value: ConstantItem) -> Constant {
        self.func.dfg.make_constant(value)
    }

    pub fn ins<'short>(&'short mut self) -> FuncInstBuilder<'short, 'a> {
        let block = self
            .position
            .expect("must be in a block to insert instructions");
        FuncInstBuilder::new(self, block)
    }

    pub(super) fn ensure_inserted_block(&mut self) {
        let block = self.position.unwrap();
        assert!(
            self.func.dfg.is_block_inserted(block),
            "current block is detached from the function!"
        );
    }

    pub(super) fn is_current_block_terminated(&self) -> bool {
        self.is_block_terminated(self.position.expand().unwrap())
    }

    pub(super) fn is_block_terminated(&self, block: Block) -> bool {
        if let Some(inst) = self.func.dfg.last_inst(block) {
            let data = &self.func.dfg[inst];
            data.opcode().is_terminator()
        } else {
            false
        }
    }
}

pub struct FuncInstBuilder<'a, 'b: 'a> {
    builder: &'a mut IrBuilder<'b>,
    block: Block,
}
impl<'a, 'b> FuncInstBuilder<'a, 'b> {
    fn new(builder: &'a mut IrBuilder<'b>, block: Block) -> Self {
        assert!(builder.func.dfg.is_block_inserted(block));

        Self { builder, block }
    }
}
impl<'a, 'b> InstBuilderBase<'a> for FuncInstBuilder<'a, 'b> {
    fn data_flow_graph(&self) -> &DataFlowGraph {
        &self.builder.func.dfg
    }

    fn data_flow_graph_mut(&mut self) -> &mut DataFlowGraph {
        &mut self.builder.func.dfg
    }

    fn build(self, data: InstData, ty: Type, span: SourceSpan) -> (Inst, &'a mut DataFlowGraph) {
        debug_assert!(
            !self.builder.is_block_terminated(self.block),
            "cannot append an instruction to a block that is already terminated"
        );

        let inst = self.builder.func.dfg.push_inst(self.block, data, span);
        self.builder.func.dfg.make_inst_results(inst, ty);

        (inst, &mut self.builder.func.dfg)
    }
}
