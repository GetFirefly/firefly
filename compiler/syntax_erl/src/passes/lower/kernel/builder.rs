use std::collections::HashMap;

use cranelift_entity::packed_option::PackedOption;

use liblumen_diagnostics::SourceSpan;
use liblumen_intern::Symbol;
use liblumen_syntax_core::*;
use log::debug;

pub struct IrBuilder<'a> {
    pub func: &'a mut Function,
    pub entry: Block,
    position: PackedOption<Block>,
    vars: HashMap<Symbol, Value>,
}
impl<'a> IrBuilder<'a> {
    pub fn new(func: &'a mut Function) -> Self {
        let entry = func.dfg.make_block();
        let position = Some(entry);

        for param_ty in func.signature.params() {
            func.dfg
                .append_block_param(entry, param_ty.clone(), SourceSpan::default());
        }

        Self {
            func,
            entry,
            position: position.into(),
            vars: HashMap::new(),
        }
    }

    pub fn current_block(&self) -> Block {
        self.position.expand().unwrap()
    }

    pub fn switch_to_block(&mut self, block: Block) {
        debug!("switching to block {:?}", block);
        self.position = PackedOption::from(block);
    }

    pub fn create_block(&mut self) -> Block {
        let block = self.func.dfg.make_block();
        debug!("created new block {:?}", block);
        block
    }

    pub fn append_block_param(&mut self, block: Block, ty: Type, span: SourceSpan) -> Value {
        self.func.dfg.append_block_param(block, ty, span)
    }

    pub fn first_result(&self, inst: Inst) -> Value {
        self.func.dfg.first_result(inst)
    }

    pub fn inst_results(&self, inst: Inst) -> &[Value] {
        self.func.dfg.inst_results(inst)
    }

    pub fn get_callee(&self, callee: FunctionName) -> Option<FuncRef> {
        self.func.dfg.get_callee(callee)
    }

    pub fn get_or_register_callee(&self, mfa: FunctionName) -> FuncRef {
        self.func.dfg.register_callee(mfa)
    }

    /// Associates `name` to the given value, returning the previous value
    /// if one was present.
    pub fn define_var(&mut self, name: Symbol, v: Value) -> Option<Value> {
        use std::collections::hash_map::Entry;

        match self.vars.entry(name) {
            Entry::Vacant(e) => {
                e.insert(v);
                None
            }
            Entry::Occupied(mut e) => Some(e.insert(v)),
        }
    }

    /// Returns the value bound to the given name
    pub fn var(&self, name: Symbol) -> Option<Value> {
        self.vars.get(&name).copied()
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
