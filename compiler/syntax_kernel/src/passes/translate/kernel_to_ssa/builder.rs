use std::collections::{HashMap, HashSet};

use cranelift_entity::packed_option::PackedOption;

use firefly_diagnostics::SourceSpan;
use firefly_intern::Symbol;
use firefly_syntax_base::*;
use firefly_syntax_ssa::*;
use log::debug;

pub struct IrBuilder<'a> {
    pub func: &'a mut Function,
    pub entry: Block,
    position: PackedOption<Block>,
    vars: HashMap<Symbol, Value>,
    var_types: HashMap<Value, Type>,
    reachable_blocks: HashSet<Block>,
}
impl<'a> IrBuilder<'a> {
    pub fn new(func: &'a mut Function) -> Self {
        let entry = func.dfg.make_block();
        let position = Some(entry);

        let mut reachable_blocks = HashSet::new();
        reachable_blocks.insert(entry);

        Self {
            func,
            entry,
            position: position.into(),
            vars: HashMap::new(),
            var_types: HashMap::new(),
            reachable_blocks,
        }
    }

    #[inline]
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

    pub fn remove_block(&mut self, block: Block) {
        assert!(
            block != self.current_block(),
            "cannot remove block the builder is currently inserting in"
        );
        self.func.dfg.remove_block(block);
    }

    #[allow(unused)]
    pub fn prune_unreachable_blocks(&mut self) {
        // Find the set of unreachable blocks
        let mut unreachable = Vec::new();
        {
            for (block, _) in self.func.dfg.blocks() {
                if !self.reachable_blocks.contains(&block) {
                    unreachable.push(block);
                }
            }
        }
        // Then remove them
        for block in unreachable.drain(..) {
            self.func.dfg.remove_block(block);
        }
    }

    #[inline]
    pub fn append_block_param(&mut self, block: Block, ty: Type, span: SourceSpan) -> Value {
        self.func.dfg.append_block_param(block, ty, span)
    }

    #[inline]
    pub fn is_block_empty(&mut self, block: Block) -> bool {
        self.func.dfg.is_block_empty(block)
    }

    #[inline]
    pub fn inst_results(&self, inst: Inst) -> &[Value] {
        self.func.dfg.inst_results(inst)
    }

    #[inline]
    pub fn first_result(&self, inst: Inst) -> Value {
        self.func.dfg.first_result(inst)
    }

    #[inline]
    pub fn get_callee(&self, callee: FunctionName) -> Option<FuncRef> {
        self.func.dfg.get_callee(callee)
    }

    #[allow(unused)]
    #[inline]
    pub fn get_callee_signature(&self, callee: FuncRef) -> std::cell::Ref<'_, Signature> {
        self.func.dfg.callee_signature(callee)
    }

    #[inline]
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

    /// Sets the known type of the given variable
    pub fn set_var_type(&mut self, var: Symbol, ty: Type) {
        let var = self.var(var).unwrap();
        self.var_types.insert(var, ty);
    }

    /// Sets the known type of the given value
    #[allow(unused)]
    pub fn set_value_type(&mut self, value: Value, ty: Type) {
        self.func.dfg.set_value_type(value, ty)
    }

    /// Returns the value bound to the given name
    pub fn var(&self, name: Symbol) -> Option<Value> {
        self.vars.get(&name).copied()
    }

    /// Returns the type (if known) bound to the given variable
    #[allow(unused)]
    pub fn var_type(&self, var: Symbol) -> Option<&Type> {
        self.var(var).and_then(|v| self.var_types.get(&v))
    }

    /// Returns the type associated with the given value
    #[allow(unused)]
    pub fn value_type(&self, var: Value) -> Type {
        match self.var_types.get(&var) {
            None => self.func.dfg.value_type(var),
            Some(t) => t.clone(),
        }
    }

    pub fn ins<'short>(&'short mut self) -> FuncInstBuilder<'short, 'a> {
        let block = self
            .position
            .expect("must be in a block to insert instructions");
        FuncInstBuilder::new(self, block)
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

        match self.builder.func.dfg.analyze_branch(inst) {
            BranchInfo::SingleDest(blk, _) => {
                self.builder.reachable_blocks.insert(blk);
            }
            BranchInfo::MultiDest(blks) => {
                for blk in blks.iter().map(|jt| jt.destination) {
                    self.builder.reachable_blocks.insert(blk);
                }
            }
            BranchInfo::NotABranch => (),
        }

        (inst, &mut self.builder.func.dfg)
    }
}
