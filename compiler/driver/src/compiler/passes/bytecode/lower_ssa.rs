use std::borrow::Borrow;
use std::mem;

use anyhow::bail;

use log::debug;

use rustc_hash::FxHasher;

use smallvec::SmallVec;

use firefly_binary::Bitstring;
use firefly_bytecode as bc;
use firefly_bytecode::ops::SpawnOpts;
use firefly_bytecode::{
    InvalidBytecodeError, LocationId, ModuleFunctionArity, Register, StandardByteCode,
};
use firefly_intern::symbols;
use firefly_number::Int;
use firefly_pass::Pass;
use firefly_session::Options;
use firefly_syntax_base::Signature;
use firefly_syntax_ssa as syntax_ssa;
use firefly_syntax_ssa::ir::instructions::*;
use firefly_syntax_ssa::DataFlowGraph;
use firefly_syntax_ssa::{ConstantItem, Immediate, ImmediateTerm};
use firefly_util::diagnostics::{
    CodeMap, DiagnosticsHandler, FileName, Severity, SourceSpan, Spanned,
};

use crate::compiler::Artifact;

type HashMap<K, V> = std::collections::HashMap<K, V, core::hash::BuildHasherDefault<FxHasher>>;

type Builder = bc::Builder<bc::AtomicStr, bc::LocalAtomTable>;
type FunctionBuilder<'a> = bc::FunctionBuilder<'a, bc::AtomicStr, bc::LocalAtomTable>;

pub struct LowerSsa<'a> {
    options: &'a Options,
    diagnostics: &'a DiagnosticsHandler,
    codemap: &'a CodeMap,
}
impl<'a> LowerSsa<'a> {
    pub fn new(
        options: &'a Options,
        diagnostics: &'a DiagnosticsHandler,
        codemap: &'a CodeMap,
    ) -> Self {
        Self {
            options,
            diagnostics,
            codemap,
        }
    }
}
impl<'m> Pass for LowerSsa<'m> {
    type Input<'a> = Vec<Artifact<syntax_ssa::Module>>;
    type Output<'a> = StandardByteCode;

    fn run<'a>(&mut self, mut modules: Self::Input<'a>) -> anyhow::Result<Self::Output<'a>> {
        debug!("building bytecode for {} modules", modules.len());

        let mut builder = BytecodeBuilder::new(self.diagnostics, self.codemap);
        let mut bytecode = Builder::new(StandardByteCode::default());

        for Artifact { output: module, .. } in modules.drain(..) {
            builder.build_module(&mut bytecode, module)?;
        }

        let module = bytecode.finish();

        match module.validate() {
            Ok(_) => {
                if let Some(path) = self.options.maybe_emit_bytecode() {
                    crate::compiler::emit_file_with_callback(path, |f| {
                        use std::io::Write;
                        f.write_fmt(format_args!("{}", &module))?;
                        Ok(())
                    })?;
                }
                Ok(module)
            }
            Err(InvalidBytecodeError::IncompleteFunction(mfa)) => {
                self.diagnostics
                    .diagnostic(Severity::Error)
                    .with_message(format!("missing function definition for {:?}", &mfa))
                    .emit();
                bail!("bytecode validation failed, see diagnostics for details");
            }
            Err(_) => unreachable!(),
        }
    }
}

struct BytecodeBuilder<'a> {
    diagnostics: &'a DiagnosticsHandler,
    codemap: &'a CodeMap,
    current_module: Option<syntax_ssa::Module>,
    // The current syntax_ssa block being translated
    current_source_block: syntax_ssa::Block,
    // The current MLIR block being built
    current_block: bc::BlockId,
    // Used to track the mapping of blocks in the current function being translated
    blocks: HashMap<syntax_ssa::Block, bc::BlockId>,
    // Used to track the mapping of values in the current function being translated
    values: HashMap<syntax_ssa::Value, bc::Register>,
}
impl<'a> BytecodeBuilder<'a> {
    fn new(diagnostics: &'a DiagnosticsHandler, codemap: &'a CodeMap) -> Self {
        Self {
            diagnostics,
            codemap,
            current_module: None,
            current_source_block: syntax_ssa::Block::default(),
            current_block: bc::BlockId::default(),
            blocks: HashMap::default(),
            values: HashMap::default(),
        }
    }

    fn build_module(
        &mut self,
        builder: &mut Builder,
        mut module: syntax_ssa::Module,
    ) -> anyhow::Result<()> {
        let module_name = module.name();
        debug!("translating {} to bytecode", module_name);

        let functions = mem::take(&mut module.functions);
        self.current_module = Some(module);

        let mut invalid = false;
        for f in functions.iter() {
            let mfa = f.signature.mfa();
            let mfa = bc::ModuleFunctionArity {
                module: builder.insert_atom(mfa.module.unwrap().as_str().get()),
                function: builder.insert_atom(mfa.function.as_str().get()),
                arity: mfa.arity,
            };
            let loc = self.location_from_span(builder, f.span);
            let result = builder.build_function(mfa, loc);
            if let Ok(mut fb) = result {
                self.build_function(&mut fb, f)?;
            } else {
                match unsafe { result.unwrap_err_unchecked() } {
                    InvalidBytecodeError::DuplicateDefinition(mfa) => {
                        let id = builder.function_by_mfa(&mfa).unwrap().id();
                        let span = builder
                            .function_location(id)
                            .and_then(|loc| {
                                let filename = FileName::real(&*loc.file);
                                self.codemap
                                    .get_file_id(&filename)
                                    .map(|file_id| (file_id, loc.line, loc.column))
                            })
                            .and_then(|(f, l, c)| self.codemap.line_column_to_span(f, l, c).ok());
                        match span {
                            None => {
                                self.diagnostics
                                    .diagnostic(Severity::Error)
                                    .with_message(format!("attempted to redefine {:?}", &mfa))
                                    .with_primary_label(f.span(), "redefinition occurs here")
                                    .emit();
                            }
                            Some(span) => {
                                self.diagnostics
                                    .diagnostic(Severity::Error)
                                    .with_message(format!("attempted to redefine {:?}", &mfa))
                                    .with_primary_label(f.span(), "redefinition occurs here")
                                    .with_secondary_label(span, "original definition occurs here")
                                    .emit();
                            }
                        }
                        invalid = true;
                        continue;
                    }
                    _ => unreachable!(),
                }
            }
        }

        if invalid {
            bail!("errors were found while lowering to bytecode, see diagnostics for details");
        }

        debug!("finished translating {}", module_name);

        Ok(())
    }

    fn find_function(&self, f: syntax_ssa::FuncRef) -> Signature {
        self.current_module
            .as_ref()
            .unwrap()
            .call_signature(f)
            .clone()
    }

    /// Switches the builder to the MLIR block corresponding to the given syntax_ssa block
    fn switch_to_block(&mut self, block: syntax_ssa::Block) {
        debug!("switching builder to block {:?}", block);
        self.current_source_block = block;
        self.current_block = self.blocks[&block];
    }

    fn location_from_span(&self, builder: &mut Builder, span: SourceSpan) -> Option<bc::Location> {
        if span.is_unknown() {
            return None;
        }
        // Get the source file in which this span belongs
        let source_file = self
            .codemap
            .get_with_span(span)
            .expect("invalid source span, no corresponding source file!");
        // Get the location (i.e. line/col index) which this span represents
        let loc = self.codemap.location_for_span(span).unwrap();
        // Convert the source file name to an Rc<str>
        let source_filename = source_file.name();
        if let Some(filename) = source_filename.as_str() {
            let file = builder.get_or_insert_file(filename);
            Some(bc::Location {
                file,
                line: loc.line.number().to_usize() as u32,
                column: (loc.column.to_usize() + 1) as u32,
            })
        } else {
            let filename = source_filename.to_string();
            let file = builder.get_or_insert_file(filename.as_str());
            Some(bc::Location {
                file,
                line: loc.line.number().to_usize() as u32,
                column: (loc.column.to_usize() + 1) as u32,
            })
        }
    }

    #[inline]
    fn location_id_from_span(
        &self,
        builder: &mut FunctionBuilder,
        span: SourceSpan,
    ) -> Option<bc::LocationId> {
        let loc = self.location_from_span(builder.builder, span)?;
        Some(builder.builder.get_or_insert_location(loc))
    }

    fn load_immediate(
        &self,
        builder: &mut FunctionBuilder,
        imm: Immediate,
        loc: Option<LocationId>,
    ) -> Register {
        match imm {
            Immediate::Term(imm_term) => match imm_term {
                ImmediateTerm::Bool(b) => builder.build_bool(b, loc),
                ImmediateTerm::Atom(a) => builder.build_atom(a.as_str().get(), loc),
                ImmediateTerm::Integer(i) => builder.build_int(i, loc),
                ImmediateTerm::Float(f) => builder.build_float(f, loc),
                ImmediateTerm::Nil => builder.build_nil(loc),
                imm => panic!("invalid immediate: {:?}", imm),
            },
            Immediate::Isize(n) => {
                builder.build_int(n.try_into().expect("invalid immediate isize"), loc)
            }
            Immediate::I64(n) if n < Int::MIN_SMALL || n > Int::MAX_SMALL => {
                builder.build_bigint(n.into(), loc)
            }
            Immediate::I64(n) => builder.build_int(n, loc),
            Immediate::I32(n) => builder.build_int(n as i64, loc),
            Immediate::I16(n) => builder.build_int(n as i64, loc),
            Immediate::I8(n) => builder.build_int(n as i64, loc),
            Immediate::I1(b) => builder.build_bool(b, loc),
            Immediate::F64(f) => builder.build_float(f, loc),
        }
    }

    fn load_constant(
        &self,
        builder: &mut FunctionBuilder,
        constant: &ConstantItem,
        loc: Option<LocationId>,
    ) -> Register {
        match constant {
            ConstantItem::Integer(Int::Small(i)) => builder.build_int(*i, loc),
            ConstantItem::Integer(Int::Big(i)) => builder.build_bigint(i.clone(), loc),
            ConstantItem::Float(f) => builder.build_float(*f, loc),
            ConstantItem::Bool(b) => builder.build_bool(*b, loc),
            ConstantItem::Atom(a) => builder.build_atom(a.as_str().get(), loc),
            ConstantItem::Bytes(data) => builder.build_raw_binary(data.as_slice(), loc),
            ConstantItem::Bitstring(ref bitvec) => {
                let selection = bitvec.select();
                let bytes = selection.to_bytes();
                let trailing_bits = selection.trailing_bits();
                builder.build_bitstring(bytes.borrow(), trailing_bits, loc)
            }
            ConstantItem::String(ref string) => builder.build_utf8_binary(string, loc),
            ConstantItem::InternedStr(ident) => {
                builder.build_utf8_binary(ident.as_str().get(), loc)
            }
        }
    }

    fn build_function(
        &mut self,
        builder: &mut FunctionBuilder,
        function: &syntax_ssa::Function,
    ) -> anyhow::Result<()> {
        debug!("building bytecode function {}", function.signature.mfa());

        // Reset the block/value maps for this function
        self.blocks.clear();
        self.values.clear();

        // If this is a NIF, mark it as such so the bytecode loader can check whether
        // to load the native implementation or the bytecoded shim
        if function.signature.is_nif() {
            builder.mark_as_nif();
        }

        // Build lookup map for syntax_ssa blocks to bytecode blocks, creating the blocks in the
        // process
        self.blocks.extend(function.dfg.blocks().map(|(b, _data)| {
            let arity = function.dfg.block_param_types(b).len() as u8;
            let bc_block = builder.create_block(arity);
            for (value, register) in function
                .dfg
                .block_params(b)
                .iter()
                .zip(builder.block_args(bc_block))
            {
                self.values.insert(*value, *register);
            }
            (b, bc_block)
        }));

        // For each block, in layout order, fill out the block with translated instructions
        for (block, block_data) in function.dfg.blocks() {
            builder.switch_to_block(self.blocks[&block]);
            self.switch_to_block(block);
            for inst in block_data.insts() {
                self.build_inst(builder, &function.dfg, inst)?;
            }
        }

        Ok(())
    }

    fn build_inst(
        &mut self,
        builder: &mut FunctionBuilder,
        dfg: &DataFlowGraph,
        inst: Inst,
    ) -> anyhow::Result<()> {
        let inst_data = &dfg[inst];
        let inst_span = inst_data.span();
        debug!(
            "translating instruction with opcode {:?} to bytecode",
            inst_data.opcode()
        );
        match inst_data.as_ref() {
            InstData::Ret(_) => {
                let loc = self.location_id_from_span(builder, inst_span);
                let args = dfg.inst_args(inst);
                assert_eq!(args.len(), 1);
                builder.build_ret(self.values[&args[0]], loc);
                Ok(())
            }
            InstData::Br(op) => self.build_br(builder, dfg, inst, inst_span, op),
            InstData::CondBr(op) => self.build_cond_br(builder, dfg, inst, inst_span, op),
            InstData::Switch(op) => self.build_switch(builder, dfg, inst, inst_span, op),
            InstData::IsType(op) => self.build_is_type(builder, dfg, inst, inst_span, op),
            InstData::Call(op) => self.build_call(builder, dfg, inst, inst_span, op),
            InstData::CallIndirect(op) => {
                self.build_call_indirect(builder, dfg, inst, inst_span, op)
            }
            InstData::SetElement(op) => self.build_setelement(builder, dfg, inst, inst_span, op),
            InstData::SetElementImm(op) => {
                self.build_setelement_imm(builder, dfg, inst, inst_span, op)
            }
            InstData::MakeFun(op) => self.build_make_fun(builder, dfg, inst, inst_span, op),
            InstData::UnaryOp(op) => self.build_unary_op(builder, dfg, inst, inst_span, op),
            InstData::UnaryOpImm(op) => self.build_unary_op_imm(builder, dfg, inst, inst_span, op),
            InstData::UnaryOpConst(op) => {
                self.build_unary_op_const(builder, dfg, inst, inst_span, op)
            }
            InstData::BinaryOp(op) => self.build_binary_op(builder, dfg, inst, inst_span, op),
            InstData::BinaryOpImm(op) => {
                self.build_binary_op_imm(builder, dfg, inst, inst_span, op)
            }
            InstData::PrimOp(op) => self.build_primop(builder, dfg, inst, inst_span, op),
            InstData::PrimOpImm(op) => self.build_primop_imm(builder, dfg, inst, inst_span, op),
            InstData::Catch(op) => {
                let loc = self.location_id_from_span(builder, inst_span);
                let dest = self.blocks[&op.dest];
                builder.build_catch(dest, loc);
                Ok(())
            }
            InstData::BitsPush(op) => self.build_bits_push(builder, dfg, inst, inst_span, op),
            InstData::BitsMatch(op) => self.build_bits_match(builder, dfg, inst, inst_span, op),
            InstData::BitsMatchSkip(op) => {
                self.build_bits_match_skip(builder, dfg, inst, inst_span, op)
            }
        }
    }

    fn build_br(
        &mut self,
        builder: &mut FunctionBuilder,
        dfg: &DataFlowGraph,
        inst: Inst,
        span: SourceSpan,
        op: &Br,
    ) -> anyhow::Result<()> {
        let loc = self.location_id_from_span(builder, span);
        let dest = self.blocks[&op.destination];
        let args = dfg.inst_args(inst);
        let args = args
            .iter()
            .map(|a| self.values[a])
            .collect::<SmallVec<[Register; 8]>>();

        match op.op {
            Opcode::Br => {
                builder.build_br(dest, args.as_slice(), loc);
            }
            Opcode::BrIf => {
                let (cond, args) = args.split_first().unwrap();
                builder.build_br_if(*cond, dest, args, loc);
            }
            Opcode::BrUnless => {
                let (cond, args) = args.split_first().unwrap();
                builder.build_br_unless(*cond, dest, args, loc);
            }
            other => unimplemented!("unrecognized branching op: {}", other),
        }

        Ok(())
    }

    fn build_cond_br(
        &mut self,
        builder: &mut FunctionBuilder,
        dfg: &DataFlowGraph,
        _inst: Inst,
        span: SourceSpan,
        op: &CondBr,
    ) -> anyhow::Result<()> {
        let loc = self.location_id_from_span(builder, span);
        let then_dest = self.blocks[&op.then_dest.0];
        let else_dest = self.blocks[&op.else_dest.0];
        let then_args = op.then_dest.1.as_slice(&dfg.value_lists);
        let else_args = op.else_dest.1.as_slice(&dfg.value_lists);
        let cond = self.values[&op.cond];

        let args = then_args
            .iter()
            .map(|a| self.values[a])
            .collect::<SmallVec<[Register; 8]>>();
        builder.build_br_if(cond, then_dest, args.as_slice(), loc);

        let args = else_args
            .iter()
            .map(|a| self.values[a])
            .collect::<SmallVec<[Register; 8]>>();
        builder.build_br(else_dest, args.as_slice(), loc);
        Ok(())
    }

    fn build_switch(
        &mut self,
        builder: &mut FunctionBuilder,
        _dfg: &DataFlowGraph,
        _inst: Inst,
        span: SourceSpan,
        op: &Switch,
    ) -> anyhow::Result<()> {
        let loc = self.location_id_from_span(builder, span);
        let arg = self.values[&op.arg];

        let mut jt = builder.build_jump_table(arg, loc);
        for (value, dest) in op.arms.iter() {
            let dest = self.blocks[dest];
            jt.entry(*value, dest);
        }
        jt.finish(self.blocks[&op.default], &[]);

        Ok(())
    }

    fn build_is_type(
        &mut self,
        builder: &mut FunctionBuilder,
        dfg: &DataFlowGraph,
        inst: Inst,
        span: SourceSpan,
        op: &IsType,
    ) -> anyhow::Result<()> {
        use core::num::NonZeroU32;

        use firefly_syntax_base::{TermType, Type};

        let loc = self.location_id_from_span(builder, span);
        let input = self.values[&op.arg];
        let is_type = match &op.ty {
            Type::Term(TermType::Any) => builder.build_bool(true, loc),
            Type::Term(TermType::List(_)) => builder.build_is_list(input, loc),
            Type::Term(TermType::Cons) => builder.build_is_cons(input, loc),
            Type::Term(TermType::Nil) => builder.build_is_nil(input, loc),
            Type::Term(TermType::Tuple(None)) => builder.build_is_tuple(input, None, loc),
            Type::Term(TermType::Tuple(Some(ref elems))) => {
                builder.build_is_tuple(input, NonZeroU32::new(elems.len() as u32), loc)
            }
            Type::Term(TermType::Map) => builder.build_is_map(input, loc),
            Type::Term(TermType::Number) => builder.build_is_number(input, loc),
            Type::Term(TermType::Integer) => builder.build_is_int(input, loc),
            Type::Term(TermType::Float) => builder.build_is_float(input, loc),
            Type::Term(TermType::Atom) => builder.build_is_atom(input, loc),
            Type::Term(TermType::Bool) => builder.build_is_bool(input, loc),
            Type::Term(TermType::Bitstring) => builder.build_is_bitstring(input, loc),
            Type::Term(TermType::Binary) => builder.build_is_binary(input, loc),
            Type::Term(TermType::Reference) => builder.build_is_reference(input, loc),
            Type::Term(TermType::Port) => builder.build_is_port(input, loc),
            Type::Term(TermType::Pid) => builder.build_is_pid(input, loc),
            Type::Term(TermType::Fun(None)) => builder.build_is_function(input, None, loc),
            Type::Term(TermType::Fun(Some(fun))) => {
                let arity = builder.build_int(fun.arity() as i64, loc);
                builder.build_is_function(input, Some(arity), loc)
            }
            Type::Term(ty) => unimplemented!("no support for type checks of {:?}", ty),
            ty => panic!("unsupported type check for {:?}", ty),
        };

        // Map syntax_ssa results to op results
        let result = dfg.first_result(inst);
        self.values.insert(result, is_type);

        Ok(())
    }

    fn build_call(
        &mut self,
        builder: &mut FunctionBuilder,
        dfg: &DataFlowGraph,
        inst: Inst,
        span: SourceSpan,
        op: &Call,
    ) -> anyhow::Result<()> {
        let loc = self.location_id_from_span(builder, span);
        let sig = self.find_function(op.callee);
        let callee = match sig.module {
            symbols::Empty => {
                // The callee is a native function that uses a different naming scheme
                builder.get_or_define_nif(sig.name.as_str(), sig.arity() as u8)
            }
            symbols::Erlang => {
                let args = dfg.inst_args(inst);
                let args = args
                    .iter()
                    .map(|a| self.values[a])
                    .collect::<SmallVec<[Register; 8]>>();
                match (sig.name, args.len()) {
                    (symbols::Apply, _) => {
                        // Calls to erlang:apply/2,3 are implemented with special instructions
                        match (op.op, args.len()) {
                            (Opcode::Call, 2) => {
                                let results = dfg.inst_results(inst);
                                assert_eq!(results.len(), 1);
                                let call_result = builder.build_call_apply2(args[0], args[1], loc);
                                self.values.insert(results[0], call_result);
                                return Ok(());
                            }
                            (Opcode::Call, 3) => {
                                let results = dfg.inst_results(inst);
                                assert_eq!(results.len(), 1);
                                let call_result =
                                    builder.build_call_apply3(args[0], args[1], args[2], loc);
                                self.values.insert(results[0], call_result);
                                return Ok(());
                            }
                            (Opcode::Enter, 2) => {
                                builder.build_enter_apply2(args[0], args[1], loc);
                                return Ok(());
                            }
                            (Opcode::Enter, 3) => {
                                builder.build_enter_apply3(args[0], args[1], args[2], loc);
                                return Ok(());
                            }
                            (_, n) => panic!("unexpected call to invalid erlang:apply/{} bif", n),
                        }
                    }
                    (symbols::Error, n) => {
                        panic!("unexpected call to invalid erlang:error/{} bif", n)
                    }
                    (symbols::Throw, n) => {
                        panic!("unexpected call to invalid erlang:throw/{} bif", n)
                    }
                    (symbols::Exit, n) => {
                        panic!("unexpected call to invalid erlang:exit/{} bif", n)
                    }
                    (symbols::Raise, n) => {
                        panic!("unexpected call to invalid erlang:raise/{} bif", n)
                    }
                    (symbols::SELF, 0) => match op.op {
                        Opcode::Call => {
                            let results = dfg.inst_results(inst);
                            assert_eq!(results.len(), 1);
                            let pid = builder.build_self(loc);
                            self.values.insert(results[0], pid);
                            return Ok(());
                        }
                        Opcode::Enter => {
                            builder.build_self(loc);
                            return Ok(());
                        }
                        _ => unreachable!(),
                    },
                    (symbols::Spawn, 1) => match op.op {
                        Opcode::Call => {
                            let results = dfg.inst_results(inst);
                            assert_eq!(results.len(), 1);
                            let pid = builder.build_spawn2(args[0], SpawnOpts::empty(), loc);
                            self.values.insert(results[0], pid);
                            return Ok(());
                        }
                        Opcode::Enter => {
                            builder.build_spawn2(args[0], SpawnOpts::empty(), loc);
                            return Ok(());
                        }
                        _ => unreachable!(),
                    },
                    (symbols::Spawn, 3) => match op.op {
                        Opcode::Call => {
                            let results = dfg.inst_results(inst);
                            assert_eq!(results.len(), 1);
                            let pid = builder.build_spawn3_indirect(
                                args[0],
                                args[1],
                                args[2],
                                SpawnOpts::empty(),
                                loc,
                            );
                            self.values.insert(results[0], pid);
                            return Ok(());
                        }
                        Opcode::Enter => {
                            builder.build_spawn3_indirect(
                                args[0],
                                args[1],
                                args[2],
                                SpawnOpts::empty(),
                                loc,
                            );
                            return Ok(());
                        }
                        _ => unreachable!(),
                    },
                    (symbols::SpawnLink, 1) => match op.op {
                        Opcode::Call => {
                            let results = dfg.inst_results(inst);
                            assert_eq!(results.len(), 1);
                            let pid = builder.build_spawn2(args[0], SpawnOpts::LINK, loc);
                            self.values.insert(results[0], pid);
                            return Ok(());
                        }
                        Opcode::Enter => {
                            builder.build_spawn2(args[0], SpawnOpts::LINK, loc);
                            return Ok(());
                        }
                        _ => unreachable!(),
                    },
                    (symbols::SpawnLink, 3) => match op.op {
                        Opcode::Call => {
                            let results = dfg.inst_results(inst);
                            assert_eq!(results.len(), 1);
                            let pid = builder.build_spawn3_indirect(
                                args[0],
                                args[1],
                                args[2],
                                SpawnOpts::LINK,
                                loc,
                            );
                            self.values.insert(results[0], pid);
                            return Ok(());
                        }
                        Opcode::Enter => {
                            builder.build_spawn3_indirect(
                                args[0],
                                args[1],
                                args[2],
                                SpawnOpts::LINK,
                                loc,
                            );
                            return Ok(());
                        }
                        _ => unreachable!(),
                    },
                    (symbols::Yield, 0) => {
                        let results = dfg.inst_results(inst);
                        assert_eq!(results.len(), 1);
                        builder.build_yield(loc);
                        let yielded = builder.build_bool(true, loc);
                        self.values.insert(results[0], yielded);
                        return Ok(());
                    }
                    _ => {
                        let name = sig.mfa();
                        let mfa = ModuleFunctionArity {
                            module: builder.insert_atom("erlang"),
                            function: builder.insert_atom(name.function.as_str().get()),
                            arity: name.arity,
                        };
                        if name.is_bif() {
                            builder.get_or_define_bif(mfa)
                        } else {
                            builder.get_or_define_function(mfa)
                        }
                    }
                }
            }
            module => {
                let name = sig.mfa();
                let mfa = ModuleFunctionArity {
                    module: builder.insert_atom(module.as_str().get()),
                    function: builder.insert_atom(name.function.as_str().get()),
                    arity: name.arity,
                };
                if name.is_bif() {
                    builder.get_or_define_bif(mfa)
                } else {
                    builder.get_or_define_function(mfa)
                }
            }
        };

        let args = dfg.inst_args(inst);
        let args = args
            .iter()
            .map(|a| self.values[a])
            .collect::<SmallVec<[Register; 8]>>();

        match op.op {
            Opcode::Call => {
                let results = dfg.inst_results(inst);
                assert_eq!(results.len(), 1);
                let call_result = builder.build_call(callee, args.as_slice(), loc);
                self.values.insert(results[0], call_result);
                Ok(())
            }
            Opcode::Enter => {
                builder.build_enter(callee, args.as_slice(), loc);
                Ok(())
            }
            op => panic!("unrecognized call opcode '{}'", &op),
        }
    }

    fn build_call_indirect(
        &mut self,
        builder: &mut FunctionBuilder,
        dfg: &DataFlowGraph,
        inst: Inst,
        span: SourceSpan,
        op: &CallIndirect,
    ) -> anyhow::Result<()> {
        let loc = self.location_id_from_span(builder, span);
        let args = dfg.inst_args(inst);
        let args = args
            .iter()
            .map(|a| self.values[a])
            .collect::<SmallVec<[Register; 8]>>();

        let callee = self.values[&op.callee];
        match op.op {
            Opcode::CallIndirect => {
                let results = dfg.inst_results(inst);
                assert_eq!(results.len(), 1);
                let call_result = builder.build_call_indirect(callee, args.as_slice(), loc);
                self.values.insert(results[0], call_result);
                Ok(())
            }
            Opcode::EnterIndirect => {
                builder.build_enter_indirect(callee, args.as_slice(), loc);
                Ok(())
            }
            op => panic!("unrecognized call.indirect opcode '{}'", &op),
        }
    }

    fn build_setelement(
        &mut self,
        builder: &mut FunctionBuilder,
        dfg: &DataFlowGraph,
        inst: Inst,
        span: SourceSpan,
        op: &SetElement,
    ) -> anyhow::Result<()> {
        let loc = self.location_id_from_span(builder, span);
        let tuple = self.values[&op.args[0]];
        let value = self.values[&op.args[1]];
        let index: usize = op
            .index
            .as_i64()
            .unwrap()
            .try_into()
            .expect("invalid index");

        match op.op {
            Opcode::SetElement => {
                let updated = builder.build_set_element(tuple, index, value, loc);
                self.values.insert(dfg.first_result(inst), updated);
            }
            Opcode::SetElementMut => {
                builder.build_set_element_mut(tuple, index, value, loc);
                self.values.insert(dfg.first_result(inst), tuple);
            }
            op => panic!("unrecognized setelement opcode: {}", op),
        }

        Ok(())
    }

    fn build_setelement_imm(
        &mut self,
        builder: &mut FunctionBuilder,
        dfg: &DataFlowGraph,
        inst: Inst,
        span: SourceSpan,
        op: &SetElementImm,
    ) -> anyhow::Result<()> {
        let loc = self.location_id_from_span(builder, span);
        let tuple = self.values[&op.arg];
        let value = self.load_immediate(builder, op.value, loc);
        let index: usize = op
            .index
            .as_i64()
            .unwrap()
            .try_into()
            .expect("invalid index");

        match op.op {
            Opcode::SetElement => {
                let updated = builder.build_set_element(tuple, index, value, loc);
                self.values.insert(dfg.first_result(inst), updated);
            }
            Opcode::SetElementMut => {
                builder.build_set_element_mut(tuple, index, value, loc);
                self.values.insert(dfg.first_result(inst), tuple);
            }
            op => panic!("unrecognized setelement opcode: {}", op),
        }

        Ok(())
    }

    fn build_make_fun(
        &mut self,
        builder: &mut FunctionBuilder,
        dfg: &DataFlowGraph,
        inst: Inst,
        span: SourceSpan,
        op: &MakeFun,
    ) -> anyhow::Result<()> {
        let loc = self.location_id_from_span(builder, span);
        let sig = self.find_function(op.callee);
        let name = sig.mfa();
        let mfa = ModuleFunctionArity {
            module: builder.insert_atom(name.module.unwrap().as_str().get()),
            function: builder.insert_atom(name.function.as_str().get()),
            arity: name.arity,
        };
        let callee = if name.is_bif() {
            builder.get_or_define_bif(mfa)
        } else {
            builder.get_or_define_function(mfa)
        };

        let args = dfg.inst_args(inst);
        let env = args
            .iter()
            .map(|a| self.values[a])
            .collect::<SmallVec<[Register; 8]>>();

        let closure = builder.build_closure(callee, env.as_slice(), loc);

        let results = dfg.inst_results(inst);
        assert_eq!(results.len(), 1);
        self.values.insert(results[0], closure);

        Ok(())
    }

    fn build_unary_op(
        &mut self,
        builder: &mut FunctionBuilder,
        dfg: &DataFlowGraph,
        inst: Inst,
        span: SourceSpan,
        op: &UnaryOp,
    ) -> anyhow::Result<()> {
        let loc = self.location_id_from_span(builder, span);
        let arg = self.values[&op.arg];
        self.do_build_unary_op(builder, dfg, inst, loc, op.op, arg)
    }

    fn build_unary_op_imm(
        &mut self,
        builder: &mut FunctionBuilder,
        dfg: &DataFlowGraph,
        inst: Inst,
        span: SourceSpan,
        op: &UnaryOpImm,
    ) -> anyhow::Result<()> {
        let loc = self.location_id_from_span(builder, span);
        match op.op {
            Opcode::Tuple => {
                let Immediate::Isize(cap) = op.imm else { unreachable!() };
                let tuple = builder.build_tuple_with_capacity(cap as usize, loc);
                let result = dfg.first_result(inst);
                self.values.insert(result, tuple);
                Ok(())
            }
            Opcode::ImmNil => {
                let nil = builder.build_nil(loc);
                let result = dfg.first_result(inst);
                self.values.insert(result, nil);
                Ok(())
            }
            Opcode::ImmAtom => {
                let Immediate::Term(ImmediateTerm::Atom(a)) = op.imm else { unreachable!() };
                let atom = builder.build_atom(a.as_str().get(), loc);
                let result = dfg.first_result(inst);
                self.values.insert(result, atom);
                Ok(())
            }
            Opcode::ImmInt => {
                let Immediate::Term(ImmediateTerm::Integer(i)) = op.imm else { unreachable!() };
                let int = builder.build_int(i, loc);
                let result = dfg.first_result(inst);
                self.values.insert(result, int);
                Ok(())
            }
            _ => {
                let arg = self.load_immediate(builder, op.imm, loc);
                self.do_build_unary_op(builder, dfg, inst, loc, op.op, arg)
            }
        }
    }

    fn build_unary_op_const(
        &mut self,
        builder: &mut FunctionBuilder,
        dfg: &DataFlowGraph,
        inst: Inst,
        span: SourceSpan,
        op: &UnaryOpConst,
    ) -> anyhow::Result<()> {
        let loc = self.location_id_from_span(builder, span);
        match op.op {
            Opcode::ConstBigInt => {
                let constant = dfg.constant(op.imm);
                let ConstantItem::Integer(Int::Big(ref i)) = *constant else { unreachable!() };
                let i = builder.build_bigint(i.clone(), loc);
                let result = dfg.first_result(inst);
                self.values.insert(result, i);
                Ok(())
            }
            _ => {
                let arg = self.load_constant(builder, &dfg.constant(op.imm), loc);
                self.do_build_unary_op(builder, dfg, inst, loc, op.op, arg)
            }
        }
    }

    fn do_build_unary_op(
        &mut self,
        builder: &mut FunctionBuilder,
        dfg: &DataFlowGraph,
        inst: Inst,
        loc: Option<LocationId>,
        op: Opcode,
        arg: Register,
    ) -> anyhow::Result<()> {
        let results = dfg.inst_results(inst);
        let returned = match op {
            // Cast is a no-op
            Opcode::Cast => arg,
            Opcode::Head => builder.build_head(arg, loc),
            Opcode::Tail => builder.build_tail(arg, loc),
            Opcode::Neg => builder.build_neg(arg, loc),
            Opcode::Not => builder.build_not(arg, loc),
            Opcode::Bnot => builder.build_bnot(arg, loc),
            other => unimplemented!("no lowering for unary op with opcode {:?}", other),
        };

        assert_eq!(results.len(), 1);
        self.values.insert(results[0], returned);
        Ok(())
    }

    fn build_binary_op(
        &mut self,
        builder: &mut FunctionBuilder,
        dfg: &DataFlowGraph,
        inst: Inst,
        span: SourceSpan,
        op: &BinaryOp,
    ) -> anyhow::Result<()> {
        let loc = self.location_id_from_span(builder, span);
        let lhs = self.values[&op.args[0]];
        let rhs = self.values[&op.args[1]];
        self.do_build_binary_op(builder, dfg, inst, loc, op.op, lhs, rhs)
    }

    fn build_binary_op_imm(
        &mut self,
        builder: &mut FunctionBuilder,
        dfg: &DataFlowGraph,
        inst: Inst,
        span: SourceSpan,
        op: &BinaryOpImm,
    ) -> anyhow::Result<()> {
        let loc = self.location_id_from_span(builder, span);
        match op.op {
            Opcode::GetElement => {
                let tuple = self.values[&op.arg];
                let Immediate::Isize(index) = op.imm else { panic!("invalid immediate for {} op", op.op); };
                let value = builder.build_get_element(tuple, index.try_into().unwrap(), loc);
                let results = dfg.inst_results(inst);
                assert_eq!(results.len(), 1);
                self.values.insert(results[0], value);
                Ok(())
            }
            Opcode::UnpackEnv => {
                let fun = self.values[&op.arg];
                let Immediate::Isize(index) = op.imm else { panic!("invalid immediate for {} op", op.op); };
                let value = builder.build_unpack_env(fun, index.try_into().unwrap(), loc);
                let results = dfg.inst_results(inst);
                assert_eq!(results.len(), 1);
                self.values.insert(results[0], value);
                Ok(())
            }
            _ => {
                let lhs = self.values[&op.arg];
                let rhs = self.load_immediate(builder, op.imm, loc);
                self.do_build_binary_op(builder, dfg, inst, loc, op.op, lhs, rhs)
            }
        }
    }

    fn do_build_binary_op(
        &mut self,
        builder: &mut FunctionBuilder,
        dfg: &DataFlowGraph,
        inst: Inst,
        loc: Option<LocationId>,
        op: Opcode,
        lhs: Register,
        rhs: Register,
    ) -> anyhow::Result<()> {
        let results = dfg.inst_results(inst);
        let returned = match op {
            Opcode::Cons => builder.build_cons(lhs, rhs, loc),
            Opcode::Send => {
                builder.build_send(lhs, rhs, loc);
                rhs
            }
            Opcode::ListConcat => builder.build_list_append(lhs, rhs, loc),
            Opcode::ListSubtract => builder.build_list_remove(lhs, rhs, loc),
            Opcode::Eq => builder.build_eq(lhs, rhs, false, loc),
            Opcode::EqExact | Opcode::IcmpEq => builder.build_eq(lhs, rhs, true, loc),
            Opcode::Neq => builder.build_neq(lhs, rhs, false, loc),
            Opcode::NeqExact | Opcode::IcmpNeq => builder.build_neq(lhs, rhs, true, loc),
            Opcode::Gt | Opcode::IcmpGt => builder.build_gt(lhs, rhs, loc),
            Opcode::Gte | Opcode::IcmpGte => builder.build_gte(lhs, rhs, loc),
            Opcode::Lt | Opcode::IcmpLt => builder.build_lt(lhs, rhs, loc),
            Opcode::Lte | Opcode::IcmpLte => builder.build_lte(lhs, rhs, loc),
            Opcode::And => builder.build_and(lhs, rhs, loc),
            Opcode::AndAlso => builder.build_andalso(lhs, rhs, loc),
            Opcode::Or => builder.build_or(lhs, rhs, loc),
            Opcode::OrElse => builder.build_orelse(lhs, rhs, loc),
            Opcode::Xor => builder.build_xor(lhs, rhs, loc),
            Opcode::Band => builder.build_band(lhs, rhs, loc),
            Opcode::Bor => builder.build_bor(lhs, rhs, loc),
            Opcode::Bxor => builder.build_bxor(lhs, rhs, loc),
            Opcode::Bsl => builder.build_bsl(lhs, rhs, loc),
            Opcode::Bsr => builder.build_bsr(lhs, rhs, loc),
            Opcode::Div => builder.build_div(lhs, rhs, loc),
            Opcode::Rem => builder.build_rem(lhs, rhs, loc),
            Opcode::Add => builder.build_add(lhs, rhs, loc),
            Opcode::Sub => builder.build_sub(lhs, rhs, loc),
            Opcode::Mul => builder.build_mul(lhs, rhs, loc),
            Opcode::Fdiv => builder.build_divide(lhs, rhs, loc),
            other => unimplemented!("no lowering for binary op with opcode {}", other),
        };

        assert_eq!(results.len(), 1);
        self.values.insert(results[0], returned);
        Ok(())
    }

    fn build_primop(
        &mut self,
        builder: &mut FunctionBuilder,
        dfg: &DataFlowGraph,
        inst: Inst,
        span: SourceSpan,
        op: &PrimOp,
    ) -> anyhow::Result<()> {
        let loc = self.location_id_from_span(builder, span);
        let args = dfg.inst_args(inst);
        let args = args
            .iter()
            .map(|a| self.values[a])
            .collect::<SmallVec<[Register; 4]>>();

        match op.op {
            Opcode::Split => {
                let (head, tail) = builder.build_split(args[0], loc);
                let results = dfg.inst_results(inst);
                self.values.insert(results[0], head);
                self.values.insert(results[1], tail);
            }
            Opcode::IsTupleFetchArity => {
                let (is_tuple, arity) = builder.build_is_tuple_fetch_arity(args[0], loc);
                let results = dfg.inst_results(inst);
                self.values.insert(results[0], is_tuple);
                self.values.insert(results[1], arity);
            }
            Opcode::IsFunctionWithArity => {
                let is_function = builder.build_is_function(args[0], Some(args[1]), loc);
                let result = dfg.first_result(inst);
                self.values.insert(result, is_function);
            }
            Opcode::MapPut => {
                let updated = builder.build_map_insert(args[0], args[1], args[2], loc);
                let result = dfg.first_result(inst);
                self.values.insert(result, updated);
            }
            Opcode::MapPutMut => {
                builder.build_map_insert_mut(args[0], args[1], args[2], loc);
            }
            Opcode::MapUpdate => {
                let updated = builder.build_map_update(args[0], args[1], args[2], loc);
                let result = dfg.first_result(inst);
                self.values.insert(result, updated);
            }
            Opcode::MapUpdateMut => {
                builder.build_map_update_mut(args[0], args[1], args[2], loc);
            }
            Opcode::MapExtendPut => {
                let (map, args) = args.split_first().unwrap();
                let updated = builder.build_map_extend_insert(*map, args, loc);
                let result = dfg.first_result(inst);
                self.values.insert(result, updated);
            }
            Opcode::MapExtendUpdate => {
                let (map, args) = args.split_first().unwrap();
                let updated = builder.build_map_extend_update(*map, args, loc);
                let result = dfg.first_result(inst);
                self.values.insert(result, updated);
            }
            Opcode::MapTryGet => {
                let (is_err, value) = builder.build_map_try_get(args[0], args[1], loc);
                let results = dfg.inst_results(inst);
                self.values.insert(results[0], is_err);
                self.values.insert(results[1], value);
            }
            Opcode::RecvNext => builder.build_recv_next(loc),
            Opcode::RecvPeek => {
                let (available, msg) = builder.build_recv_peek(loc);
                let results = dfg.inst_results(inst);
                self.values.insert(results[0], available);
                self.values.insert(results[1], msg);
            }
            Opcode::RecvPop => builder.build_recv_pop(loc),
            Opcode::RecvWaitTimeout => {
                let timed_out = builder.build_recv_wait_timeout(args[0], loc);
                let result = dfg.first_result(inst);
                self.values.insert(result, timed_out);
            }
            Opcode::BitsMatchStart => {
                let (is_err, bin) = builder.build_bs_match_start(args[0], loc);
                let results = dfg.inst_results(inst);
                self.values.insert(results[0], is_err);
                self.values.insert(results[1], bin);
            }
            Opcode::BitsInit => {
                let bin_builder = builder.build_bs_init(loc);
                let result = dfg.first_result(inst);
                self.values.insert(result, bin_builder);
            }
            Opcode::BitsFinish => {
                let bin = builder.build_bs_finish(args[0], loc);
                let result = dfg.first_result(inst);
                self.values.insert(result, bin);
            }
            Opcode::EndCatch => builder.build_end_catch(loc),
            Opcode::Raise => {
                let trace = if args.len() == 3 { Some(args[2]) } else { None };
                let opts = if args.len() == 4 { Some(args[3]) } else { None };
                let badarg = builder.build_raise(args[0], args[1], trace, opts, loc);
                let result = dfg.first_result(inst);
                self.values.insert(result, badarg);
            }
            Opcode::Throw => builder.build_throw(args[0], loc),
            Opcode::Error => builder.build_error(args[0], loc),
            Opcode::Exit1 => builder.build_exit1(args[0], loc),
            Opcode::Exit2 => {
                let success = builder.build_exit2(args[0], args[1], loc);
                let result = dfg.first_result(inst);
                self.values.insert(result, success);
            }
            Opcode::Halt => match args.len() {
                0 => builder.build_halt(None, None, loc),
                1 => builder.build_halt(Some(args[0]), None, loc),
                2 => builder.build_halt(Some(args[0]), Some(args[1]), loc),
                n => unimplemented!("no support for halt/{}", n),
            },
            Opcode::BuildStacktrace => {
                let trace = builder.build_stacktrace(loc);
                let result = dfg.first_result(inst);
                self.values.insert(result, trace);
            }
            Opcode::Yield => {
                builder.build_yield(loc);
                let result = dfg.first_result(inst);
                let yielded = builder.build_bool(true, loc);
                self.values.insert(result, yielded);
            }
            Opcode::GarbageCollect => {
                builder.build_garbage_collect(loc);
                let result = dfg.first_result(inst);
                let success = builder.build_bool(true, loc);
                self.values.insert(result, success);
            }
            other => unimplemented!("unrecognized primop: {}", other),
        }

        Ok(())
    }

    fn build_primop_imm(
        &mut self,
        builder: &mut FunctionBuilder,
        dfg: &DataFlowGraph,
        inst: Inst,
        span: SourceSpan,
        op: &PrimOpImm,
    ) -> anyhow::Result<()> {
        let loc = self.location_id_from_span(builder, span);
        let args = dfg.inst_args(inst);
        let args = args
            .iter()
            .map(|a| self.values[a])
            .collect::<SmallVec<[Register; 4]>>();

        match op.op {
            Opcode::Map => {
                let Immediate::Isize(cap) = op.imm else { unreachable!() };
                let map = builder.build_map(cap as usize, loc);
                let result = dfg.first_result(inst);
                self.values.insert(result, map);
            }
            Opcode::BitsTestTail => {
                let Immediate::Isize(size) = op.imm else { unreachable!() };
                let test_result = builder.build_bs_test_tail(args[0], size as usize, loc);
                let result = dfg.first_result(inst);
                self.values.insert(result, test_result);
            }
            op => unimplemented!("unrecognized primop w/immediate: {}", op),
        }

        Ok(())
    }

    fn build_bits_match(
        &mut self,
        builder: &mut FunctionBuilder,
        dfg: &DataFlowGraph,
        inst: Inst,
        span: SourceSpan,
        op: &BitsMatch,
    ) -> anyhow::Result<()> {
        let loc = self.location_id_from_span(builder, span);
        let args = dfg.inst_args(inst);
        let context = self.values[&args[0]];
        let size = if args.len() > 1 {
            Some(self.values[&args[1]])
        } else {
            None
        };

        let (is_err, value, next) = builder.build_bs_match(context, size, op.spec, loc);
        let results = dfg.inst_results(inst);
        self.values.insert(results[0], is_err);
        self.values.insert(results[1], value);
        self.values.insert(results[2], next);
        Ok(())
    }

    fn build_bits_match_skip(
        &mut self,
        builder: &mut FunctionBuilder,
        dfg: &DataFlowGraph,
        inst: Inst,
        span: SourceSpan,
        op: &BitsMatchSkip,
    ) -> anyhow::Result<()> {
        let loc = self.location_id_from_span(builder, span);
        let args = dfg.inst_args(inst);
        let context = self.values[&args[0]];
        let size = self.values[&args[1]];
        let value = self.load_immediate(builder, op.value, loc);

        let (is_err, next) = builder.build_bs_match_skip(context, size, op.spec, value, loc);
        let results = dfg.inst_results(inst);
        self.values.insert(results[0], is_err);
        self.values.insert(results[1], next);
        Ok(())
    }

    fn build_bits_push(
        &mut self,
        builder: &mut FunctionBuilder,
        dfg: &DataFlowGraph,
        inst: Inst,
        span: SourceSpan,
        op: &BitsPush,
    ) -> anyhow::Result<()> {
        let loc = self.location_id_from_span(builder, span);
        let args = dfg.inst_args(inst);
        let bin_builder = self.values[&args[0]];
        let value = self.values[&args[1]];
        let size = if args.len() > 2 {
            Some(self.values[&args[2]])
        } else {
            None
        };

        let updated = builder.build_bs_push(bin_builder, value, size, op.spec, loc);
        let result = dfg.first_result(inst);
        self.values.insert(result, updated);
        Ok(())
    }
}
