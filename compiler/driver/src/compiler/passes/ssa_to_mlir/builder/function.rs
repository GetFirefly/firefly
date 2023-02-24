use core::mem;

use firefly_binary::{Bitstring, Encoding};
use firefly_diagnostics::{SourceSpan, Spanned};
use firefly_llvm::Linkage;
use firefly_mlir::llvm::{ICmpPredicate, LinkageAttr};
use firefly_mlir::*;
use firefly_number::Int;
use firefly_syntax_base::{self as syntax_base, Signature};
use firefly_syntax_ssa::{self as syntax_ssa, ir::instructions::*, DataFlowGraph};
use firefly_syntax_ssa::{ConstantItem, Immediate, ImmediateTerm};

use log::debug;

use super::*;

impl<'m> ModuleBuilder<'m> {
    /// Lowers the definition of a syntax_ssa function to CIR dialect
    pub(super) fn build_function(&mut self, function: &syntax_ssa::Function) -> anyhow::Result<()> {
        debug!("building mlir function {}", function.signature.mfa());

        // Reset the block/value maps for this function
        self.blocks.clear();
        self.values.clear();

        // Declare the function
        let function_loc = self.location_from_span(function.span);
        let name = function.signature.mfa().to_string();
        let func = self.builder.get_func_by_symbol(name.as_str()).unwrap();

        // If the function was declared/detected as implemented by a NIF, set the linkage
        // to linkonce, this should allow the default code in the module to be used if at link
        // time there is no stronger definition found
        //
        // This linkage type cannot be set on declarations, hence why we only set this attribute
        // when generating the definition of the function.
        if function.signature.is_nif() {
            func.set_attribute_by_name(
                "llvm.linkage",
                LinkageAttr::get(self.builder.context(), Linkage::LinkOnceAny),
            );
        }

        let function_body = func.get_region(0);
        let mut tx_param_types = Vec::with_capacity(8);
        let mut tx_param_locs = Vec::with_capacity(8);
        // Build lookup map for syntax_ssa blocks to MLIR blocks, creating the blocks in the process
        {
            let builder = CirBuilder::new(&self.builder);
            self.blocks.extend(function.dfg.blocks().map(|(b, _data)| {
                tx_param_types.clear();
                tx_param_locs.clear();
                let orig_param_types = function.dfg.block_param_types(b);
                for t in orig_param_types.iter() {
                    tx_param_types.push(translate_ir_type(
                        &self.module,
                        &self.options,
                        &builder,
                        t,
                    ));
                    tx_param_locs.push(function_loc);
                }
                // Create the corresponding MLIR block
                let mlir_block = builder.create_block_in_region(
                    function_body,
                    tx_param_types.as_slice(),
                    tx_param_locs.as_slice(),
                );
                // Map all syntax_ssa block parameters to their MLIR block argument values
                for (value, mlir_value) in function
                    .dfg
                    .block_params(b)
                    .iter()
                    .zip(mlir_block.arguments())
                {
                    self.values.insert(*value, mlir_value.base());
                }
                (b, mlir_block)
            }));
        }

        // For each block, in layout order, fill out the block with translated instructions
        for (block, block_data) in function.dfg.blocks() {
            self.switch_to_block(block);
            for inst in block_data.insts() {
                self.build_inst(&function.dfg, inst)?;
            }
        }

        Ok(())
    }

    /// Switches the builder to the MLIR block corresponding to the given syntax_ssa block
    fn switch_to_block(&mut self, block: syntax_ssa::Block) {
        debug!("switching builder to block {:?}", block);
        self.current_source_block = block;
        self.current_block = self.blocks[&block];
        self.builder.set_insertion_point_to_end(self.current_block);
    }

    /// Obtains the MLIR value corresponding to the first argument of the current function,
    /// which in the Erlang calling convention must always be a pointer to the current process
    fn get_process_argument(&self) -> mlir::ValueBase {
        let op = self.current_block.operation().unwrap();
        let body = op.get_body().unwrap();
        body.get_argument(0).base()
    }

    /// Lowers the declaration of a syntax_ssa function to CIR dialect
    pub(super) fn declare_function(
        &self,
        span: SourceSpan,
        sig: &Signature,
    ) -> anyhow::Result<FuncOp> {
        debug!("declaring function {}", sig.mfa());
        // Generate the symbol name for the function, e.g. module:function/arity
        let name = match sig.module {
            symbols::Empty => sig.name.to_string(),
            _ => sig.mfa().to_string(),
        };
        let builder = self.cir();
        let ty = signature_to_fn_type(self.module, self.options, &builder, &sig);
        let vis = if sig.visibility.is_public() && !sig.visibility.is_externally_defined() {
            Visibility::Public
        } else {
            Visibility::Private
        };
        let _ip = self.builder.insertion_guard();
        self.builder
            .set_insertion_point_to_end(self.mlir_module.body());
        let loc = self.location_from_span(span);
        let cc = sig.cc.to_string();
        let cc_attr = self
            .builder
            .get_named_attr("cc", self.builder.get_string_attr(cc.as_str()));
        let op = self
            .builder
            .build_func(loc, name.as_str(), ty, &[cc_attr], &[]);
        op.set_visibility(vis);
        Ok(op)
    }

    fn immediate_to_constant(&self, loc: Location, imm: Immediate) -> ValueBase {
        let builder = CirBuilder::new(&self.builder);
        match imm {
            Immediate::Term(imm_term) => match imm_term {
                ImmediateTerm::Bool(b) => {
                    let op = builder.build_constant(
                        loc,
                        builder.get_cir_bool_type(),
                        builder.get_cir_bool_attr(b),
                    );
                    op.get_result(0).base()
                }
                ImmediateTerm::Atom(a) => {
                    let ty = builder.get_cir_atom_type();
                    let op = builder.build_constant(loc, ty, builder.get_atom_attr(a, ty));
                    op.get_result(0).base()
                }
                ImmediateTerm::Integer(i) => {
                    let op = builder.build_constant(
                        loc,
                        builder.get_cir_isize_type(),
                        builder.get_isize_attr(i.try_into().unwrap()),
                    );
                    op.get_result(0).base()
                }
                ImmediateTerm::Float(f) => {
                    let op = builder.build_constant(
                        loc,
                        builder.get_cir_float_type(),
                        builder.get_float_attr(f),
                    );
                    op.get_result(0).base()
                }
                ImmediateTerm::Nil => {
                    let ty = builder.get_cir_nil_type();
                    let op = builder.build_constant(loc, ty, builder.get_nil_attr());
                    op.get_result(0).base()
                }
                ImmediateTerm::None => {
                    let ty = builder.get_cir_none_type();
                    let op = builder.build_constant(loc, ty, builder.get_none_attr());
                    op.get_result(0).base()
                }
            },
            Immediate::I1(b) => {
                let op =
                    builder.build_constant(loc, builder.get_i1_type(), builder.get_bool_attr(b));
                op.get_result(0).base()
            }
            Immediate::I8(n) => {
                let op = builder.build_constant(loc, builder.get_i8_type(), builder.get_i8_attr(n));
                op.get_result(0).base()
            }
            Immediate::I16(n) => {
                let op =
                    builder.build_constant(loc, builder.get_i16_type(), builder.get_i16_attr(n));
                op.get_result(0).base()
            }
            Immediate::I32(n) => {
                let op =
                    builder.build_constant(loc, builder.get_i32_type(), builder.get_i32_attr(n));
                op.get_result(0).base()
            }
            Immediate::I64(n) => {
                let op =
                    builder.build_constant(loc, builder.get_i64_type(), builder.get_i64_attr(n));
                op.get_result(0).base()
            }
            Immediate::Isize(n) => {
                let op = builder.build_constant(
                    loc,
                    builder.get_index_type(),
                    builder.get_index_attr(n as i64),
                );
                op.get_result(0).base()
            }
            Immediate::F64(n) => {
                let op =
                    builder.build_constant(loc, builder.get_f64_type(), builder.get_float_attr(n));
                op.get_result(0).base()
            }
        }
    }

    fn const_to_constant(&self, loc: Location, constant: &ConstantItem) -> ValueBase {
        match constant {
            ConstantItem::Integer(Int::Small(i)) => {
                let builder = CirBuilder::new(&self.builder);
                let op = builder.build_constant(
                    loc,
                    builder.get_cir_isize_type(),
                    builder.get_isize_attr((*i).try_into().unwrap()),
                );
                op.get_result(0).base()
            }
            ConstantItem::Integer(Int::Big(i)) => {
                let builder = CirBuilder::new(&self.builder);
                let ty = builder
                    .get_cir_box_type(builder.get_cir_bigint_type())
                    .base();
                let op = builder.build_constant(loc, ty, builder.get_bigint_attr(i, ty));
                op.get_result(0).base()
            }
            ConstantItem::Float(f) => {
                let builder = CirBuilder::new(&self.builder);
                let op = builder.build_constant(
                    loc,
                    builder.get_cir_float_type(),
                    builder.get_float_attr(*f),
                );
                op.get_result(0).base()
            }
            ConstantItem::Bool(b) => {
                let builder = CirBuilder::new(&self.builder);
                let op = builder.build_constant(
                    loc,
                    builder.get_cir_bool_type(),
                    builder.get_cir_bool_attr(*b),
                );
                op.get_result(0).base()
            }
            ConstantItem::Atom(a) => {
                let builder = CirBuilder::new(&self.builder);
                let ty = builder.get_cir_atom_type();
                let op = builder.build_constant(loc, ty, builder.get_atom_attr(*a, ty));
                op.get_result(0).base()
            }
            ConstantItem::Bytes(const_data) => {
                self.bitstring_to_constant(loc, const_data.as_slice())
            }
            ConstantItem::Bitstring(ref bitvec) => self.bitstring_to_constant(loc, bitvec),
            ConstantItem::String(string) => self.bitstring_to_constant(loc, string.as_str()),
            ConstantItem::InternedStr(ident) => {
                self.bitstring_to_constant(loc, ident.as_str().get())
            }
        }
    }

    fn bitstring_to_constant<B: ?Sized + Bitstring>(
        &self,
        loc: Location,
        bitstring: &B,
    ) -> ValueBase {
        assert!(
            bitstring.is_aligned(),
            "bitstring constants must be aligned for codegen"
        );

        let builder = CirBuilder::new(&self.builder);
        let ty = builder.get_cir_box_type(builder.get_cir_bits_type());
        let bytes = unsafe { bitstring.as_bytes_unchecked() };
        let attr = StringAttr::get_with_type(bytes, ty.base());
        let op = builder.build_constant(loc, ty, attr);
        op.set_attribute_by_name(
            "byte_size",
            builder.get_isize_attr(bitstring.byte_size() as u64),
        );
        op.set_attribute_by_name(
            "bit_size",
            builder.get_isize_attr(bitstring.bit_size() as u64),
        );
        let encoding = if bitstring.is_binary() {
            Encoding::detect(bytes)
        } else {
            Encoding::Raw
        };
        match encoding {
            Encoding::Raw => {
                op.set_attribute_by_name("utf8", builder.get_bool_attr(false));
                op.set_attribute_by_name("latin1", builder.get_bool_attr(false));
            }
            Encoding::Latin1 => {
                op.set_attribute_by_name("utf8", builder.get_bool_attr(false));
                op.set_attribute_by_name("latin1", builder.get_bool_attr(true));
            }
            Encoding::Utf8 => {
                op.set_attribute_by_name("utf8", builder.get_bool_attr(true));
                op.set_attribute_by_name("latin1", builder.get_bool_attr(true));
            }
            _ => unreachable!(),
        }
        op.get_result(0).base()
    }

    /// Lowers a single syntax_ssa instruction to the corresponding CIR dialect operation
    fn build_inst(&mut self, dfg: &DataFlowGraph, inst: Inst) -> anyhow::Result<()> {
        let inst_data = &dfg[inst];
        let inst_span = inst_data.span();
        debug!(
            "translating instruction with opcode {:?} to mlir",
            inst_data.opcode()
        );
        match inst_data.as_ref() {
            InstData::UnaryOp(op) => self.build_unary_op(dfg, inst, inst_span, op),
            InstData::UnaryOpImm(op) => self.build_unary_op_imm(dfg, inst, inst_span, op),
            InstData::UnaryOpConst(op) => self.build_unary_op_const(dfg, inst, inst_span, op),
            InstData::BinaryOp(op) => self.build_binary_op(dfg, inst, inst_span, op),
            InstData::BinaryOpImm(op) => self.build_binary_op_imm(dfg, inst, inst_span, op),
            InstData::Ret(op) => self.build_ret(dfg, inst, inst_span, op),
            InstData::RetImm(op) => self.build_ret_imm(dfg, inst, inst_span, op),
            InstData::CondBr(op) => self.build_cond_br(dfg, inst, inst_span, op),
            InstData::Br(op) => self.build_br(dfg, inst, inst_span, op),
            InstData::Switch(op) => self.build_switch(dfg, inst, inst_span, op),
            InstData::IsType(op) => self.build_is_type(dfg, inst, inst_span, op),
            InstData::PrimOp(op) => self.build_primop(dfg, inst, inst_span, op),
            InstData::PrimOpImm(op) => self.build_primop_imm(dfg, inst, inst_span, op),
            InstData::Call(op) => self.build_call(dfg, inst, inst_span, op),
            InstData::CallIndirect(op) => self.build_call_indirect(dfg, inst, inst_span, op),
            InstData::MakeFun(op) => self.build_make_fun(dfg, inst, inst_span, op),
            InstData::SetElement(op) => self.build_setelement(dfg, inst, inst_span, op),
            InstData::SetElementImm(op) => self.build_setelement_imm(dfg, inst, inst_span, op),
            InstData::BitsPush(op) => self.build_bits_push(dfg, inst, inst_span, op),
            InstData::BitsMatch(op) => self.build_bits_match(dfg, inst, inst_span, op),
            InstData::BitsMatchSkip(op) => self.build_bits_match_skip(dfg, inst, inst_span, op),
        }
    }

    fn build_unary_op(
        &mut self,
        dfg: &DataFlowGraph,
        inst: Inst,
        span: SourceSpan,
        op: &UnaryOp,
    ) -> anyhow::Result<()> {
        let loc = self.location_from_span(span);
        let arg = self.values[&op.arg];
        let results = dfg.inst_results(inst);
        let mlir_op = match op.op {
            Opcode::IsNull => self.cir().build_is_null(loc, arg).base(),
            Opcode::Cast => {
                let builder = self.cir();
                let result = dfg.first_result(inst);
                let ty =
                    translate_ir_type(self.module, self.options, &builder, &dfg.value_type(result));
                builder.build_cast(loc, arg, ty).base()
            }
            Opcode::Trunc => {
                let builder = self.cir();
                let result = dfg.first_result(inst);
                let ty =
                    translate_ir_type(self.module, self.options, &builder, &dfg.value_type(result));
                builder.build_trunc(loc, arg, ty).base()
            }
            Opcode::Zext => {
                let builder = self.cir();
                let result = dfg.first_result(inst);
                let ty =
                    translate_ir_type(self.module, self.options, &builder, &dfg.value_type(result));
                builder.build_zext(loc, arg, ty).base()
            }
            Opcode::Head => self.cir().build_head(loc, arg).base(),
            Opcode::Tail => self.cir().build_tail(loc, arg).base(),
            Opcode::Neg => {
                let neg1 = self.get_or_declare_builtin("erlang:-/1").unwrap();
                let op = self.cir().build_call(loc, neg1, &[arg]).base();
                self.values.insert(results[0], op.get_result(1).base());
                return Ok(());
            }
            Opcode::Not => self.cir().build_not(loc, arg).base(),
            Opcode::Bnot => {
                let bnot1 = self.get_or_declare_builtin("erlang:bnot/1").unwrap();
                let op = self.cir().build_call(loc, bnot1, &[arg]).base();
                self.values.insert(results[0], op.get_result(1).base());
                return Ok(());
            }
            other => unimplemented!("no lowering for unary op with opcode {:?}", other),
        };

        assert_eq!(results.len(), mlir_op.num_results());
        for (value, op_result) in results.iter().copied().zip(mlir_op.results()) {
            self.values.insert(value, op_result.base());
        }
        Ok(())
    }

    fn build_unary_op_imm(
        &mut self,
        dfg: &DataFlowGraph,
        inst: Inst,
        span: SourceSpan,
        op: &UnaryOpImm,
    ) -> anyhow::Result<()> {
        let loc = self.location_from_span(span);
        let results = dfg.inst_results(inst);
        let mlir_op = match op.op {
            Opcode::ImmNull => {
                let builder = self.cir();
                let result = dfg.first_result(inst);
                let ty =
                    translate_ir_type(self.module, self.options, &builder, &dfg.value_type(result));
                let null = builder.build_null(loc, ty);
                self.values.insert(result, null.get_result(0).base());
                return Ok(());
            }
            Opcode::Zext => {
                use firefly_syntax_base::{PrimitiveType, Type as CoreType};

                let builder = self.cir();
                let result = dfg.first_result(inst);
                let i = op.imm.as_i64().expect("expected integer immediate");
                let CoreType::Primitive(prim) = dfg.value_type(result) else { panic!("expected primitive type"); };
                let (ty, attr) = match prim {
                    PrimitiveType::I1 => (
                        builder.get_i1_type().base(),
                        builder.get_bool_attr(i > 0).base(),
                    ),
                    PrimitiveType::I8 => (
                        builder.get_i8_type().base(),
                        builder.get_i8_attr(i.try_into().unwrap()).base(),
                    ),
                    PrimitiveType::I16 => (
                        builder.get_i16_type().base(),
                        builder.get_i16_attr(i.try_into().unwrap()).base(),
                    ),
                    PrimitiveType::I32 => (
                        builder.get_i32_type().base(),
                        builder.get_i32_attr(i.try_into().unwrap()).base(),
                    ),
                    PrimitiveType::I64 => (
                        builder.get_i64_type().base(),
                        builder.get_i64_attr(i).base(),
                    ),
                    PrimitiveType::Isize => (
                        builder.get_index_type().base(),
                        builder.get_index_attr(i).base(),
                    ),
                    _ => panic!("expected primitive integer type"),
                };
                builder.build_constant(loc, ty, attr).base()
            }
            Opcode::ImmInt
            | Opcode::ImmFloat
            | Opcode::ImmBool
            | Opcode::ImmAtom
            | Opcode::ImmNil
            | Opcode::ImmNone => {
                let imm = self.immediate_to_constant(loc, op.imm);
                self.values.insert(dfg.first_result(inst), imm);
                return Ok(());
            }
            Opcode::Tuple => match op.imm {
                arity @ Immediate::Isize(_) => {
                    let imm = self.immediate_to_constant(loc, arity);
                    let callee = self.get_or_declare_native(symbols::NifMakeTuple).unwrap();
                    self.cir().build_call(loc, callee, &[imm]).base()
                }
                other => panic!(
                    "invalid tuple size, expected isize immediate, got {:?}",
                    other
                ),
            },
            Opcode::Neg => {
                let imm = self.immediate_to_constant(loc, op.imm);
                let neg1 = self.get_or_declare_builtin("erlang:-/1").unwrap();
                let op = self.cir().build_call(loc, neg1, &[imm]).base();
                self.values.insert(results[0], op.get_result(1).base());
                return Ok(());
            }
            Opcode::Not => {
                let imm = self.immediate_to_constant(loc, op.imm);
                self.cir().build_not(loc, imm).base()
            }
            Opcode::Bnot => {
                let imm = self.immediate_to_constant(loc, op.imm);
                let bnot1 = self.get_or_declare_builtin("erlang:bnot/1").unwrap();
                let op = self.cir().build_call(loc, bnot1, &[imm]).base();
                self.values.insert(results[0], op.get_result(1).base());
                return Ok(());
            }
            other => unimplemented!("no lowering for unary op immediate with opcode {}", other),
        };

        assert_eq!(results.len(), mlir_op.num_results());
        for (value, op_result) in results.iter().copied().zip(mlir_op.results()) {
            self.values.insert(value, op_result.base());
        }
        Ok(())
    }

    fn build_unary_op_const(
        &mut self,
        dfg: &DataFlowGraph,
        inst: Inst,
        span: SourceSpan,
        op: &UnaryOpConst,
    ) -> anyhow::Result<()> {
        let loc = self.location_from_span(span);
        let imm = self.const_to_constant(loc, &dfg.constant(op.imm));
        let results = dfg.inst_results(inst);
        let mlir_op = match op.op {
            Opcode::ConstBigInt | Opcode::ConstBinary => {
                self.values.insert(dfg.first_result(inst), imm);
                return Ok(());
            }
            Opcode::Neg => {
                let neg1 = self.get_or_declare_builtin("erlang:-/1").unwrap();
                let op = self.cir().build_call(loc, neg1, &[imm]).base();
                self.values.insert(results[0], op.get_result(1).base());
                return Ok(());
            }
            Opcode::Not => self.cir().build_not(loc, imm).base(),
            Opcode::Bnot => {
                let bnot1 = self.get_or_declare_builtin("erlang:bnot/1").unwrap();
                let op = self.cir().build_call(loc, bnot1, &[imm]).base();
                self.values.insert(results[0], op.get_result(1).base());
                return Ok(());
            }
            other => unimplemented!("no lowering for unary op constant with opcode {}", other),
        };

        assert_eq!(results.len(), mlir_op.num_results());
        for (value, op_result) in results.iter().copied().zip(mlir_op.results()) {
            self.values.insert(value, op_result.base());
        }
        Ok(())
    }

    fn build_binary_op(
        &mut self,
        dfg: &DataFlowGraph,
        inst: Inst,
        span: SourceSpan,
        op: &BinaryOp,
    ) -> anyhow::Result<()> {
        let loc = self.location_from_span(span);
        let lhs = self.values[&op.args[0]];
        let rhs = self.values[&op.args[1]];
        let results = dfg.inst_results(inst);
        let mlir_op = match op.op {
            Opcode::Cons => {
                let process = self.get_process_argument();
                self.cir().build_cons(loc, process, lhs, rhs).base()
            }
            Opcode::ListConcat => {
                let callee = self.get_or_declare_builtin("erlang:++/2").unwrap();
                let op = self.cir().build_call(loc, callee, &[lhs, rhs]).base();
                self.values.insert(results[0], op.get_result(1).base());
                return Ok(());
            }
            Opcode::ListSubtract => {
                let callee = self.get_or_declare_builtin("erlang:--/2").unwrap();
                let op = self.cir().build_call(loc, callee, &[lhs, rhs]).base();
                self.values.insert(results[0], op.get_result(1).base());
                return Ok(());
            }
            Opcode::IcmpEq => self
                .llvm()
                .build_icmp(loc, ICmpPredicate::Eq, lhs, rhs)
                .base(),
            Opcode::IcmpNeq => self
                .llvm()
                .build_icmp(loc, ICmpPredicate::Neq, lhs, rhs)
                .base(),
            Opcode::IcmpGt => self
                .llvm()
                .build_icmp(loc, ICmpPredicate::Ugt, lhs, rhs)
                .base(),
            Opcode::IcmpGte => self
                .llvm()
                .build_icmp(loc, ICmpPredicate::Uge, lhs, rhs)
                .base(),
            Opcode::IcmpLt => self
                .llvm()
                .build_icmp(loc, ICmpPredicate::Ult, lhs, rhs)
                .base(),
            Opcode::IcmpLte => self
                .llvm()
                .build_icmp(loc, ICmpPredicate::Ule, lhs, rhs)
                .base(),
            Opcode::Eq => {
                let callee = self.get_or_declare_builtin("erlang:==/2").unwrap();
                let op = self.cir().build_call(loc, callee, &[lhs, rhs]).base();
                self.values.insert(results[0], op.get_result(1).base());
                return Ok(());
            }
            Opcode::EqExact => {
                let callee = self.get_or_declare_builtin("erlang:=:=/2").unwrap();
                let op = self.cir().build_call(loc, callee, &[lhs, rhs]).base();
                self.values.insert(results[0], op.get_result(1).base());
                return Ok(());
            }
            Opcode::Neq => {
                let callee = self.get_or_declare_builtin("erlang:/=/2").unwrap();
                let op = self.cir().build_call(loc, callee, &[lhs, rhs]).base();
                self.values.insert(results[0], op.get_result(1).base());
                return Ok(());
            }
            Opcode::NeqExact => {
                let callee = self.get_or_declare_builtin("erlang:=/=/2").unwrap();
                let op = self.cir().build_call(loc, callee, &[lhs, rhs]).base();
                self.values.insert(results[0], op.get_result(1).base());
                return Ok(());
            }
            Opcode::Gt => {
                let callee = self.get_or_declare_builtin("erlang:>/2").unwrap();
                let op = self.cir().build_call(loc, callee, &[lhs, rhs]).base();
                self.values.insert(results[0], op.get_result(1).base());
                return Ok(());
            }
            Opcode::Gte => {
                let callee = self.get_or_declare_builtin("erlang:>=/2").unwrap();
                let op = self.cir().build_call(loc, callee, &[lhs, rhs]).base();
                self.values.insert(results[0], op.get_result(1).base());
                return Ok(());
            }
            Opcode::Lt => {
                let callee = self.get_or_declare_builtin("erlang:</2").unwrap();
                let op = self.cir().build_call(loc, callee, &[lhs, rhs]).base();
                self.values.insert(results[0], op.get_result(1).base());
                return Ok(());
            }
            Opcode::Lte => {
                let callee = self.get_or_declare_builtin("erlang:=</2").unwrap();
                let op = self.cir().build_call(loc, callee, &[lhs, rhs]).base();
                self.values.insert(results[0], op.get_result(1).base());
                return Ok(());
            }
            Opcode::And => self.cir().build_and(loc, lhs, rhs).base(),
            Opcode::AndAlso => self.cir().build_andalso(loc, lhs, rhs).base(),
            Opcode::Or => self.cir().build_or(loc, lhs, rhs).base(),
            Opcode::OrElse => self.cir().build_orelse(loc, lhs, rhs).base(),
            Opcode::Xor => self.cir().build_xor(loc, lhs, rhs).base(),
            Opcode::Band => {
                let callee = self.get_or_declare_builtin("erlang:band/2").unwrap();
                let op = self.cir().build_call(loc, callee, &[lhs, rhs]).base();
                self.values.insert(results[0], op.get_result(1).base());
                return Ok(());
            }
            Opcode::Bor => {
                let callee = self.get_or_declare_builtin("erlang:bor/2").unwrap();
                let op = self.cir().build_call(loc, callee, &[lhs, rhs]).base();
                self.values.insert(results[0], op.get_result(1).base());
                return Ok(());
            }
            Opcode::Bxor => {
                let callee = self.get_or_declare_builtin("erlang:bxor/2").unwrap();
                let op = self.cir().build_call(loc, callee, &[lhs, rhs]).base();
                self.values.insert(results[0], op.get_result(1).base());
                return Ok(());
            }
            Opcode::Bsl => {
                let callee = self.get_or_declare_builtin("erlang:bsl/2").unwrap();
                let op = self.cir().build_call(loc, callee, &[lhs, rhs]).base();
                self.values.insert(results[0], op.get_result(1).base());
                return Ok(());
            }
            Opcode::Bsr => {
                let callee = self.get_or_declare_builtin("erlang:bsr/2").unwrap();
                let op = self.cir().build_call(loc, callee, &[lhs, rhs]).base();
                self.values.insert(results[0], op.get_result(1).base());
                return Ok(());
            }
            Opcode::Div => {
                let callee = self.get_or_declare_builtin("erlang:div/2").unwrap();
                let op = self.cir().build_call(loc, callee, &[lhs, rhs]).base();
                self.values.insert(results[0], op.get_result(1).base());
                return Ok(());
            }
            Opcode::Rem => {
                let callee = self.get_or_declare_builtin("erlang:rem/2").unwrap();
                let op = self.cir().build_call(loc, callee, &[lhs, rhs]).base();
                self.values.insert(results[0], op.get_result(1).base());
                return Ok(());
            }
            Opcode::Add => {
                let callee = self.get_or_declare_builtin("erlang:+/2").unwrap();
                let op = self.cir().build_call(loc, callee, &[lhs, rhs]).base();
                self.values.insert(results[0], op.get_result(1).base());
                return Ok(());
            }
            Opcode::Sub => {
                let callee = self.get_or_declare_builtin("erlang:-/2").unwrap();
                let op = self.cir().build_call(loc, callee, &[lhs, rhs]).base();
                self.values.insert(results[0], op.get_result(1).base());
                return Ok(());
            }
            Opcode::Mul => {
                let callee = self.get_or_declare_builtin("erlang:*/2").unwrap();
                let op = self.cir().build_call(loc, callee, &[lhs, rhs]).base();
                self.values.insert(results[0], op.get_result(1).base());
                return Ok(());
            }
            Opcode::Fdiv => {
                let callee = self.get_or_declare_builtin("erlang:fdiv/2").unwrap();
                let op = self.cir().build_call(loc, callee, &[lhs, rhs]).base();
                self.values.insert(results[0], op.get_result(1).base());
                return Ok(());
            }
            other => unimplemented!("no lowering for binary op with opcode {}", other),
        };

        assert_eq!(results.len(), mlir_op.num_results());
        for (value, op_result) in results.iter().copied().zip(mlir_op.results()) {
            self.values.insert(value, op_result.base());
        }
        Ok(())
    }

    fn build_binary_op_imm(
        &mut self,
        dfg: &DataFlowGraph,
        inst: Inst,
        span: SourceSpan,
        op: &BinaryOpImm,
    ) -> anyhow::Result<()> {
        let loc = self.location_from_span(span);
        let lhs = self.values[&op.arg];
        let results = dfg.inst_results(inst);
        let mlir_op = match op.op {
            Opcode::Cons => {
                let process = self.get_process_argument();
                let rhs = self.immediate_to_constant(loc, op.imm);
                self.cir().build_cons(loc, process, lhs, rhs).base()
            }
            Opcode::GetElement => {
                let index = op.imm.as_i64().expect("invalid get_element immediate argument, only integer immediates are supported");
                let rhs = self.cir().get_index_attr(index);
                self.cir().build_get_element(loc, lhs, rhs).base()
            }
            Opcode::IsTaggedTuple => {
                match op.imm {
                    Immediate::Term(ImmediateTerm::Atom(a)) => self.cir().build_is_tagged_tuple(loc, lhs, a).base(),
                    _ => panic!("invalid is_tagged_tuple immediate argument, only atom immediates are supported"),
                }
            }
            Opcode::BitsTestTail => {
                let size = op.imm.as_i64().expect("invalid bs_test_tail immediate argument, only integer immediates are supported");
                let rhs = self.cir().get_index_attr(size);
                self.cir().build_bs_test_tail(loc, lhs, rhs).base()
            }
            Opcode::UnpackEnv => {
                let index = op.imm.as_i64().expect("invalid unpack_env immediate argument, only integer immediates are supported");
                let rhs = self.cir().get_index_attr(index);
                self.cir().build_unpack_env(loc, lhs, rhs).base()
            }
            Opcode::IcmpEq => {
                let rhs = self.immediate_to_constant(loc, op.imm);
                self.llvm().build_icmp(loc, ICmpPredicate::Eq, lhs, rhs).base()
            }
            Opcode::IcmpNeq => {
                let rhs = self.immediate_to_constant(loc, op.imm);
                self.llvm().build_icmp(loc, ICmpPredicate::Neq, lhs, rhs).base()
            }
            Opcode::IcmpGt => {
                let rhs = self.immediate_to_constant(loc, op.imm);
                self.llvm().build_icmp(loc, ICmpPredicate::Ugt, lhs, rhs).base()
            }
            Opcode::IcmpGte => {
                let rhs = self.immediate_to_constant(loc, op.imm);
                self.llvm().build_icmp(loc, ICmpPredicate::Uge, lhs, rhs).base()
            }
            Opcode::IcmpLt => {
                let rhs = self.immediate_to_constant(loc, op.imm);
                self.llvm().build_icmp(loc, ICmpPredicate::Ult, lhs, rhs).base()
            }
            Opcode::IcmpLte => {
                let rhs = self.immediate_to_constant(loc, op.imm);
                self.llvm().build_icmp(loc, ICmpPredicate::Ule, lhs, rhs).base()
            }
            Opcode::Eq => {
                let rhs = self.immediate_to_constant(loc, op.imm);
                let callee = self.get_or_declare_builtin("erlang:==/2").unwrap();
                let op = self.cir().build_call(loc, callee, &[lhs, rhs]).base();
                self.values.insert(results[0], op.get_result(1).base());
                return Ok(());
            }
            Opcode::EqExact => {
                let rhs = self.immediate_to_constant(loc, op.imm);
                let callee = self.get_or_declare_builtin("erlang:=:=/2").unwrap();
                let op = self.cir().build_call(loc, callee, &[lhs, rhs]).base();
                self.values.insert(results[0], op.get_result(1).base());
                return Ok(());
            }
            Opcode::Neq => {
                let rhs = self.immediate_to_constant(loc, op.imm);
                let callee = self.get_or_declare_builtin("erlang:/=/2").unwrap();
                let op = self.cir().build_call(loc, callee, &[lhs, rhs]).base();
                self.values.insert(results[0], op.get_result(1).base());
                return Ok(());
            }
            Opcode::NeqExact => {
                let rhs = self.immediate_to_constant(loc, op.imm);
                let callee = self.get_or_declare_builtin("erlang:=/=/2").unwrap();
                let op = self.cir().build_call(loc, callee, &[lhs, rhs]).base();
                self.values.insert(results[0], op.get_result(1).base());
                return Ok(());
            }
            Opcode::Gt => {
                let rhs = self.immediate_to_constant(loc, op.imm);
                let callee = self.get_or_declare_builtin("erlang:>/2").unwrap();
                let op = self.cir().build_call(loc, callee, &[lhs, rhs]).base();
                self.values.insert(results[0], op.get_result(1).base());
                return Ok(());
            }
            Opcode::Gte => {
                let rhs = self.immediate_to_constant(loc, op.imm);
                let callee = self.get_or_declare_builtin("erlang:>=/2").unwrap();
                let op = self.cir().build_call(loc, callee, &[lhs, rhs]).base();
                self.values.insert(results[0], op.get_result(1).base());
                return Ok(());
            }
            Opcode::Lt => {
                let rhs = self.immediate_to_constant(loc, op.imm);
                let callee = self.get_or_declare_builtin("erlang:</2").unwrap();
                let op = self.cir().build_call(loc, callee, &[lhs, rhs]).base();
                self.values.insert(results[0], op.get_result(1).base());
                return Ok(());
            }
            Opcode::Lte => {
                let rhs = self.immediate_to_constant(loc, op.imm);
                let callee = self.get_or_declare_builtin("erlang:=</2").unwrap();
                let op = self.cir().build_call(loc, callee, &[lhs, rhs]).base();
                self.values.insert(results[0], op.get_result(1).base());
                return Ok(());
            }
            Opcode::And => {
                assert!(op.imm.as_bool().is_some(), "invalid binary immediate argument, requires a boolean");
                self.cir().build_and(loc, lhs, self.immediate_to_constant(loc, op.imm)).base()
            }
            Opcode::AndAlso => {
                assert!(op.imm.as_bool().is_some(), "invalid binary immediate argument, requires a boolean");
                self.cir().build_andalso(loc, lhs, self.immediate_to_constant(loc, op.imm)).base()
            }
            Opcode::Or => {
                assert!(op.imm.as_bool().is_some(), "invalid binary immediate argument, requires a boolean");
                self.cir().build_or(loc, lhs, self.immediate_to_constant(loc, op.imm)).base()
            }
            Opcode::OrElse => {
                assert!(op.imm.as_bool().is_some(), "invalid binary immediate argument, requires a boolean");
                self.cir().build_orelse(loc, lhs, self.immediate_to_constant(loc, op.imm)).base()
            }
            Opcode::Xor => {
                assert!(op.imm.as_bool().is_some(), "invalid binary immediate argument, requires a boolean");
                self.cir().build_xor(loc, lhs, self.immediate_to_constant(loc, op.imm)).base()
            }
            Opcode::Band => {
                let rhs = self.immediate_to_constant(loc, op.imm);
                let callee = self.get_or_declare_builtin("erlang:band/2").unwrap();
                let op = self.cir().build_call(loc, callee, &[lhs, rhs]).base();
                self.values.insert(results[0], op.get_result(1).base());
                return Ok(());
            }
            Opcode::Bor => {
                let rhs = self.immediate_to_constant(loc, op.imm);
                let callee = self.get_or_declare_builtin("erlang:bor/2").unwrap();
                let op = self.cir().build_call(loc, callee, &[lhs, rhs]).base();
                self.values.insert(results[0], op.get_result(1).base());
                return Ok(());
            }
            Opcode::Bxor => {
                let rhs = self.immediate_to_constant(loc, op.imm);
                let callee = self.get_or_declare_builtin("erlang:bxor/2").unwrap();
                let op = self.cir().build_call(loc, callee, &[lhs, rhs]).base();
                self.values.insert(results[0], op.get_result(1).base());
                return Ok(());
            }
            Opcode::Bsl => {
                let rhs = self.immediate_to_constant(loc, op.imm);
                let callee = self.get_or_declare_builtin("erlang:bsl/2").unwrap();
                let op = self.cir().build_call(loc, callee, &[lhs, rhs]).base();
                self.values.insert(results[0], op.get_result(1).base());
                return Ok(());
            }
            Opcode::Bsr => {
                let rhs = self.immediate_to_constant(loc, op.imm);
                let callee = self.get_or_declare_builtin("erlang:bsr/2").unwrap();
                let op = self.cir().build_call(loc, callee, &[lhs, rhs]).base();
                self.values.insert(results[0], op.get_result(1).base());
                return Ok(());
            }
            Opcode::Div => {
                let rhs = self.immediate_to_constant(loc, op.imm);
                let callee = self.get_or_declare_builtin("erlang:div/2").unwrap();
                let op = self.cir().build_call(loc, callee, &[lhs, rhs]).base();
                self.values.insert(results[0], op.get_result(1).base());
                return Ok(());
            }
            Opcode::Rem => {
                let rhs = self.immediate_to_constant(loc, op.imm);
                let callee = self.get_or_declare_builtin("erlang:rem/2").unwrap();
                let op = self.cir().build_call(loc, callee, &[lhs, rhs]).base();
                self.values.insert(results[0], op.get_result(1).base());
                return Ok(());
            }
            Opcode::Add => {
                let rhs = self.immediate_to_constant(loc, op.imm);
                let callee = self.get_or_declare_builtin("erlang:+/2").unwrap();
                let op = self.cir().build_call(loc, callee, &[lhs, rhs]).base();
                self.values.insert(results[0], op.get_result(1).base());
                return Ok(());
            }
            Opcode::Sub => {
                let rhs = self.immediate_to_constant(loc, op.imm);
                let callee = self.get_or_declare_builtin("erlang:-/2").unwrap();
                let op = self.cir().build_call(loc, callee, &[lhs, rhs]).base();
                self.values.insert(results[0], op.get_result(1).base());
                return Ok(());
            }
            Opcode::Mul => {
                let rhs = self.immediate_to_constant(loc, op.imm);
                let callee = self.get_or_declare_builtin("erlang:*/2").unwrap();
                let op = self.cir().build_call(loc, callee, &[lhs, rhs]).base();
                self.values.insert(results[0], op.get_result(1).base());
                return Ok(());
            }
            Opcode::Fdiv => {
                let rhs = self.immediate_to_constant(loc, op.imm);
                let callee = self.get_or_declare_builtin("erlang:fdiv/2").unwrap();
                let op = self.cir().build_call(loc, callee, &[lhs, rhs]).base();
                self.values.insert(results[0], op.get_result(1).base());
                return Ok(());
            }
            other => unimplemented!("no lowering for binary immediate op with opcode {}", other),
        };

        assert_eq!(results.len(), mlir_op.num_results());
        for (value, op_result) in results.iter().copied().zip(mlir_op.results()) {
            self.values.insert(value, op_result.base());
        }
        Ok(())
    }

    fn build_ret(
        &self,
        dfg: &DataFlowGraph,
        inst: Inst,
        span: SourceSpan,
        _op: &Ret,
    ) -> anyhow::Result<()> {
        let loc = self.location_from_span(span);
        let args = dfg.inst_args(inst);
        let current_function: FuncOp = self.current_block.operation().unwrap().try_into().unwrap();
        let func_type = current_function.get_type();
        let mut mapped_args = Vec::with_capacity(args.len());
        for (i, mapped_arg) in args.iter().map(|a| self.values[a]).enumerate() {
            let arg_type = mapped_arg.get_type();
            let return_type = func_type.get_result(i).unwrap();

            if arg_type == return_type {
                mapped_args.push(mapped_arg);
            } else {
                let cast = self.cir().build_cast(loc, mapped_arg, return_type);
                mapped_args.push(cast.get_result(0).base());
            }
        }
        self.cir().build_return(loc, mapped_args.as_slice());
        Ok(())
    }

    fn build_ret_imm(
        &self,
        _dfg: &DataFlowGraph,
        _inst: Inst,
        span: SourceSpan,
        op: &RetImm,
    ) -> anyhow::Result<()> {
        let loc = self.location_from_span(span);
        let current_function: FuncOp = self.current_block.operation().unwrap().try_into().unwrap();
        let func_type = current_function.get_type();
        let arg = self.values[&op.arg];
        // Only Bool is supported as an immediate for this op currently
        let flag = op.imm.as_bool().expect("expected boolean immediate");
        let arg_type = arg.get_type();
        let expected_imm_type = func_type.get_result(0).unwrap();
        let expected_arg_type = func_type.get_result(1).unwrap();

        let builder = self.cir();
        let arg = if arg_type == expected_arg_type {
            arg
        } else {
            let cast = builder.build_cast(loc, arg, expected_arg_type);
            cast.get_result(0).base()
        };

        let imm = builder.build_constant(
            loc,
            expected_imm_type,
            builder.get_integer_attr(builder.get_i1_type(), flag as i64),
        );

        builder.build_return(loc, &[imm.get_result(0).base(), arg]);
        Ok(())
    }

    fn build_switch(
        &mut self,
        _dfg: &DataFlowGraph,
        _inst: Inst,
        span: SourceSpan,
        op: &Switch,
    ) -> anyhow::Result<()> {
        let loc = self.location_from_span(span);
        let arg = self.values[&op.arg];
        let builder = CirBuilder::new(&self.builder);
        let mut mlir_op = builder.build_switch(loc, arg);
        for (value, dest) in op.arms.iter() {
            let dest = self.blocks[dest];
            mlir_op.with_case(*value, dest, &[]);
        }
        let default = self.blocks[&op.default];
        mlir_op.with_default(default, &[]);
        mlir_op.build();
        Ok(())
    }

    fn build_cond_br(
        &mut self,
        dfg: &DataFlowGraph,
        _inst: Inst,
        span: SourceSpan,
        op: &CondBr,
    ) -> anyhow::Result<()> {
        let loc = self.location_from_span(span);
        let then_dest = self.blocks[&op.then_dest.0];
        let else_dest = self.blocks[&op.else_dest.0];
        let then_args = op.then_dest.1.as_slice(&dfg.value_lists);
        let else_args = op.else_dest.1.as_slice(&dfg.value_lists);
        let builder = CirBuilder::new(&self.builder);
        let i1ty = builder.get_i1_type().base();
        let cond = self.values[&op.cond];
        let cond = if cond.get_type() != i1ty {
            let cond_cast = builder.build_cast(loc, cond, builder.get_i1_type());
            cond_cast.get_result(0).base()
        } else {
            cond
        };
        let mut then_args_mapped = Vec::with_capacity(then_args.len());
        for (i, mapped_arg) in then_args.iter().map(|a| self.values[a]).enumerate() {
            let expected_ty = then_dest.get_argument(i).get_type();
            if mapped_arg.get_type() == expected_ty {
                then_args_mapped.push(mapped_arg.base());
            } else {
                let cast = builder.build_cast(loc, mapped_arg, expected_ty);
                then_args_mapped.push(cast.get_result(0).base());
            }
        }
        let mut else_args_mapped = Vec::with_capacity(else_args.len());
        for (i, mapped_arg) in else_args.iter().map(|a| self.values[a]).enumerate() {
            let expected_ty = else_dest.get_argument(i).get_type();
            if mapped_arg.get_type() == expected_ty {
                else_args_mapped.push(mapped_arg.base());
            } else {
                let cast = builder.build_cast(loc, mapped_arg, expected_ty);
                else_args_mapped.push(cast.get_result(0).base());
            }
        }
        builder.build_cond_branch(
            loc,
            cond,
            then_dest,
            then_args_mapped.as_slice(),
            else_dest,
            else_args_mapped.as_slice(),
        );
        Ok(())
    }

    fn build_br(
        &mut self,
        dfg: &DataFlowGraph,
        inst: Inst,
        span: SourceSpan,
        op: &Br,
    ) -> anyhow::Result<()> {
        let loc = self.location_from_span(span);
        let dest = self.blocks[&op.destination];
        let args = dfg.inst_args(inst);
        let mut mapped_args = Vec::with_capacity(args.len());
        let builder = CirBuilder::new(&self.builder);
        let i1ty = builder.get_i1_type().base();
        for (i, mapped_arg) in args.iter().map(|a| self.values[a]).enumerate() {
            if i == 0 && op.op != Opcode::Br {
                let cond = if mapped_arg.get_type() != i1ty {
                    let cond_cast = builder.build_cast(loc, mapped_arg, builder.get_i1_type());
                    cond_cast.get_result(0).base()
                } else {
                    mapped_arg
                };
                mapped_args.push(cond);
                continue;
            }
            let index = if op.op == Opcode::Br { i } else { i - 1 };
            let expected_ty = dest.get_argument(index).get_type();
            if mapped_arg.get_type() == expected_ty {
                mapped_args.push(mapped_arg.base());
            } else {
                let cast = builder.build_cast(loc, mapped_arg, expected_ty);
                mapped_args.push(cast.get_result(0).base());
            }
        }
        match op.op {
            Opcode::Br => {
                self.cir().build_branch(loc, dest, mapped_args.as_slice());

                Ok(())
            }
            Opcode::BrIf | Opcode::BrUnless => {
                // In syntax_ssa, control continues after the conditional jump, but in mlir
                // we need to split the original block in two, and jump to either the desired
                // destination block, or the latter half of the original block where we will
                // resume building
                let split_block = {
                    let region = self.current_block.region().unwrap();
                    let block = OwnedBlock::default();
                    let block_ref = block.base();
                    region.insert_after(self.current_block, block);
                    block_ref
                };
                let dest_args = &mapped_args[1..];
                let cond = mapped_args[0];
                if op.op == Opcode::BrIf {
                    builder.build_cond_branch(loc, cond, dest, dest_args, split_block, &[]);
                } else {
                    // For BrUnless, we need to invert the condition
                    builder.build_cond_branch(loc, cond, split_block, &[], dest, dest_args);
                };
                builder.set_insertion_point_to_end(split_block);
                self.current_block = split_block;

                Ok(())
            }
            other => unimplemented!("unrecognized branching op: {}", other),
        }
    }

    fn build_is_type(
        &mut self,
        dfg: &DataFlowGraph,
        inst: Inst,
        span: SourceSpan,
        op: &IsType,
    ) -> anyhow::Result<()> {
        use firefly_syntax_base::{TermType, Type as CoreType};

        let builder = CirBuilder::new(&self.builder);
        let loc = self.location_from_span(span);
        let input = self.values[&op.arg];
        let op = match op.ty {
            CoreType::Term(TermType::List(_)) => builder.build_is_list(loc, input).base(),
            CoreType::Term(TermType::Cons) => builder.build_is_nonempty_list(loc, input).base(),
            CoreType::Term(TermType::Nil) => {
                let raw = builder.build_cast(loc, input, builder.get_i64_type());
                let llvm = builder.llvm();
                let i1 = llvm.get_i64_type();
                let nil = llvm.build_constant(
                    loc,
                    i1,
                    llvm.get_i64_attr(unsafe { mem::transmute::<f64, i64>(f64::INFINITY) })
                        .base(),
                );
                llvm.build_icmp(
                    loc,
                    ICmpPredicate::Eq,
                    raw.get_result(0).base(),
                    nil.get_result(0).base(),
                )
                .base()
            }
            CoreType::Term(TermType::Number) => builder.build_is_number(loc, input).base(),
            CoreType::Term(TermType::Integer) => builder.build_is_integer(loc, input).base(),
            CoreType::Term(TermType::Float) => builder.build_is_float(loc, input).base(),
            CoreType::Term(TermType::Atom) => builder.build_is_atom(loc, input).base(),
            CoreType::Term(TermType::Bool) => builder.build_is_bool(loc, input).base(),
            _ => {
                let ty = translate_ir_type(&self.module, &self.options, &builder, &op.ty);
                builder.build_is_type(loc, input, ty).base()
            }
        };

        // Map syntax_ssa results to MLIR results
        let result = dfg.first_result(inst);
        let mlir_result = op.get_result(0);
        self.values.insert(result, mlir_result.base());

        Ok(())
    }

    fn build_primop(
        &mut self,
        dfg: &DataFlowGraph,
        inst: Inst,
        span: SourceSpan,
        op: &PrimOp,
    ) -> anyhow::Result<()> {
        let loc = self.location_from_span(span);
        let args = dfg.inst_args(inst);
        let builder = CirBuilder::new(&self.builder);
        let mlir_op = match op.op {
            Opcode::BitsMatchStart => {
                let bin = self.values[&args[0]];
                builder.build_bs_match_start(loc, bin).base()
            }
            Opcode::RecvStart => builder.build_recv_start(loc, self.values[&args[0]]).base(),
            Opcode::RecvNext => builder.build_recv_next(loc, self.values[&args[0]]).base(),
            Opcode::RecvPeek => builder.build_recv_next(loc, self.values[&args[0]]).base(),
            Opcode::RecvPop => builder.build_recv_pop(loc, self.values[&args[0]]).base(),
            Opcode::RecvWait => builder.build_yield(loc).base(),
            Opcode::Raise => {
                let class = self.values[&args[0]];
                let reason = self.values[&args[1]];
                let trace = self.values[&args[2]];
                builder.build_raise(loc, class, reason, trace).base()
            }
            Opcode::ExceptionClass => builder
                .build_exception_class(loc, self.values[&args[0]])
                .base(),
            Opcode::ExceptionReason => builder
                .build_exception_reason(loc, self.values[&args[0]])
                .base(),
            Opcode::ExceptionTrace => builder
                .build_exception_trace(loc, self.values[&args[0]])
                .base(),
            other => unimplemented!("unrecognized primop: {}", other),
        };

        let results = dfg.inst_results(inst);
        assert_eq!(
            results.len(),
            mlir_op.num_results(),
            "representation of instruction {} in ssa does not match mlir",
            &op.op
        );
        for (value, op_result) in results.iter().copied().zip(mlir_op.results()) {
            self.values.insert(value, op_result.base());
        }
        Ok(())
    }

    fn build_primop_imm(
        &mut self,
        dfg: &DataFlowGraph,
        inst: Inst,
        span: SourceSpan,
        op: &PrimOpImm,
    ) -> anyhow::Result<()> {
        let loc = self.location_from_span(span);
        let imm = self.immediate_to_constant(loc, op.imm);
        let args = dfg.inst_args(inst);
        let builder = CirBuilder::new(&self.builder);
        let mlir_op = match op.op {
            Opcode::RecvStart => builder.build_recv_start(loc, imm).base(),
            Opcode::Raise => {
                let class = imm;
                let reason = self.values[&args[1]];
                let trace = self.values[&args[2]];
                builder.build_raise(loc, class, reason, trace).base()
            }
            other => unimplemented!("unrecognized primop immediate op: {}", other),
        };

        let results = dfg.inst_results(inst);
        assert_eq!(
            results.len(),
            mlir_op.num_results(),
            "representation of instruction {} in ssa does not match mlir",
            &op.op
        );
        for (value, op_result) in results.iter().copied().zip(mlir_op.results()) {
            self.values.insert(value, op_result.base());
        }
        Ok(())
    }

    fn build_make_fun(
        &mut self,
        dfg: &DataFlowGraph,
        inst: Inst,
        span: SourceSpan,
        op: &MakeFun,
    ) -> anyhow::Result<()> {
        let loc = self.location_from_span(span);
        let sig = self.find_function(op.callee);
        let name = sig.mfa().to_string();

        let callee = self.get_or_declare_function(name.as_str()).unwrap();

        let args = dfg.inst_args(inst);
        let process = self.values[&args[0]];
        let env = args
            .iter()
            .skip(1)
            .map(|a| self.values[a])
            .collect::<Vec<_>>();
        let mlir_op = self.cir().build_fun(loc, callee, process, env.as_slice());

        let results = dfg.inst_results(inst);
        assert_eq!(results.len(), mlir_op.num_results());
        for (value, op_result) in results.iter().copied().zip(mlir_op.results()) {
            self.values.insert(value, op_result.base());
        }
        Ok(())
    }

    fn build_call(
        &mut self,
        dfg: &DataFlowGraph,
        inst: Inst,
        span: SourceSpan,
        op: &Call,
    ) -> anyhow::Result<()> {
        let loc = self.location_from_span(span);
        let sig = self.find_function(op.callee);
        let callee = match sig.module {
            symbols::Empty => {
                // The callee is a native function that uses a different naming scheme
                self.get_or_declare_native(sig.name).unwrap()
            }
            _ => {
                let name = sig.mfa().to_string();
                self.get_or_declare_function(name.as_str()).unwrap()
            }
        };

        let func_type = callee.get_type();

        let builder = CirBuilder::new(&self.builder);
        let args = dfg.inst_args(inst);
        let mut mapped_args = Vec::with_capacity(args.len());
        for (i, mapped_arg) in args.iter().map(|a| self.values[a]).enumerate() {
            let arg_type = mapped_arg.get_type();
            let expected_type = func_type.get_input(i).unwrap();

            if arg_type == expected_type {
                mapped_args.push(mapped_arg);
            } else {
                let cast = builder.build_cast(loc, mapped_arg, expected_type);
                mapped_args.push(cast.get_result(0).base());
            }
        }

        let is_async = op.cc.is_async();
        let mlir_op = match op.op {
            Opcode::Call => {
                let mlir_op = builder.build_call(loc, callee, mapped_args.as_slice());
                if is_async {
                    mlir_op.set_attribute_by_name("async", builder.get_unit_attr());
                }
                mlir_op.base()
            }
            Opcode::Enter => {
                let mlir_op = builder.build_enter(loc, callee, mapped_args.as_slice());
                if is_async {
                    mlir_op.set_attribute_by_name("async", builder.get_unit_attr());
                }
                mlir_op.set_attribute_by_name("tail", builder.get_unit_attr());
                mlir_op.set_attribute_by_name("musttail", builder.get_unit_attr());
                mlir_op.base()
            }
            op => panic!("unrecognized call opcode '{}'", &op),
        };

        let results = dfg.inst_results(inst);
        assert_eq!(results.len(), mlir_op.num_results());
        for (value, op_result) in results.iter().copied().zip(mlir_op.results()) {
            self.values.insert(value, op_result.base());
        }
        Ok(())
    }

    fn build_call_indirect(
        &mut self,
        dfg: &DataFlowGraph,
        inst: Inst,
        span: SourceSpan,
        op: &CallIndirect,
    ) -> anyhow::Result<()> {
        let loc = self.location_from_span(span);
        let args = dfg.inst_args(inst);
        let mapped_args = args.iter().map(|a| self.values[a]).collect::<Vec<_>>();

        let builder = CirBuilder::new(&self.builder);
        let callee = self.values[&op.callee];
        let is_async = op.cc.is_async();
        let mlir_op = match op.op {
            Opcode::CallIndirect => {
                // This is an indirect call in tail position, and the callee is known statically
                // to be a fun, so we can skip the overhead of erlang:apply/3
                let mlir_op = builder.build_call_indirect(loc, callee, mapped_args.as_slice());
                if is_async {
                    mlir_op.set_attribute_by_name("async", builder.get_unit_attr());
                }
                mlir_op.base()
            }
            Opcode::EnterIndirect => {
                // This is an indirect call, not in tail position, but the callee is known statically
                // to be a fun, so we can skip the overhead of erlang:apply/3
                let mlir_op = builder.build_enter_indirect(loc, callee, mapped_args.as_slice());
                if is_async {
                    mlir_op.set_attribute_by_name("async", builder.get_unit_attr());
                }
                mlir_op.set_attribute_by_name("tail", builder.get_unit_attr());
                mlir_op.set_attribute_by_name("musttail", builder.get_unit_attr());
                mlir_op.base()
            }
            op => panic!("unrecognized call.indirect opcode '{}'", &op),
        };

        let results = dfg.inst_results(inst);
        assert_eq!(results.len(), mlir_op.num_results());
        for (value, op_result) in results.iter().copied().zip(mlir_op.results()) {
            self.values.insert(value, op_result.base());
        }
        Ok(())
    }

    fn build_setelement(
        &mut self,
        dfg: &DataFlowGraph,
        inst: Inst,
        span: SourceSpan,
        op: &SetElement,
    ) -> anyhow::Result<()> {
        let loc = self.location_from_span(span);
        let tuple = self.values[&op.args[0]];
        let value = self.values[&op.args[1]];

        let builder = CirBuilder::new(&self.builder);
        let index = builder.get_index_attr(
            op.index
                .as_i64()
                .unwrap()
                .try_into()
                .expect("index too large"),
        );
        let mlir_op = match op.op {
            Opcode::SetElement => builder.build_set_element(loc, tuple, index, value),
            Opcode::SetElementMut => builder.build_set_element_mut(loc, tuple, index, value),
            op => panic!("unrecognized setelement opcode: {}", op),
        };
        self.values
            .insert(dfg.first_result(inst), mlir_op.get_result(0).base());

        Ok(())
    }

    fn build_setelement_imm(
        &mut self,
        dfg: &DataFlowGraph,
        inst: Inst,
        span: SourceSpan,
        op: &SetElementImm,
    ) -> anyhow::Result<()> {
        let loc = self.location_from_span(span);
        let tuple = self.values[&op.arg];
        let value = self.immediate_to_constant(loc, op.value);

        let builder = CirBuilder::new(&self.builder);
        let index = builder.get_index_attr(
            op.index
                .as_i64()
                .unwrap()
                .try_into()
                .expect("index too large"),
        );
        let mlir_op = match op.op {
            Opcode::SetElement => builder.build_set_element(loc, tuple, index, value),
            Opcode::SetElementMut => builder.build_set_element_mut(loc, tuple, index, value),
            op => panic!("unrecognized setelement opcode: {}", op),
        };
        self.values
            .insert(dfg.first_result(inst), mlir_op.get_result(0).base());

        Ok(())
    }

    fn build_bits_match(
        &mut self,
        dfg: &DataFlowGraph,
        inst: Inst,
        span: SourceSpan,
        op: &BitsMatch,
    ) -> anyhow::Result<()> {
        let loc = self.location_from_span(span);
        let args = dfg.inst_args(inst);
        let spec = op.spec;
        let src = self.values[&args[0]];
        let size = if args.len() > 1 {
            Some(self.values[&args[1]])
        } else {
            None
        };
        let builder = CirBuilder::new(&self.builder);
        let mlir_op = builder.build_bs_match(loc, src, spec, size);

        let results = dfg.inst_results(inst);
        assert_eq!(results.len(), mlir_op.num_results());
        for (value, op_result) in results.iter().copied().zip(mlir_op.results()) {
            self.values.insert(value, op_result.base());
        }
        Ok(())
    }

    fn build_bits_match_skip(
        &mut self,
        dfg: &DataFlowGraph,
        inst: Inst,
        span: SourceSpan,
        op: &BitsMatchSkip,
    ) -> anyhow::Result<()> {
        let loc = self.location_from_span(span);
        let args = dfg.inst_args(inst);
        let spec = op.spec;
        let src = self.values[&args[0]];
        let size = self.values[&args[1]];
        let value = self.immediate_to_constant(loc, op.value);

        let builder = CirBuilder::new(&self.builder);
        let mlir_op = builder.build_bs_match_skip(loc, src, spec, size, value);

        let results = dfg.inst_results(inst);
        assert_eq!(results.len(), mlir_op.num_results());
        for (value, op_result) in results.iter().copied().zip(mlir_op.results()) {
            self.values.insert(value, op_result.base());
        }
        Ok(())
    }

    fn build_bits_push(
        &mut self,
        dfg: &DataFlowGraph,
        inst: Inst,
        span: SourceSpan,
        op: &BitsPush,
    ) -> anyhow::Result<()> {
        let loc = self.location_from_span(span);
        let args = dfg.inst_args(inst);
        let spec = op.spec;
        let bin = self.values[&args[0]];
        let value = self.values[&args[1]];
        let size = if args.len() > 2 {
            Some(self.values[&args[2]])
        } else {
            None
        };
        let builder = CirBuilder::new(&self.builder);
        let mlir_op = builder.build_bs_push(loc, bin, spec, value, size);

        let results = dfg.inst_results(inst);
        assert_eq!(results.len(), mlir_op.num_results());
        for (value, op_result) in results.iter().copied().zip(mlir_op.results()) {
            self.values.insert(value, op_result.base());
        }
        Ok(())
    }
}

/// Translates a type to an equivalent MLIR type
fn translate_ir_type<'a, B: OpBuilder>(
    module: &syntax_ssa::Module,
    options: &Options,
    builder: &CirBuilder<'a, B>,
    ty: &syntax_base::Type,
) -> TypeBase {
    use firefly_syntax_base::Type as CoreType;

    debug!("translating syntax_base type {:?} to mlir type", ty);
    match ty {
        CoreType::Unit | CoreType::Never => builder.get_cir_none_type().base(),
        CoreType::Primitive(ref ty) => translate_primitive_ir_type(builder, ty),
        CoreType::Term(ref ty) => translate_term_ir_type(module, options, builder, ty),
        CoreType::Function(ref ty) => {
            let param_types = ty
                .params()
                .iter()
                .map(|t| translate_ir_type(module, options, builder, t))
                .collect::<Vec<_>>();
            let result_types = ty
                .results()
                .iter()
                .map(|t| translate_ir_type(module, options, builder, t))
                .collect::<Vec<_>>();
            builder
                .get_function_type(param_types.as_slice(), result_types.as_slice())
                .base()
        }
        CoreType::Future(_) => {
            //let inner = translate_ir_type(module, options, builder, ty);
            //builder.get_cir_future_type(inner).base()
            unimplemented!()
        }
        CoreType::Process => builder
            .get_cir_ptr_type(builder.get_cir_process_type())
            .base(),
        CoreType::Exception => builder
            .get_cir_ptr_type(builder.get_cir_exception_type())
            .base(),
        CoreType::ExceptionTrace => builder
            .get_cir_ptr_type(builder.get_cir_trace_type())
            .base(),
        CoreType::RecvContext => builder.get_cir_recv_context_type().base(),
        CoreType::RecvState => builder.get_i8_type().base(),
        CoreType::BinaryBuilder => builder
            .get_cir_ptr_type(builder.get_cir_binary_builder_type())
            .base(),
        CoreType::MatchContext => builder
            .get_cir_ptr_type(builder.get_cir_match_context_type())
            .base(),
        // NOTE: For now, unknown type is assumed to be a term
        CoreType::Unknown => builder.get_cir_term_type().base(),
    }
}

fn translate_primitive_ir_type<'a, B: OpBuilder>(
    builder: &CirBuilder<'a, B>,
    ty: &syntax_base::PrimitiveType,
) -> TypeBase {
    use firefly_syntax_base::PrimitiveType;
    match ty {
        PrimitiveType::I1 => builder.get_i1_type().base(),
        PrimitiveType::I8 => builder.get_i8_type().base(),
        PrimitiveType::I16 => builder.get_i16_type().base(),
        PrimitiveType::I32 => builder.get_i32_type().base(),
        PrimitiveType::I64 => builder.get_i64_type().base(),
        PrimitiveType::Isize => builder.get_index_type().base(),
        PrimitiveType::F64 => builder.get_f64_type().base(),
        PrimitiveType::Ptr(inner) => {
            let inner_ty = translate_primitive_ir_type(builder, &inner);
            builder.get_cir_ptr_type(inner_ty).base()
        }
        PrimitiveType::Struct(fields) => {
            let fields = fields
                .iter()
                .map(|t| translate_primitive_ir_type(builder, t))
                .collect::<Vec<_>>();
            builder.get_struct_type(fields.as_slice()).base()
        }
        PrimitiveType::Array(inner, arity) => {
            let inner_ty = translate_primitive_ir_type(builder, &inner);
            builder.get_array_type(inner_ty, *arity).base()
        }
    }
}

fn translate_term_ir_type<'a, B: OpBuilder>(
    module: &syntax_ssa::Module,
    options: &Options,
    builder: &CirBuilder<'a, B>,
    ty: &syntax_base::TermType,
) -> TypeBase {
    use firefly_syntax_base::TermType;

    let use_boxed_floats = !options.target.term_encoding().is_nanboxed();
    match ty {
        TermType::Any => builder.get_cir_term_type().base(),
        TermType::Bool => builder.get_cir_bool_type().base(),
        TermType::Integer => builder.get_cir_integer_type().base(),
        TermType::Float if use_boxed_floats => builder
            .get_cir_box_type(builder.get_cir_float_type())
            .base(),
        TermType::Float => builder.get_cir_float_type().base().base(),
        TermType::Number => builder.get_cir_number_type().base(),
        TermType::Atom => builder.get_cir_atom_type().base(),
        TermType::Bitstring | TermType::Binary => builder
            .get_cir_box_type(builder.get_cir_binary_type())
            .base(),
        TermType::Nil => builder.get_cir_nil_type().base(),
        TermType::Cons => builder.get_cir_box_type(builder.get_cir_cons_type()).base(),
        TermType::List(_) | TermType::MaybeImproperList => {
            // Since lists can be either nil or a boxed cons cell, we use the generic term type
            builder.get_cir_term_type().base()
        }
        TermType::Tuple(None) => builder.get_cir_box_type(builder.get_tuple_type(&[])).base(),
        TermType::Tuple(Some(ref elems)) => {
            let element_types = elems
                .iter()
                .map(|t| translate_term_ir_type(module, options, builder, t))
                .collect::<Vec<_>>();
            builder
                .get_cir_box_type(builder.get_tuple_type(element_types.as_slice()))
                .base()
        }
        TermType::Map => builder.get_cir_box_type(builder.get_cir_map_type()).base(),
        TermType::Reference => builder.get_cir_reference_type().base(),
        TermType::Port => builder.get_cir_port_type().base(),
        TermType::Pid => builder.get_cir_pid_type().base(),
        TermType::Fun(None) => builder.get_cir_term_type().base(),
        TermType::Fun(Some(ty)) => {
            let param_types = ty
                .params()
                .iter()
                .map(|t| translate_ir_type(module, options, builder, t))
                .collect::<Vec<_>>();
            let result_types = ty
                .results()
                .iter()
                .map(|t| translate_ir_type(module, options, builder, t))
                .collect::<Vec<_>>();
            builder
                .get_function_type(param_types.as_slice(), result_types.as_slice())
                .base()
        }
    }
}

/// Converts a Signature to its corresponding MLIR function type
fn signature_to_fn_type<'a, B: OpBuilder>(
    module: &syntax_ssa::Module,
    options: &Options,
    builder: &CirBuilder<'a, B>,
    sig: &Signature,
) -> FunctionType {
    let param_types = sig
        .params()
        .iter()
        .map(|t| translate_ir_type(module, options, builder, t))
        .collect::<Vec<_>>();
    let result_types = sig
        .results()
        .iter()
        .map(|t| translate_ir_type(module, options, builder, t))
        .collect::<Vec<_>>();
    builder.get_function_type(param_types.as_slice(), result_types.as_slice())
}
