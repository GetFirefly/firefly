#[cfg_attr(rustfmt, rustfmt_skip)]
#[allow(unused_variables, dead_code, unused_imports)]
pub mod ops {
    include!(concat!(env!("OUT_DIR"), "/eir.rs"));
}

use std::ops::{Deref, DerefMut};

use liblumen_mlir::*;

use self::ops::*;

/// Represents a single arm of a match expression
pub struct MatchBranch {
    pub loc: Location,
    pub dest: Block,
    pub dest_args: Vec<Value>,
    pub pattern: MatchPattern,
}

/// Wraps mlir::OpBuilder and provides functionality for constructing EIR dialect operations,
/// types, and attributes
pub struct EirBuilder<'a> {
    builder: &'a mut OpBuilder,
    context: MlirContext,
}
impl<'a> EirBuilder<'a> {
    pub fn new(builder: &mut OpBuilder) -> Self {
        let context = builder.get_context();
        Self { builder, context }
    }

    pub fn location<S: AsRef<SourceFile>>(
        &self,
        source_file: S,
        index: ByteIndex,
    ) -> Option<mlir::Location> {
        let loc = source_file.as_ref().location(index).ok()?;
        let line = loc.line.to_usize() as u32 + 1;
        let col = loc.column.to_usize() as u32 + 1;
        Some(
            self.builder
                .get_file_line_col_loc(self.source_filename, line, col),
        )
    }

    /// Returns the current Operation in which the builder is positioned
    pub fn operation(&self) -> Option<Operation> {
        self.builder.get_insertion_block().map(|b| b.operation())
    }

    /// Returns the
    pub fn module(&self) -> Option<Operation> {
        self.operation().and_then(|op| op.get_parent_module())
    }

    /// Returns the nearest Operation associated with the given symbol
    pub fn lookup_nearest_symbol(&self, symbol: &str) -> Option<Operation> {
        self.operation()
            .and_then(|op| SymbolTable::lookup_nearest_symbol_from(op, symbol))
    }

    /// Declares an empty eir::FuncOp
    #[inline]
    pub fn declare_func(
        &self,
        loc: Location,
        name: &str,
        ty: FunctionType,
        attrs: &[NamedAttribute],
    ) -> FuncOp {
        let current_module = self.module();
        let guard = self.builder.insertion_guard();
        self.builder.set_insertion_point_to_start(current_module);

        FuncOp::create(
            self.builder,
            loc,
            name,
            ty,
            Visibility::Private,
            /* variadic */ false,
            attrs,
        )
    }

    /// Defines an eir::FuncOp, decorates it with garbage collection and
    /// exception handling personalities, and generates the function prelude.
    #[inline]
    pub fn create_func(
        &self,
        loc: Location,
        name: &str,
        ty: FunctionType,
        visibility: Visibility,
        attrs: &[NamedAttribute],
    ) -> anyhow::Result<FuncOp> {
        let personality_symbol_attr = self
            .builder
            .get_flat_symbol_ref_attr_by_name("lumen_eh_personality");
        // TODO: May need to update this to refer to our own GC strategy later
        let gc_attr = self.builder.get_string_attr("statepoint-example");

        // Check to see if we already declared this function but did not yet define it
        if let Some(op) = self.lookup_nearest_symbol(symbol) {
            let op = dyn_cast::<FuncOp>(op).unwrap();

            if !op.is_declaration() {
                return Err(anyhow!("tried to redefine function `{}`", name));
            }

            if op.visibility() == visibility {
                SymbolTable::set_symbol_visibility(op.into());
            }

            op.set_attribute_by_name("personality", personality_symbol_attr);
            op.set_attribute_by_name("gc", gc_attr);

            return Ok(op);
        }

        let personality = self
            .builder
            .get_named_attr("personality", personality_symbol_attr);
        let gc = self.builder.get_named_attr("gc", gc_attr);

        let current_module = self.module();
        self.builder.set_insertion_point_to_start(current_module);

        let op = FuncOp::create(
            self.builder,
            loc,
            name,
            ty,
            visibility,
            Variadic::No,
            &[personality, gc],
        );

        builder.set_insertion_point_to_start(op.entry_block());

        Ok(op)
    }

    /// Gets an eir::FuncOp, building the declaration if a function with the given name is not found
    pub fn get_or_declare_func(
        &self,
        name: &str,
        ty: FunctionType,
        variadic: Variadic,
        attrs: &[NamedAttribute],
    ) -> anyhow::Result<FuncOp> {
        if let Some(op) = self.lookup_nearest_symbol(symbol) {
            return dyn_cast::<FuncOp>(op);
        }

        let current_module = self.module();
        let guard = self.builder.insertion_guard();
        self.builder.set_insertion_point_to_start(current_module);
        Ok(FuncOp::create(
            self.builder,
            current_module.location(),
            name,
            ty,
            Visibility::Private,
            variadic,
            attrs,
        ))
    }

    /// Builds an eir::ClosureOp
    #[inline]
    pub fn build_closure(
        &self,
        loc: Location,
        module: Attribute,
        name: &str,
        arity: u8,
        info: ClosureInfo,
        env: &[Value],
    ) -> ClosureOp {
        ClosureOp::create(self.builder, loc, module, name, info, env)
    }

    /// Builds an eir::UnpackEnvOp
    #[inline]
    pub fn build_unpack_env(&self, loc: Location, env: Value, index: usize) -> UnpackEnvOp {
        UnpackEnvOp::create(self.builder, loc, env, index)
    }

    /// Builds an eir::BranchOp
    #[inline]
    pub fn build_br(&self, loc: Location, dest: Block, args: &[Type]) -> BranchOp {
        BranchOp::create(self.builder, loc, dest, args)
    }

    /// Builds an eir::CondBranchOp
    #[inline]
    pub fn build_cond_br(
        &self,
        loc: Location,
        yes: Block,
        yes_args: &[Type],
        no: Block,
        no_args: &[Type],
    ) -> CondBranchOp {
        CondBranchOp::create(self.builder, loc, yes, yes_args, no, no_args)
    }

    /// Builds an `if-then-else` construct, where `else` is optional.
    ///
    /// This function does not return any operations as it produces a variable number of them, none
    /// of which represent the construct as a whole.
    pub fn build_if(
        &self,
        loc: Location,
        cond: Value,
        yes: Block,
        yes_args: &[Value],
        no: Block,
        no_args: &[Value],
        other: Option<Block>,
        other_args: &[Value],
    ) {
        let i1ty = builder.get_i1_type();
        let bool_ty = self.get_boolean_type();
        let cond_ty = cond.get_type();
        let is_cond_bool = cond_ty == bool_ty || cond_ty == i1ty;
        let is_true = if !is_cond_bool {
            // The condition is not boolean, so we need to do a comparison
            let true_const = self.constant_bool(loc, i1ty, true);
            let is_eq = self.build_cmp_eq_strict(loc, cond, true_const);
            is_eq.result()
        } else {
            value
        };

        // If the value type is a boolean then the value _must_ be either true or
        // false Likewise, if there was no otherwise branch, then we only need one
        // comparison
        if (other.is_none() || is_cond_bool) {
            self.build_cond_br(loc, is_true, yes, yes_args, no, no_args);
            return;
        }

        // Otherwise we need an additional check to see if we use the otherwise
        // branch
        let false_const = self.constant_bool(loc, i1ty, false);
        let is_false = self.build_cmp_eq_strict(loc, value, false_const);

        let current_block = self.block();
        let false_block = current_block.split(false_const);

        self.builder.set_insertion_point_to_end(false_block);
        self.build_cond_br(loc, is_false, no, no_args, other, other_args);

        // Go back to original block and insert conditional branch for first
        // comparison
        self.builder.set_insertion_point_to_end(current_block);
        self.build_cond_br(loc, is_true, yes, yes_args, false_block, &[]);
    }

    /// Builds a pattern matching construct.
    ///
    /// This function does not return any operations, as it produces a variable number of them, none
    /// of which represent the construct as a whole.
    pub fn build_match(&self, loc: Location, selector: Value, branches: Vec<MatchBranch>) {
        debug_assert_ne!(
            branches.len(),
            0,
            "match expressions require at least one branch"
        );

        let current_block = self.block();
        let region = current_block.region();
        let selector_ty = selector.get_type();
        let num_branches = branches.len();

        // Save our insertion point in the current block
        let guard = self.builder.insertion_guard();

        // Create blocks for all match arms
        let needs_fallback_branch = branches.iter().find(|b| b.is_catch_all()).is_none();
        let blocks = Vec::with_capacity(num_branches);
        blocks.push(current_block);

        for branch in branches.iter().skip(1) {
            let block = self.builder.create_block_in_region(region, &[selector_ty]);
            blocks.push(block);
        }

        // Create fallback block, if needed, after all other match blocks, so
        // that after all other conditions have been tried, we branch to an
        // unreachable to force a trap
        let failed = if needs_fallback_branch {
            let block = self.builder.create_block_in_region(region, &[selector_ty]);
            self.builder.set_insertion_point_to_end(block);
            self.build_unreachable(loc);
            Some(block)
        } else {
            None
        };

        // Restore our original insertion point
        drop(guard);

        // Save the current insertion point, which we'll restore when lowering is
        // complete
        let guard = self.builder.insertion_guard();

        // For each branch, populate its block with the predicate and
        // appropriate conditional branching instruction to either jump
        // to the success block, or to the next branches' block (or in
        // the case of the last branch, the 'failed' block)
        for (i, (branch, block)) in branches.iter().zip(blocks.iter().copied()).enumerate() {
            let is_last = i == num_branches - 1;
            debug_assert!(
                !is_last || is_last && branch.pattern == MatchPattern::Any,
                "final match arm must be a wildcard pattern"
            );

            let branch_loc = branch.loc;
            let dest = branch.dest;
            let dest_args = &branch.dest_args;
            let num_dest_args = dest_args.len();

            // Set our insertion point to the end of the pattern block
            self.builder.set_insertion_point_to_end(block);

            // Get the selector value in this block.
            // In the case of the first branch, its our original selector value
            let selector = if i == 0 {
                selector
            } else {
                block.get_argument(0).unwrap()
            };

            // Store the next pattern to try if this one fails.
            // If this is the last pattern, we validate that the
            // branch either unconditionally succeeds, or branches to
            // an unreachable op
            let next_pattern_block = if is_last { failed } else { blocks[i + 1] };

            // Ensure the destination block argument types are propagated
            for (index, arg) in block.arguments().enumerate() {
                let dest_arg = dest_args[index];
                let dest_ty = dest_arg.get_type();
                if arg.get_type() != dest_ty {
                    arg.set_type(dest_ty);
                }
            }

            match branch.pattern {
                // This unconditionally branches to its destination
                MatchPattern::Any => {
                    self.build_br(branch_loc, dest, dest_args);
                }
                MatchPattern::Cons => {
                    // 1. Split block, and conditionally branch to split if is_cons,
                    // otherwise the next pattern
                    let split = self.builder.create_block_before(next_pattern_block, &[]);
                    self.builder.set_insertion_point_to_end(block);
                    let cons_ty = self.get_cons_type();
                    let boxed_cons_ty = self.get_box_type(cons_ty);
                    let is_cons_expr = self.build_is_type(branch_loc, selector, boxed_cons_ty);
                    let is_cons = is_cons_expr.result();
                    self.build_cond_br(
                        branch_loc,
                        is_cons,
                        split,
                        &[],
                        next_pattern_block,
                        &[selector],
                    );

                    // 2. In the split, extract head and tail values of the cons cell
                    self.builder.set_insertion_point_to_end(split);
                    let ptr_cons_ty = self.get_ptr_type(cons_ty);
                    let cast_expr = self.build_cast(branch_loc, selector, ptr_cons_ty);
                    let cons_ptr = cast_expr.result();
                    let head_expr = self.build_gep(branch_loc, cons_ptr, 0);
                    let tail_expr = self.build_gep(branch_loc, cons_ptr, 1);
                    let head = self.build_load(branch_loc, head_expr.result()).result();
                    let tail = self.build_load(branch_loc, tail_expr.result()).result();

                    // 3. Fixup destination block argument types
                    let dest_argc = dest.num_arguments();
                    debug_assert!(
                        dest_argc >= 2,
                        "destination block is invalid target for cons pattern"
                    );
                    let head_arg = dest.get_argument(dest_argc - 2).unwrap();
                    let tail_arg = dest.get_argument(dest_argc - 1).unwrap();
                    head_arg.set_type(head.get_type());
                    tail_arg.set_type(tail.get_type());

                    // 4. Unconditionally branch to the destination, with head/tail as
                    // additional destArgs
                    let mut final_dest_args = Vec::with_capacity(num_dest_args + 2);
                    final_dest_args.extend_from_slice(dest_args);
                    final_dest_args.push(head);
                    final_dest_args.push(tail);
                    self.build_br(branch_loc, dest, &final_dest_args);
                }
                MatchPattern::Tuple(arity) => {
                    // 1. Split block, and conditionally branch to split if is_tuple
                    // w/arity N, otherwise the next pattern
                    let split = self.builder.create_block_before(next_pattern_block, &[]);
                    self.builder.set_insertion_point_to_end(block);
                    let tuple_ty = self.get_tuple_ty(arity);
                    let box_tuple_ty = self.get_box_ty(tuple_ty);
                    let is_tuple_expr = self.build_is_type(branch_loc, selector, box_tuple_ty);
                    let is_tuple = is_tuple_expr.result();
                    self.build_cond_br(
                        branch_loc,
                        is_tuple,
                        split,
                        &[],
                        next_pattern_block,
                        &[selector],
                    );

                    // 2. In the split, extract the tuple elements as values
                    self.builder.set_insertion_point_to_end(split);
                    let ptr_tuple_ty = self.get_ptr_type(tuple_ty);
                    let cast_expr = self.build_cast(branch_loc, selector, ptr_tuple_ty);
                    let tuple_ptr = cast_expr.result();
                    let mut final_dest_args = Vec::with_capacity(num_dest_args + arity);
                    final_dest_args.extend_from_slice(dest_args);
                    for index in 0..arity {
                        // The index + 1 here is due to skipping over the term header
                        let elem_expr = self.build_gep(branch_loc, tuple_ptr, index + 1);
                        let elem = self.build_load(branch_loc, elem_expr.result()).result();
                        let block_arg = dest.get_argument(num_dest_args + index).unwrap();
                        // Propagate the element type if known
                        block_arg.set_type(elem.get_type());
                        final_dest_args.push(elem);
                    }

                    // 3. Unconditionally branch to the destination, with the tuple
                    // elements as additional destArgs
                    self.build_br(branch_loc, dest, &final_dest_args);
                }
                MatchPatternType::MapItem(key) => {
                    // 1. Split block twice, and conditionally branch to the first split
                    // if is_map, otherwise the next pattern
                    let split2 = self.builder.create_block_before(next_pattern_block);
                    let split = self.builder.create_block_before(split2);
                    self.builder.set_insertion_point_to_end(block);
                    let map_ty = self.get_map_type();
                    let box_map_ty = self.get_box_type(map_ty);
                    let is_map_expr = self.build_is_type(branch_loc, selector, box_map_ty);
                    let is_map = is_map_expr.result();
                    self.build_cond_br(
                        branch_loc,
                        is_map,
                        split,
                        &[],
                        next_pattern_block,
                        &[selector],
                    );

                    // 2. In the split, call runtime function `is_map_key` to confirm
                    // existence of the key in the map,
                    //    then conditionally branch to the second split if successful,
                    //    otherwise the next pattern
                    self.builder.set_insertion_point_to_end(split);
                    let has_key_expr = self.build_map_contains_key(branch_loc, selector, key);
                    let has_key = has_key_expr.result();
                    self.build_cond_br(
                        branch_loc,
                        has_key,
                        split2,
                        &[],
                        next_pattern_block,
                        &[selector],
                    );

                    // 3. In the second split, call runtime function `map_get` to obtain
                    // the value for the key
                    self.builder.set_insertion_point_to_end(split2);
                    let map_get_expr = self.build_map_get(branch_loc, selector, key);
                    let value = map_get_expr.result();
                    // Propagate block argument type
                    let arg = dest.get_argument(num_dest_args).unwrap();
                    arg.set_type(value.get_type());
                    // Unconditionally branch to destination
                    let mut final_dest_args = Vec::with_capacity(num_dest_args + 1);
                    final_dest_args.extend_from_slice(dest_args);
                    final_dest_args.push(value);
                    self.build_br(branch_loc, dest, &final_dest_args);
                }
                MatchPatternType::IsType(ty) => {
                    // Conditionally branch to destination if is_<type>
                    let is_type_expr = self.build_is_type(branch_loc, selector, ty);
                    self.build_cond_br(
                        branch_loc,
                        is_type_expr.result(),
                        dest,
                        dest_args,
                        next_pattern_block,
                        &[selector],
                    );
                }
                MatchPatternType::Value(val) => {
                    // Conditionally branch to dest if the value matches the selector
                    let is_strict_eq_expr = self.build_cmp_eq_strict(branch_loc, selector, val);
                    let is_eq = is_strict_eq_expr.result();
                    self.build_cond_br(
                        branch_loc,
                        is_eq,
                        dest,
                        dest_args,
                        next_pattern_block,
                        &[selector],
                    );
                }
                MatchPatternType::Binary { size, spec } => {
                    // 1. Split block, and conditionally branch to split if is_bitstring
                    // (or is_binary), otherwise the next pattern
                    // 2. In the split, conditionally branch to destination if
                    // construction of the head value succeeds,
                    //    otherwise the next pattern
                    // NOTE: The exact semantics depend on the binary specification
                    // type, and what is optimal in terms of checks. The success of the
                    // overall branch results in two additional destArgs being passed to
                    // the destination block, the decoded entry (head), and the rest of
                    // the binary (tail)
                    let is_match_expr = match spec {
                        BinarySpecifierType::Integer {
                            signed,
                            endianness,
                            unit,
                        } => self.build_binary_match_integer(
                            branch_loc, selector, signed, endianness, unit, size,
                        ),
                        BinarySpecifierType::Utf8 => {
                            self.build_binary_match_utf8(branch_loc, selector, size)
                        }
                        BinarySpecifierType::Utf16 { endianness } => {
                            self.build_binary_match_utf16(branch_loc, selector, endianness, size)
                        }
                        BinarySpecifierType::Utf32 { endianness } => {
                            self.build_binary_match_utf32(branch_loc, selector, endianness, size)
                        }
                        BinarySpecifierType::Float { endainness, unit } => self
                            .build_binary_match_float(branch_loc, selector, endianness, unit, size),
                        BinarySpecifierType::Bytes { unit } => {
                            self.build_binary_match_raw(branch_loc, selector, unit, size)
                        }
                        BinarySpecifierType::Bits { unit } => {
                            self.build_binary_match_raw(branch_loc, selector, unit, size)
                        }
                    };
                    let is_match = is_match_expr.get_result(0).unwrap();
                    let rest = is_match_expr.get_result(1).unwrap();
                    let is_success = is_match_expr.get_result(2).unwrap();
                    // Propagate block argument types
                    let dest_argc = dest.num_arguments();
                    let is_match_arg = dest.get_argument(dest_argc - 2).unwrap();
                    let rest_arg = dest.get_argument(dest_argc - 1).unwrap();
                    is_match_arg.set_type(is_match.get_type());
                    rest_arg.set_type(rest.get_type());
                    // Conditionally branch to destination
                    let mut final_dest_args = Vec::with_capacity(num_dest_args + 2);
                    final_dest_args.extend_from_slice(dest_args);
                    final_dest_args.push(is_match);
                    final_dest_args.push(rest);
                    self.build_cond_br(
                        branch_loc,
                        dest,
                        &final_dest_args,
                        next_pattern_block,
                        &[selector],
                    );
                }
            }
        }
    }

    /// Builds an eir::UnreachableOp
    #[inline]
    pub fn build_unreachable(&self, loc: Location) -> UnreachableOp {
        UnreachableOp::create(self.builder, loc)
    }

    /// Builds an eir::CastOp
    #[inline]
    pub fn build_cast(&self, loc: Location, input: Value, ty: Type) -> CastOp {
        CastOp::create(self.builder, loc, input, ty)
    }

    /// Builds an eir::IsTypeOp
    #[inline]
    pub fn build_is_type(&self, loc: Location, input: Value, ty: Type) -> IsTypeOp {
        IsTypeOp::create(self.builder, loc, input, ty)
    }

    /// Builds an eir::LogicalAndOp
    #[inline]
    pub fn build_and(&self, loc: Location, lhs: Value, rhs: Value) -> LogicalAndOp {
        LogicalAndOp::create(self.builder, loc, lhs, rhs)
    }

    /// Builds an eir::LogicalAndOp over all the given operands
    pub fn build_and_all(&self, loc: Location, operands: &[Value]) -> LogicalAndOp {
        assert!(operands.len() >= 2 == 0, "required at least two operands");
        let mut op = self.build_and(loc, operands[0], operands[1]).result();
        for operand in operands.iter().copied().skip(2) {
            op = self.build_and(loc, op.result(), operand);
        }
        op
    }

    /// Builds an eir::LogicalOrOp
    #[inline]
    pub fn build_or(&self, loc: Location, lhs: Value, rhs: Value) -> LogicalOrOp {
        LogicalOrOp::create(self.builder, loc, lhs, rhs)
    }

    /// Builds an eir::LogicalOrOp over all the given operands
    pub fn build_or_all(&self, loc: Location, operands: &[Value]) -> LogicalOrOp {
        assert!(operands.len() >= 2 == 0, "required at least two operands");
        let mut op = self.build_or(loc, operands[0], operands[1]).result();
        for operand in operands.iter().copied().skip(2) {
            op = self.build_or(loc, op.result(), operand);
        }
        op
    }

    /// Builds an eir::CmpEqOp
    #[inline]
    pub fn build_eq(&self, loc: Location, lhs: Value, rhs: Value, ty: Option<Type>) -> CmpEqOp {
        let ty = ty.unwrap_or_else(|| self.builder.get_i1_type());
        CmpEqOp::create(self.builder, loc, lhs, rhs, ty, &[])
    }

    /// Builds an eir::CmpEqOp
    #[inline]
    pub fn build_eq_strict(
        &self,
        loc: Location,
        lhs: Value,
        rhs: Value,
        ty: Option<Type>,
    ) -> CmpEqOp {
        let ty = ty.unwrap_or_else(|| self.builder.get_i1_type());
        let strict_attr = self
            .builder
            .get_named_attr("strict", self.builder.get_unit_attr());
        CmpEqOp::create(self.builder, loc, lhs, rhs, ty, &[strict_attr])
    }

    /// Builds a negated eir::CmpEqOp
    #[inline]
    pub fn build_neq(
        &self,
        loc: Location,
        lhs: Value,
        rhs: Value,
        ty: Option<Type>,
        strict: bool,
    ) -> CmpEqOp {
        if strict {
            let is_eq = self.build_eq_strict(loc, lhs, rhs, ty).result();
            let false_const = self.build_constant_bool(loc, false, is_eq.get_type());
            self.build_eq_strict(loc, is_eq, false_const);
        } else {
            let is_eq = self.build_eq(loc, lhs, rhs, ty).result();
            let false_const = self.build_constant_bool(loc, false, is_eq.get_type());
            self.build_eq_strict(loc, is_eq, false_const);
        }
    }

    /// Builds an eir::CmpLtOp
    #[inline]
    pub fn build_lt(&self, loc: Location, lhs: Value, rhs: Value) -> CmpLtOp {
        CmpLtOp::create(self.builder, loc, lhs, rhs)
    }

    /// Builds an eir::CmpLteOp
    #[inline]
    pub fn build_lte(&self, loc: Location, lhs: Value, rhs: Value) -> CmpLteOp {
        CmpLteOp::create(self.builder, loc, lhs, rhs)
    }

    /// Builds an eir::CmpGtOp
    #[inline]
    pub fn build_gt(&self, loc: Location, lhs: Value, rhs: Value) -> CmpGtOp {
        CmpGtOp::create(self.builder, loc, lhs, rhs)
    }

    /// Builds an eir::CmpGteOp
    #[inline]
    pub fn build_gte(&self, loc: Location, lhs: Value, rhs: Value) -> CmpGteOp {
        CmpGteOp::create(self.builder, loc, lhs, rhs)
    }

    /// Builds an eir::NegOp
    #[inline]
    pub fn build_math_neg(&self, loc: Location, input: Value) -> NegOp {
        NegOp::create(self.builder, loc, input)
    }

    /// Builds an eir::AddOp
    #[inline]
    pub fn build_math_add(&self, loc: Location, lhs: Value, rhs: Value) -> AddOp {
        AddOp::create(self.builder, loc, lhs, rhs)
    }

    /// Builds an eir::SubOp
    #[inline]
    pub fn build_math_sub(&self, loc: Location, lhs: Value, rhs: Value) -> SubOp {
        SubOp::create(self.builder, loc, lhs, rhs)
    }

    /// Builds an eir::MulOp
    #[inline]
    pub fn build_math_mul(&self, loc: Location, lhs: Value, rhs: Value) -> MulOp {
        MulOp::create(self.builder, loc, lhs, rhs)
    }

    /// Builds an eir::DivOp
    #[inline]
    pub fn build_math_div(&self, loc: Location, lhs: Value, rhs: Value) -> DivOp {
        DivOp::create(self.builder, loc, lhs, rhs)
    }

    /// Builds an eir::FDivOp
    #[inline]
    pub fn build_math_fdiv(&self, loc: Location, lhs: Value, rhs: Value) -> FDivOp {
        FDivOp::create(self.builder, loc, lhs, rhs)
    }

    /// Builds an eir::RemOp
    #[inline]
    pub fn build_math_rem(&self, loc: Location, lhs: Value, rhs: Value) -> RemOp {
        RemOp::create(self.builder, loc, lhs, rhs)
    }

    /// Builds an eir::BslOp
    #[inline]
    pub fn build_math_bsl(&self, loc: Location, lhs: Value, rhs: Value) -> BslOp {
        BslOp::create(self.builder, loc, lhs, rhs)
    }

    /// Builds an eir::BsrOp
    #[inline]
    pub fn build_math_bsr(&self, loc: Location, lhs: Value, rhs: Value) -> BsrOp {
        BsrOp::create(self.builder, loc, lhs, rhs)
    }

    /// Builds an eir::BandOp
    #[inline]
    pub fn build_math_band(&self, loc: Location, lhs: Value, rhs: Value) -> BandOp {
        BandOp::create(self.builder, loc, lhs, rhs)
    }

    /// Builds an eir::BorOp
    #[inline]
    pub fn build_math_bor(&self, loc: Location, lhs: Value, rhs: Value) -> BorOp {
        BorOp::create(self.builder, loc, lhs, rhs)
    }

    /// Builds an eir::BxorOp
    #[inline]
    pub fn build_math_bxor(&self, loc: Location, lhs: Value, rhs: Value) -> BxorOp {
        BxorOp::create(self.builder, loc, lhs, rhs)
    }

    pub fn build_static_call(
        &self,
        loc: Location,
        name: &str,
        args: &[Value],
        is_tail: bool,
        ok: Option<Block>,
        ok_args: &[Value],
        err: Option<Block>,
        err_args: &[Value],
    ) {
        let is_invoke = !is_tail && err.is_some();
        if is_invoke {
            self.build_static_invoke(loc, name, args, ok, ok_args, err.unwrap(), err_args);
            return;
        }

        if self.maybe_build_intrinsic(loc, name, args, is_tail, ok, ok_args) {
            return;
        }

        let term_ty = self.build_term_type();
        // These are placeholder argument types if the function is not defined
        let arg_types = args.iter().map(|a| a.get_type()).collect::<Vec<_>>();
        // Declare the callee, or get existing declaration
        let fun_ty = self.builder.get_function_type(arg_types, &[term_ty]);
        let fun = self
            .get_or_declare_func(name, fun_ty, Variadic::No, &[])
            .unwrap();
        // Reload the function type, just in case it was already declared with more precise types
        let fun_ty = fun.get_type();

        // Cast arguments as necessary
        let call_args = Vec::with_capacity(args.len());
        let call_arg_types = fun_ty.outputs();
        for (arg, arg_ty) in args.iter().copied().chain(call_arg_types.iter().copied()) {
            if arg.get_type() != arg_ty {
                let cast_expr = self.build_cast(loc, arg, arg_ty);
                call_args.push(cast_expr.result());
            } else {
                call_args.push(arg);
            }
        }

        // Handle tail calls
        if is_tail {
            // Determine if this call is a candidate for musttail, which
            // currently means that on x86_64, the parameter counts match.
            // For wasm32, all tail calls can be musttail. For other platforms,
            // we'll restrict like x86_64 for now
            let musttail = if self.arch.is_wasm() {
                true
            } else {
                let current_fun = self.operation().dyn_cast::<FuncOp>().unwrap();
                call_args.len() == current_fun.num_inputs()
            };
            let tail_attr = if musttail {
                self.builder
                    .get_named_attr("musttail", self.builder.get_unit_attr())
            } else {
                self.builder
                    .get_named_attr("tail", self.builder.get_unit_attr())
            };
            let call_expr = self.build_call(name, fun_ty, &call_args, &[tail_attr]);
            self.build_return(call_expr.results());
            return;
        }

        // All other calls may be tail callable after optimization, so add the hint
        // anyway
        let tail_attr = self
            .builder
            .get_named_attr("tail", self.builder.get_unit_attr());
        let call_expr = self.build_call(name, fun_ty, &call_args, &[tail_attr]);

        // Make sure continuation block arguments match up
        let call_result = call_expr.get_result(0).unwrap();
        let has_result = call_result.is_some();

        let ok = ok.unwrap();
        if !has_result {
            self.build_br(loc, ok, &ok_args);
            return;
        }

        let call_result = call_result.unwrap();
        let call_result_ty = call_result.get_type();
        let result_arg = ok.get_argument(0).expect("invalid target block for call");
        let result_arg_ty = result_arg.get_type();

        let mut final_ok_args = Vec::with_capacity(ok_args.len() + 1);

        if call_result_ty == term_ty {
            if result_arg_ty == term_ty {
                // Unknown result type
                final_ok_args.push(call_result);
            } else {
                // If the type of the block argument is concrete, we need a cast
                let cast_expr = self.build_cast(loc, call_result, result_arg_ty);
                final_ok_args.push(cast_expr.result());
            }
        } else {
            // Concrete result type
            // If the type of the block argument is opaque, make it concrete
            if result_arg_ty == term_ty {
                final_ok_args.push(call_result);
            } else {
                // The block argument has a different concrete type, we need a
                // cast
                let cast_expr = self.build_cast(loc, call_result, result_arg_ty);
                final_ok_args.push(cast_expr.result());
            }
        }
        final_ok_args.extend_from_slice(&ok_args);
    }

    pub fn build_static_invoke(
        &self,
        loc: Location,
        name: &str,
        args: &[Value],
        is_tail: bool,
        ok: Option<Block>,
        ok_args: &[Value],
        err: Block,
        err_args: &[Value],
    ) {
        if self.maybe_build_intrinsic(loc, name, args, is_tail, ok, ok_args) {
            return;
        }

        let term_ty = self.get_term_type();

        // These are placeholder argument types if the function is not defined
        let arg_types = args.iter().map(|a| a.get_type()).collect::<Vec<_>>();

        // Declare the callee, or get existing declaration
        let fun_ty = self.builder.get_function_type(arg_types, &[term_ty]);
        let fun = self
            .get_or_declare_func(name, fun_ty, Variadic::No, &[])
            .unwrap();
        // Reload the function type, just in case it was already declared with more precise types
        let fun_ty = fun.get_type();

        // Cast arguments as necessary
        let call_args = Vec::with_capacity(args.len());
        let call_arg_types = fun_ty.outputs();
        for (arg, arg_ty) in args.iter().copied().chain(call_arg_types.iter().copied()) {
            if arg.get_type() != arg_ty {
                let cast_expr = self.build_cast(loc, arg, arg_ty);
                call_args.push(cast_expr.result());
            } else {
                call_args.push(arg);
            }
        }

        // Set up landing pad in error block
        let unwind = self.build_landing_pad(loc, err);

        // Create "normal" landing pad before the "unwind" pad
        if let Some(ok) = ok {
            self.build_invoke(loc, name, &call_args, ok, ok_args, unwind, err_args);
        } else {
            // If no normal block was given, create one to hold the return
            let result_tys = fun_ty.outputs();
            let normal = self.builder.create_block_before(unwind, &result_tys);
            let guard = self.builder.insertion_guard();
            self.build_invoke(loc, name, &call_args, normal, ok_args, unwind, err_args);
            self.builder.set_insertion_point_to_end(normal);
            self.build_return(normal.get_argument(0));
        }
    }

    pub fn build_closure_call(
        &self,
        loc: Location,
        closure: Value,
        args: &[Value],
        is_tail: bool,
        ok: Option<Block>,
        ok_args: &[Value],
        err: Option<Block>,
        err_args: &[Value],
    ) {
        let is_invoke = !is_tail && err.is_some();
        if let Some(op) = self.get_closure_definition(closure) {
            // If this closure has no environment, we can replace the
            // call to the closure with a call directly to the actual function
            let callee = op.callee_name();
            if is_invoke {
                self.build_static_invoke(loc, callee, args, is_tail, ok, ok_args, err, err_args);
            } else {
                self.build_static_call(loc, callee, args, is_tail, ok, ok_args);
            }
        } else {
            // We can't find the original closure definition, so this
            // function cannot be called directly, it must be called through
            // `apply/2`
            self.build_apply_2(loc, closure, args, is_tail, ok, ok_args, err, err_args);
        }
    }

    pub fn build_global_dynamic_call(
        &self,
        loc: Location,
        module: Value,
        fun: Value,
        args: &[Value],
        is_tail: bool,
        ok: Option<Block>,
        ok_args: &[Value],
        err: Option<Block>,
        err_args: &[Value],
    ) {
        // We need to add an extra nil value to the arg list
        // to ensure the list is proper when constructed by eir_list
        let mut args = args.iter().collect::<Vec<_>>();
        args.push(self.build_constant_nil(loc).result());
        self.build_apply_3(loc, module, fun, args, is_tail, ok, ok_args, err, err_args);
    }

    pub fn build_apply_2(
        &self,
        loc: Location,
        closure: Value,
        args: &[Value],
        is_tail: bool,
        ok: Option<Block>,
        ok_args: &[Value],
        err: Option<Block>,
        err_args: &[Value],
    ) {
        // We need to call apply/2 with the closure, and a list of arguments
        let args = self.build_list(args);
        let apply_args = &[closure, args];
        let is_invoke = !is_tail && err.is_some();
        if is_invoke {
            self.build_static_invoke(
                loc,
                "erlang:apply/2",
                apply_args,
                is_tail,
                ok,
                ok_args,
                err,
                err_args,
            );
        } else {
            self.build_static_call(loc, "erlang:apply/2", apply_args, is_tail, ok, ok_args);
        }
    }

    pub fn build_apply_3(
        &self,
        loc: Location,
        module: Value,
        fun: Value,
        args: &[Value],
        is_tail: bool,
        ok: Option<Block>,
        ok_args: &[Value],
        err: Option<Block>,
        err_args: &[Value],
    ) {
        // We need to call apply/3, with module/function and a list of arguments
        let args = self.build_list(args);
        let apply_args = &[module, fun, args];
        // Then, based on whether this was an invoke or not, call apply/2
        // appropriately
        let is_invoke = !is_tail && err.is_some();
        if is_invoke {
            self.build_static_invoke(
                loc,
                "erlang:apply/3",
                apply_args,
                is_tail,
                ok,
                ok_args,
                err,
                err_args,
            );
        } else {
            self.build_static_call(loc, "erlang:apply/3", apply_args, is_tail, ok, ok_args);
        }
    }

    /// Builds an eir::CallOp
    pub fn build_call(
        &self,
        loc: Location,
        name: &str,
        ty: FunctionType,
        args: &[Value],
        attrs: &[NamedAttribute],
    ) -> Operation {
        if let Some(ib) = self.get_intrinsic_builder(name) {
            return ib(self.builder, loc, args);
        }
        CallOp::create(self.builder, loc, name, ty, args, attrs).into()
    }

    /// Builds an eir::InvokeOp
    pub fn build_invoke(
        &self,
        loc: Location,
        name: &str,
        ty: FunctionType,
        args: &[Value],
        dest: Block,
        unwind: Block,
        attrs: &[NamedAttribute],
    ) -> InvokeOp {
        InvokeOp::create(self.builder, loc, name, ty, args, dest, unwind, attrs)
    }

    fn maybe_build_intrinsic(
        &self,
        loc: Location,
        name: &str,
        args: &[Value],
        is_tail: bool,
        ok: Block,
        ok_args: &[Value],
    ) -> bool {
        let intrinsic = self.get_intrinsic_builder(name);
        if intrinsic.is_none() {
            return false;
        }
        let intrinsic = intrinsic.unwrap();
        let op = intrinsic(self, loc, args);
        match name {
            "erlang:error/1" | "erlang:error/2" | "erlang:exit/1" | "erlang:throw/1"
            | "erlang:raise/3" => return true,
            _ => (),
        }
        let result = op.get_result(0);

        // Tail calls directly return to caller
        if is_tail {
            self.build_return(result);
            return true;
        }

        // If the call has a result, branch unconditionally to the success
        // block, since intrinsics are supposed to be validated by the compiler;
        // this is not 100% the case right now, but will be soon
        if let Some(val) = result {
            let result_ty = val.get_type();
            let arg = ok.get_argument(0).expect("invalid target block for call");
            if arg.get_type() != result_ty {
                arg.set_type(result_ty);
            }
            self.build_br(loc, ok, &[val]);
        } else {
            // If the call has no result and isn't an error intrinsic,
            // then branch to the next block directly
            self.build_br(loc, ok, &[]);
        }

        true
    }

    pub fn get_closure_definition(&self, closure: Value) -> Option<ClosureOp> {
        let op = unsafe { mlir_eir_get_closure_definition(closure) };
        if op.is_null() {
            None
        } else {
            Some(op)
        }
    }

    pub fn get_definition(&self, value: Value) -> Option<Operation> {
        let op = unsafe { mlir_eir_get_definition(value) };
        if op.is_null() {
            None
        } else {
            Some(op)
        }
    }

    /// Builds an eir::LandingPadOp
    pub fn build_landing_pad(&self, loc: Location, err: Block) -> Block {
        let guard = self.builder.insertion_guard();

        // This block is intended as the LLVM landing pad, and exists to
        // ensure that exception unwinding is handled properly
        let unwind = self.builder.create_block_before(err);

        // Obtain catch type value from entry block
        let region = unwind.region();
        let entry = region.entry();

        // There are two ways to define the type we catch, depending
        // on whether the exceptions are SEH or DWARF/EHABI
        let mut catch_type = None;
        if !self.arch.is_like_msvc() {
            for op in entry.iter().filter(|op| op.isa::<LLVMNullOp>()) {
                catch_type = Some(op.result());
                break;
            }
            // If not defined yet, we need to insert the operation
            if catch_type.is_none() {
                self.builder.set_insertion_point_to_start(entry);
                let llvm_builder = self.llvm();
                let i8ty = llvm_builder.get_i8_type();
                let i8_ptr_ty = llvm_builder.get_ptr_type(i8ty);
                let null_expr = llvm_builder.build_null(i8_ptr_ty);
                catch_type = Some(null_expr);
            }
        } else {
            for op in entry.iter().filter(|op| op.isa::<LLVMBitcastOp>()) {
                if let Some(maybe_addrof) = self.get_definition(op.get_operand(0).unwrap()) {
                    if let Some(addrof) = maybe_addrof.dyn_cast::<LLVMAddressOfOp>() {
                        if addrof.global_name() == "__lumen_erlang_error_type_info" {
                            catch_type = Some(op.result());
                            break;
                        }
                    }
                }
            }
            // If not defined yet, we need to insert the operation
            if catch_type.is_none() {
                self.builder.set_insertion_point_to_start(entry);
                let llvm_builder = self.llvm();
                let i8ty = llvm_builder.get_i8_type();
                let i8_ptr_ty = llvm_builder.get_ptr_type(i8ty);
                if let Some(op) = self.lookup_nearest_symbol("__lumen_erlang_error_type_info") {
                    let bitcast_expr =
                        llvm_builder.build_bitcast(llvm_builder.build_addressof(op), i8_ptr_ty);
                    catch_type = Some(bitcast_expr.result())
                } else {
                    // type_name = lumen_panic\0
                    let typename_ty = llvm_builder.get_array_type(i8ty, 12);
                    let typename_ptr_ty = llvm_builder.get_ptr_type(typename_ty);
                    let typeinfo_ty = llvm_builder.get_struct_type(
                        Some("type_info"),
                        &[i8_ptr_ty, i8_ptr_ty, typename_ptr_ty],
                    );
                    let global = llvm_builder.get_or_insert_global(
                        loc,
                        "__lumen_erlang_error_type_info",
                        typeinfo_ty,
                        false,
                        Linkage::External,
                        ThreadLocalMode::NotThreadLocal,
                        None,
                    );
                    let bitcast_expr = llvm_builder.build_bitcast(global, i8_ptr_ty);
                    catch_type = Some(bitcast_expr.result())
                }
            }
        };

        // Insert the landing pad in the unwind block
        self.builder.set_insertion_point_to_end(unwind);

        let op = LandingPadOp::create(self.builder, loc, &[catch_type.unwrap()]);
        let results = op.results();
        self.build_br(err, &results);

        let trace_ref_ty = self.get_trace_ref_type();
        let trace_arg = err.get_argument(2).unwrap();
        trace_arg.set_type(trace_ref_ty);

        unwind
    }

    /// Builds an eir::ReturnOp
    pub fn build_return(&self, loc: Location, value: Option<Value>) -> ReturnOp {
        ReturnOp::create(self.builder, loc, value)
    }

    /// Builds an eir::YieldOp
    pub fn build_yield(&self, loc: Location) -> YieldOp {
        YieldOp::create(self.builder, loc)
    }

    /// Builds an eir::YieldCheckOp
    pub fn build_yield_check(
        &self,
        loc: Location,
        max: Value,
        success: Block,
        success_args: &[Value],
        fail: Block,
        fail_args: &[Value],
    ) -> YieldCheckOp {
        YieldCheckOp::create(
            self.builder,
            loc,
            max,
            success,
            success_args,
            fail,
            fail_args,
        )
    }

    /// Builds an eir::ThrowOp
    pub fn build_throw(&self, loc: Location, kind: Value, reason: Value, trace: Value) -> ThrowOp {
        ThrowOp::create(self.builder, loc, kind, reason, trace)
    }

    /// Builds an eir::IncrementReductionsOp
    pub fn build_increment_reductions(
        &self,
        loc: Location,
        increment: usize,
    ) -> IncrementReductionsOp {
        IncrementReductionsOp::create(self.builder, loc, increment)
    }

    /// Builds an eir::MallocOp for a single object of type `ty`
    pub fn build_malloc(&self, loc: Location, ty: Type, align: Option<usize>) -> MallocOp {
        MallocOp::create(self.builder, ty, 1, align)
    }

    /// Builds an eir::MallocOp for an array of objects of type `ty`
    pub fn build_malloc_array(
        &self,
        loc: Location,
        ty: Type,
        len: usize,
        align: Option<usize>,
    ) -> MallocOp {
        MallocOp::create(self.builder, ty, len, align)
    }

    /// Builds an eir::LoadOp
    pub fn build_load(&self, loc: Location, ptr: Value) -> LoadOp {
        LoadOp::create(self.builder, loc, ptr)
    }

    /// Builds an eir::GetElementPtrOp
    pub fn build_gep(&self, loc: Location, ptr: Value, index: usize) -> GetElementPtrOp {
        GetElementPtrOp::create(self.builder, loc, ptr, index)
    }

    /// Builds an eir::PrintOp
    pub fn build_print(&self, loc: Location, value: Value) -> PrintOp {
        PrintOp::create(self.builder, loc, value)
    }

    /// Builds an eir::NullOp
    pub fn build_null(&self, loc: Location, ty: Type) -> NullOp {
        NullOp::create(self.builder, loc, ty)
    }

    /// Builds an eir::ConstantIntOp
    pub fn build_constant_int(&self, loc: Location, value: usize) -> ConstantOp {
        ConstantOp::create_fixnum(self.builder, loc, value)
    }

    /// Builds an eir::ConstantBigIntOp
    pub fn build_constant_bigint(&self, loc: Location, value: num_bigint::BigInt) -> ConstantOp {
        ConstantOp::create_bigint(self.builder, loc, value)
    }

    /// Builds an eir::ConstantFloatOp
    pub fn build_constant_float(&self, loc: Location, value: f64) -> ConstantOp {
        ConstantOp::create_float(self.builder, loc, value)
    }

    /// Builds an eir::ConstantBoolOp
    pub fn build_constant_bool(&self, loc: Location, value: bool) -> ConstantOp {
        ConstantOp::create_bool(self.builder, loc, value)
    }

    /// Builds an eir::ConstantAtomOp
    pub fn build_constant_atom(&self, loc: Location, id: usize, name: &str) -> ConstantOp {
        ConstantOp::create_atom(self.builder, loc, id, name)
    }

    /// Builds an eir::ConstantBinaryOp
    pub fn build_constant_binary(
        &self,
        loc: Location,
        bin: &[u8],
        flags: BinaryFlags,
    ) -> ConstantOp {
        ConstantOp::create_binary(self.builder, loc, bin, flags)
    }

    /// Builds an eir::ConstantNilOp
    pub fn build_constant_nil(&self, loc: Location) -> ConstantOp {
        ConstantOp::create_nil(self.builder, loc)
    }

    /// Builds an eir::ConstantNoneOp
    pub fn build_constant_none(&self, loc: Location) -> ConstantOp {
        ConstantOp::create_none(self.builder, loc)
    }

    /// Builds an eir::ConstantMapOp
    pub fn build_constant_map(
        &self,
        loc: Location,
        items: &[(Attribute, Attribute)],
    ) -> ConstantOp {
        ConstantOp::create_map(self.builder, loc, items)
    }

    /// Builds an eir::ConstantTupleOp
    pub fn build_constant_tuple(&self, loc: Location, elements: &[Attribute]) -> ConstantOp {
        ConstantOp::create_tuple(self.builder, loc, elements)
    }

    /// Builds an eir::ConstantListOp
    pub fn build_constant_list(&self, loc: Location, elements: &[Attribute]) -> ConstantOp {
        ConstantOp::create_list(self.builder, loc, elements)
    }

    /// Builds an eir::ConsOp
    pub fn build_cons(&self, loc: Location, head: Value, tail: Value) -> ConsOp {
        let term_ty = self.get_term_type();
        let head = if head.get_type() != term_ty {
            self.build_cast(head, term_ty).result()
        } else {
            head
        };
        let tail = if tail.get_type() != term_ty {
            self.build_cast(tail, term_ty).result()
        } else {
            tail
        };
        ConsOp::create(self.builder, loc, head, tail)
    }

    /// Builds an eir::ListOp
    pub fn build_list(&self, loc: Location, elements: &[Value]) -> ListOp {
        ListOp::create(self.builder, loc, elements)
    }

    /// Builds an eir::TupleOp
    pub fn build_tuple(&self, loc: Location, elements: &[Value]) -> TupleOp {
        TupleOp::create(self.builder, loc, elements)
    }

    /// Builds an eir::MapOp
    pub fn build_map(&self, loc: Location, items: &[(Value, Value)]) -> MapOp {
        MapOp::create(self.builder, loc, items)
    }

    /// Builds an eir::MapInsertOp
    pub fn build_map_insert(
        &self,
        loc: Location,
        map: Value,
        key: Value,
        value: Value,
    ) -> MapInsertOp {
        MapInsertOp::create(self.builder, loc, map, key, value)
    }

    /// Builds an eir::MapUpdateOp
    pub fn build_map_update(
        &self,
        loc: Location,
        map: Value,
        key: Value,
        value: Value,
    ) -> MapUpdateOp {
        MapUpdateOp::create(self.builder, loc, map, key, value)
    }

    /// Generates IR that performs a series of updates/inserts against a map
    ///
    /// The builder will be positioned at the end of the `ok` block on return
    pub fn build_map_updates(
        &self,
        loc: Location,
        map: Value,
        actions: &[MapAction],
        ok: Block,
        err: Block,
    ) {
        let map_ty = self.get_map_type();
        let box_map_ty = self.get_box_type(map_ty);

        // Each insert or update implicitly branches to a continuation block for the
        // next insert/update; the last continuation block simply branches
        // unconditionally to the ok block
        let current_block = self.current_block();

        // For empty maps, we simply branch to the continuation block
        let num_actions = actions.len();
        if num_actions == 0 {
            self.build_br(loc, ok, &[map]);
            return;
        }

        let mut map = map;
        for (i, action) in actions.iter().enumerate() {
            let is_last = i == num_actions - 1;
            // Create the continuation block, which expects the updated map as arg;
            // as well as the error block. Use the ok/error blocks provided as part
            // of the op if this is the last action being generated
            let ok = if is_last {
                ok
            } else {
                self.builder.create_block_in_region(region, &[box_map_ty])
            };
            // Make sure the successor block argument has the right type
            let map_arg = ok
                .get_argument(0)
                .expect("invalid target block for map update");
            map_arg.set_type(box_map_ty);
            // Perform the map action, and conditionally branch to the next ok
            // block if successful
            match action {
                MapAction::Insert { key, value } => {
                    let insert_expr = self.build_map_insert_op(loc, map, key, val);
                    let map = insert_expr.map();
                    let inserted = insert_expr.success();
                    self.build_cond_br(loc, inserted, ok, &[map], err, &[key]);
                }
                MapAction::Update { key, value } => {
                    let update_expr = self.build_map_update_op(loc, map, key, val);
                    let map = update_expr.map();
                    let updated = update_expr.success();
                    self.build_cond_br(loc, updated, ok, &[map], err, &[key]);
                }
            }

            // Each action occurs in the next `ok` block
            self.builder.set_insertion_point_to_end(ok);
            // We need to update our map reference to the one passed
            // to the current block, otherwise we'll refer to a binding
            // that is missing the most recent update
            map = map_arg;
        }
    }

    /// Builds an eir::MapContainsKeyOp
    pub fn build_map_contains_key(
        &self,
        loc: Location,
        map: Value,
        key: Value,
    ) -> MapContainsKeyOp {
        MapContainsKeyOp::create(self.builder, loc, map, key)
    }

    /// Builds an eir::MapGetKeyOp
    pub fn build_map_get_key(&self, loc: Location, map: Value, key: Value) -> MapGetKeyOp {
        MapGetKeyOp::create(self.builder, loc, map, key)
    }

    /// Builds an eir::BinaryStartOp
    pub fn build_binary_start(&self, loc: Location) -> BinaryStartOp {
        BinaryStartOp::create(self.builder, loc)
    }

    /// Builds an eir::BinaryFinishOp
    pub fn build_binary_finish(&self, loc: Location, bin: Value) -> BinaryFinishOp {
        BinaryFinishOp::create(self.builder, loc, bin)
    }

    /// Builds an eir::BinaryPushOp
    pub fn build_binary_push(
        &self,
        loc: Location,
        bin: Value,
        value: Value,
        size: Option<Value>,
    ) -> BinaryPushOp {
        BinaryPushOp::create(self.builder, loc, bin, value, size)
    }

    /// Builds an eir::BinaryMatchRawOp
    pub fn build_binary_match_raw(
        &self,
        loc: Location,
        bin: Value,
        unit: usize,
        size: Option<Value>,
    ) -> BinaryMatchRawOp {
        BinaryMatchRawOp::create(self.builder, loc, bin, value, size)
    }

    /// Builds an eir::BinaryMatchIntegerOp
    pub fn build_binary_match_integer(
        &self,
        loc: Location,
        bin: Value,
        signed: bool,
        endianess: Endianness,
        unit: usize,
        size: Option<Value>,
    ) -> BinaryMatchIntegerOp {
        BinaryMatchIntegerOp::create(self.builder, loc, bin, signed, endianness, unit, size)
    }

    /// Builds an eir::BinaryMatchFloatOp
    pub fn build_binary_match_float(
        &self,
        loc: Location,
        bin: Value,
        endianess: Endianness,
        unit: usize,
        size: Option<Value>,
    ) -> BinaryMatchFloatOp {
        BinaryMatchFloatOp::create(self.builder, loc, bin, endianness, unit, size)
    }

    /// Builds an eir::BinaryMatchUtf8Op
    pub fn build_binary_match_utf8(
        &self,
        loc: Location,
        bin: Value,
        size: Option<Value>,
    ) -> BinaryMatchUtf8Op {
        BinaryMatchUtf8Op::create(self.builder, loc, bin, size)
    }

    /// Builds an eir::BinaryMatchUtf16Op
    pub fn build_binary_match_utf16(
        &self,
        loc: Location,
        bin: Value,
        endianness: Endianess,
        size: Option<Value>,
    ) -> BinaryMatchUtf16Op {
        BinaryMatchUtf16Op::create(self.builder, loc, bin, endianness, size)
    }

    /// Builds an eir::BinaryMatchUtf32Op
    pub fn build_binary_match_utf32(
        &self,
        loc: Location,
        bin: Value,
        endianness: Endianess,
        size: Option<Value>,
    ) -> BinaryMatchUtf32Op {
        BinaryMatchUtf32Op::create(self.builder, loc, bin, endianness, size)
    }

    /// Builds an eir::ReceiveStartOp
    pub fn build_receive_start(&self, loc: Location, timeout: Value) -> ReceiveStartOp {
        ReceiveStartOp::create(self.builder, loc, timeout)
    }

    /// Builds an eir::ReceiveWaitOp
    pub fn build_receive_wait(&self, loc: Location, recv: Value) -> ReceiveWaitOp {
        ReceiveWaitOp::create(self.builder, loc, recv)
    }

    /// Builds an eir::ReceiveMessageOp
    pub fn build_receive_message(&self, loc: Location, recv: Value) -> ReceiveMessageOp {
        ReceiveMessageOp::create(self.builder, loc, recv)
    }

    /// Builds an eir::ReceiveDoneOp
    pub fn build_receive_done(&self, loc: Location, recv: Value) -> ReceiveDoneOp {
        ReceiveDoneOp::create(self.builder, loc, recv)
    }

    /// Builds a trace capture operation, branching to the given destination block
    pub fn build_trace_capture(&self, loc: Location, dest: Block, dest_args: &[Value]) {
        let trace_ref_type = self.builder.get_trace_ref_type();
        let trace_arg = dest
            .get_argument(0)
            .expect("invalid target block for trace capture");
        trace_arg.set_type(trace_ref_type);

        let trace_expr = TraceCaptureOp::create(self.builder, loc);
        let trace = &[trace_expr.result()];
        let dest_args = trace.iter().chain(dest_args.iter()).collect::<Vec<_>>();
        self.build_br(loc, dest, &dest_args);
    }

    /// Builds an eir::TracePrintOp
    pub fn build_trace_print(
        &self,
        loc: Location,
        kind: Value,
        reason: Value,
        trace: Value,
    ) -> TracePrintOp {
        TracePrintOp::create(self.builder, loc, kind, reason, trace)
    }

    /// Builds an eir::TraceConstructOp
    pub fn build_trace_construct(&self, loc: Location, trace: Value) -> TraceConstructOp {
        TraceConstructOp::create(self.builder, loc, trace)
    }

    /// Creates a new Block in `f` with the given argument types.
    ///
    /// Positions the builder at the end of the new block
    pub fn append_block(&self, f: &FuncOp, args: &[Type]) -> Block {
        let region = f.as_operation().get_region(0).unwrap();
        let block = self.builder.create_block_in_region(region, args);
        self.builder.set_insertion_point_to_end(block);
        block
    }

    /// Builds a runtime type check for integers
    #[inline(always)]
    pub fn build_is_integer(&self, loc: Location, input: Value) -> IsTypeOp {
        self.build_is_type(loc, input, self.get_integer_type())
    }

    /// Builds a runtime type check for integers
    #[inline(always)]
    pub fn build_is_integer(&self, loc: Location, input: Value) -> IsTypeOp {
        self.build_is_type(loc, input, self.get_integer_type())
    }

    /// Builds a runtime type check for numbers
    #[inline(always)]
    pub fn build_is_number(&self, loc: Location, input: Value) -> IsTypeOp {
        self.build_is_type(loc, input, self.get_number_type())
    }

    /// Builds a runtime type check for floats
    #[inline(always)]
    pub fn build_is_float(&self, loc: Location, input: Value) -> IsTypeOp {
        self.build_is_type(loc, input, self.get_float_type())
    }

    /// Builds a runtime type check for atoms
    #[inline(always)]
    pub fn build_is_atom(&self, loc: Location, input: Value) -> IsTypeOp {
        self.build_is_type(loc, input, self.get_atom_type())
    }

    /// Builds a runtime type check for booleans
    #[inline(always)]
    pub fn build_is_bool(&self, loc: Location, input: Value) -> IsTypeOp {
        self.build_is_type(loc, input, self.get_boolean_type())
    }

    /// Builds a runtime type check for nil
    #[inline(always)]
    pub fn build_is_nil(&self, loc: Location, input: Value) -> IsTypeOp {
        self.build_is_type(loc, input, self.get_nil_type())
    }

    /// Builds a runtime type check for lists
    #[inline(always)]
    pub fn build_is_list(&self, loc: Location, input: Value) -> IsTypeOp {
        self.build_is_type(loc, input, self.get_nil_type())
    }

    /// Builds a runtime type check for binaries
    #[inline(always)]
    pub fn build_is_binary(&self, loc: Location, input: Value) -> IsTypeOp {
        self.build_is_type(loc, input, self.get_binary_type())
    }

    /// Builds a runtime type check for maps
    #[inline(always)]
    pub fn build_is_map(&self, loc: Location, input: Value) -> IsTypeOp {
        self.build_is_type(loc, input, self.get_map_type())
    }

    /// Builds a runtime type check for references
    #[inline(always)]
    pub fn build_is_reference(&self, loc: Location, input: Value) -> IsTypeOp {
        self.build_is_type(loc, input, self.get_reference_type())
    }

    /// Builds a runtime type check for pids
    #[inline(always)]
    pub fn build_is_pid(&self, loc: Location, input: Value) -> IsTypeOp {
        self.build_is_type(loc, input, self.get_pid_type())
    }

    #[inline]
    pub fn build_intrinsic_error1(&self, loc: Location, reason: Value) {
        let kind = self
            .build_constant_atom(loc, self.get_atom("error"))
            .result();
        let trace = self.build_trace_capture(loc).result();
        self.build_throw(loc, kind, reason, trace);
    }

    #[inline]
    pub fn build_intrinsic_error2(&self, loc: Location, reason: Value, trace: Value) {
        let kind = self
            .build_constant_atom(loc, self.get_atom("error"))
            .result();
        self.build_throw(loc, kind, reason, trace);
    }

    #[inline]
    pub fn build_intrinsic_exit1(&self, loc: Location, reason: Value) {
        let kind = self
            .build_constant_atom(loc, self.get_atom("exit"))
            .result();
        let trace = self.build_trace_capture(loc).result();
        self.build_throw(loc, kind, reason, trace);
    }

    #[inline]
    pub fn build_intrinsic_throw1(&self, loc: Location, reason: Value) {
        let kind = self
            .build_constant_atom(loc, self.get_atom("throw"))
            .result();
        let trace = self.build_trace_capture(loc).result();
        self.build_throw(loc, kind, reason, trace);
    }

    #[inline]
    pub fn get_intrinsic_builder(
        &self,
        name: &str,
    ) -> Option<Fn(&Self, Location, &[Value]) -> Operation> {
        let builder = match name {
            "erlang:error/1" => {
                |builder, loc, args| builder.build_intrinsic_error1(loc, args[0]).into()
            }
            "erlang:error/2" => {
                |builder, loc, args| builder.build_intrinsic_error2(loc, args[0], args[1]).into()
            }
            "erlang:exit/1" => {
                |builder, loc, args| builder.build_intrinsic_exit1(loc, args[0]).into()
            }
            "erlang:throw/1" => {
                |builder, loc, args| builder.build_intrinsic_throw1(loc, args[0]).into()
            }
            "erlang:raise/3" => {
                |builder, loc, args| builder.build_throw(loc, args[0], args[1], args[2]).into()
            }
            "erlang:print/1" => |builder, loc, args| builder.build_print(loc, args[0]).into(),
            "erlang:+/2" => {
                |builder, loc, args| builder.build_math_add(loc, args[0], args[1]).into()
            }
            "erlang:-/1" => |builder, loc, args| builder.build_math_neg(loc, args[0]).into(),
            "erlang:+/2" => {
                |builder, loc, args| builder.build_math_sub(loc, args[0], args[1]).into()
            }
            "erlang:*/2" => {
                |builder, loc, args| builder.build_math_div(loc, args[0], args[1]).into()
            }
            "erlang:div/2" => {
                |builder, loc, args| builder.build_math_div(loc, args[0], args[1]).into()
            }
            "erlang:rem/2" => {
                |builder, loc, args| builder.build_math_rem(loc, args[0], args[1]).into()
            }
            "erlang://2" => {
                |builder, loc, args| builder.build_math_fdiv(loc, args[0], args[1]).into()
            }
            "erlang:bsl/2" => {
                |builder, loc, args| builder.build_math_bsl(loc, args[0], args[1]).into()
            }
            "erlang:bsr/2" => {
                |builder, loc, args| builder.build_math_bsr(loc, args[0], args[1]).into()
            }
            "erlang:band/2" => {
                |builder, loc, args| builder.build_math_band(loc, args[0], args[1]).into()
            }
            "erlang:bor/2" => {
                |builder, loc, args| builder.build_math_bor(loc, args[0], args[1]).into()
            }
            "erlang:bxor/2" => {
                |builder, loc, args| builder.build_math_bxor(loc, args[0], args[1]).into()
            }
            "erlang:and/2" => |builder, loc, args| builder.build_and(loc, args[0], args[1]).into(),
            "erlang:or/2" => |builder, loc, args| builder.build_or(loc, args[0], args[1]).into(),
            "erlang:=:=/2" => {
                |builder, loc, args| builder.build_eq_strict(loc, args[0], args[1], None).into()
            }
            "erlang:=/=/2" => |builder, loc, args| {
                builder
                    .build_neq(loc, args[0], args[1], None, /* strict */ true)
                    .into()
            },
            "erlang:==/2" => {
                |builder, loc, args| builder.build_eq(loc, args[0], args[1], None).into()
            }
            "erlang:/=/2" => |builder, loc, args| {
                builder
                    .build_neq(loc, args[0], args[1], None, /* strict */ false)
                    .into()
            },
            "erlang:</2" => |builder, loc, args| builder.build_lt(loc, args[0], args[1]).into(),
            "erlang:=</2" => |builder, loc, args| builder.build_lte(loc, args[0], args[1]).into(),
            "erlang:>/2" => |builder, loc, args| builder.build_gt(loc, args[0], args[1]).into(),
            "erlang:>=/2" => |builder, loc, args| builder.build_gte(loc, args[0], args[1]).into(),
            "erlang:is_integer/1" => {
                |builder, loc, args| builder.build_is_integer(loc, args[0]).into()
            }
            "erlang:is_number/1" => {
                |builder, loc, args| builder.build_is_number(loc, args[0]).into()
            }
            "erlang:is_float/1" => |builder, loc, args| builder.build_is_float(loc, args[0]).into(),
            "erlang:is_atom/1" => |builder, loc, args| builder.build_is_atom(loc, args[0]).into(),
            "erlang:is_boolean/1" => {
                |builder, loc, args| builder.build_is_boolean(loc, args[0]).into()
            }
            "erlang:is_tuple/1" => {
                |builder, loc, args| builder.build_is_tuple(loc, args[0], None).into()
            }
            "erlang:is_tuple/2" => {
                |builder, loc, args| builder.build_is_tuple(loc, args[0], Some(args[1])).into()
            }
            "erlang:is_nil/1" => |builder, loc, args| builder.build_is_nil(loc, args[0]).into(),
            "erlang:is_list/1" => |builder, loc, args| builder.build_is_list(loc, args[0]).into(),
            "erlang:is_map/1" => |builder, loc, args| builder.build_is_map(loc, args[0]).into(),
            "erlang:is_binary/1" => {
                |builder, loc, args| builder.build_is_binary(loc, args[0]).into()
            }
            "erlang:is_function/1" => {
                |builder, loc, args| builder.build_is_function(loc, args[0], None).into()
            }
            "erlang:is_function/2" => |builder, loc, args| {
                builder
                    .build_is_function(loc, args[0], Some(args[1]))
                    .into()
            },
            "erlang:is_reference/1" => {
                |builder, loc, args| builder.build_is_reference(loc, args[0]).into()
            }
            "erlang:is_pid/1" => |builder, loc, args| builder.build_is_pid(loc, args[0]).into(),
            _ => return None,
        };
        Some(builder)
    }

    #[inline]
    pub fn get_none_type(&self) -> Type {
        unsafe { mlir_eir_get_type_none(self.builder.as_ref()) }
    }

    /// Gets a reference to the eir::TermType type
    #[inline]
    pub fn get_term_type(&self) -> Type {
        unsafe { mlir_eir_get_type_term(self.builder.as_ref()) }
    }

    #[inline]
    pub fn get_list_type(&self) -> Type {
        unsafe { mlir_eir_get_type_list(self.builder.as_ref()) }
    }

    #[inline]
    pub fn get_number_type(&self) -> Type {
        unsafe { mlir_eir_get_type_number(self.builder.as_ref()) }
    }

    #[inline]
    pub fn get_integer_type(&self) -> Type {
        unsafe { mlir_eir_get_type_integer(self.builder.as_ref()) }
    }

    #[inline]
    pub fn get_float_type(&self) -> Type {
        unsafe { mlir_eir_get_type_float(self.builder.as_ref()) }
    }

    #[inline]
    pub fn get_atom_type(&self) -> Type {
        unsafe { mlir_eir_get_type_atom(self.builder.as_ref()) }
    }

    #[inline]
    pub fn get_boolean_type(&self) -> Type {
        unsafe { mlir_eir_get_type_boolean(self.builder.as_ref()) }
    }

    #[inline]
    pub fn get_fixnum_type(&self) -> Type {
        unsafe { mlir_eir_get_type_fixnum(self.builder.as_ref()) }
    }

    #[inline]
    pub fn get_bigint_type(&self) -> Type {
        unsafe { mlir_eir_get_type_bigint(self.builder.as_ref()) }
    }

    #[inline]
    pub fn get_nil_type(&self) -> Type {
        unsafe { mlir_eir_get_type_nil(self.builder.as_ref()) }
    }

    #[inline]
    pub fn get_cons_type(&self) -> Type {
        unsafe { mlir_eir_get_type_cons(self.builder.as_ref()) }
    }

    #[inline]
    pub fn get_map_type(&self) -> Type {
        unsafe { mlir_eir_get_type_map(self.builder.as_ref()) }
    }

    #[inline]
    pub fn get_binary_type(&self) -> Type {
        unsafe { mlir_eir_get_type_binary(self.builder.as_ref()) }
    }

    #[inline]
    pub fn get_heapbin_type(&self) -> Type {
        unsafe { mlir_eir_get_type_heap_binary(self.builder.as_ref()) }
    }

    #[inline]
    pub fn get_procbin_type(&self) -> Type {
        unsafe { mlir_eir_get_type_proc_binary(self.builder.as_ref()) }
    }

    #[inline]
    pub fn get_pid_type(&self) -> Type {
        unsafe { mlir_eir_get_type_pid(self.builder.as_ref()) }
    }

    #[inline]
    pub fn get_reference_type(&self) -> Type {
        unsafe { mlir_eir_get_type_reference(self.builder.as_ref()) }
    }

    /// Gets a reference to the eir::TupleType type
    #[inline]
    pub fn get_tuple_type(&self, arity: Option<usize>) -> Type {
        match arity {
            None => unsafe { mlir_eir_get_type_tuple(self.builder.as_ref()) },
            Some(n) => unsafe { mlir_eir_get_type_tuple_with_arity(self.builder.as_ref(), n) },
        }
    }

    /// Gets a reference to the eir::TupleType type with specific element types for each env value
    #[inline]
    pub fn get_tuple_type_with_elements(&self, elements: &[Type]) -> Type {
        unsafe {
            mlir_eir_get_type_tuple_with_elements(
                self.builder.as_ref(),
                elements.as_ptr(),
                elements.len(),
            )
        }
    }

    /// Gets a reference to the eir::ClosureType type
    #[inline]
    pub fn get_closure_type(&self, arity: Option<usize>) -> Type {
        match arity {
            None => unsafe { mlir_eir_get_type_closure(self.builder.as_ref()) },
            Some(n) => unsafe { mlir_eir_get_type_closure_with_arity(self.builder.as_ref(), n) },
        }
    }

    /// Gets a reference to the eir::ClosureType type with specific element types for each env value
    #[inline]
    pub fn get_closure_type_with_elements(&self, elements: &[Type]) -> Type {
        unsafe {
            mlir_eir_get_type_closure_with_elements(
                self.builder.as_ref(),
                elements.as_ptr(),
                elements.len(),
            )
        }
    }

    /// Gets a reference to a pointer of type `ty`
    #[inline]
    pub fn get_ptr_type(&self, ty: Type) -> Type {
        unsafe { mlir_eir_get_type_ptr(self.builder.as_ref(), ty) }
    }

    /// Gets a reference to a boxed value of type `ty`
    #[inline]
    pub fn get_box_type(&self, ty: Type) -> Type {
        unsafe { mlir_eir_get_type_box(self.builder.as_ref(), ty) }
    }

    #[inline]
    pub fn get_trace_ref_type(&self) -> Type {
        unsafe { mlir_eir_get_type_trace_ref(self.builder.as_ref()) }
    }

    #[inline]
    pub fn get_receive_ref_type(&self) -> Type {
        unsafe { mlir_eir_get_type_receive_ref(self.builder.as_ref()) }
    }

    #[inline]
    pub fn try_get_pointee_type(&self, ty: Type, index: Option<usize>) -> Option<Type> {
        let pointee =
            unsafe { mlir_eir_get_pointee_type(self.builder.as_ref(), ty, index.unwrap_or(0)) };
        if pointee.is_null() {
            None
        } else {
            Some(pointee)
        }
    }
}
impl AsRef<MlirBuilder> for EirBuilder<'_> {
    fn as_ref(&self) -> &MlirBuilder {
        self.builder.as_ref()
    }
}
impl Deref for EirBuilder<'_> {
    type Target = OpBuilder;

    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        self.0
    }
}
impl DerefMut for EirBuilder<'_> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0
    }
}

extern "C" {
    #[link_name = "mlirEirGetTypeNone"]
    fn mlir_eir_get_type_none(builder: &MlirBuilder) -> Type;
    #[link_name = "mlirEirGetTypeTerm"]
    fn mlir_eir_get_type_term(builder: &MlirBuilder) -> Type;
    #[link_name = "mlirEirGetTypeList"]
    fn mlir_eir_get_type_list(builder: &MlirBuilder) -> Type;
    #[link_name = "mlirEirGetTypeNumber"]
    fn mlir_eir_get_type_number(builder: &MlirBuilder) -> Type;
    #[link_name = "mlirEirGetTypeInteger"]
    fn mlir_eir_get_type_integer(builder: &MlirBuilder) -> Type;
    #[link_name = "mlirEirGetTypeFloat"]
    fn mlir_eir_get_type_float(builder: &MlirBuilder) -> Type;
    #[link_name = "mlirEirGetTypeAtom"]
    fn mlir_eir_get_type_atom(builder: &MlirBuilder) -> Type;
    #[link_name = "mlirEirGetTypeBoolean"]
    fn mlir_eir_get_type_boolean(builder: &MlirBuilder) -> Type;
    #[link_name = "mlirEirGetTypeFixnum"]
    fn mlir_eir_get_type_fixnum(builder: &MlirBuilder) -> Type;
    #[link_name = "mlirEirGetTypeBigInt"]
    fn mlir_eir_get_type_bigint(builder: &MlirBuilder) -> Type;
    #[link_name = "mlirEirGetTypeNil"]
    fn mlir_eir_get_type_nil(builder: &MlirBuilder) -> Type;
    #[link_name = "mlirEirGetTypeCons"]
    fn mlir_eir_get_type_cons(builder: &MlirBuilder) -> Type;
    #[link_name = "mlirEirGetTypeMap"]
    fn mlir_eir_get_type_map(builder: &MlirBuilder) -> Type;
    #[link_name = "mlirEirGetTypeBinary"]
    fn mlir_eir_get_type_binary(builder: &MlirBuilder) -> Type;
    #[link_name = "mlirEirGetTypeHeapBin"]
    fn mlir_eir_get_type_heap_binary(builder: &MlirBuilder) -> Type;
    #[link_name = "mlirEirGetTypeProcBin"]
    fn mlir_eir_get_type_proc_binary(builder: &MlirBuilder) -> Type;
    #[link_name = "mlirEirGetTypePid"]
    fn mlir_eir_get_type_pid(builder: &MlirBuilder) -> Type;
    #[link_name = "mlirEirGetTypeReference"]
    fn mlir_eir_get_type_reference(builder: &MlirBuilder) -> Type;
    #[link_name = "mlirEirGetTypeTuple"]
    fn mlir_eir_get_type_tuple(builder: &MlirBuilder) -> Type;
    #[link_name = "mlirEirGetTypeTupleWithArity"]
    fn mlir_eir_get_type_tuple_with_arity(builder: &MlirBuilder, arity: usize) -> Type;
    #[link_name = "mlirEirGetTypeTupleWithElements"]
    fn mlir_eir_get_type_tuple_with_elements(
        builder: &MlirBuilder,
        elements: *const Type,
        len: usize,
    ) -> Type;
    #[link_name = "mlirEirGetTypeClosure"]
    fn mlir_eir_get_type_closure(builder: &MlirBuilder) -> Type;
    #[link_name = "mlirEirGetTypeClosureWithArity"]
    fn mlir_eir_get_type_closure_with_arity(builder: &MlirBuilder, arity: usize) -> Type;
    #[link_name = "mlirEirGetTypeClosureWithElements"]
    fn mlir_eir_get_type_closure_with_elements(
        builder: &MlirBuilder,
        elements: *const Type,
        len: usize,
    ) -> Type;
    #[link_name = "mlirEirGetTypePtr"]
    fn mlir_eir_get_type_ptr(builder: &MlirBuilder, ty: Type) -> Type;
    #[link_name = "mlirEirGetTypeBox"]
    fn mlir_eir_get_type_box(builder: &MlirBuilder, ty: Type) -> Type;
    #[link_name = "mlirEirGetTypeTraceRef"]
    fn mlir_eir_get_type_trace_ref(builder: &MlirBuilder) -> Type;
    #[link_name = "mlirEirGetTypeReceiveRef"]
    fn mlir_eir_get_type_receive_ref(builder: &MlirBuilder) -> Type;
    #[link_name = "mlirEirGetPointeeType"]
    fn mlir_eir_get_pointee_type(builder: &MlirBuilder, ty: Type, index: usize) -> Type;
    #[link_name = "mlirEirGetClosureDefinition"]
    fn mlir_eir_get_closure_definition(closure: Value) -> ClosureOp;
    #[link_name = "mlirEirGetDefinition"]
    fn mlir_eir_get_definition(value: Value) -> Operation;
}
