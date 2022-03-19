use std::ops::Deref;

use liblumen_compiler_macros::operation;

use liblumen_mlir::*;

/// Represents an EIR function
#[operation("eir.func")]
pub struct FuncOp;
impl FuncOp {
    /// Returns the entry block for this function
    pub fn entry_block(&self) -> Block {
        let region = self.get_region(0).unwrap();
        region.entry().unwrap()
    }

    fn create(
        builder: &mut OpBuilder,
        loc: Location,
        name: &str,
        ty: FunctionType,
        visibility: Visibility,
        variadic: Variadic,
        attrs: &[NamedAttribute],
    ) -> Self {
        let context = builder.get_context();

        // Build the operation
        let mut state = Self::build(loc);

        // Set function type, create region
        let type_attr = builder.get_named_attr("type", builder.get_type_attr(ty));
        state.add_attributes(&[type_attr]);
        if attrs.len() > 0 {
            state.add_attributes(attrs);
        }

        // Add default region
        state.add_regions(&[Region::new()]);

        // Construct FuncOp
        let op = builder.create_operation(state);
        debug_assert!(!op.is_null(), "expected valid operation, but got null");
        debug_assert_eq!(
            op.num_regions(),
            1,
            "expected function to contain one region"
        );
        debug_assert!(
            op.get_region(0).unwrap().entry().is_some(),
            "expected function region to contain entry block"
        );

        if variadic == Variadic::Yes {
            op.set_attribute_by_name("std.varargs", builder.get_bool_attr(true));
        }

        // Set the symbol name and visibility for this function
        SymbolTable::set_symbol_name(op, name);
        SymbolTable::set_symbol_visibility(op, visibility);

        Self(op)
    }
}

/// This operation type is used to represent a closure capture
#[operation("eir.closure")]
pub struct ClosureOp;
impl ClosureOp {
    #[inline]
    pub fn result(&self) -> Value {
        self.0.get_result(0).unwrap()
    }

    fn create(
        builder: &mut OpBuilder,
        loc: Location,
        module_attr: Attribute,
        name: &str,
        arity: u8,
        info: ClosureInfo,
        env: &[Value],
    ) -> Self {
        let i8ty = builder.get_i8_type();
        let i32ty = builder.get_i32_type();
        let closure_type = builder.get_closure_type_with_arity(env.len());
        let closure_ptr_type = builder.get_ptr_type(closure_type);
        let callee_symbol_attr = builder.get_symbol_ref_attr(name, &[]);

        // Build the operation
        let mut state = Self::build(loc);

        let callee = builder.get_named_attr("callee", callee_symbol_attr);
        let module = builder.get_named_attr("module", module_attr);
        let arity = builder.get_named_attr("arity", builder.get_i8_attr(arity));
        let env_len = builder.get_named_attr("env_len", builder.get_i8_attr(env.len()));
        let index = builder.get_named_attr("index", builder.get_i32_attr(info.index));
        let old_unique = builder.get_named_attr("old_unique", builder.get_i32_attr(info.index));
        let unique =
            builder.get_named_attr("unique", builder.get_string_attr_from_bytes(info.unique));

        state.add_attributes(&[callee, module, arity, env_len, index, old_unique, unique]);
        state.add_operands(env);
        state.add_results(&[closure_ptr_type]);

        Self(builder.create_operation(state))
    }
}

/// This operation type is used to represent the extraction of values
/// from a closure environment in the function prelude.
#[operation("eir.closure.unpack")]
pub struct UnpackEnvOp;
impl UnpackEnvOp {
    /// Returns the result of the operation: the unpacked value
    pub fn result(&self) -> Value {
        self.get_result(0).unwrap()
    }

    fn create(builder: &mut OpBuilder, loc: Location, env: Value, index: usize) -> Self {
        let index_attr = builder.get_named_attr("index", builder.get_i32_attr(index));

        let mut state = Self::build(loc);
        state.add_attributes(&[index_attr]);
        state.add_results(&[term_ty]);
        state.add_operands(&[env]);

        Self(builder.create_operation(state))
    }
}

/// This operation type is used to type check values at runtime
#[operation("eir.is_type")]
pub struct IsTypeOp;
impl IsTypeOp {
    #[inline(always)]
    pub fn result(&self) -> Value {
        self.get_result(0).unwrap()
    }

    fn create(builder: &mut OpBuilder, loc: Location, input: Value, ty: Type) -> Self {
        let i1ty = builder.get_i1_type();
        let type_attr = builder.get_named_attr("type", builder.get_type_attr(ty));

        let mut state = Self::build(loc);
        state.add_attributes(&[type_attr]);
        state.add_operands(&[input]);
        state.add_results(&[i1ty]);

        Self(builder.create_operation(state))
    }
}

#[operation("eir.logical.and")]
pub struct LogicalAndOp;
impl LogicalAndOp {
    #[inline(always)]
    pub fn result(&self) -> Value {
        self.get_result(0).unwrap()
    }

    fn create(builder: &mut OpBuilder, loc: Location, lhs: Value, rhs: Value) -> Self {
        Self(self::create_binary_operator_op(
            Self::build(loc),
            lhs,
            rhs,
            builder.get_i1_type(),
        ))
    }
}

#[operation("eir.logical.or")]
pub struct LogicalOrOp;
impl LogicalOrOp {
    #[inline(always)]
    pub fn result(&self) -> Value {
        self.get_result(0).unwrap()
    }

    fn create(builder: &mut OpBuilder, loc: Location, lhs: Value, rhs: Value) -> Self {
        Self(self::create_binary_operator_op(
            builder,
            Self::build(loc),
            lhs,
            rhs,
            builder.get_i1_type(),
        ))
    }
}

#[operation("eir.cmp.eq")]
pub struct CmpEqOp;
impl CmpEqOp {
    #[inline(always)]
    pub fn result(&self) -> Value {
        self.get_result(0).unwrap()
    }

    fn create<B>(
        builder: &mut OpBuilder,
        loc: Location,
        lhs: Value,
        rhs: Value,
        ty: Type,
        attrs: &[NamedAttribute],
    ) -> Self {
        let mut state = Self::build(loc);
        state.add_results(&[ty]);
        state.add_operands(&[lhs, rhs]);
        state.add_attributes(attrs);

        Self(builder.create_operation(state))
    }
}

#[operation("eir.cmp.lt")]
pub struct CmpLtOp;
impl CmpLtOp {
    #[inline(always)]
    pub fn result(&self) -> Value {
        self.get_result(0).unwrap()
    }

    fn create(builder: &mut OpBuilder, loc: Location, lhs: Value, rhs: Value, ty: Type) -> Self {
        Self(self::create_binary_operator_op(
            builder,
            Self::build(loc),
            lhs,
            rhs,
            ty,
        ))
    }
}

#[operation("eir.cmp.lte")]
pub struct CmpLteOp;
impl CmpLteOp {
    #[inline(always)]
    pub fn result(&self) -> Value {
        self.get_result(0).unwrap()
    }

    fn create(builder: &mut OpBuilder, loc: Location, lhs: Value, rhs: Value, ty: Type) -> Self {
        Self(self::create_binary_operator_op(
            builder,
            Self::build(loc),
            lhs,
            rhs,
            ty,
        ))
    }
}

#[operation("eir.cmp.gt")]
pub struct CmpGtOp;
impl CmpGtOp {
    #[inline(always)]
    pub fn result(&self) -> Value {
        self.get_result(0).unwrap()
    }

    fn create(builder: &mut OpBuilder, loc: Location, lhs: Value, rhs: Value, ty: Type) -> Self {
        Self(self::create_binary_operator_op(
            builder,
            Self::build(loc),
            lhs,
            rhs,
            ty,
        ))
    }
}

#[operation("eir.cmp.gte")]
pub struct CmpGteOp;
impl CmpGteOp {
    #[inline(always)]
    pub fn result(&self) -> Value {
        self.get_result(0).unwrap()
    }

    fn create(builder: &mut OpBuilder, loc: Location, lhs: Value, rhs: Value, ty: Type) -> Self {
        Self(self::create_binary_operator_op(
            builder,
            Self::build(loc),
            lhs,
            rhs,
            ty,
        ))
    }
}

#[operation("eir.math.neg")]
pub struct NegOp;
impl NegOp {
    #[inline(always)]
    pub fn result(&self) -> Value {
        self.get_result(0).unwrap()
    }

    fn create(builder: &mut OpBuilder, loc: Location, input: Value) -> Self {
        let ty = input.get_type();
        Self(self::create_unary_operator_op(
            builder,
            Self::build(loc),
            input,
            ty,
        ))
    }
}

#[operation("eir.math.add")]
pub struct AddOp;
impl AddOp {
    #[inline(always)]
    pub fn result(&self) -> Value {
        self.get_result(0).unwrap()
    }

    fn create(builder: &mut OpBuilder, loc: Location, lhs: Value, rhs: Value) -> Self {
        let ty = lhs.get_type();
        Self(self::create_binary_operator_op(
            builder,
            Self::build(loc),
            lhs,
            rhs,
            ty,
        ))
    }
}

#[operation("eir.math.sub")]
pub struct SubOp;
impl SubOp {
    #[inline(always)]
    pub fn result(&self) -> Value {
        self.get_result(0).unwrap()
    }

    fn create(builder: &mut OpBuilder, loc: Location, lhs: Value, rhs: Value) -> Self {
        let ty = lhs.get_type();
        Self(self::create_binary_operator_op(
            builder,
            Self::build(loc),
            lhs,
            rhs,
            ty,
        ))
    }
}

#[operation("eir.math.mul")]
pub struct MulOp;
impl MulOp {
    #[inline(always)]
    pub fn result(&self) -> Value {
        self.get_result(0).unwrap()
    }

    fn create(builder: &mut OpBuilder, loc: Location, lhs: Value, rhs: Value) -> Self {
        let ty = lhs.get_type();
        Self(self::create_binary_operator_op(
            builder,
            Self::build(loc),
            lhs,
            rhs,
            ty,
        ))
    }
}

#[operation("eir.math.div")]
pub struct DivOp;
impl DivOp {
    #[inline(always)]
    pub fn result(&self) -> Value {
        self.get_result(0).unwrap()
    }

    fn create(builder: &mut OpBuilder, loc: Location, lhs: Value, rhs: Value) -> Self {
        let ty = lhs.get_type();
        Self(self::create_binary_operator_op(
            builder,
            Self::build(loc),
            lhs,
            rhs,
            ty,
        ))
    }
}

#[operation("eir.math.fdiv")]
pub struct FDivOp;
impl FDivOp {
    #[inline(always)]
    pub fn result(&self) -> Value {
        self.get_result(0).unwrap()
    }

    fn create(builder: &mut OpBuilder, loc: Location, lhs: Value, rhs: Value) -> Self {
        let ty = lhs.get_type();
        Self(self::create_binary_operator_op(
            builder,
            Self::build(loc),
            lhs,
            rhs,
            ty,
        ))
    }
}

#[operation("eir.math.rem")]
pub struct RemOp;
impl RemOp {
    #[inline(always)]
    pub fn result(&self) -> Value {
        self.get_result(0).unwrap()
    }

    fn create(builder: &mut OpBuilder, loc: Location, lhs: Value, rhs: Value) -> Self {
        let ty = lhs.get_type();
        Self(self::create_binary_operator_op(
            builder,
            Self::build(loc),
            lhs,
            rhs,
            ty,
        ))
    }
}

#[operation("eir.math.bsl")]
pub struct BslOp;
impl BslOp {
    #[inline(always)]
    pub fn result(&self) -> Value {
        self.get_result(0).unwrap()
    }

    fn create(builder: &mut OpBuilder, loc: Location, lhs: Value, rhs: Value) -> Self {
        let ty = lhs.get_type();
        Self(self::create_binary_operator_op(
            builder,
            Self::build(loc),
            lhs,
            rhs,
            ty,
        ))
    }
}

#[operation("eir.math.bsr")]
pub struct BsrOp;
impl BsrOp {
    #[inline(always)]
    pub fn result(&self) -> Value {
        self.get_result(0).unwrap()
    }

    fn create(builder: &mut OpBuilder, loc: Location, lhs: Value, rhs: Value) -> Self {
        let ty = lhs.get_type();
        Self(self::create_binary_operator_op(
            builder,
            Self::build(loc),
            lhs,
            rhs,
            ty,
        ))
    }
}

#[operation("eir.math.band")]
pub struct BandOp;
impl BandOp {
    #[inline(always)]
    pub fn result(&self) -> Value {
        self.get_result(0).unwrap()
    }

    fn create(builder: &mut OpBuilder, loc: Location, lhs: Value, rhs: Value) -> Self {
        let ty = lhs.get_type();
        Self(self::create_binary_operator_op(
            builder,
            Self::build(loc),
            lhs,
            rhs,
            ty,
        ))
    }
}

#[operation("eir.math.bor")]
pub struct BorOp;
impl BorOp {
    #[inline(always)]
    pub fn result(&self) -> Value {
        self.get_result(0).unwrap()
    }

    fn create(builder: &mut OpBuilder, loc: Location, lhs: Value, rhs: Value) -> Self {
        let ty = lhs.get_type();
        Self(self::create_binary_operator_op(
            builder,
            Self::build(loc),
            lhs,
            rhs,
            ty,
        ))
    }
}

#[operation("eir.math.bxor")]
pub struct BxorOp;
impl BxorOp {
    #[inline(always)]
    pub fn result(&self) -> Value {
        self.get_result(0).unwrap()
    }

    fn create(builder: &mut OpBuilder, loc: Location, lhs: Value, rhs: Value) -> Self {
        let ty = lhs.get_type();
        Self(self::create_binary_operator_op(
            builder,
            Self::build(loc),
            lhs,
            rhs,
            ty,
        ))
    }
}

#[operation("eir.br")]
pub struct BranchOp;
impl BranchOp {
    fn create(builder: &mut OpBuilder, loc: Location, dest: Block, args: &[Value]) -> Self {
        // Build the operation
        let mut state = Self::build(loc);
        state.add_successors(&[dest]);
        state.add_operands(args);

        Self(builder.create_operation(state))
    }
}

#[operation("eir.cond_br")]
pub struct CondBranchOp;
impl CondBranchOp {
    fn create(
        builder: &mut OpBuilder,
        loc: Location,
        cond: Value,
        yes: Block,
        yes_args: &[Value],
        no: Block,
        no_args: &[Value],
    ) -> Self {
        // Build the operation
        let mut state = Self::build(loc);
        state.add_successors(&[yes, no]);
        state.add_operands(yes_args);
        state.add_operands(no_args);

        Self(builder.create_operation(state))
    }
}

#[operation("eir.call")]
pub struct CallOp;
impl CallOp {
    fn create(
        builder: &mut OpBuilder,
        loc: Location,
        name: &str,
        ty: FunctionType,
        args: &[Value],
        attrs: &[NamedAttribute],
    ) -> Self {
        let mut state = Self::build(loc);
        let callee =
            builder.get_named_attr("callee", builder.get_flat_symbol_ref_attr_by_name(name));
        state.add_attributes(attrs);
        state.add_attributes(&[callee]);
        state.add_operands(args);
        state.add_results(&ty.outputs());

        Self(builder.create_operation(state))
    }
}

#[operation("eir.invoke")]
pub struct InvokeOp;
impl InvokeOp {
    fn create(
        builder: &mut OpBuilder,
        loc: Location,
        name: &str,
        ty: FunctionType,
        args: &[Value],
        dest: Block,
        unwind: Block,
        attrs: &[NamedAttribute],
    ) -> Self {
        let mut state = Self::build(loc);
        let callee =
            builder.get_named_attr("callee", builder.get_flat_symbol_ref_attr_by_name(name));
        state.add_attributes(attrs);
        state.add_attributes(&[callee]);
        state.add_operands(args);
        state.add_successors(&[dest, unwind]);

        Self(builder.create_operation(state))
    }
}

#[operation("eir.landing_pad")]
pub struct LandingPadOp;
impl LandingPadOp {
    pub fn kind(&self) -> Value {
        self.get_result(0).unwrap();
    }
    pub fn reason(&self) -> Value {
        self.get_result(1).unwrap();
    }
    pub fn trace(&self) -> Value {
        self.get_result(2).unwrap();
    }
    fn create(builder: &mut OpBuilder, loc: Location, clauses: &[Value]) -> Self {
        let mut state = Self::build(loc);
        state.add_operands(clauses);

        Self(builder.create_operation(state))
    }
}

#[operation("eir.return")]
pub struct ReturnOp;
impl ReturnOp {
    fn create(builder: &mut OpBuilder, loc: Location, value: Option<Value>) -> Self {
        let mut state = Self::build(loc);
        if let Some(ret) = value {
            state.add_results(&[ret.get_type()]);
        }
        Self(builder.create_operation(state))
    }
}

#[operation("eir.yield")]
pub struct YieldOp;
impl YieldOp {
    fn create(builder: &mut OpBuilder, loc: Location) -> Self {
        Self(builder.create_operation(Self::build(loc)))
    }
}

#[operation("eir.yield_check")]
pub struct YieldCheckOp;
impl YieldCheckOp {
    fn create(
        builder: &mut OpBuilder,
        loc: Location,
        max: Value,
        success: Block,
        success_args: &[Value],
        fail: Block,
        fail_args: &[Value],
    ) -> Self {
        let mut state = Self::build(loc);
        state.add_successors(&[success, fail]);
        state.add_operands(&[max]);
        Self(builder.create_operation(state))
    }
}

/// This operation type is used to represent unreachable code
#[operation("eir.unreachable")]
pub struct UnreachableOp;
impl UnreachableOp {
    fn create(builder: &mut OpBuilder, loc: Location) -> Self {
        Self(builder.create_operation(Self::build(loc)))
    }
}

#[operation("eir.throw")]
pub struct ThrowOp;
impl ThrowOp {
    fn create(
        builder: &mut OpBuilder,
        loc: Location,
        kind: Value,
        reason: Value,
        trace: Value,
    ) -> Self {
        let mut state = Self::build(loc);
        state.add_operands(&[kind, reason, trace]);
        Self(builder.create_operation(state))
    }
}

#[operation("eir.reductions.inc")]
pub struct IncrementReductionsOp;
impl IncrementReductionsOp {
    fn create(builder: &mut OpBuilder, loc: Location, increment: usize) -> Self {
        let increment = builder.get_named_attr("increment", builder.get_i32_attr(increment));
        let mut state = Self::build(loc);
        state.add_attributes(&[increment]);
        Self(builder.create_operation(state))
    }
}

/// This operation type is used to cast values from one type
/// to another. In some cases this performs actual conversion,
/// in others it is used to simply change how the bits are interpreted
#[operation("eir.cast")]
pub struct CastOp;
impl CastOp {
    #[inline(always)]
    pub fn result(&self) -> Value {
        self.get_result(0).unwrap()
    }

    fn create(builder: &mut OpBuilder, loc: Location, input: Value, ty: Type) -> Self {
        let from_type_attr = builder.get_named_attr("from", builder.get_type_attr(src_ty));
        let to_type_attr = builder.get_named_attr("to", builder.get_type_attr(ty));

        let mut state = Self::build(loc);
        state.add_attributes(&[from_type_attr, to_type_attr]);
        state.add_results(&[ty]);
        state.add_operands(&[input]);

        Self(builder.create_operation(state))
    }
}

#[operation("eir.malloc")]
pub struct MallocOp;
impl MallocOp {
    #[inline(always)]
    pub fn result(&self) -> Value {
        self.get_result(0).unwrap()
    }

    fn create(
        builder: &mut OpBuilder,
        loc: Location,
        arity: Option<Value>,
        align: Option<usize>,
    ) -> Self {
        let mut state = Self::build(loc);
        if let Some(n) = arity {
            state.add_operands(&[n]);
        }
        if let Some(n) = align {
            let attr = builder.get_named_attr("alignment", builder.get_i32_attr(n));
            state.add_attributes(&[attr]);
        }
        state.add_attributes(&[align]);
        state.add_results(&[ty]);
        Self(builder.create_operation(state))
    }
}

#[operation("eir.load")]
pub struct LoadOp;
impl LoadOp {
    #[inline(always)]
    pub fn result(&self) -> Value {
        self.get_result(0).unwrap()
    }

    fn create(builder: &mut OpBuilder, loc: Location, ptr: Value) -> Self {
        let mut state = Self::build(loc);
        state.add_operands(&[ptr]);
        state.add_results(&[ptr.get_type()]);
        Self(builder.create_operation(state))
    }
}

#[operation("eir.getelementptr")]
pub struct GetElementPtrOp;
impl GetElementPtrOp {
    #[inline(always)]
    pub fn result(&self) -> Value {
        self.get_result(0).unwrap()
    }

    fn create(builder: &mut OpBuilder, loc: Location, ptr: Value, index: usize) -> Self {
        let index_attr = builder.get_named_attr("index", builder.get_index_attr(index));
        let mut state = Self::build(loc);
        state.add_attributes(&[index_attr]);
        state.add_operands(&[ptr]);
        let base_type = ptr.get_type();
        let pointee_type = builder
            .try_get_pointee_type(base_type, Some(index))
            .map(|t| builder.get_ptr_type(t))
            .unwrap_or_else(|| builder.get_none_type());

        state.add_results(&[pointee_type]);
        Self(builder.create_operation(state))
    }
}

#[operation("eir.print")]
pub struct PrintOp;
impl PrintOp {
    #[inline(always)]
    pub fn result(&self) -> Value {
        self.get_result(0).unwrap()
    }

    fn create(builder: &mut OpBuilder, loc: Location, value: Value) -> Self {
        let term_ty = builder.get_term_type();
        let mut state = Self::build(loc);
        state.add_operands(&[value]);
        state.add_results(&[term_ty]);
        Self(builder.create_operation(state))
    }
}

#[operation("eir.null")]
pub struct NullOp;
impl NullOp {
    #[inline(always)]
    pub fn result(&self) -> Value {
        self.get_result(0).unwrap()
    }

    fn create(builder: &mut OpBuilder, loc: Location, ty: Type) -> Self {
        let mut state = Self::build(loc);
        state.add_results(&[ty]);
        Self(builder.create_operation(state))
    }
}

#[operation("eir.constant")]
pub struct ConstantOp;
impl ConstantOp {
    #[inline(always)]
    pub fn result(&self) -> Value {
        self.get_result(0).unwrap()
    }

    fn create_fixnum(builder: &mut OpBuilder, loc: Location, value: usize) -> Self {
        let ty = builder.get_fixnum_type();
        let value_attr = builder.get_named_attr("value", builder.get_apint_attr(value, fixnum_ty));
        Self::create(builder, loc, "eir.constant.int", value_attr, ty)
    }

    fn create_bigint(builder: &mut OpBuilder, loc: Location, value: num_bigint::BigInt) -> Self {
        let ty = builder.get_bigint_type();
        let value_attr = builder.get_named_attr("value", builder.get_apint_attr(value));
        Self::create(builder, loc, "eir.constant.bigint", value_attr, ty)
    }

    fn create_float(builder: &mut OpBuilder, loc: Location, value: f64) -> Self {
        let ty = builder.get_float_type();
        let value_attr = builder.get_named_attr("value", builder.get_apfloat_attr(value));
        Self::create(builder, loc, "eir.constant.float", value_attr, ty)
    }

    fn create_bool(builder: &mut OpBuilder, loc: Location, value: bool) -> Self {
        let ty = builder.get_boolean_type();
        let value_attr = builder.get_named_attr("value", builder.get_bool_attr(value));
        Self::create(builder, loc, "eir.constant.bool", value_attr, ty)
    }

    fn create_atom(builder: &mut OpBuilder, loc: Location, id: usize, name: &str) -> Self {
        let ty = builder.get_atom_type();
        let value_attr = builder.get_named_attr("value", builder.get_atom_attr(id, name));
        Self::create(builder, loc, "eir.constant.atom", value_attr, ty)
    }

    fn create_binary(builder: &mut OpBuilder, loc: Location, bytes: &[u8], flags: BinaryFlags) -> Self {
        let ty = builder.get_binary_type();
        let value_attr = builder.get_named_attr("value", builder.get_binary_attr(bytes, flags));
        Self::create(builder, loc, "eir.constant.binary", value_attr, ty)
    }

    fn create_nil(builder: &mut OpBuilder, loc: Location) -> Self {
        let ty = builder.get_nil_type();
        let value_attr = builder.get_named_attr("value", builder.get_unit_attr());
        Self::create(builder, loc, "eir.constant.nil", value_attr, ty)
    }

    fn create_none(builder: &mut OpBuilder, loc: Location) -> Self {
        let ty = builder.get_none_type();
        let value_attr = builder.get_named_attr("value", builder.get_unit_attr());
        Self::create(builder, loc, "eir.constant.none", value_attr, ty)
    }

    fn create_tuple(builder: &mut OpBuilder, loc: Location, elements: &[Attribute]) -> Self {
        let ty = builder.get_tuple_type(elements.len());
        let value_attr = builder.get_named_attr("value", builder.get_seq_attr(elements));
        Self::create(builder, loc, "eir.constant.tuple", value_attr, ty)
    }

    fn create_list(builder: &mut OpBuilder, loc: Location, elements: &[Attribute]) -> Self {
        let ty = builder.get_list_type();
        let value_attr = builder.get_named_attr("value", builder.get_seq_attr(elements));
        Self::create(builder, loc, "eir.constant.list", value_attr, ty)
    }

    fn create_map(builder: &mut OpBuilder, loc: Location, elements: &[Attribute]) -> Self {
        let ty = builder.get_map_type();
        let value_attr = builder.get_named_attr("value", builder.get_seq_attr(elements));
        Self::create(builder, loc, "eir.constant.map", value_attr, ty)
    }

    fn create(builder: &mut OpBuilder, loc: Location, name: &'str, attr: Attribute, ty: Type) -> Self {
        let mut state = OperationState::new(name, loc);
        state.add_results(&[ty]);
        state.add_attributes(&[attr]);
        Self(builder.create_operation(state))
    }
}

#[operation("eir.cons")]
pub struct ConsOp;
impl ConsOp {
    #[inline(always)]
    pub fn result(&self) -> Value {
        self.get_result(0).unwrap()
    }

    crate fn create(builder: &mut OpBuilder, loc: Location, head: Value, tail: Value) -> Self {
        let box_cons_ty = builder.get_box_type(builder.get_cons_type());
        let mut state = Self::build(loc);
        state.add_operands(&[head, tail]);
        state.add_results(&[box_cons_ty]);
        Self(builder.create_operation(state))
    }
}

#[operation("eir.list")]
pub struct ListOp;
impl ListOp {
    #[inline(always)]
    pub fn result(&self) -> Value {
        self.get_result(0).unwrap()
    }

    crate fn create(builder: &mut OpBuilder, loc: Location, elements: &[Value]) -> Self {
        let box_cons_ty = builder.get_box_type(builder.get_cons_type());
        let mut state = Self::build(loc);
        state.add_operands(elements);
        state.add_results(&[box_cons_ty]);
        Self(builder.create_operation(state))
    }
}

#[operation("eir.tuple")]
pub struct TupleOp;
impl TupleOp {
    #[inline(always)]
    pub fn result(&self) -> Value {
        self.get_result(0).unwrap()
    }

    crate fn create(builder: &mut OpBuilder, loc: Location, elements: &[Value]) -> Self {
        let element_types = elements.map(|e| e.get_type()).collect::<Vec<_>>();
        let box_tuple_ty = builder.get_box_type(builder.get_tuple_type(&element_types));
        let mut state = Self::build(loc);
        state.add_operands(elements);
        state.add_results(&[box_tuple_ty]);
        Self(builder.create_operation(state))
    }
}

#[operation("eir.trace.capture")]
pub struct TraceCaptureOp;
impl TraceCaptureOp {
    #[inline(always)]
    pub fn result(&self) -> Value {
        self.get_result(0).unwrap()
    }

    crate fn create(builder: &mut OpBuilder, loc: Location) -> Self {
        let trace_ref_ty = builder.get_trace_ref_type();
        let mut state = Self::build(loc);
        state.add_results(&[box_trace_ref_ty]);
        Self(builder.create_operation(state))
    }
}

#[operation("eir.trace.print")]
pub struct TracePrintOp;
impl TracePrintOp {
    crate fn create(builder: &mut OpBuilder, loc: Location, kind: Value, reason: Value, trace: Value) -> Self {
        let mut state = Self::build(loc);
        state.add_operands(&[kind, reason, trace]);
        Self(builder.create_operation(state))
    }
}

#[operation("eir.trace.construct")]
pub struct TraceConstructOp;
impl TraceConstructOp {
    #[inline(always)]
    pub fn result(&self) -> Value {
        self.get_result(0).unwrap()
    }

    crate fn create(builder: &mut OpBuilder, loc: Location, trace: Value) -> Self {
        let box_cons_ty = builder.get_box_type(builder.get_cons_type());
        let mut state = Self::build(loc);
        state.add_operands(&[trace]);
        state.add_results(&[box_cons_ty]);
        Self(builder.create_operation(state))
    }
}

#[operation("eir.map.new")]
pub struct MapOp;
impl MapOp {
    #[inline(always)]
    pub fn result(&self) -> Value {
        self.get_result(0).unwrap()
    }

    crate fn create(builder: &mut OpBuilder, loc: Location, elements: &[Value]) -> Self {
        let box_map_ty = builder.get_box_type(builder.get_map_type());
        let mut state = Self::build(loc);
        state.add_operands(elements);
        state.add_results(&[box_map_ty]);
        Self(builder.create_operation(state))
    }
}

#[operation("eir.map.insert")]
pub struct MapInsertOp;
impl MapInsertOp {
    #[inline(always)]
    pub fn map(&self) -> Value {
        self.get_result(0).unwrap()
    }

    #[inline(always)]
    pub fn err(&self) -> Value {
        self.get_result(1).unwrap()
    }

    crate fn create(builder: &mut OpBuilder, loc: Location, map: Value, key: Value, value: Value) -> Self {
        let box_map_ty = builder.get_box_type(builder.get_map_type());
        let term_ty = builder.get_term_type();
        let mut state = Self::build(loc);
        state.add_operands(&[map, key, value]);
        state.add_results(&[box_map_ty, term_ty]);
        Self(builder.create_operation(state))
    }
}

#[operation("eir.map.update")]
pub struct MapUpdateOp;
impl MapUpdateOp {
    #[inline(always)]
    pub fn map(&self) -> Value {
        self.get_result(0).unwrap()
    }

    #[inline(always)]
    pub fn err(&self) -> Value {
        self.get_result(1).unwrap()
    }

    crate fn create(builder: &mut OpBuilder, loc: Location, map: Value, key: Value, value: Value) -> Self {
        let box_map_ty = builder.get_box_type(builder.get_map_type());
        let term_ty = builder.get_term_type();
        let mut state = Self::build(loc);
        state.add_operands(&[map, key, value]);
        state.add_results(&[box_map_ty, term_ty]);
        Self(builder.create_operation(state))
    }
}

#[operation("eir.map.contains")]
pub struct MapContainsKeyOp;
impl MapContainsKeyOp {
    #[inline(always)]
    pub fn result(&self) -> Value {
        self.get_result(0).unwrap()
    }

    crate fn create(builder: &mut OpBuilder, loc: Location, map: Value, key: Value, value: Value) -> Self {
        let i1ty = builder.get_i1_type();
        let mut state = Self::build(loc);
        state.add_operands(&[map, key]);
        state.add_results(&[i1ty]);
        Self(builder.create_operation(state))
    }
}

#[operation("eir.map.get")]
pub struct MapGetKeyOp;
impl MapGetKeyOp {
    #[inline(always)]
    pub fn result(&self) -> Value {
        self.get_result(0).unwrap()
    }

    crate fn create(builder: &mut OpBuilder, loc: Location, map: Value, key: Value, value: Value) -> Self {
        let term_ty = builder.get_term_type();
        let mut state = Self::build(loc);
        state.add_operands(&[map, key]);
        state.add_results(&[term_ty]);
        Self(builder.create_operation(state))
    }
}

#[operation("eir.binary.start")]
pub struct BinaryStartOp;
impl BinaryStartOp {
    #[inline(always)]
    pub fn result(&self) -> Value {
        self.get_result(0).unwrap()
    }

    crate fn create(builder: &mut OpBuilder, loc: Location) -> Self {
        let term_ty = builder.get_term_type();
        let mut state = Self::build(loc);
        state.add_results(&[term_ty]);
        Self(builder.create_operation(state))
    }
}

#[operation("eir.binary.finish")]
pub struct BinaryFinishOp;
impl BinaryFinishOp {
    #[inline(always)]
    pub fn result(&self) -> Value {
        self.get_result(0).unwrap()
    }

    crate fn create(builder: &mut OpBuilder, loc: Location, bin: Value) -> Self {
        let binary_ty = builder.get_binary_type();
        let box_bin_ty = builder.get_box_type(binary_ty);
        let mut state = Self::build(loc);
        state.add_operands(&[bin]);
        state.add_results(&[box_bin_ty]);
        Self(builder.create_operation(state))
    }
}

#[operation("eir.binary.push")]
pub struct BinaryPushOp;
impl BinaryPushOp {
    #[inline(always)]
    pub fn bin(&self) -> Value {
        self.get_result(0).unwrap()
    }

    #[inline(always)]
    pub fn push_succeeded(&self) -> Value {
        self.get_result(1).unwrap()
    }

    crate fn create(builder: &mut OpBuilder, loc: Location, bin: Value, value: Value, size: Option<Value>) -> Self {
        let i1ty = builder.get_i1_type();
        let term_ty = builder.get_term_type();
        let mut state = Self::build(loc);
        if let Some(size) = size {
            state.add_operands(&[bin, value, size]);
        } else {
            state.add_operands(&[bin, value]);
        }
        state.add_results(&[term_ty, i1ty]);
        Self(builder.create_operation(state))
    }
}

#[operation("eir.binary.match.raw")]
pub struct BinaryMatchRawOp;
impl BinaryMatchRawOp {
    #[inline(always)]
    pub fn matched(&self) -> Value {
        self.get_result(0).unwrap()
    }

    #[inline(always)]
    pub fn rest(&self) -> Value {
        self.get_result(1).unwrap()
    }

    #[inline(always)]
    pub fn match_succeeded(&self) -> Value {
        self.get_result(2).unwrap()
    }

    crate fn create(builder: &mut OpBuilder, loc: Location, bin: Value, unit: usize, size: Option<Value>) -> Self {
        let i1ty = builder.get_i1_type();
        let term_ty = builder.get_term_type();
        let unit_attr = builder.get_named_attr("unit", builder.get_i8_attr(unit));
        let mut state = Self::build(loc);
        if let Some(size) = size {
            state.add_operands(&[bin, size]);
        } else {
            state.add_operands(&[bin]);
        }
        state.add_results(&[term_ty, term_ty, i1ty]);
        Self(builder.create_operation(state))
    }
}

#[operation("eir.binary.match.integer")]
pub struct BinaryMatchIntegerOp;
impl BinaryMatchIntegerOp {
    #[inline(always)]
    pub fn matched(&self) -> Value {
        self.get_result(0).unwrap()
    }

    #[inline(always)]
    pub fn rest(&self) -> Value {
        self.get_result(1).unwrap()
    }

    #[inline(always)]
    pub fn match_succeeded(&self) -> Value {
        self.get_result(2).unwrap()
    }

    crate fn create(builder: &mut OpBuilder, loc: Location, bin: Value, signed: bool, endianness: Endianness, unit: usize, size: Option<Value>) -> Self {
        let i1ty = builder.get_i1_type();
        let term_ty = builder.get_term_type();
        let unit_attr = builder.get_named_attr("unit", builder.get_i8_attr(unit));
        let is_signed_attr = builder.get_named_attr("is_signed", builder.get_bool_attr(signed));
        let endianness_attr = builder.get_named_attr("endianness", builder.get_i8_attr(endianness.into()));
        let mut state = Self::build(loc);
        if let Some(size) = size {
            state.add_operands(&[bin, size]);
        } else {
            state.add_operands(&[bin]);
        }
        state.add_results(&[term_ty, term_ty, i1ty]);
        Self(builder.create_operation(state))
    }
}

#[operation("eir.binary.match.float")]
pub struct BinaryMatchFloatOp;
impl BinaryMatchFloatOp {
    #[inline(always)]
    pub fn matched(&self) -> Value {
        self.get_result(0).unwrap()
    }

    #[inline(always)]
    pub fn rest(&self) -> Value {
        self.get_result(1).unwrap()
    }

    #[inline(always)]
    pub fn match_succeeded(&self) -> Value {
        self.get_result(2).unwrap()
    }

    crate fn create(builder: &mut OpBuilder, loc: Location, bin: Value, endianness: Endianness, unit: usize, size: Option<Value>) -> Self {
        let i1ty = builder.get_i1_type();
        let term_ty = builder.get_term_type();
        let unit_attr = builder.get_named_attr("unit", builder.get_i8_attr(unit));
        let endianness_attr = builder.get_named_attr("endianness", builder.get_i8_attr(endianness.into()));
        let mut state = Self::build(loc);
        if let Some(size) = size {
            state.add_operands(&[bin, size]);
        } else {
            state.add_operands(&[bin]);
        }
        state.add_results(&[term_ty, term_ty, i1ty]);
        Self(builder.create_operation(state))
    }
}

#[operation("eir.binary.match.utf8")]
pub struct BinaryMatchUtf8Op;
impl BinaryMatchUtf8Op {
    #[inline(always)]
    pub fn matched(&self) -> Value {
        self.get_result(0).unwrap()
    }

    #[inline(always)]
    pub fn rest(&self) -> Value {
        self.get_result(1).unwrap()
    }

    #[inline(always)]
    pub fn match_succeeded(&self) -> Value {
        self.get_result(2).unwrap()
    }

    crate fn create(builder: &mut OpBuilder, loc: Location, bin: Value, size: Option<Value>) -> Self {
        let i1ty = builder.get_i1_type();
        let term_ty = builder.get_term_type();
        let mut state = Self::build(loc);
        if let Some(size) = size {
            state.add_operands(&[bin, size]);
        } else {
            state.add_operands(&[bin]);
        }
        state.add_results(&[term_ty, term_ty, i1ty]);
        Self(builder.create_operation(state))
    }
}

#[operation("eir.binary.match.utf16")]
pub struct BinaryMatchUtf16Op;
impl BinaryMatchUtf16Op {
    #[inline(always)]
    pub fn matched(&self) -> Value {
        self.get_result(0).unwrap()
    }

    #[inline(always)]
    pub fn rest(&self) -> Value {
        self.get_result(1).unwrap()
    }

    #[inline(always)]
    pub fn match_succeeded(&self) -> Value {
        self.get_result(2).unwrap()
    }

    crate fn create(builder: &mut OpBuilder, loc: Location, bin: Value, endianness: Endianness, size: Option<Value>) -> Self {
        let i1ty = builder.get_i1_type();
        let term_ty = builder.get_term_type();
        let endianness_attr = builder.get_named_attr("endianness", builder.get_i8_attr(endianness.into()));
        let mut state = Self::build(loc);
        if let Some(size) = size {
            state.add_operands(&[bin, size]);
        } else {
            state.add_operands(&[bin]);
        }
        state.add_results(&[term_ty, term_ty, i1ty]);
        Self(builder.create_operation(state))
    }
}

#[operation("eir.binary.match.utf16")]
pub struct BinaryMatchUtf32Op;
impl BinaryMatchUtf32Op {
    #[inline(always)]
    pub fn matched(&self) -> Value {
        self.get_result(0).unwrap()
    }

    #[inline(always)]
    pub fn rest(&self) -> Value {
        self.get_result(1).unwrap()
    }

    #[inline(always)]
    pub fn match_succeeded(&self) -> Value {
        self.get_result(2).unwrap()
    }

    crate fn create(builder: &mut OpBuilder, loc: Location, bin: Value, endianness: Endianness, size: Option<Value>) -> Self {
        let i1ty = builder.get_i1_type();
        let term_ty = builder.get_term_type();
        let endianness_attr = builder.get_named_attr("endianness", builder.get_i8_attr(endianness.into()));
        let mut state = Self::build(loc);
        if let Some(size) = size {
            state.add_operands(&[bin, size]);
        } else {
            state.add_operands(&[bin]);
        }
        state.add_results(&[term_ty, term_ty, i1ty]);
        Self(builder.create_operation(state))
    }
}

#[operation("eir.receive.start")]
pub struct ReceiveStartOp;
impl ReceiveStartOp {
    #[inline(always)]
    pub fn result(&self) -> Value {
        self.get_result(0).unwrap()
    }

    crate fn create(builder: &mut OpBuilder, loc: Location, timeout: Value) -> Self {
        let recv_ref_type = builder.get_receive_ref_type();
        let mut state = Self::build(loc);
        state.add_operands(&[timeout]);
        state.add_results(&[recv_ref_type]);
        Self(builder.create_operation(state))
    }
}

#[operation("eir.receive.wait")]
pub struct ReceiveWaitOp;
impl ReceiveWaitOp {
    #[inline(always)]
    pub fn result(&self) -> Value {
        self.get_result(0).unwrap()
    }

    crate fn create(builder: &mut OpBuilder, loc: Location, recv: Value) -> Self {
        let recv_status_type = builder.get_i8_type();
        let mut state = Self::build(loc);
        state.add_operands(&[recv]);
        state.add_results(&[recv_status_type]);
        Self(builder.create_operation(state))
    }
}

#[operation("eir.receive.message")]
pub struct ReceiveMessageOp;
impl ReceiveMessageOp {
    #[inline(always)]
    pub fn result(&self) -> Value {
        self.get_result(0).unwrap()
    }

    crate fn create(builder: &mut OpBuilder, loc: Location, recv: Value) -> Self {
        let term_ty = builder.get_term_type();
        let mut state = Self::build(loc);
        state.add_operands(&[recv]);
        state.add_results(&[term_ty]);
        Self(builder.create_operation(state))
    }
}

#[operation("eir.receive.done")]
pub struct ReceiveDoneOp;
impl ReceiveDoneOp {
    crate fn create(builder: &mut OpBuilder, loc: Location, recv: Value) -> Self {
        let mut state = Self::build(loc);
        state.add_operands(&[recv]);
        Self(builder.create_operation(state))
    }
}

#[inline(always)]
fn create_unary_operator_op(
    builder: &mut OpBuilder,
    state: OperationState,
    input: Value,
    ty: Type,
) -> Operation {
    state.add_operands(&[input]);
    state.add_results(&[ty]);
    builder.create_operation(state)
}

#[inline(always)]
fn create_binary_operator_op(
    builder: &mut OpBuilder,
    state: OperationState,
    lhs: Value,
    rhs: Value,
    ty: Type,
) -> Operation {
    state.add_operands(&[lhs, rhs]);
    state.add_results(&[ty]);
    builder.create_operation(state)
}
