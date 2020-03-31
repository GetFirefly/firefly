use super::*;

pub struct IsTypeBuilder;

impl IsTypeBuilder {
    pub fn build<'f, 'o>(
        builder: &mut ScopedFunctionBuilder<'f, 'o>,
        ir_value: Option<ir::Value>,
        op: IsType,
    ) -> Result<Option<Value>> {
        let loc = op.loc;
        let builder_ref = builder.as_ref();
        let value_ref = builder.value_ref(op.value);
        let result_ref = match op.expected {
            Type::Tuple(arity) => unsafe {
                MLIRBuildIsTypeTupleWithArity(builder_ref, loc, value_ref, arity)
            },
            Type::List => unsafe { MLIRBuildIsTypeList(builder_ref, loc, value_ref) },
            Type::Cons => unsafe { MLIRBuildIsTypeNonEmptyList(builder_ref, loc, value_ref) },
            Type::Nil => unsafe { MLIRBuildIsTypeNil(builder_ref, loc, value_ref) },
            Type::Map => unsafe { MLIRBuildIsTypeMap(builder_ref, loc, value_ref) },
            Type::Number => unsafe { MLIRBuildIsTypeNumber(builder_ref, loc, value_ref) },
            Type::Float => unsafe { MLIRBuildIsTypeFloat(builder_ref, loc, value_ref) },
            Type::Integer => unsafe { MLIRBuildIsTypeInteger(builder_ref, loc, value_ref) },
            Type::Fixnum => unsafe { MLIRBuildIsTypeFixnum(builder_ref, loc, value_ref) },
            Type::BigInt => unsafe { MLIRBuildIsTypeBigInt(builder_ref, loc, value_ref) },
            _ => unreachable!("unsupported type used in is_type operation"),
        };

        assert!(!result_ref.is_null());

        let result = builder.new_value(ir_value, result_ref, ValueDef::Result(0));
        Ok(Some(result))
    }
}

pub struct MatchBuilder<'a, 'f, 'o> {
    builder: &'a mut ScopedFunctionBuilder<'f, 'o>,
    arglists: Vec<Vec<ValueRef>>,
}

macro_rules! assert_no_arguments {
    ($args:expr, $msg:expr) => {
        if !$args.is_empty() {
            return Err(anyhow!(
                "expected {} to have no arguments, but got {}",
                $msg,
                $args.len()
            ));
        }
    };
}

macro_rules! assert_optional_argument {
    ($builder:expr, $args:ident, $msg:expr) => {
        match $args.len() {
            0 => {
                debug_in!($builder, "pattern has no arguments");
                Default::default()
            }
            1 => {
                let arg = $args[0];
                debug_in!($builder, "pattern has one argument: {:?}", arg);
                let v = $builder.build_value(arg)?;
                $builder.value_ref(v)
            }
            n => {
                return Err(anyhow!(
                    "expected {} to have no more than 1 argument, but got {}",
                    $msg,
                    n
                ))
            }
        }
    };
}

macro_rules! assert_single_argument {
    ($builder:expr, $args:ident, $msg:expr) => {
        if $args.len() != 1 {
            return Err(anyhow!(
                "expected {} to have 1 argument, but got {}",
                $msg,
                $args.len()
            ));
        } else {
            let arg = $args[0];
            debug_in!($builder, "pattern argument is {:?}", arg);
            let v = $builder.build_value(arg)?;
            $builder.value_ref(v)
        }
    };
}

impl<'a, 'f, 'o> MatchBuilder<'a, 'f, 'o> {
    /// Lowers an EIR match operation to its MLIR equivalent
    pub fn build(
        builder: &'a mut ScopedFunctionBuilder<'f, 'o>,
        op: Match,
    ) -> Result<Option<Value>> {
        let this = MatchBuilder {
            builder,
            arglists: Vec::with_capacity(op.branches.len()),
        };
        this.do_build(op)
    }

    fn do_build(mut self, mut op: Match) -> Result<Option<Value>> {
        debug_in!(self.builder, "building match op");
        debug_assert!(
            op.branches.len() > 0,
            "invalid match operation, no patterns defined"
        );

        // Get match inputs
        let selector = self.builder.value_ref(op.selector);
        debug_in!(self.builder, "selector is {:?}", op.selector);
        let reads = op.reads.as_slice();
        debug_in!(self.builder, "reads are {:?}", reads);
        let mut branches = Vec::with_capacity(op.branches.len());
        for Pattern {
            loc,
            kind,
            block,
            args,
        } in op.branches.drain(..)
        {
            debug_in!(
                self.builder,
                "lowering pattern ({:?}) for block {:?}",
                kind,
                block
            );
            branches.push(self.translate_branch_kind(loc, kind, block, args.as_slice(), reads)?);
        }

        let match_op = MatchOp {
            loc: op.loc,
            selector,
            branches: branches.as_ptr(),
            num_branches: branches.len() as libc::c_uint,
        };

        unsafe { MLIRBuildMatchOp(self.builder.as_ref(), match_op) }

        Ok(None)
    }

    fn translate_branch_kind(
        &mut self,
        loc: LocationRef,
        kind: ir::MatchKind,
        block: Block,
        args: &[ir::Value],
        reads: &[ir::Value],
    ) -> Result<MatchBranch> {
        use ir::MatchKind;

        // Translate the pattern type for MLIR
        let pattern = match kind {
            MatchKind::Value => {
                let expected = assert_single_argument!(self.builder, args, "value pattern");
                MatchPattern::Value(expected)
            }
            MatchKind::Type(ty) => {
                assert_no_arguments!(args, "type pattern");
                MatchPattern::IsType(ty.into())
            }
            MatchKind::Binary(ref spec) => {
                let size = assert_optional_argument!(self.builder, args, "binary pattern");
                MatchPattern::Binary {
                    size,
                    spec: spec.into(),
                }
            }
            MatchKind::Tuple(arity) => {
                assert_no_arguments!(args, "type pattern");
                MatchPattern::Tuple(arity as libc::c_uint)
            }
            MatchKind::ListCell => {
                assert_no_arguments!(args, "type pattern");
                MatchPattern::Cons
            }
            MatchKind::MapItem => {
                let key = assert_single_argument!(self.builder, args, "map pattern");
                MatchPattern::MapItem(key)
            }
            MatchKind::Wildcard => {
                assert_no_arguments!(args, "wildcard pattern");
                MatchPattern::Any
            }
        };

        debug_in!(
            self.builder,
            "pattern is valid, building successor block arguments"
        );

        // Move ownership of block arguments vector to builder
        let arglist = self
            .builder
            .build_target_block_args(block, reads)
            .drain(..)
            .map(|v| self.builder.value_ref(v))
            .collect::<Vec<_>>();
        debug_in!(
            self.builder,
            "destination args for {:?}: {:?}",
            &pattern,
            &arglist
        );
        self.arglists.push(arglist);

        // Then store the argc/argv in the branch
        let args = self.arglists.last().unwrap();
        let dest_argv = args.as_ptr();
        let dest_argc = args.len() as libc::c_uint;

        Ok(MatchBranch {
            loc,
            dest: self.builder.block_ref(block),
            dest_argv,
            dest_argc,
            pattern,
        })
    }
}
