#![allow(unused)]

use std::ffi::CString;
use std::fmt;
use std::ptr;

use anyhow::anyhow;

use cranelift_entity::packed_option::PackedOption;
use cranelift_entity::{PrimaryMap, SecondaryMap};

use log::debug;

use libeir_ir as ir;
use libeir_ir::FunctionIdent;

use liblumen_core::symbols::FunctionSymbol;
use liblumen_mlir as mlir;
use liblumen_mlir::symbols::Visibility;
use liblumen_util::diagnostics::SourceSpan;

use crate::builder::block::*;
use crate::builder::dialect::eir::*;
use crate::builder::ffi::{self, Span};
use crate::builder::value::*;
use crate::builder::ModuleBuilder;
use crate::Result;

pub struct Function {
    // MFA of this function
    name: ir::FunctionIdent,
    pub(super) span: SourceSpan,
    // Signature of this function
    signature: Signature,
    visibility: Visibility,
    // Primary storage for blocks
    pub(super) blocks: PrimaryMap<Block, BlockData>,
    // Mapping of EIR blocks to MLIR blocks
    pub(super) block_mapping: SecondaryMap<ir::Block, Block>,
    // Primary storage for MLIR values -> value metadata
    pub(super) values: PrimaryMap<Value, ValueData>,
    // Mapping of EIR values to MLIR values by block
    //
    // Access is of the form "given value V from EIR, give me the MLIR value
    // which corresponds to that value in block B". This question is necessary
    // since MLIR requires values to be passed explicitly to successor blocks as
    // arguments, while EIR does not. Each time a value is used as an implicit argument
    // in EIR, we extend the block argument list in MLIR with a new value, and track
    // the connection between that new argument and its original definition. When we
    // encounter a usage of a value in EIR, we can do the following:
    //
    // - Find the value in the current block which represents the EIR value
    // - Efficiently look up the value metadata from the definition
    // - Perform data flow analysis and type synthesis
    pub(super) value_mapping: SecondaryMap<ir::Value, SecondaryMap<Block, PackedOption<Value>>>,

    ret: Value,
    ret_ir: ir::Value,
    esc: Value,
    esc_ir: ir::Value,
}
impl Function {
    /// Initializes this function definition with the given name and signature
    ///
    /// NOTE: This does not construct the function in MLIR
    pub fn with_name_signature(
        span: SourceSpan,
        name: FunctionIdent,
        signature: Signature,
        visibility: Visibility,
    ) -> Self {
        Self {
            name,
            span,
            signature,
            visibility,
            blocks: PrimaryMap::new(),
            block_mapping: SecondaryMap::new(),
            values: PrimaryMap::new(),
            value_mapping: SecondaryMap::with_default(SecondaryMap::new()),
            ret: Default::default(),
            ret_ir: Default::default(),
            esc: Default::default(),
            esc_ir: Default::default(),
        }
    }

    /// Sets this functions return continuation
    pub fn set_return_continuation(&mut self, ret: ir::Value, entry: Block) -> Value {
        let value = self.values.push(ValueData {
            value: Default::default(),
            ir_value: PackedOption::from(ret),
            data: ValueDef::Return,
        });
        self.value_mapping[ret][entry] = PackedOption::from(value);
        self.ret_ir = ret;
        self.ret = value;
        self.ret
    }

    /// Sets this functions escape continuation
    pub fn set_escape_continuation(&mut self, esc: ir::Value, entry: Block) -> Value {
        let value = self.values.push(ValueData {
            value: Default::default(),
            ir_value: PackedOption::from(esc),
            data: ValueDef::Escape,
        });
        self.value_mapping[esc][entry] = PackedOption::from(value);
        self.esc_ir = esc;
        self.esc = value;
        self.esc
    }

    /// Constructs this function definition in MLIR
    pub fn build<'m>(
        &self,
        builder: &mut EirBuilder<'m>,
        module_info: &mut ModuleInfo,
    ) -> Result<(FuncOp, mlir::ir::Block)> {
        // Create function in MLIR
        debug!(
            "creating function {} with {} parameters and return types {:?}",
            &self.name,
            self.signature.inputs.len(),
            self.signature.outputs.len(),
        );

        let loc = module_info
            .location_from_index(self.span.start().index())
            .expect("missing source location for function");
        let name = self.name.to_string();
        let ty = builder.get_function_type(&self.signature.inputs, &self.signature.outputs);
        let op = builder.build_func(loc, &name, ty, self.visibility, &self.attrs)?;
        let block = op.entry_block();

        // Register function symbol globally
        module_info.symbols_mut().insert(FunctionSymbol {
            module: self.name.module.name.as_usize(),
            function: self.name.name.name.as_usize(),
            arity: self.name.arity as u8,
            ptr: ptr::null(),
        });

        Ok((op, entry_block))
    }

    /// Returns a reference to the current functions' identifier
    #[inline]
    pub fn name(&self) -> &ir::FunctionIdent {
        &self.name
    }

    /// Returns true if the given value is this functions' return continuation
    #[inline]
    pub fn is_return(&self, value: Value, block: Block) -> bool {
        if let Some(expected) = self.value_mapping[self.ret_ir][block].expand() {
            value == expected
        } else {
            false
        }
    }

    /// Returns true if the given EIR value is this functions' return continuation
    #[inline]
    pub fn is_return_ir(&self, value: ir::Value) -> bool {
        self.ret_ir == value
    }

    /// Returns true if the given value is this functions' escape continuation
    #[inline]
    pub fn is_throw(&self, value: Value, block: Block) -> bool {
        if let Some(expected) = self.value_mapping[self.esc_ir][block].expand() {
            value == expected
        } else {
            false
        }
    }

    /// Returns true if the given EIR value is this functions' escape continuation
    #[inline]
    pub fn is_throw_ir(&self, value: ir::Value) -> bool {
        self.esc_ir == value
    }

    /// Creates a new block in this function, optionally tracked against the given EIR block
    ///
    /// The block is intantiated with an empty parameter list
    pub fn new_block(&mut self, ir_block: Option<ir::Block>, block: mlir::ir::Block) -> Block {
        self.new_block_with_params(ir_block, block, &[])
    }

    /// Same as `new_block`, but takes parameter metadata which specifies which arguments the
    /// block takes, and how they map back to the original EIR
    pub fn new_block_with_params(
        &mut self,
        ir_block: Option<ir::Block>,
        block: mlir::ir::Block,
        params: &[Option<ir::Value>],
    ) -> Block {
        // Construct the block so we get a block handle to use
        let block_handle = self.blocks.push(BlockData {
            block,
            ir_block: PackedOption::from(ir_block),
        });

        // Update the block data with the mapped parameter values
        for (i, ir_value) in params.iter().enumerate() {
            let arg = block
                .get_argument(i)
                .expect("unexpected mismatch between block parameter lists");
            let value = self.new_value(block_handle, ir_value, arg.into(), ValueDef::Param(i));
            if let Some(irv) = ir_value {
                // Register EIR value in this block
                self.value_mapping[irv][block_handle] = PackedOption::from(value);
            }
        }

        // Map the original EIR block to the MLIR block
        if let Some(irb) = ir_block {
            self.block_mapping[irb] = block_handle;
        }
        block_handle
    }

    /// Returns the block data for the given block
    #[inline]
    pub fn block_data(&self, block_handle: Block) -> &BlockData {
        assert!(
            self.blocks.is_valid(block_handle),
            "invalid block requested {:?}",
            block_handle
        );
        self.blocks.get(block_handle).unwrap()
    }

    /// Returns the block data for the given block mutably
    #[inline]
    pub fn block_data_mut(&mut self, block_handle: Block) -> &mut BlockData {
        assert!(
            self.blocks.is_valid(block_handle),
            "invalid block requested {:?}",
            block_handle
        );
        self.blocks.get_mut(block_handle).unwrap()
    }

    /// Returns the arity of the given block
    pub fn block_arity(&self, block_handle: Block) -> usize {
        self.block_to_block_ref(block_handle).num_arguments()
    }

    /// Maps the given block back to its original EIR block
    pub fn block_to_ir_block(&self, block_handle: Block) -> Option<ir::Block> {
        self.block_data(block_handle).ir_block.into()
    }

    /// Maps the given block to its underlying MLIR block reference
    #[inline]
    pub fn block_to_block_ref(&self, block_handle: Block) -> mlir::ir::Block {
        self.block_data(block_handle).block
    }

    /// Returns the parameter info for the given block
    pub fn block_argument(&self, block_handle: Block, index: usize) -> Option<BlockArgument> {
        self.block_to_block_ref(block_handle).get_argument(index)
    }

    /// Registers a new value in the given block, optionally tracking an EIR value
    pub fn new_value(
        &mut self,
        block_handle: Block,
        ir_value: Option<libeir_ir::Value>,
        value: mlir::ir::Value,
        data: ValueDef,
    ) -> Value {
        let result = self.values.push(ValueData {
            value,
            ir_value: PackedOption::from(ir_value),
            data,
        });
        if let Some(irv) = ir_value {
            self.value_mapping[irv][block] = PackedOption::from(value);
        }
        result
    }

    /// Returns the value metadata for the given value
    #[inline]
    pub fn value_data(&self, value_handle: Value) -> &ValueData {
        self.values.get(value_handle).unwrap()
    }

    /// Returns the value metadata for the given value, mutably
    #[inline]
    pub fn value_data_mut(&mut self, value_handle: Value) -> &mut ValueData {
        self.values.get_mut(value_handle).unwrap()
    }

    /// Returns the EIR value the given value corresponds to
    pub fn value_to_ir_value(&self, value_handle: Value) -> Option<ir::Value> {
        self.value_data(value_handle).ir_value.into()
    }

    /// Returns the MLIR value reference for the given value
    pub fn value_to_value_ref(&self, value_handle: Value) -> mlir::ir::Value {
        self.value_data(value_handle).value
    }

    /// Returns the ValueDef structure for the given value
    pub fn value_to_value_def(&self, value_handle: Value) -> ValueDef {
        self.value_data(value_handle).data
    }
}

/// Function signature.
///
/// The function signature describes the types of formal parameters and return values along with
/// other details that are needed to call a function correctly.
///
/// A signature can optionally include ISA-specific ABI information which specifies exactly how
/// arguments and return values are passed.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Signature {
    /// The arguments passed to the function.
    pub inputs: Vec<mlir::ir::Type>,
    /// Values returned from the function.
    pub outputs: Vec<mlir::ir::Type>,
    /// Calling convention.
    pub call_conv: CallConv,
}
impl Signature {
    pub fn new(cc: CallConv) -> Self {
        Self {
            inputs: vec![],
            outputs: vec![],
            call_conv: cc,
        }
    }
}

#[derive(Debug, Copy, Clone, PartialEq, Eq, Hash)]
pub enum CallConv {
    Fast,
    Rust,
    C,
}
impl Default for CallConv {
    fn default() -> Self {
        Self::Fast
    }
}
