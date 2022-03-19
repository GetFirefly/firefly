use std::ffi::c_void;
use std::fmt::{self, Display};

use super::*;
use crate::support::{self, MlirStringCallback};
use crate::Context;

extern "C" {
    type MlirAffineMap;
}

#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct AffineMap(*mut MlirAffineMap);
impl AffineMap {
    /// Creates a zero-result affine map with no dimensions or symbols
    pub fn get_empty(context: Context) -> Self {
        unsafe { mlir_affine_map_empty_get(context) }
    }

    /// Creates a zero-result affine map with the given dimensions and symbols
    pub fn get_zero_result(context: Context, dims: usize, symbols: usize) -> Self {
        unsafe { mlir_affine_map_zero_result_get(context, dims, symbols) }
    }

    /// Creates an affine map with results defined by the given list of affine expressions.
    /// The resulting map also has the requested number of input dimensions and symbols,
    /// regardless of them being used in the results.
    pub fn get(context: Context, dims: usize, symbols: usize, exprs: &[AffineExprBase]) -> Self {
        unsafe { mlir_affine_map_get(context, dims, symbols, exprs.len(), exprs.as_ptr()) }
    }

    /// Creates a single constant result affine map
    pub fn get_constant(context: Context, value: u64) -> Self {
        unsafe { mlir_affine_map_constant_get(context, value) }
    }

    /// Creates an affine map with `dims` identity
    pub fn get_multi_dim_identity(context: Context, dims: usize) -> Self {
        unsafe { mlir_affine_map_multi_dim_identity_get(context, dims) }
    }

    /// Creates an identity affine map on the most minor dimensions.
    /// This function will panic if the number of dimensions is greater or equal to the number of results
    pub fn get_minor_identity(context: Context, dims: usize, results: usize) -> Self {
        assert!(
            dims < results,
            "the number of dimensions must be less than the number of results"
        );
        unsafe { mlir_affine_map_minor_identity_get(context, dims, results) }
    }

    /// Creates an affine map with a permutation expression and its size.
    ///
    /// The permutation expression is a non-empty vector of integers.
    /// The elements of the permutation vector must be continuous from 0 and cannot be repeated,
    /// i.e. `[1, 2, 0]` is a valid permutation, `[2, 0]` or `[1, 1, 2]` are not.
    pub fn get_permutation(context: Context, permutation: Permutation) -> Self {
        unsafe { mlir_affine_map_permutation_get(context, permutation.len(), permutation.as_ptr()) }
    }

    /// Returns true if this is an identity affine map
    pub fn is_identity(self) -> bool {
        unsafe { mlir_affine_map_is_identity(self) }
    }

    /// Returns true if this is a minor identity affine map
    pub fn is_minor_identity(self) -> bool {
        unsafe { mlir_affine_map_is_minor_identity(self) }
    }

    /// Returns true if this is an empty affine map
    pub fn is_empty(self) -> bool {
        unsafe { mlir_affine_map_is_empty(self) }
    }

    /// Returns true if this is a single-result constant affine map
    pub fn is_single_constant(self) -> bool {
        unsafe { mlir_affine_map_is_single_constant(self) }
    }

    /// Returns the constant result of this affine map.
    ///
    /// NOTE: This function will panic if the underlying affine map is not constant
    pub fn get_constant_result(self) -> u64 {
        assert!(
            self.is_single_constant(),
            "cannot get constant result from non-constant affine map"
        );
        unsafe { mlir_affine_map_get_single_constant_result(self) }
    }

    /// Returns the number of dimensions
    pub fn num_dimensions(self) -> usize {
        unsafe { mlir_affine_map_get_num_dims(self) }
    }

    /// Returns the number of symbols
    pub fn num_symbols(self) -> usize {
        unsafe { mlir_affine_map_get_num_symbols(self) }
    }

    /// Returns the number of results
    pub fn num_results(self) -> usize {
        unsafe { mlir_affine_map_get_num_results(self) }
    }

    /// Returns the number of inputs (dimensions + symbols)
    pub fn num_inputs(self) -> usize {
        unsafe { mlir_affine_map_get_num_inputs(self) }
    }

    /// Returns the result at the given position
    pub fn result(self, index: usize) -> AffineExprBase {
        unsafe { mlir_affine_map_get_result(self, index) }
    }

    /// Returns true if this affine map represents a subset of a symbol-less permutation map
    pub fn is_projected_permutation(self) -> bool {
        unsafe { mlir_affine_map_is_projected_permutation(self) }
    }

    /// Returns true if this affine map represents a symbol-less permutation map
    pub fn is_permutation(self) -> bool {
        unsafe { mlir_affine_map_is_permutation(self) }
    }

    /// Returns a new affine map consisting of the given results
    pub fn subset(self, results: &[usize]) -> Self {
        unsafe { mlir_affine_map_get_sub_map(self, results.len(), results.as_ptr()) }
    }

    /// Returns a new affine map consisting of the most major `n` results
    /// Returns a null affine map if `n` is zero
    /// Returns the input map if `n` is >= the number of its results
    pub fn get_major_subset(self, n: usize) -> Self {
        unsafe { mlir_affine_map_get_major_sub_map(self, n) }
    }

    /// Returns a new affine map consisting of the most minor `n` results
    /// Returns a null affine map if `n` is zero
    /// Returns the input map if `n` is >= the number of its results
    pub fn get_minor_subset(self, n: usize) -> Self {
        unsafe { mlir_affine_map_get_minor_sub_map(self, n) }
    }

    /// Returns a new affine map resulting from applying the given affine expression to each of the
    /// results and with the specified number of dimensions and symbols
    pub fn replace(
        self,
        expression: AffineExprBase,
        replacement: AffineExprBase,
        dims: usize,
        symbols: usize,
    ) -> Self {
        unsafe { mlir_affine_map_replace(self, expression, replacement, dims, symbols) }
    }

    /// Returns the simplified affine map resulting from dropping the symbols that do
    /// not appear in any of the individual maps in `maps`.
    ///
    /// Asserts that all maps in `maps` are normalized to the same number of dimensions and symbols.
    pub fn compress_unused_symbols(maps: &[Self]) -> Vec<Self> {
        extern "C" fn populate_result(results: &mut Vec<AffineMap>, _index: usize, map: AffineMap) {
            results.push(map)
        }
        let mut out = Vec::new();
        unsafe {
            mlir_affine_map_compress_unused_symbols(
                maps.as_ptr(),
                maps.len(),
                &mut out,
                populate_result,
            )
        }
        out
    }

    pub fn context(self) -> Context {
        unsafe { mlir_affine_map_get_context(self) }
    }

    pub fn is_null(&self) -> bool {
        self.0.is_null()
    }

    pub fn dump(self) {
        unsafe { mlir_affine_map_dump(self) }
    }
}
impl Default for AffineMap {
    fn default() -> Self {
        Self(unsafe { std::mem::transmute::<*mut (), *mut MlirAffineMap>(::core::ptr::null_mut()) })
    }
}
impl Display for AffineMap {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        unsafe {
            mlir_affine_map_print(
                *self,
                support::write_to_formatter,
                f as *mut _ as *mut c_void,
            );
        }
        Ok(())
    }
}
impl fmt::Pointer for AffineMap {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:p}", self.0)
    }
}
impl Eq for AffineMap {}
impl PartialEq for AffineMap {
    fn eq(&self, other: &Self) -> bool {
        unsafe { mlir_affine_map_equal(*self, *other) }
    }
}

extern "C" {
    #[link_name = "mlirAffineMapGetContext"]
    fn mlir_affine_map_get_context(map: AffineMap) -> Context;
    #[link_name = "mlirAffineMapEqual"]
    fn mlir_affine_map_equal(a: AffineMap, b: AffineMap) -> bool;
    #[link_name = "mlirAffineMapPrint"]
    fn mlir_affine_map_print(map: AffineMap, callback: MlirStringCallback, userdata: *const c_void);
    #[link_name = "mlirAffineMapDump"]
    fn mlir_affine_map_dump(map: AffineMap);
    #[link_name = "mlirAffineMapEmptyGet"]
    fn mlir_affine_map_empty_get(context: Context) -> AffineMap;
    #[link_name = "mlirAffineMapZeroResultGet"]
    fn mlir_affine_map_zero_result_get(context: Context, dims: usize, symbols: usize) -> AffineMap;
    #[link_name = "mlirAffineMapGet"]
    fn mlir_affine_map_get(
        context: Context,
        dims: usize,
        symbols: usize,
        num_exprs: usize,
        exprs: *const AffineExprBase,
    ) -> AffineMap;
    #[link_name = "mlirAffineMapMinorIdentityGet"]
    fn mlir_affine_map_minor_identity_get(
        context: Context,
        dims: usize,
        results: usize,
    ) -> AffineMap;
    #[link_name = "mlirAffineMapConstantGet"]
    fn mlir_affine_map_constant_get(context: Context, value: u64) -> AffineMap;
    #[link_name = "mlirAffineMapMultiDimIdentityGet"]
    fn mlir_affine_map_multi_dim_identity_get(context: Context, num_dims: usize) -> AffineMap;
    #[link_name = "mlirAffineMapPermutationGet"]
    fn mlir_affine_map_permutation_get(
        context: Context,
        num_permutations: usize,
        permutations: *const u32,
    ) -> AffineMap;
    #[link_name = "mlirAffineMapIsIdentity"]
    fn mlir_affine_map_is_identity(map: AffineMap) -> bool;
    #[link_name = "mlirAffineMapIsMinorIdentity"]
    fn mlir_affine_map_is_minor_identity(map: AffineMap) -> bool;
    #[link_name = "mlirAffineMapIsEmpty"]
    fn mlir_affine_map_is_empty(map: AffineMap) -> bool;
    #[link_name = "mlirAffineMapIsSingleConstant"]
    fn mlir_affine_map_is_single_constant(map: AffineMap) -> bool;
    #[link_name = "mlirAffineMapGetSingleConstantResult"]
    fn mlir_affine_map_get_single_constant_result(map: AffineMap) -> u64;
    #[link_name = "mlirAffineMapGetResult"]
    fn mlir_affine_map_get_result(map: AffineMap, index: usize) -> AffineExprBase;
    #[link_name = "mlirAffineMapGetNumDims"]
    fn mlir_affine_map_get_num_dims(map: AffineMap) -> usize;
    #[link_name = "mlirAffineMapGetNumSymbols"]
    fn mlir_affine_map_get_num_symbols(map: AffineMap) -> usize;
    #[link_name = "mlirAffineMapGetNumResults"]
    fn mlir_affine_map_get_num_results(map: AffineMap) -> usize;
    #[link_name = "mlirAffineMapGetNumInputs"]
    fn mlir_affine_map_get_num_inputs(map: AffineMap) -> usize;
    #[link_name = "mlirAffineMapIsProjectedPermutation"]
    fn mlir_affine_map_is_projected_permutation(map: AffineMap) -> bool;
    #[link_name = "mlirAffineMapIsPermutation"]
    fn mlir_affine_map_is_permutation(map: AffineMap) -> bool;
    #[link_name = "mlirAffineMapGetSubMap"]
    fn mlir_affine_map_get_sub_map(
        map: AffineMap,
        num_results: usize,
        results: *const usize,
    ) -> AffineMap;
    #[link_name = "mlirAffineMapGetMajorSubMap"]
    fn mlir_affine_map_get_major_sub_map(map: AffineMap, results: usize) -> AffineMap;
    #[link_name = "mlirAffineMapGetMinorSubMap"]
    fn mlir_affine_map_get_minor_sub_map(map: AffineMap, results: usize) -> AffineMap;
    #[link_name = "mlirAffineMapReplace"]
    fn mlir_affine_map_replace(
        map: AffineMap,
        expression: AffineExprBase,
        replacement: AffineExprBase,
        dims: usize,
        symbols: usize,
    ) -> AffineMap;
    #[allow(improper_ctypes)]
    #[link_name = "mlirAffineMapCompressUnusedSymbols"]
    fn mlir_affine_map_compress_unused_symbols(
        maps: *const AffineMap,
        num_maps: usize,
        result: &mut Vec<AffineMap>,
        callback: extern "C" fn(&mut Vec<AffineMap>, usize, AffineMap),
    );
}

/// Represents an affine map permutation expression.
///
/// A permutation expression is a non-empty vector of integers which
/// contains a continuous set of integers starting at 0 with no duplicates.
pub struct Permutation(Vec<u32>);
impl Permutation {
    /// Construct a new random permutation of the given size
    ///
    /// NOTE: This function asserts that the size is greater than 0
    pub fn new(size: usize) -> Self {
        use rand::prelude::*;

        assert!(size > 0, "permutations cannot be empty");
        let size: u32 = size.try_into().unwrap();

        let mut rng = rand::thread_rng();
        let mut values: Vec<u32> = (0..size).collect();
        values.shuffle(&mut rng);

        Self(values)
    }

    /// Constructs a permutation directly from the given `Vec`.
    ///
    /// # Safety
    ///
    /// It is up to the caller to ensure the invariants of this data structure
    /// are met, otherwise use of this is likely to assert somewhere in MLIR or
    /// potentially result in miscompilations. It is recommended that you only
    /// use this when reproducing a previously constructed permutation for tests
    /// or development.
    pub unsafe fn from_vec(elements: Vec<u32>) -> Self {
        Self(elements)
    }

    /// Returns the size of this permutation
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns a raw pointer to the beginning of the underlying vector
    pub fn as_ptr(&self) -> *const u32 {
        self.0.as_ptr()
    }

    /// Returns the underlying vector as a slice
    pub fn as_slice(&self) -> &[u32] {
        self.0.as_slice()
    }
}
