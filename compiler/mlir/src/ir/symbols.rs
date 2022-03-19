use crate::*;

extern "C" {
    type MlirSymbolTable;
}

/// This enum represents the visibility of a symbol in a symbol table
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u32)]
pub enum Visibility {
    /// The symbol is public and may be referenced anywhere internal or external
    /// to the visible references in the IR.
    Public = 0,
    /// The symbol is private and may only be referenced by SymbolRefAttrs local
    /// to the operations within the current symbol table.
    Private,
    /// The symbol is visible to the current IR, which may include operations in
    /// symbol tables above the one that owns the current symbol. `Nested`
    /// visibility allows for referencing a symbol outside of its current symbol
    /// table, while retaining the ability to observe all uses.
    Nested,
}

/// The `SymbolTable` API provides facilities for looking up symbols in the current
/// symbol table, or in other symbol tables, and setting symbol names and visibility.
#[repr(transparent)]
#[derive(Copy, Clone)]
pub struct SymbolTable(*const MlirSymbolTable);
impl SymbolTable {
    #[inline]
    pub fn is_null(&self) -> bool {
        self.0.is_null()
    }

    /// Creates a symbol table for the given operation.
    ///
    /// If the operation does not have the C++ SymbolTable trait, returns None
    pub fn create<O: Operation>(op: O) -> Option<Self> {
        let this = unsafe { mlir_symbol_table_create(op.base()) };
        if this.is_null() {
            None
        } else {
            Some(this)
        }
    }

    /// Destroys this symbol table
    ///
    /// NOTE: This does not affect operations in the table
    pub fn destroy(self) {
        unsafe { mlir_symbol_table_destroy(self) }
    }

    /// Looks up a symbol with the given name in this symbol table and
    /// returns the operation that corresponds to it.
    ///
    /// If the symbol cannot be found, returns None
    pub fn lookup<S: Into<StringRef>>(&self, name: S) -> Option<OperationBase> {
        let name = name.into();
        let op = unsafe { mlir_symbol_table_lookup(*self, name) };
        if op.is_null() {
            None
        } else {
            Some(op)
        }
    }

    /// Inserts the given operation in this symbol table.
    ///
    /// The operation must have the C++ Symbol trait.
    ///
    /// If the symbol table already has a symbol with the same name, renames
    /// the symbol being inserted to ensure uniqueness. NOTE: This does not move
    /// the operation itself into the block of the operation holding the symbol table,
    /// this should be done separately.
    ///
    /// Returns the name of the symbol after insertion.
    pub fn insert<O: Operation>(&self, op: O) -> StringAttr {
        unsafe { mlir_symbol_table_insert(*self, op.base()) }
    }

    /// Removes the given operation from this symbol table
    pub fn erase<O: Operation>(&self, op: O) {
        unsafe { mlir_symbol_table_erase(*self, op.base()) }
    }

    /// Returns the nearest Operation that is a symbol table, if one exists
    pub fn get_nearest<O: Operation>(from: O) -> Option<OperationBase> {
        let op = unsafe { mlir_symbol_table_get_nearest_symbol_table(from.base()) };
        if op.is_null() {
            None
        } else {
            Some(op)
        }
    }

    /// Returns the nearest Operation that has the given symbol name, if it exists
    pub fn lookup_nearest_symbol_from<O: Operation, S: Into<StringRef>>(
        from: O,
        symbol: S,
    ) -> Option<OperationBase> {
        let op =
            unsafe { mlir_symbol_table_lookup_nearest_symbol_from(from.base(), symbol.into()) };
        if op.is_null() {
            None
        } else {
            Some(op)
        }
    }

    /// Returns the symbol with the given name, in the symbol table of the provided Operation, if it
    /// exists
    pub fn lookup_symbol_in<O: Operation, S: Into<StringRef>>(
        op: O,
        symbol: S,
    ) -> Option<OperationBase> {
        let op = unsafe { mlir_symbol_table_lookup_in(op.base(), symbol.into()) };
        if op.is_null() {
            None
        } else {
            Some(op)
        }
    }

    /// Return whether the given symbol is known to have no uses that are nested
    /// within the given operation 'from'. This does not traverse into any nested
    /// symbol tables. This function will also return false if there are any
    /// unknown operations that may potentially be symbol tables. This doesn't
    /// necessarily mean that there are no uses, we just can't conservatively
    /// prove it.
    pub fn has_uses_from<O: Operation, S: Into<StringRef>>(from: O, symbol: S) -> bool {
        unsafe { mlir_symbol_table_known_use_empty(symbol.into(), from.base()) }
    }

    /// Attempt to replace all uses that are nested within the given operation with symbol `old`,
    /// with symbol `new`.
    ///
    /// This does not traverse into nested symbol tables.
    ///
    /// Will fail atomically if there are any unknown operations that may be potential symbol tables.
    pub fn replace_all_uses<O: Operation, S: Into<StringRef>>(from: O, old: S, new: S) -> bool {
        let result = unsafe {
            mlir_symbol_table_replace_all_symbol_uses(old.into(), new.into(), from.base())
        };
        result.into()
    }

    /// Get the symbol name for the given Operation
    pub fn get_symbol_name<O: Operation>(op: O) -> StringRef {
        unsafe { mlir_symbol_table_get_symbol_name(op.base()) }
    }

    /// Set the symbol name for the given Operation
    pub fn set_symbol_name<O: Operation, S: Into<StringRef>>(op: O, name: S) {
        unsafe {
            mlir_symbol_table_set_symbol_name(op.base(), name.into());
        }
    }

    /// Get the symbol visibility for the given Operation
    pub fn get_symbol_visibility<O: Operation>(op: O) -> Visibility {
        unsafe { mlir_symbol_table_get_symbol_visibility(op.base()) }
    }

    /// Set the symbol visibility for the given Operation
    pub fn set_symbol_visibility<O: Operation>(op: O, visibility: Visibility) {
        unsafe {
            mlir_symbol_table_set_symbol_visibility(op.base(), visibility);
        }
    }

    /// Get the attribute name used to store symbol names on an Operation
    pub fn get_symbol_attr_name() -> &'static str {
        // The string reference returned here is known to be utf-8, and have static lifetime
        let sr = unsafe { mlir_symbol_table_get_symbol_attr_name() };
        let s = sr.try_into().unwrap();
        unsafe { core::mem::transmute::<&str, &'static str>(s) }
    }

    /// Get the attribute name used to store symbol visibility on an Operation
    pub fn get_symbol_visibility_name() -> &'static str {
        // The string reference returned here is known to be utf-8, and have static lifetime
        let sr = unsafe { mlir_symbol_table_get_visibility_attr_name() };
        let s = sr.try_into().unwrap();
        unsafe { core::mem::transmute::<&str, &'static str>(s) }
    }
}

extern "C" {
    #[link_name = "mlirSymbolTableCreate"]
    fn mlir_symbol_table_create(op: OperationBase) -> SymbolTable;
    #[link_name = "mlirSymbolTableDestroy"]
    fn mlir_symbol_table_destroy(table: SymbolTable);
    #[link_name = "mlirSymbolTableLookup"]
    fn mlir_symbol_table_lookup(table: SymbolTable, name: StringRef) -> OperationBase;
    #[link_name = "mlirSymbolTableInsert"]
    fn mlir_symbol_table_insert(table: SymbolTable, op: OperationBase) -> StringAttr;
    #[link_name = "mlirSymbolTableErase"]
    fn mlir_symbol_table_erase(table: SymbolTable, op: OperationBase);
    #[link_name = "mlirSymbolTableGetNearestSymbolTable"]
    fn mlir_symbol_table_get_nearest_symbol_table(from: OperationBase) -> OperationBase;
    #[link_name = "mlirSymbolTableLookupNearestSymbolFrom"]
    fn mlir_symbol_table_lookup_nearest_symbol_from(
        from: OperationBase,
        symbol: StringRef,
    ) -> OperationBase;
    #[link_name = "mlirSymbolTableLookupIn"]
    fn mlir_symbol_table_lookup_in(op: OperationBase, symbol: StringRef) -> OperationBase;
    #[link_name = "mlirSymbolTableSymbolKnownUseEmpty"]
    fn mlir_symbol_table_known_use_empty(symbol: StringRef, from: OperationBase) -> bool;
    #[link_name = "mlirSymbolTableReplaceAllSymbolUses"]
    fn mlir_symbol_table_replace_all_symbol_uses(
        old: StringRef,
        new: StringRef,
        from: OperationBase,
    ) -> LogicalResult;
    #[link_name = "mlirSymbolTableGetSymbolVisibility"]
    fn mlir_symbol_table_get_symbol_visibility(symbol: OperationBase) -> Visibility;
    #[link_name = "mlirSymbolTableSetSymbolVisibility"]
    fn mlir_symbol_table_set_symbol_visibility(symbol: OperationBase, visibility: Visibility);
    #[link_name = "mlirSymbolTableGetSymbolName"]
    fn mlir_symbol_table_get_symbol_name(symbol: OperationBase) -> StringRef;
    #[link_name = "mlirSymbolTableSetSymbolName"]
    fn mlir_symbol_table_set_symbol_name(symbol: OperationBase, name: StringRef);
    #[link_name = "mlirSymbolTableGetSymbolAttributeName"]
    fn mlir_symbol_table_get_symbol_attr_name() -> StringRef;
    #[link_name = "mlirSymbolTableGetVisibilityAttributeName"]
    fn mlir_symbol_table_get_visibility_attr_name() -> StringRef;
}
