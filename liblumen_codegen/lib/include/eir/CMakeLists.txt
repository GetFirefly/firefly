function(add_dialect_tablegen_item name)
  set(LLVM_TARGET_DEFINITIONS ${name}.td)
  mlir_tablegen(${name}.h.inc -gen-op-decls)
  mlir_tablegen(${name}.cpp.inc -gen-op-defs)
  add_public_tablegen_target(${name}IncGen)
endfunction()

add_dialect_tablegen_item(Ops)
