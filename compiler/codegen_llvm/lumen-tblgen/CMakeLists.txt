set(LLVM_LINK_COMPONENTS
    Demangle
    Support
    TableGen)

add_tablegen(lumen-tblgen LUMEN
    lumen-tblgen.cpp
    EncodingGen.cpp
    )

target_link_libraries(lumen-tblgen PRIVATE MLIRTableGen)
llvm_update_compile_flags(lumen-tblgen)

set_property(TARGET lumen-tblgen PROPERTY CXX_STANDARD ${LUMEN_CXX_STANDARD})
set_property(TARGET lumen-tblgen PROPERTY CXX_STANDARD_REQUIRED ON)
