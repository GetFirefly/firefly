project(codegen
  VERSION 0.1
  LANGUAGES C CXX
)

add_library(codegen STATIC
  CodeGen.cpp
)
llvm_map_components_to_libnames(llvm_libs all)
target_include_directories(codegen PUBLIC include/)
target_include_directories(codegen PRIVATE ${LLVM_INCLUDE_DIRS})
target_compile_definitions(codegen PRIVATE ${LLVM_DEFINITIONS})
target_compile_options(codegen PUBLIC -g -fno-rtti)
target_link_libraries(codegen PRIVATE ${llvm_libs})

#if (ENABLE_TESTING)
    #add_executable(codegen_test tests.cpp)
    #target_include_directories(codegen_test PUBLIC include/)
    #target_link_libraries(codegen_test codegen)
    #gtest_discover_tests(codegen_test)
#endif()