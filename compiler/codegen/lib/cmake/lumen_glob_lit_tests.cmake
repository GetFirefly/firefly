# lumen_glob_lit_tests()

function(lumen_glob_lit_tests)
  if(NOT LUMEN_BUILD_TESTS)
    return()
  endif()

  lumen_package_name(_PACKAGE_NAME)
  file(GLOB_RECURSE _TEST_FILES *.mlir)
  set(_TOOL_DEPS lumen_tool_lumen-opt LumenFileCheck)

  foreach(_TEST_FILE ${_TEST_FILES})
    get_filename_component(_TEST_FILE_LOCATION ${_TEST_FILE} DIRECTORY)
    get_filename_component(_TEST_NAME ${_TEST_FILE} NAME_WE)
    set(_NAME "${_PACKAGE_NAME}_${_TEST_NAME}")

    add_test(NAME ${_NAME} COMMAND ${CMAKE_SOURCE_DIR}/tools/run_lit.sh ${_TEST_FILE} ${CMAKE_SOURCE_DIR}/tools/LumenFileCheck.sh WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/tools)
    set_tests_properties(${_NAME} PROPERTIES DEPENDS _TOOL_DEPS)
  endforeach()
endfunction()
