configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py

  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
  )

set(CIR_TEST_DEPENDS
  FileCheck count not
  standalone-capi-test
  )

add_lit_testsuite(check-cir "Running the CIR regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${CIR_TEST_DEPENDS}
  )

set_target_properties(check-cir PROPERTIES FOLDER "Tests")

add_lit_testsuites(CIR ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${CIR_TEST_DEPENDS})

add_subdirectory(CAPI)
