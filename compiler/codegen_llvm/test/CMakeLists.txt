if(${LUMEN_BUILD_TESTS})
  add_custom_target(LumenFileCheck ALL
    COMMAND ${CMAKE_COMMAND} -E create_symlink ${CMAKE_CURRENT_SOURCE_DIR}/LumenFileCheck.sh LumenFileCheck
  )
  add_custom_target(LLVMFileCheck ALL
    COMMAND ${CMAKE_COMMAND} -E create_symlink $<TARGET_FILE:FileCheck> FileCheck
    DEPENDS FileCheck
  )
endif()
