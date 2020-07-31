include(CMakeParseArguments)

if (NOT DEFINED _LUMEN_CC_DYLIB_NAMES)
  set(_LUMEN_CC_DYLIB_NAMES "")
endif()

# lumen_cc_library()
#
# CMake function to construct a library target
#
# Parameters:
# NAME: name of target (see Note)
# HDRS: List of public header files for the library
# TEXTUAL_HDRS: List of public header files that cannot be compiled on their own
# SRCS: List of source files for the library
# DEPS: List of other libraries to be linked in to the binary targets
# DEPENDS: List of other libraries that are dependencies, but not linked in
# COPTS: List of private compile options
# DEFINES: List of public defines
# INCLUDES: Include directories to add to dependencies
# LINKOPTS: List of link options
# ALWAYSLINK: Always link the library into any binary with a direct dep.
# PUBLIC: Add this so that this library will be exported under lumen::
# Also in IDE, target will appear in Lumen folder while non PUBLIC will be in Lumen/internal.
# TESTONLY: When added, this target will only be built if user passes -DLUMEN_BUILD_TESTS=ON to CMake.
#
# Note:
# By default, lumen_cc_library will always create a library named lumen_${NAME},
# and alias target lumen::${NAME}. The lumen:: form should always be used.
# This is to reduce namespace pollution.
#
# lumen_cc_library(
#   NAME
#     awesome
#   HDRS
#     "a.h"
#   SRCS
#     "a.cc"
# )
# lumen_cc_library(
#   NAME
#     fantastic_lib
#   SRCS
#     "b.cc"
#   DEPS
#     lumen::package::awesome # not "awesome" !
#   PUBLIC
# )
#
# lumen_cc_library(
#   NAME
#     main_lib
#   ...
#   DEPS
#     lumen::package::fantastic_lib
# )
function(lumen_cc_library)
  cmake_parse_arguments(
    _RULE
    "PUBLIC;ALWAYSLINK;TESTONLY;SHARED"
    "NAME"
    "HDRS;TEXTUAL_HDRS;SRCS;COPTS;DEFINES;LINKOPTS;DEPS;DEPENDS;INCLUDES"
    ${ARGN}
  )

  lumen_package_ns(_PACKAGE_NS)
  # Replace dependencies passed by ::name with ::lumen::package::name
  list(TRANSFORM _RULE_DEPS REPLACE "^::" "${_PACKAGE_NS}::")
  list(TRANSFORM _RULE_DEPENDS REPLACE "^::" "${_PACKAGE_NS}::")

  if(NOT _RULE_TESTONLY OR LUMEN_BUILD_TESTS)
    # Prefix the library with the package name, so we get: lumen_package_name.
    lumen_package_name(_PACKAGE_NAME)
    if(NOT _RULE_NAME)
      set(_NAME "${_PACKAGE_NAME}")
    else()
      set(_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")
    endif()

    # Check if this is a header-only library.
    # Note that as of February 2019, many popular OS's (for example, Ubuntu
    # 16.04 LTS) only come with cmake 3.5 by default.  For this reason, we can't
    # use list(FILTER...)
    set(_CC_SRCS "${_RULE_SRCS}")
    foreach(src_file IN LISTS _CC_SRCS)
      if(${src_file} MATCHES ".*\\.(h|inc)")
        list(REMOVE_ITEM _CC_SRCS "${src_file}")
      endif()
    endforeach()
    if("${_CC_SRCS}" STREQUAL "")
      set(_RULE_IS_INTERFACE 1)
    else()
      set(_RULE_IS_INTERFACE 0)
    endif()

    if(NOT _RULE_IS_INTERFACE)
      if (NOT _RULE_SHARED)
        add_library(${_NAME} STATIC "")
      else()
        add_library(${_NAME} SHARED "")
      endif()
      target_sources(${_NAME}
        PRIVATE
          ${_RULE_SRCS}
          ${_RULE_TEXTUAL_HDRS}
          ${_RULE_HDRS}
      )
      target_include_directories(${_NAME}
        PUBLIC
          "$<BUILD_INTERFACE:${LUMEN_COMMON_INCLUDE_DIRS}>"
          "$<BUILD_INTERFACE:${_RULE_INCLUDES}>"
      )
      target_compile_options(${_NAME}
        PRIVATE
          ${_RULE_COPTS}
          ${LUMEN_DEFAULT_COPTS}
      )

      if (NOT _RULE_SHARED)
        target_link_libraries(${_NAME}
          PUBLIC
            ${_RULE_DEPS}
          PRIVATE
            ${_RULE_LINKOPTS}
            ${LUMEN_DEFAULT_LINKOPTS}
        )
      endif()
      if ("${_RULE_DEPENDS}")
        add_dependencies(${_NAME}
          ${_RULE_DEPENDS}
        )
      endif()
      target_compile_definitions(${_NAME}
        PUBLIC
          ${_RULE_DEFINES}
      )

      if(DEFINED _RULE_ALWAYSLINK)
        set_property(TARGET ${_NAME} PROPERTY ALWAYSLINK 1)
      endif()

      # Add all Lumen targets to a folder in the IDE for organization.
      if(_RULE_PUBLIC)
        set_property(TARGET ${_NAME} PROPERTY FOLDER ${LUMEN_IDE_FOLDER})
      elseif(_RULE_TESTONLY)
        set_property(TARGET ${_NAME} PROPERTY FOLDER ${LUMEN_IDE_FOLDER}/test)
      else()
        set_property(TARGET ${_NAME} PROPERTY FOLDER ${LUMEN_IDE_FOLDER}/internal)
      endif()

      # INTERFACE libraries can't have the CXX_STANDARD property set.
      set_property(TARGET ${_NAME} PROPERTY CXX_STANDARD ${LUMEN_CXX_STANDARD})
      set_property(TARGET ${_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)
    else()
      # Generating header-only library.
      add_library(${_NAME} INTERFACE)
      target_include_directories(${_NAME}
        INTERFACE
          "$<BUILD_INTERFACE:${LUMEN_COMMON_INCLUDE_DIRS}>"
      )
      target_compile_options(${_NAME}
        INTERFACE
          ${_RULE_COPTS}
          ${LUMEN_DEFAULT_COPTS}
      )
      target_link_libraries(${_NAME}
        INTERFACE
          ${_RULE_DEPS}
          ${_RULE_LINKOPTS}
          ${LUMEN_DEFAULT_LINKOPTS}
      )
      target_compile_definitions(${_NAME}
        INTERFACE
          ${_RULE_DEFINES}
      )
    endif()

    install(TARGETS ${_NAME}
      ARCHIVE DESTINATION lib
      LIBRARY DESTINATION lib
    )

    # Alias the lumen_package_name library to lumen::package::name.
    # This lets us more clearly map to Bazel and makes it possible to
    # disambiguate the underscores in paths vs. the separators.
    if(NOT _RULE_NAME)
      add_library(${_PACKAGE_NS} ALIAS ${_NAME})
    else()
      add_library(${_PACKAGE_NS}::${_RULE_NAME} ALIAS ${_NAME})
    endif()
    lumen_package_dir(_PACKAGE_DIR)

    if (_RULE_SHARED)
      # Defer computing transitive dependencies and calling target_link_libraries()
      # until all shared libraries have been declared.
      # Track target and deps, use in lumen_complete_dylib_link_options() later.
      set_property(GLOBAL APPEND PROPERTY _LUMEN_CC_DYLIB_NAMES "${_NAME}")
      set_property(TARGET ${_NAME} PROPERTY DIRECT_DEPS ${_RULE_DEPS})
      set_property(TARGET ${_NAME} PROPERTY DIRECT_LINKOPTS ${_RULE_LINKOPTS})
    endif()
  endif()
endfunction()

function(lumen_complete_dylib_link_options)
  get_property(_NAMES GLOBAL PROPERTY _LUMEN_CC_DYLIB_NAMES)

  foreach(_NAME ${_NAMES})
    get_target_property(_DIRECT_DEPS ${_NAME} DIRECT_DEPS)
    get_target_property(_DIRECT_LINKOPTS ${_NAME} DIRECT_LINKOPTS)

    # List all dependencies, including transitive dependencies, then split the
    # dependency list into one for whole archive (ALWAYSLINK) and one for
    # standard linking (which only links in symbols that are directly used).
    _lumen_transitive_dependencies("${_DIRECT_DEPS}" _TRANSITIVE_DEPS)
    set(_ALWAYS_LINK_DEPS "")
    set(_STANDARD_DEPS "")
    foreach(_DEP ${_TRANSITIVE_DEPS})
      # Check if _DEP is a library with the ALWAYSLINK property set.
      set(_DEP_IS_ALWAYSLINK OFF)
      if (TARGET ${_DEP})
        get_target_property(_DEP_TYPE ${_DEP} TYPE)
        if(${_DEP_TYPE} STREQUAL "INTERFACE_LIBRARY")
          # Can't be ALWAYSLINK since it's an INTERFACE library.
          # We also can't even query for the property, since it isn't whitelisted.
        else()
          get_target_property(_DEP_IS_ALWAYSLINK ${_DEP} ALWAYSLINK)
        endif()
      endif()

      # Append to the corresponding list of deps.
      if(_DEP_IS_ALWAYSLINK)
        list(APPEND _ALWAYS_LINK_DEPS ${_DEP})

        # For MSVC, also add a `-WHOLEARCHIVE:` version of the dep.
        # CMake treats -WHOLEARCHIVE[:lib] as a link flag and will not actually
        # try to link the library in, so we need the flag *and* the dependency.
        if(MSVC)
          get_target_property(_ALIASED_TARGET ${_DEP} ALIASED_TARGET)
          if (_ALIASED_TARGET)
            list(APPEND _ALWAYS_LINK_DEPS "-WHOLEARCHIVE:${_ALIASED_TARGET}")
          else()
            list(APPEND _ALWAYS_LINK_DEPS "-WHOLEARCHIVE:${_DEP}")
          endif()
        endif()
      else()
        list(APPEND _STANDARD_DEPS ${_DEP})
      endif()
    endforeach(_DEP)

    # Call into target_link_libraries with the lists of deps.
    # TODO: `-Wl,-force_load` version
    if(MSVC)
      target_link_libraries(${_NAME}
        PUBLIC
          ${_ALWAYS_LINK_DEPS}
          ${_STANDARD_DEPS}
        PRIVATE
          ${_DIRECT_LINKOPTS}
          ${_RULE_LINKOPTS}
      )
    else()
      if("${CMAKE_SYSTEM_NAME}" STREQUAL "Darwin")
        #set(_ALWAYS_LINK_DEPS_W_FLAGS "-Wl,-all_load" ${_ALWAYS_LINK_DEPS} "-Wl,-noall_load")
        foreach(_ALWAYS_LINK_DEPS_DEP ${_ALWAYS_LINK_DEPS})
          list(APPEND _ALWAYS_LINK_DEPS_W_FLAGS "-Wl,-force_load")
          list(APPEND _ALWAYS_LINK_DEPS_W_FLAGS "${_ALWAYS_LINK_DEPS_DEP}")
        endforeach(_ALWAYS_LINK_DEPS_DEP)
      else()
        set(_ALWAYS_LINK_DEPS_W_FLAGS "-Wl,--whole-archive" ${_ALWAYS_LINK_DEPS} "-Wl,--no-whole-archive")
      endif()
      target_link_libraries(${_NAME}
        PUBLIC
          ${_ALWAYS_LINK_DEPS_W_FLAGS}
          ${_STANDARD_DEPS}
        PRIVATE
          ${_DIRECT_LINKOPTS}
          ${_RULE_LINKOPTS}
      )
    endif()
  endforeach(_NAME)
endfunction()
