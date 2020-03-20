include(CMakeParseArguments)

if (NOT DEFINED _LUMEN_CC_BINARY_NAMES)
  set(_LUMEN_CC_BINARY_NAMES "")
endif()

# lumen_cc_binary()
#
# Parameters:
# NAME: name of target (see Usage below)
# SRCS: List of source files for the binary
# DEPS: List of other libraries to be linked in to the binary targets
# COPTS: List of private compile options
# DEFINES: List of public defines
# LINKOPTS: List of link options
#
# Note:
# By default, lumen_cc_binary will always create a binary named lumen_${NAME}.
#
# Usage:
# lumen_cc_library(
#   NAME
#     awesome
#   HDRS
#     "a.h"
#   SRCS
#     "a.cc"
#   PUBLIC
# )
#
# lumen_cc_binary(
#   NAME
#     awesome_tool
#   OUT
#     awesome-tool
#   SRCS
#     "awesome_tool_main.cc"
#   DEPS
#     lumen::awesome
# )
function(lumen_cc_binary)
  cmake_parse_arguments(
    _RULE
    ""
    "NAME;OUT"
    "SRCS;COPTS;DEFINES;LINKOPTS;DEPS"
    ${ARGN}
  )

  # Prefix the library with the package name, so we get: lumen_package_name
  lumen_package_name(_PACKAGE_NAME)
  set(_NAME "${_PACKAGE_NAME}_${_RULE_NAME}")

  add_executable(${_NAME} "")
  if(_RULE_SRCS)
    target_sources(${_NAME}
      PRIVATE
        ${_RULE_SRCS}
    )
  else()
    set(_DUMMY_SRC "${CMAKE_CURRENT_BINARY_DIR}/${_NAME}_dummy.cc")
    file(WRITE ${_DUMMY_SRC} "")
    target_sources(${_NAME}
      PRIVATE
        ${_DUMMY_SRC}
    )
  endif()
  if(_RULE_OUT)
    set_target_properties(${_NAME} PROPERTIES OUTPUT_NAME "${_RULE_OUT}")
  endif()
  target_include_directories(${_NAME}
    PUBLIC
      ${LUMEN_COMMON_INCLUDE_DIRS}
    PRIVATE
      ${GTEST_INCLUDE_DIRS}
  )
  target_compile_definitions(${_NAME}
    PUBLIC
      ${_RULE_DEFINES}
  )
  target_compile_options(${_NAME}
    PRIVATE
      ${_RULE_COPTS}
  )

  lumen_package_ns(_PACKAGE_NS)
  # Replace dependencies passed by ::name with ::lumen::package::name
  list(TRANSFORM _RULE_DEPS REPLACE "^::" "${_PACKAGE_NS}::")

  # Add all LUMEN targets to a folder in the IDE for organization.
  set_property(TARGET ${_NAME} PROPERTY FOLDER ${LUMEN_IDE_FOLDER}/binaries)

  set_property(TARGET ${_NAME} PROPERTY CXX_STANDARD ${LUMEN_CXX_STANDARD})
  set_property(TARGET ${_NAME} PROPERTY CXX_STANDARD_REQUIRED ON)

  # Defer computing transitive dependencies and calling target_link_libraries()
  # until all libraries have been declared.
  # Track target and deps, use in lumen_complete_binary_link_options() later.
  set_property(GLOBAL APPEND PROPERTY _LUMEN_CC_BINARY_NAMES "${_NAME}")
  set_property(TARGET ${_NAME} PROPERTY DIRECT_DEPS ${_RULE_DEPS})
endfunction()

# Lists all transitive dependencies of DIRECT_DEPS in TRANSITIVE_DEPS.
function(_lumen_transitive_dependencies DIRECT_DEPS TRANSITIVE_DEPS)
  set(_TRANSITIVE "")

  foreach(_DEP ${DIRECT_DEPS})
    _lumen_transitive_dependencies_helper(${_DEP} _TRANSITIVE)
  endforeach(_DEP)

  set(${TRANSITIVE_DEPS} "${_TRANSITIVE}" PARENT_SCOPE)
endfunction()

# Recursive helper function for _lumen_transitive_dependencies.
# Performs a depth-first search through the dependency graph, appending all
# dependencies of TARGET to the TRANSITIVE_DEPS list.
function(_lumen_transitive_dependencies_helper TARGET TRANSITIVE_DEPS)
  if (NOT TARGET "${TARGET}")
    # Excluded from the project, or invalid name? Just ignore.
    return()
  endif()

  # Resolve aliases, canonicalize name formatting.
  get_target_property(_ALIASED_TARGET ${TARGET} ALIASED_TARGET)
  if(_ALIASED_TARGET)
    set(_TARGET_NAME ${_ALIASED_TARGET})
  else()
    string(REPLACE "::" "_" _TARGET_NAME ${TARGET})
  endif()

  set(_RESULT "${${TRANSITIVE_DEPS}}")
  if (${_TARGET_NAME} IN_LIST _RESULT)
    # Already visited, ignore.
    return()
  endif()

  # Append this target to the list. Dependencies of this target will be added
  # (if valid and not already visited) in recursive function calls.
  list(APPEND _RESULT ${_TARGET_NAME})

  # Check for non-target identifiers again after resolving the alias.
  if (NOT TARGET ${_TARGET_NAME})
    return()
  endif()

  # Get the list of direct dependencies for this target.
  get_target_property(_TARGET_TYPE ${_TARGET_NAME} TYPE)
  if(NOT ${_TARGET_TYPE} STREQUAL "INTERFACE_LIBRARY")
    get_target_property(_TARGET_DEPS ${_TARGET_NAME} LINK_LIBRARIES)
  else()
    get_target_property(_TARGET_DEPS ${_TARGET_NAME} INTERFACE_LINK_LIBRARIES)
  endif()

  if(_TARGET_DEPS)
    # Recurse on each dependency.
    foreach(_TARGET_DEP ${_TARGET_DEPS})
      _lumen_transitive_dependencies_helper(${_TARGET_DEP} _RESULT)
    endforeach(_TARGET_DEP)
  endif()

  # Propagate the augmented list up to the parent scope.
  set(${TRANSITIVE_DEPS} "${_RESULT}" PARENT_SCOPE)
endfunction()

# Sets target_link_libraries() on all registered binaries.
# This must be called after all libraries have been declared.
function(lumen_complete_binary_link_options)
  get_property(_NAMES GLOBAL PROPERTY _LUMEN_CC_BINARY_NAMES)

  foreach(_NAME ${_NAMES})
    get_target_property(_DIRECT_DEPS ${_NAME} DIRECT_DEPS)

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
          ${_RULE_LINKOPTS}
      )
    else()
      if("${CMAKE_SYSTEM_NAME}" STREQUAL "Darwin")
        set(_ALWAYS_LINK_DEPS_W_FLAGS "-Wl,-all_load" ${_ALWAYS_LINK_DEPS} "-Wl,-noall_load")
        #foreach(_always_link_dep ${_ALWAYS_LINK_DEPS})
        # list(APPEND _ALWAYS_LINK_DEPS_W_FLAGS ${_ALWAYS_LINK_DEPS_W_FLAGS} "-Wl,-force_load" ${_always_link_dep} )
        #endforeach(_always_link_dep)
      else()
        set(_ALWAYS_LINK_DEPS_W_FLAGS "-Wl,--whole-archive" ${_ALWAYS_LINK_DEPS} "-Wl,--no-whole-archive")
      endif()
      target_link_libraries(${_NAME}
        PUBLIC
          ${_ALWAYS_LINK_DEPS_W_FLAGS}
          ${_STANDARD_DEPS}
        PRIVATE
          ${_RULE_LINKOPTS}
      )
    endif()
  endforeach(_NAME)
endfunction()
