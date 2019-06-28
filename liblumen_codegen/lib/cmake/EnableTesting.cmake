include(FetchContent)

enable_testing()

find_program(VALGRIND valgrind)
if (VALGRIND)
    set(MEMORYCHECK_COMMAND "${VALGRIND}")
endif()

FetchContent_Declare(googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG master
)

FetchContent_MakeAvailable(googletest)

include(GoogleTest)
