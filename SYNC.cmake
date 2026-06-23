if(NOT DEFINED REPO_ROOT)
    get_filename_component(REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
endif()

set(MULTIRL_BACKEND_SRC_DIR "${REPO_ROOT}/Cpp/src")
set(
    MULTIRL_BACKEND_INCLUDE_DIR
    "${REPO_ROOT}/Cpp/include/multiRL"
)

set(MULTIRL_R_SRC_DEST "${REPO_ROOT}/R/src/cpp")
set(MULTIRL_R_INCLUDE_DEST "${REPO_ROOT}/R/inst/include/multiRL")
set(MULTIRL_PY_SRC_DEST "${REPO_ROOT}/Python/src/cpp")
set(
    MULTIRL_PY_INCLUDE_DEST
    "${REPO_ROOT}/Python/src/include/multiRL"
)

foreach(sync_dir
        "${MULTIRL_R_SRC_DEST}"
        "${MULTIRL_R_INCLUDE_DEST}"
        "${MULTIRL_PY_SRC_DEST}"
        "${MULTIRL_PY_INCLUDE_DEST}")
    file(REMOVE_RECURSE "${sync_dir}")
    file(MAKE_DIRECTORY "${sync_dir}")
endforeach()

file(
    GLOB MULTIRL_BACKEND_SOURCES
    "${MULTIRL_BACKEND_SRC_DIR}/*.cpp"
)

file(
    GLOB MULTIRL_BACKEND_HEADERS
    "${MULTIRL_BACKEND_INCLUDE_DIR}/*.hpp"
    "${MULTIRL_BACKEND_INCLUDE_DIR}/*.h"
)

foreach(source_file ${MULTIRL_BACKEND_SOURCES})
    file(COPY "${source_file}" DESTINATION "${MULTIRL_R_SRC_DEST}")
    file(COPY "${source_file}" DESTINATION "${MULTIRL_PY_SRC_DEST}")
endforeach()

foreach(header_file ${MULTIRL_BACKEND_HEADERS})
    file(COPY "${header_file}" DESTINATION "${MULTIRL_R_INCLUDE_DEST}")
    file(COPY "${header_file}" DESTINATION "${MULTIRL_PY_INCLUDE_DEST}")
endforeach()

set(MULTIRL_SYNC_NOTE
"Do not edit files in this directory directly.
Edit root Cpp/src and Cpp/include/multiRL, then run CMake to synchronize.
")

file(WRITE "${MULTIRL_R_SRC_DEST}/README.md" "${MULTIRL_SYNC_NOTE}")
file(
    WRITE
    "${MULTIRL_R_INCLUDE_DEST}/README.md"
    "${MULTIRL_SYNC_NOTE}"
)
file(WRITE "${MULTIRL_PY_SRC_DEST}/README.md" "${MULTIRL_SYNC_NOTE}")
file(
    WRITE
    "${MULTIRL_PY_INCLUDE_DEST}/README.md"
    "${MULTIRL_SYNC_NOTE}"
)

message(STATUS "multiRL: synchronized backend C++ sources and headers.")

# Synchronize R tests from root tests/testthat to R/tests/testthat
set(MULTIRL_R_TEST_SRC "${REPO_ROOT}/tests/testthat")
set(MULTIRL_R_TEST_DEST "${REPO_ROOT}/R/tests/testthat")

file(REMOVE_RECURSE "${MULTIRL_R_TEST_DEST}")
file(MAKE_DIRECTORY "${MULTIRL_R_TEST_DEST}")

file(
    GLOB R_TEST_SOURCES
    "${MULTIRL_R_TEST_SRC}/test-*.R"
)

foreach(test_file ${R_TEST_SOURCES})
    file(COPY "${test_file}" DESTINATION "${MULTIRL_R_TEST_DEST}")
endforeach()

file(WRITE "${MULTIRL_R_TEST_DEST}/README.md"
"Do not edit files in this directory directly.
Edit root tests/testthat, then run CMake to synchronize.
")

message(STATUS "multiRL: synchronized R tests from root tests/testthat.")

