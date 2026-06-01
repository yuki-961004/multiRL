if(NOT DEFINED REPO_ROOT)
    get_filename_component(REPO_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
endif()

set(MULTIRLCPP_BACKEND_SRC_DIR "${REPO_ROOT}/Cpp/src")
set(
    MULTIRLCPP_BACKEND_INCLUDE_DIR
    "${REPO_ROOT}/Cpp/include/multiRLcpp"
)

set(MULTIRLCPP_R_SRC_DEST "${REPO_ROOT}/R/src/cpp")
set(MULTIRLCPP_R_INCLUDE_DEST "${REPO_ROOT}/R/inst/include/multiRLcpp")
set(MULTIRLCPP_PY_SRC_DEST "${REPO_ROOT}/Python/src/cpp")
set(
    MULTIRLCPP_PY_INCLUDE_DEST
    "${REPO_ROOT}/Python/src/include/multiRLcpp"
)

foreach(sync_dir
        "${MULTIRLCPP_R_SRC_DEST}"
        "${MULTIRLCPP_R_INCLUDE_DEST}"
        "${MULTIRLCPP_PY_SRC_DEST}"
        "${MULTIRLCPP_PY_INCLUDE_DEST}")
    file(REMOVE_RECURSE "${sync_dir}")
    file(MAKE_DIRECTORY "${sync_dir}")
endforeach()

file(
    GLOB MULTIRLCPP_BACKEND_SOURCES
    "${MULTIRLCPP_BACKEND_SRC_DIR}/*.cpp"
)

file(
    GLOB MULTIRLCPP_BACKEND_HEADERS
    "${MULTIRLCPP_BACKEND_INCLUDE_DIR}/*.hpp"
)

foreach(source_file ${MULTIRLCPP_BACKEND_SOURCES})
    file(COPY "${source_file}" DESTINATION "${MULTIRLCPP_R_SRC_DEST}")
    file(COPY "${source_file}" DESTINATION "${MULTIRLCPP_PY_SRC_DEST}")
endforeach()

foreach(header_file ${MULTIRLCPP_BACKEND_HEADERS})
    file(COPY "${header_file}" DESTINATION "${MULTIRLCPP_R_INCLUDE_DEST}")
    file(COPY "${header_file}" DESTINATION "${MULTIRLCPP_PY_INCLUDE_DEST}")
endforeach()

set(MULTIRLCPP_SYNC_NOTE
"Do not edit files in this directory directly.
Edit root Cpp/src and Cpp/include/multiRLcpp, then run CMake to synchronize.
")

file(WRITE "${MULTIRLCPP_R_SRC_DEST}/README.md" "${MULTIRLCPP_SYNC_NOTE}")
file(
    WRITE
    "${MULTIRLCPP_R_INCLUDE_DEST}/README.md"
    "${MULTIRLCPP_SYNC_NOTE}"
)
file(WRITE "${MULTIRLCPP_PY_SRC_DEST}/README.md" "${MULTIRLCPP_SYNC_NOTE}")
file(
    WRITE
    "${MULTIRLCPP_PY_INCLUDE_DEST}/README.md"
    "${MULTIRLCPP_SYNC_NOTE}"
)

message(STATUS "multiRLcpp: synchronized backend C++ sources and headers.")
