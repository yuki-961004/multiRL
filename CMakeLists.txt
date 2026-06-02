cmake_minimum_required(VERSION 3.15...3.27)

project(multiRL VERSION 0.5.0 LANGUAGES CXX)

include(GNUInstallDirs)
include(FetchContent)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenMP)

option(
    MULTIRL_SYNC_PACKAGE_SOURCES
    "Sync C++ sources into R and Python package directories"
    ON
)

option(
    MULTIRL_BUILD_PYTHON
    "Build the pybind11 Python shell wrapper"
    OFF
)

option(
    MULTIRL_ENABLE_NLOPT
    "Enable NLopt-backed estimators"
    OFF
)

option(
    MULTIRL_FETCH_NLOPT
    "Fetch NLopt instead of using a system installation"
    OFF
)

if(MULTIRL_SYNC_PACKAGE_SOURCES)
    set(
        MULTIRL_SYNC_SCRIPT
        "${CMAKE_CURRENT_SOURCE_DIR}/SYNC.cmake"
    )

    execute_process(
        COMMAND
            "${CMAKE_COMMAND}"
            "-DREPO_ROOT=${CMAKE_CURRENT_SOURCE_DIR}"
            -P "${MULTIRL_SYNC_SCRIPT}"
        RESULT_VARIABLE MULTIRL_SYNC_RESULT
    )

    if(NOT MULTIRL_SYNC_RESULT EQUAL 0)
        message(
            FATAL_ERROR
            "multiRL package source synchronization failed."
        )
    endif()

    add_custom_target(
        multiRL_sync_package_sources ALL
        COMMAND
            "${CMAKE_COMMAND}"
            "-DREPO_ROOT=${CMAKE_CURRENT_SOURCE_DIR}"
            -P "${MULTIRL_SYNC_SCRIPT}"
        COMMENT "Syncing multiRL package C++ sources"
    )
endif()

if(MSVC)
    add_compile_options(/utf-8 /wd4819 /wd4267 /wd4101)
endif()

set(MULTIRL_NLOPT_TARGET "")
set(MULTIRL_NLOPT_INCLUDE_DIRS "")

if(MULTIRL_ENABLE_NLOPT)
    if(MULTIRL_FETCH_NLOPT)
        set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
        set(NLOPT_SHARED OFF CACHE BOOL "" FORCE)
        set(NLOPT_TESTS OFF CACHE BOOL "" FORCE)
        set(NLOPT_CXX ON CACHE BOOL "" FORCE)
        set(NLOPT_CXX11 OFF CACHE BOOL "" FORCE)
        set(NLOPT_FORTRAN OFF CACHE BOOL "" FORCE)
        set(NLOPT_PYTHON OFF CACHE BOOL "" FORCE)
        set(NLOPT_GUILE OFF CACHE BOOL "" FORCE)
        set(NLOPT_MATLAB OFF CACHE BOOL "" FORCE)
        set(NLOPT_OCTAVE OFF CACHE BOOL "" FORCE)
        set(NLOPT_SWIG OFF CACHE BOOL "" FORCE)

        FetchContent_Declare(
            nlopt
            GIT_REPOSITORY https://github.com/stevengj/nlopt.git
            GIT_TAG master
            GIT_SHALLOW TRUE
        )
        FetchContent_MakeAvailable(nlopt)

        set(MULTIRL_NLOPT_TARGET nlopt)
        list(APPEND MULTIRL_NLOPT_INCLUDE_DIRS "${nlopt_BINARY_DIR}")
        list(APPEND MULTIRL_NLOPT_INCLUDE_DIRS "${nlopt_SOURCE_DIR}/src/api")
    else()
        find_package(NLopt REQUIRED)

        if(TARGET NLopt::nlopt)
            set(MULTIRL_NLOPT_TARGET NLopt::nlopt)
        elseif(TARGET nlopt)
            set(MULTIRL_NLOPT_TARGET nlopt)
        else()
            message(FATAL_ERROR "NLopt has no usable CMake target.")
        endif()
    endif()
endif()

set(
    MULTIRL_SOURCES
    Cpp/src/func_alpha.cpp
    Cpp/src/func_beta.cpp
    Cpp/src/func_gamma.cpp
    Cpp/src/func_delta.cpp
    Cpp/src/func_epsilon.cpp
    Cpp/src/func_zeta.cpp
    Cpp/src/algorithm_nlopt.cpp
    Cpp/src/estimate_map.cpp
    Cpp/src/estimate_mle.cpp
    Cpp/src/info_nlopt.cpp
    Cpp/src/modify_control.cpp
    Cpp/src/process_MDP_free.cpp
    Cpp/src/modify_params.cpp
    Cpp/src/modify_priors.cpp
    Cpp/src/task_builder.cpp
    Cpp/src/shell_run_m.cpp
)

add_library(multiRL STATIC ${MULTIRL_SOURCES})

set_property(TARGET multiRL PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(
    multiRL
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Cpp/include>
        $<INSTALL_INTERFACE:include>
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(multiRL PUBLIC OpenMP::OpenMP_CXX)
endif()

if(MULTIRL_ENABLE_NLOPT)
    target_compile_definitions(multiRL PUBLIC MULTIRL_HAS_NLOPT)
    target_link_libraries(multiRL PUBLIC ${MULTIRL_NLOPT_TARGET})

    foreach(nlopt_include_dir ${MULTIRL_NLOPT_INCLUDE_DIRS})
        target_include_directories(
            multiRL
            PUBLIC
                $<BUILD_INTERFACE:${nlopt_include_dir}>
        )
    endforeach()
endif()

if(MULTIRL_BUILD_PYTHON)
    find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

    execute_process(
        COMMAND
            "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
        OUTPUT_VARIABLE MULTIRL_PYBIND11_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE MULTIRL_PYBIND11_CMAKE_RESULT
    )

    if(NOT MULTIRL_PYBIND11_CMAKE_RESULT EQUAL 0)
        message(FATAL_ERROR "pybind11 is required for Python wrapper tests.")
    endif()

    list(APPEND CMAKE_PREFIX_PATH "${MULTIRL_PYBIND11_CMAKE_DIR}")
    find_package(pybind11 CONFIG REQUIRED)

    pybind11_add_module(
        _shell_run_m
        MODULE
            Python/src/py_shell_run_m.cpp
            ${MULTIRL_SOURCES}
    )

    target_include_directories(
        _shell_run_m
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/Cpp/include"
            "${CMAKE_CURRENT_SOURCE_DIR}/Python/src/include"
    )

    set_target_properties(
        _shell_run_m
        PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python"
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/python"
    )

    if(OpenMP_CXX_FOUND)
        target_link_libraries(_shell_run_m PRIVATE OpenMP::OpenMP_CXX)
    endif()

    if(MULTIRL_ENABLE_NLOPT)
        target_compile_definitions(_shell_run_m PRIVATE MULTIRL_HAS_NLOPT)
        target_link_libraries(_shell_run_m PRIVATE ${MULTIRL_NLOPT_TARGET})
    endif()
endif()

install(
    TARGETS multiRL
    EXPORT multiRL_Targets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(DIRECTORY Cpp/include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
