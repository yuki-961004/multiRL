cmake_minimum_required(VERSION 3.15...3.27)

project(multiRL_Python_Wrapper VERSION 0.5.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS OFF)

set(
    MULTIRL_PY_PACKAGE_NAME
    "multiRL"
    CACHE STRING
    "Python package name used by the multiRL frontend"
)

if(MSVC)
    add_compile_options(/utf-8 /wd4819 /wd4267 /wd4101)
endif()

find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

execute_process(
    COMMAND
        "${Python3_EXECUTABLE}" -m pybind11 --cmakedir
    OUTPUT_VARIABLE MULTIRL_PYBIND11_CMAKE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE MULTIRL_PYBIND11_CMAKE_RESULT
)

if(NOT MULTIRL_PYBIND11_CMAKE_RESULT EQUAL 0)
    message(FATAL_ERROR "pybind11 is required for the Python wrapper.")
endif()

list(APPEND CMAKE_PREFIX_PATH "${MULTIRL_PYBIND11_CMAKE_DIR}")
find_package(pybind11 CONFIG REQUIRED)
find_package(OpenMP)

option(
    MULTIRL_ENABLE_NLOPT
    "Enable NLopt-backed estimators"
    OFF
)

if(MULTIRL_ENABLE_NLOPT)
    find_package(NLopt REQUIRED)

    if(TARGET NLopt::nlopt)
        set(MULTIRL_NLOPT_TARGET NLopt::nlopt)
    elseif(TARGET nlopt)
        set(MULTIRL_NLOPT_TARGET nlopt)
    else()
        message(FATAL_ERROR "NLopt has no usable CMake target.")
    endif()
endif()

set(
    MULTIRL_PY_BACKEND_SOURCES
    src/cpp/func_alpha.cpp
    src/cpp/func_beta.cpp
    src/cpp/func_gamma.cpp
    src/cpp/func_delta.cpp
    src/cpp/func_epsilon.cpp
    src/cpp/func_zeta.cpp
    src/cpp/algorithm_nlopt.cpp
    src/cpp/estimate_map.cpp
    src/cpp/estimate_mle.cpp
    src/cpp/info_nlopt.cpp
    src/cpp/modify_control.cpp
    src/cpp/process_MDP_free.cpp
    src/cpp/modify_params.cpp
    src/cpp/modify_priors.cpp
    src/cpp/task_builder.cpp
    src/cpp/shell_run_m.cpp
)

add_library(multiRL_python_core STATIC ${MULTIRL_PY_BACKEND_SOURCES})
set_property(TARGET multiRL_python_core PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(
    multiRL_python_core
    PUBLIC
        "${CMAKE_CURRENT_SOURCE_DIR}/src/include"
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(
        multiRL_python_core
        PUBLIC
            OpenMP::OpenMP_CXX
    )
endif()

if(MULTIRL_ENABLE_NLOPT)
    target_compile_definitions(multiRL_python_core PUBLIC MULTIRL_HAS_NLOPT)
    target_link_libraries(
        multiRL_python_core
        PUBLIC
            ${MULTIRL_NLOPT_TARGET}
    )
endif()

function(add_multiRL_module module_name source_file)
    pybind11_add_module(${module_name} MODULE ${source_file})

    target_link_libraries(
        ${module_name}
        PRIVATE
            multiRL_python_core
    )

    target_include_directories(
        ${module_name}
        PRIVATE
            "${CMAKE_CURRENT_SOURCE_DIR}/src/include"
    )

    install(TARGETS ${module_name} DESTINATION ${MULTIRL_PY_PACKAGE_NAME})
endfunction()

add_multiRL_module(_backend src/py_shell_run_m.cpp)
