cmake_minimum_required(VERSION 3.15...3.27)

project(multiRL_Python_Wrapper VERSION 0.5.0 LANGUAGES CXX)

include(FetchContent)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_DISABLE_FIND_PACKAGE_Java ON)
set(CMAKE_DISABLE_FIND_PACKAGE_JNI ON)

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
find_package(OpenMP REQUIRED)

option(
    MULTIRL_ENABLE_NLOPT
    "Enable NLopt-backed estimators"
    ON
)

option(
    MULTIRL_FETCH_NLOPT
    "Fetch NLopt instead of using a system installation"
    ON
)

option(
    MULTIRL_ENABLE_MCMC
    "Build MCMC estimator with Stan Math support"
    ON
)

set(
    MULTIRL_STAN_MATH_TAG
    "latest"
    CACHE STRING
    "Stan Math tag to fetch, or 'latest' to query GitHub releases"
)

if(MULTIRL_ENABLE_MCMC AND MULTIRL_STAN_MATH_TAG STREQUAL "latest")
    file(
        DOWNLOAD
        "https://api.github.com/repos/stan-dev/math/releases/latest"
        "${CMAKE_BINARY_DIR}/stan_release.json"
        STATUS MULTIRL_STAN_RELEASE_STATUS
        TLS_VERIFY ON
    )

    list(GET MULTIRL_STAN_RELEASE_STATUS 0 MULTIRL_STAN_RELEASE_CODE)
    list(GET MULTIRL_STAN_RELEASE_STATUS 1 MULTIRL_STAN_RELEASE_MESSAGE)

    if(NOT MULTIRL_STAN_RELEASE_CODE EQUAL 0)
        message(
            FATAL_ERROR
            "Failed to resolve latest Stan Math release: "
            "${MULTIRL_STAN_RELEASE_MESSAGE}. "
            "Set -DMULTIRL_STAN_MATH_TAG=v5.3.0 for an offline build."
        )
    endif()

    file(READ "${CMAKE_BINARY_DIR}/stan_release.json" MULTIRL_STAN_RELEASE)
    string(
        JSON MULTIRL_STAN_MATH_TAG
        GET "${MULTIRL_STAN_RELEASE}" "tag_name"
    )

    message(STATUS "Stan Math auto-resolved tag: ${MULTIRL_STAN_MATH_TAG}")
endif()

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

set(MULTIRL_STAN_MATH_INCLUDE_DIRS "")

if(MULTIRL_ENABLE_MCMC)
    FetchContent_Declare(
        stan_math
        GIT_REPOSITORY https://github.com/stan-dev/math.git
        GIT_TAG ${MULTIRL_STAN_MATH_TAG}
        GIT_SHALLOW TRUE
        GIT_SUBMODULES_RECURSE TRUE
    )
    FetchContent_MakeAvailable(stan_math)

    file(
        GLOB MULTIRL_STAN_EIGEN_DIRS
        CONFIGURE_DEPENDS
        "${stan_math_SOURCE_DIR}/lib/eigen_*"
    )
    file(
        GLOB MULTIRL_STAN_BOOST_DIRS
        CONFIGURE_DEPENDS
        "${stan_math_SOURCE_DIR}/lib/boost_*"
    )
    file(
        GLOB MULTIRL_STAN_SUNDIALS_DIRS
        CONFIGURE_DEPENDS
        "${stan_math_SOURCE_DIR}/lib/sundials_*"
    )
    file(
        GLOB MULTIRL_STAN_TBB_DIRS
        CONFIGURE_DEPENDS
        "${stan_math_SOURCE_DIR}/lib/tbb_*"
    )

    list(APPEND MULTIRL_STAN_MATH_INCLUDE_DIRS "${stan_math_SOURCE_DIR}")
    list(APPEND MULTIRL_STAN_MATH_INCLUDE_DIRS ${MULTIRL_STAN_EIGEN_DIRS})
    list(APPEND MULTIRL_STAN_MATH_INCLUDE_DIRS ${MULTIRL_STAN_BOOST_DIRS})

    foreach(stan_sundials_dir ${MULTIRL_STAN_SUNDIALS_DIRS})
        list(
            APPEND MULTIRL_STAN_MATH_INCLUDE_DIRS
            "${stan_sundials_dir}/include"
        )
    endforeach()

    foreach(stan_tbb_dir ${MULTIRL_STAN_TBB_DIRS})
        list(
            APPEND MULTIRL_STAN_MATH_INCLUDE_DIRS
            "${stan_tbb_dir}/include"
        )
    endforeach()
endif()

set(
    MULTIRL_PY_BASE_SOURCES
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
    src/cpp/modify_context.cpp
    src/cpp/modify_funcs.cpp
    src/cpp/modify_outputs.cpp
    src/cpp/process_MDP_free.cpp
    src/cpp/modify_params.cpp
    src/cpp/modify_priors.cpp
    src/cpp/task_builder.cpp
    src/cpp/shell_run_m.cpp
)

set(
    MULTIRL_PY_BACKEND_SOURCES
    ${MULTIRL_PY_BASE_SOURCES}
)

if(MULTIRL_ENABLE_MCMC)
    list(
        APPEND MULTIRL_PY_BACKEND_SOURCES
        src/cpp/algorithm_stan.cpp
        src/cpp/estimate_mcmc.cpp
    )
endif()

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

if(MULTIRL_ENABLE_MCMC)
    target_compile_definitions(multiRL_python_core PUBLIC MULTIRL_HAS_STAN)
    target_include_directories(
        multiRL_python_core
        PRIVATE
            ${MULTIRL_STAN_MATH_INCLUDE_DIRS}
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

    if(MULTIRL_ENABLE_MCMC)
        target_compile_definitions(${module_name} PRIVATE MULTIRL_HAS_STAN)
        target_include_directories(
            ${module_name}
            PRIVATE
                ${MULTIRL_STAN_MATH_INCLUDE_DIRS}
        )
    endif()

    install(TARGETS ${module_name} DESTINATION ${MULTIRL_PY_PACKAGE_NAME})
endfunction()

add_multiRL_module(_backend src/py_shell_run_m.cpp)

message(STATUS "Stan Math tag ${MULTIRL_STAN_MATH_TAG}")
