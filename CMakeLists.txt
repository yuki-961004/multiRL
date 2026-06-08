cmake_minimum_required(VERSION 3.15...3.27)

project(multiRL VERSION 0.5.0 LANGUAGES CXX)
# -- Workaround for CMake 4.3 RC detection bug with MSVC
if(MSVC AND CMAKE_VERSION VERSION_GREATER_EQUAL "4.3")
    enable_language(RC)
endif()

include(GNUInstallDirs)
include(FetchContent)

function(multirl_read_define out_var file_path define_name)
    set(version_value "unknown")

    if(EXISTS "${file_path}")
        file(
            STRINGS "${file_path}" version_line
            REGEX "#define[ \t]+${define_name}[ \t]+"
            LIMIT_COUNT 1
        )

        if(version_line)
            string(
                REGEX REPLACE
                ".*#define[ \t]+${define_name}[ \t]+\"?([^\" \t]+)\"?.*"
                "\\1"
                version_value
                "${version_line}"
            )
        endif()
    endif()

    set(${out_var} "${version_value}" PARENT_SCOPE)
endfunction()

function(multirl_read_package_version out_var file_path)
    set(version_value "unknown")

    if(EXISTS "${file_path}")
        file(
            STRINGS "${file_path}" version_line
            REGEX "PACKAGE_VERSION"
            LIMIT_COUNT 1
        )

        if(version_line)
            string(
                REGEX REPLACE
                ".*\"([^\"]+)\".*"
                "\\1"
                version_value
                "${version_line}"
            )
        endif()
    endif()

    set(${out_var} "${version_value}" PARENT_SCOPE)
endfunction()

function(multirl_read_project_version out_var file_path)
    set(version_value "unknown")

    if(EXISTS "${file_path}")
        file(
            STRINGS "${file_path}" version_line
            REGEX "project\\([^)]*VERSION[ \t]+"
            LIMIT_COUNT 1
        )

        if(version_line)
            string(
                REGEX REPLACE
                ".*VERSION[ \t]+([^ \t\\)]+).*"
                "\\1"
                version_value
                "${version_line}"
            )
        endif()
    endif()

    set(${out_var} "${version_value}" PARENT_SCOPE)
endfunction()

function(multirl_read_nlopt_version out_var source_dir)
    set(version_value "unknown")
    set(version_file "${source_dir}/CMakeLists.txt")

    if(EXISTS "${version_file}")
        foreach(version_part MAJOR MINOR BUGFIX)
            file(
                STRINGS "${version_file}" version_line
                REGEX "set[ \t]*\\([ \t]*NLOPT_${version_part}_VERSION"
                LIMIT_COUNT 1
            )

            if(version_line)
                string(
                    REGEX REPLACE
                    ".*\"([0-9]+)\".*"
                    "\\1"
                    "version_${version_part}"
                    "${version_line}"
                )
            endif()
        endforeach()

        if(DEFINED version_MAJOR
                AND DEFINED version_MINOR
                AND DEFINED version_BUGFIX)
            set(
                version_value
                "${version_MAJOR}.${version_MINOR}.${version_BUGFIX}"
            )
        endif()
    endif()

    set(${out_var} "${version_value}" PARENT_SCOPE)
endfunction()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)
set(CMAKE_CXX_EXTENSIONS OFF)
# -- Install to local directory (avoids admin permission issues)
if(CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_BINARY_DIR}/install" CACHE PATH "Local install prefix" FORCE)
endif()
# -- CMake 4.3 workaround: enable C before FetchContent sub-builds need it
if(MSVC)
    enable_language(C)
endif()

set(CMAKE_DISABLE_FIND_PACKAGE_Java ON)
set(CMAKE_DISABLE_FIND_PACKAGE_JNI ON)

find_package(OpenMP REQUIRED)

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

option(
    MULTIRL_ENABLE_RNN_TORCH
    "Build RNN estimator with LibTorch support"
    ON
)

option(
    MULTIRL_FETCH_LIBTORCH
    "Fetch LibTorch when MULTIRL_ENABLE_RNN_TORCH is ON"
    ON
)

set(
    MULTIRL_LIBTORCH_URL
    ""
    CACHE STRING
    "Optional LibTorch archive URL. Empty uses a platform default."
)

set(
    MULTIRL_LIBTORCH_CUDA
    "auto"
    CACHE STRING
    "LibTorch CUDA variant: auto, cpu, cu132, cu130, cu128, etc."
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
            WARNING
            "Failed to resolve latest Stan Math release: "
            "${MULTIRL_STAN_RELEASE_MESSAGE}. "
            "Falling back to v5.3.0."
        )
        set(MULTIRL_STAN_MATH_TAG "v5.3.0")
    else()
        file(READ "${CMAKE_BINARY_DIR}/stan_release.json" MULTIRL_STAN_RELEASE)
        string(
            JSON MULTIRL_STAN_MATH_TAG
            GET "${MULTIRL_STAN_RELEASE}" "tag_name"
        )
        message(STATUS "Stan Math auto-resolved tag: ${MULTIRL_STAN_MATH_TAG}")
    endif()
endif()
# -- Validate Stan tag (shells may truncate v5.3.0 to v5)
string(REGEX MATCH "^v[0-9]+\\.[0-9]+\\.[0-9]+$" _stan_tag_ok "${MULTIRL_STAN_MATH_TAG}")
if(NOT _stan_tag_ok AND NOT MULTIRL_STAN_MATH_TAG STREQUAL "latest")
    message(WARNING "Stan Math tag '${MULTIRL_STAN_MATH_TAG}' may have been truncated by the shell. Use quotes: -DMULTIRL_STAN_MATH_TAG=\"v5.3.0\"")
endif()


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

    if(MSVC)
        set_target_properties(
            multiRL_sync_package_sources
            PROPERTIES
                VS_GLOBAL_IntDir
                    "$(Platform)\\$(Configuration)\\multiRL_sync_package_sources\\"
        )
    endif()
endif()

if(MSVC)
    add_compile_options(/utf-8 /wd4819 /wd4267 /wd4101)
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

if(MULTIRL_ENABLE_RNN_TORCH)
    if(MULTIRL_FETCH_LIBTORCH)
        if(MULTIRL_LIBTORCH_URL STREQUAL "")
            set(MULTIRL_EFFECTIVE_LIBTORCH_CUDA "${MULTIRL_LIBTORCH_CUDA}")

            if(MULTIRL_EFFECTIVE_LIBTORCH_CUDA STREQUAL "auto")
                find_program(MULTIRL_NVCC nvcc)

                if(MULTIRL_NVCC)
                    execute_process(
                        COMMAND "${MULTIRL_NVCC}" --version
                        OUTPUT_VARIABLE MULTIRL_NVCC_OUTPUT
                        OUTPUT_STRIP_TRAILING_WHITESPACE
                    )
                    string(
                        REGEX MATCH
                        "release [0-9]+\\.[0-9]+"
                        MULTIRL_CUDA_RELEASE
                        "${MULTIRL_NVCC_OUTPUT}"
                    )
                    string(
                        REGEX REPLACE
                        "release "
                        ""
                        MULTIRL_CUDA_VERSION
                        "${MULTIRL_CUDA_RELEASE}"
                    )
                    string(
                        REPLACE
                        "."
                        ""
                        MULTIRL_CUDA_SHORT
                        "${MULTIRL_CUDA_VERSION}"
                    )
                    set(MULTIRL_EFFECTIVE_LIBTORCH_CUDA "cu${MULTIRL_CUDA_SHORT}")
                else()
                    set(MULTIRL_EFFECTIVE_LIBTORCH_CUDA "cpu")
                endif()
            endif()

            if(WIN32)
                if(MULTIRL_EFFECTIVE_LIBTORCH_CUDA STREQUAL "cpu")
                    set(
                        MULTIRL_LIBTORCH_URL
                        "https://download.pytorch.org/libtorch/cpu/libtorch-win-shared-with-deps-latest.zip"
                        CACHE STRING
                        "Optional LibTorch archive URL. Empty uses a platform default."
                        FORCE
                    )
                else()
                    set(
                        MULTIRL_LIBTORCH_URL
                        "https://download.pytorch.org/libtorch/nightly/${MULTIRL_EFFECTIVE_LIBTORCH_CUDA}/libtorch-win-shared-with-deps-latest.zip"
                        CACHE STRING
                        "Optional LibTorch archive URL. Empty uses a platform default."
                        FORCE
                    )
                endif()
            elseif(APPLE)
                set(
                    MULTIRL_LIBTORCH_URL
                    "https://download.pytorch.org/libtorch/cpu/libtorch-macos-arm64-latest.zip"
                    CACHE STRING
                    "Optional LibTorch archive URL. Empty uses a platform default."
                    FORCE
                )
            else()
                if(MULTIRL_EFFECTIVE_LIBTORCH_CUDA STREQUAL "cpu")
                    set(
                        MULTIRL_LIBTORCH_URL
                        "https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-latest.zip"
                        CACHE STRING
                        "Optional LibTorch archive URL. Empty uses a platform default."
                        FORCE
                    )
                else()
                    set(
                        MULTIRL_LIBTORCH_URL
                        "https://download.pytorch.org/libtorch/nightly/${MULTIRL_EFFECTIVE_LIBTORCH_CUDA}/libtorch-shared-with-deps-latest.zip"
                        CACHE STRING
                        "Optional LibTorch archive URL. Empty uses a platform default."
                        FORCE
                    )
                endif()
            endif()
        endif()

        message(STATUS "LibTorch URL: ${MULTIRL_LIBTORCH_URL}")

        FetchContent_Declare(
            libtorch
            URL ${MULTIRL_LIBTORCH_URL}
        )

        FetchContent_MakeAvailable(libtorch)
        FetchContent_GetProperties(libtorch)
        list(APPEND CMAKE_PREFIX_PATH "${libtorch_SOURCE_DIR}")
    endif()

    if(WIN32 AND NOT MULTIRL_EFFECTIVE_LIBTORCH_CUDA STREQUAL "cpu")
        set(TORCH_CUDA_ARCH_LIST "8.6" CACHE STRING "" FORCE)
        unset(CMAKE_CUDA_ARCHITECTURES)
        unset(CMAKE_CUDA_ARCHITECTURES CACHE)
    endif()

    set(MULTIRL_PREVIOUS_MESSAGE_LOG_LEVEL "${CMAKE_MESSAGE_LOG_LEVEL}")
    set(CMAKE_MESSAGE_LOG_LEVEL ERROR)
    find_package(Torch CONFIG REQUIRED)
    if(MULTIRL_PREVIOUS_MESSAGE_LOG_LEVEL STREQUAL "")
        unset(CMAKE_MESSAGE_LOG_LEVEL)
    else()
        set(
            CMAKE_MESSAGE_LOG_LEVEL
            "${MULTIRL_PREVIOUS_MESSAGE_LOG_LEVEL}"
        )
    endif()
    unset(MULTIRL_PREVIOUS_MESSAGE_LOG_LEVEL)

    set(MULTIRL_TORCH_VERSION "unknown")

    if(DEFINED Torch_VERSION)
        set(MULTIRL_TORCH_VERSION "${Torch_VERSION}")
    elseif(DEFINED TORCH_VERSION)
        set(MULTIRL_TORCH_VERSION "${TORCH_VERSION}")
    elseif(DEFINED libtorch_SOURCE_DIR)
        multirl_read_package_version(
            MULTIRL_TORCH_VERSION
            "${libtorch_SOURCE_DIR}/share/cmake/Torch/TorchConfigVersion.cmake"
        )
    endif()
endif()

set(MULTIRL_EIGEN_SOURCE_DIR "${CMAKE_BINARY_DIR}/_deps/eigen-src")
set(MULTIRL_EIGEN_BINARY_DIR "${CMAKE_BINARY_DIR}/_deps/eigen-build")
set(MULTIRL_EIGEN_SUBBUILD_DIR "${CMAKE_BINARY_DIR}/_deps/eigen-subbuild")

if(NOT EXISTS "${MULTIRL_EIGEN_SOURCE_DIR}/Eigen")
    FetchContent_Populate(
        eigen
        GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
        GIT_TAG master
        GIT_SHALLOW TRUE
        SOURCE_DIR "${MULTIRL_EIGEN_SOURCE_DIR}"
        BINARY_DIR "${MULTIRL_EIGEN_BINARY_DIR}"
        SUBBUILD_DIR "${MULTIRL_EIGEN_SUBBUILD_DIR}"
    )
else()
    set(eigen_SOURCE_DIR "${MULTIRL_EIGEN_SOURCE_DIR}")
endif()

multirl_read_define(
    MULTIRL_EIGEN_VERSION
    "${eigen_SOURCE_DIR}/Eigen/Version"
    "EIGEN_VERSION_STRING"
)

multirl_read_project_version(
    MULTIRL_ABCPP_VERSION
    "${CMAKE_CURRENT_SOURCE_DIR}/abcpp/CMakeLists.txt"
)

set(MULTIRL_NLOPT_TARGET "")
set(MULTIRL_NLOPT_INCLUDE_DIRS "")
set(MULTIRL_NLOPT_VERSION "unknown")

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

        set(MULTIRL_PREVIOUS_MESSAGE_LOG_LEVEL "${CMAKE_MESSAGE_LOG_LEVEL}")
        set(CMAKE_MESSAGE_LOG_LEVEL WARNING)
        FetchContent_MakeAvailable(nlopt)

        if(MULTIRL_PREVIOUS_MESSAGE_LOG_LEVEL STREQUAL "")
            unset(CMAKE_MESSAGE_LOG_LEVEL)
        else()
            set(
                CMAKE_MESSAGE_LOG_LEVEL
                "${MULTIRL_PREVIOUS_MESSAGE_LOG_LEVEL}"
            )
        endif()
        unset(MULTIRL_PREVIOUS_MESSAGE_LOG_LEVEL)

        set(MULTIRL_NLOPT_TARGET nlopt)
        list(APPEND MULTIRL_NLOPT_INCLUDE_DIRS "${nlopt_BINARY_DIR}")
        list(APPEND MULTIRL_NLOPT_INCLUDE_DIRS "${nlopt_SOURCE_DIR}/src/api")

        if(DEFINED NLOPT_VERSION_STRING)
            set(MULTIRL_NLOPT_VERSION "${NLOPT_VERSION_STRING}")
        endif()

        if(MULTIRL_NLOPT_VERSION STREQUAL "unknown")
            multirl_read_nlopt_version(
                MULTIRL_NLOPT_VERSION
                "${nlopt_SOURCE_DIR}"
            )
        endif()
    else()
        find_package(NLopt REQUIRED)

        if(TARGET NLopt::nlopt)
            set(MULTIRL_NLOPT_TARGET NLopt::nlopt)
        elseif(TARGET nlopt)
            set(MULTIRL_NLOPT_TARGET nlopt)
        else()
            message(FATAL_ERROR "NLopt has no usable CMake target.")
        endif()

        if(DEFINED NLopt_VERSION)
            set(MULTIRL_NLOPT_VERSION "${NLopt_VERSION}")
        elseif(DEFINED NLOPT_VERSION)
            set(MULTIRL_NLOPT_VERSION "${NLOPT_VERSION}")
        endif()
    endif()
endif()

set(
    MULTIRL_BASE_SOURCES
    Cpp/src/func_alpha.cpp
    Cpp/src/func_beta.cpp
    Cpp/src/func_gamma.cpp
    Cpp/src/func_delta.cpp
    Cpp/src/func_epsilon.cpp
    Cpp/src/func_zeta.cpp
    Cpp/src/algorithm_nlopt.cpp
    Cpp/src/estimate_abc.cpp
    Cpp/src/estimate_map.cpp
    Cpp/src/estimate_mle.cpp
    Cpp/src/estimate_rnn.cpp
    Cpp/src/info_nlopt.cpp
    Cpp/src/modify_control.cpp
    Cpp/src/modify_context.cpp
    Cpp/src/modify_funcs.cpp
    Cpp/src/process_model_free.cpp
    Cpp/src/modify_params.cpp
    Cpp/src/modify_priors.cpp
    Cpp/src/task_sampler.cpp
    Cpp/src/task_builder.cpp
    Cpp/src/shell_fit_p.cpp
    Cpp/src/shell_rcv_d.cpp
    Cpp/src/shell_run_m.cpp
)

set(
    MULTIRL_SOURCES
    ${MULTIRL_BASE_SOURCES}
)

if(MULTIRL_ENABLE_MCMC)
    list(
        APPEND MULTIRL_SOURCES
        Cpp/src/algorithm_stan.cpp
        Cpp/src/estimate_mcmc.cpp
    )
endif()

add_library(multiRL STATIC ${MULTIRL_SOURCES})

set_property(TARGET multiRL PROPERTY POSITION_INDEPENDENT_CODE ON)

target_include_directories(
    multiRL
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/Cpp/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/abcpp/Cpp/include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/abcpp/R/inst/include>
        $<BUILD_INTERFACE:${eigen_SOURCE_DIR}>
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

if(MULTIRL_ENABLE_MCMC)
    target_include_directories(
        multiRL
        PRIVATE
            ${MULTIRL_STAN_MATH_INCLUDE_DIRS}
    )

    target_compile_definitions(multiRL PUBLIC MULTIRL_HAS_STAN)
endif()

if(MULTIRL_ENABLE_RNN_TORCH)
    target_compile_definitions(multiRL PUBLIC MULTIRL_HAS_TORCH)
    target_link_libraries(multiRL PUBLIC "${TORCH_LIBRARIES}")
    target_compile_options(multiRL PUBLIC ${TORCH_CXX_FLAGS})
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

    if(MULTIRL_ENABLE_MCMC)
        target_compile_definitions(_shell_run_m PRIVATE MULTIRL_HAS_STAN)
        target_include_directories(
            _shell_run_m
            PRIVATE
                ${MULTIRL_STAN_MATH_INCLUDE_DIRS}
        )
    endif()

    if(MULTIRL_ENABLE_RNN_TORCH)
        target_compile_definitions(_shell_run_m PRIVATE MULTIRL_HAS_TORCH)
        target_link_libraries(_shell_run_m PRIVATE "${TORCH_LIBRARIES}")
        target_compile_options(_shell_run_m PRIVATE ${TORCH_CXX_FLAGS})
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
install(DIRECTORY ${eigen_SOURCE_DIR}/Eigen DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

message(STATUS "multiRL dependency: NLopt ${MULTIRL_NLOPT_VERSION}")
message(STATUS "multiRL dependency: abcpp ${MULTIRL_ABCPP_VERSION}")
message(STATUS "multiRL dependency: Eigen ${MULTIRL_EIGEN_VERSION}")

if(MULTIRL_ENABLE_MCMC)
    message(STATUS "multiRL dependency: Stan Math ${MULTIRL_STAN_MATH_TAG}")
endif()

if(MULTIRL_ENABLE_RNN_TORCH)
    message(STATUS "multiRL dependency: LibTorch ${MULTIRL_TORCH_VERSION}")
endif()
