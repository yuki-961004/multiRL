# Windows R Dynamic DLL Architecture for LibTorch Backend

This document describes the double-DLL isolation architecture designed for `multiRL` on Windows, addressing the toolchain and ABI incompatibility between Rtools (GCC) and LibTorch (MSVC).

---

## 1. Problem Statement

On Windows, the standard R package build toolchain is **Rtools**, which utilizes the **GCC (MinGW-w64)** compiler.
However, official Windows binaries of **LibTorch** (PyTorch C++ frontend) are compiled using **MSVC (Microsoft Visual Studio)**.

Mixing GCC and MSVC binaries on Windows leads to severe runtime crashes and linker errors due to:
1. **ABI Mismatch**: GCC and MSVC use different name mangling schemes, structure alignments, and calling conventions.
2. **Standard Library Incompatibilities**: `std::vector`, `std::string`, and memory allocators (MSVCRT vs. UCRT/MinGW) are not binary-compatible between standard libraries (`libc++`/`libstdc++` vs. MSVC's `msvcprt`).
3. **Thread Local Storage (TLS) & Exceptions**: Exception handling (`throw`/`catch`) and TLS mechanisms differ incompatibly.

As a result, compiling a single unified DLL for the R package that links against MSVC LibTorch using Rtools GCC is not feasible.

---

## 2. Double-DLL Isolation Architecture

To support Windows R users without requiring them to compile using MSVC (which is not standard for R package builds), we separate the codebase into two distinct physical binaries:

1. **`multiRL.dll` (The Main Package DLL)**:
   - Compiled by Rtools GCC.
   - Contains all core RL models (NLopt, Stan, math utils).
   - Compiled with `MULTIRL_ENABLE_RNN=OFF`.
   - Contains the wrapper functions `estimate_rnn` which act as a router.

2. **`multiRL_torch_backend.dll` (The Torch Backend DLL)**:
   - Compiled using MSVC (via a separate workflow or pre-compiled binary release).
   - Contains the full LibTorch library, helper algorithms (`algorithm_torch.cpp`), and training/inference execution logic.
   - Links dynamically/statically against MSVC LibTorch.

```mermaid
graph TD
    subgraph R Session (Rtools GCC Environment)
        R_Code[R estimate_rnn Wrapper] -->|dyn.load / LoadLibrary| Main_DLL[multiRL.dll]
    end

    subgraph MSVC Environment (Prebuilt/Downloadable)
        Main_DLL -->|Flat C ABI calls| Torch_DLL[multiRL_torch_backend.dll]
        Torch_DLL -->|C++ ABI calls| LibTorch[LibTorch DLLs]
    end
```

---

## 3. Flat C ABI Boundary

To safely cross the GCC-MSVC compiler boundary, the communication interface between `multiRL.dll` and `multiRL_torch_backend.dll` must use a **flat C ABI** (no C++ features, classes, templates, STL containers, or `std::exception` propagation).

### C API Declarations

The `multiRL_torch_backend.dll` will export the following flat C symbols:

```c
#if defined(_WIN32)
#  define MULTIRL_EXPORT __declspec(dllexport)
#else
#  define MULTIRL_EXPORT
#endif

extern "C" {

/**
 * Trains an RNN model and returns an opaque pointer to the trained model.
 * Returns 0 on success, or a non-zero error code.
 */
MULTIRL_EXPORT int rnn_backend_train(
    const double* x_data,       // Flattened training features [n_draws * n_trials * n_features]
    int n_draws,
    int n_trials,
    int n_features,
    const double* y_data,       // Flattened training targets [n_draws * n_params]
    int n_params,
    const char* layer,          // "gru", "lstm", etc.
    const char* loss,           // "mse", "nll", etc.
    int epochs,
    int batch_size,
    int units,
    int layers,
    double dropout,
    double learning_rate,
    int seed,
    int threads,
    int interop_threads,
    const char* device,         // "cpu" or "gpu"
    void** out_model_handle,    // Pointer to hold opaque handle to the trained model
    char* out_error_msg,        // Buffer for error message if any
    int error_msg_len
);

/**
 * Predicts parameters for a single subject.
 * Returns 0 on success, or a non-zero error code.
 */
MULTIRL_EXPORT int rnn_backend_predict(
    void* model_handle,         // Opaque model pointer
    const double* x_data,       // Flattened sequence features [n_trials * n_features]
    int n_trials,
    int n_features,
    int n_params,
    const char* loss,           // Needed for prediction output decoding (QRL/MDN/NLL)
    double* out_predictions,    // Output buffer for point estimates [n_params]
    char* out_error_msg,
    int error_msg_len
);

/**
 * Frees the memory allocated for the trained model.
 */
MULTIRL_EXPORT void rnn_backend_free_model(
    void* model_handle
);

}
```

### Safety Rules at the Boundary
- **Memory Management**: Memory allocated inside `multiRL_torch_backend.dll` (like the model parameters or internal buffers) must be freed by `rnn_backend_free_model` inside the same DLL. Never cross-deallocate.
- **Exception Boundary**: Every function inside `multiRL_torch_backend.dll` must wrap its body in a `try-catch` block. Any exception caught must be translated to a numeric return code and an error string copied to the caller's buffer.
- **Structure Padding**: Only primitive types (`double`, `int`, `char*`, `void*`) or arrays thereof are passed. Structs are avoided to prevent alignment/padding mismatches between GCC and MSVC.

---

## 4. Loading Workflow in R

When the user calls `estimate_rnn(..., control = list(backend = "torch"))` on Windows R:

1. **Detect DLL**: The main package checks for the presence of `multiRL_torch_backend.dll` in the package library folder or user-defined path.
2. **Dynamic Loading**:
   - If not found, it prompts the user with instructions to download the backend DLL (or falls back gracefully to a non-RNN mode).
   - If found, R uses `dyn.load()` or R's C API internal functions (`GetProcAddress`) to load the DLL and resolve symbols.
3. **Execute**: The main package calls `rnn_backend_train` and `rnn_backend_predict` via standard R/C `.Call` interfaces using the dynamic function pointers.
4. **Unload**: Once computation is completed (or when the R package is detached), `dyn.unload()` is called.
