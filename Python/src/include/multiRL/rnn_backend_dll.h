#pragma once

#if defined(_WIN32)
#  define MULTIRL_EXPORT __declspec(dllexport)
#else
#  define MULTIRL_EXPORT
#endif

extern "C" {

MULTIRL_EXPORT int rnn_backend_train(
    const double* x_data,
    int n_draws,
    int n_trials,
    int n_features,
    const double* y_data,
    int n_params,
    const char* architecture,
    const char* loss,
    const char* regularization,
    double penalty,
    int epochs,
    int batch_size,
    int units,
    int layers,
    double dropout,
    double learning_rate,
    int seed,
    int threads,
    int interop_threads,
    const char* device,
    void** out_model_handle,
    char* out_error_msg,
    int error_msg_len
);

MULTIRL_EXPORT int rnn_backend_predict(
    void* model_handle,
    const double* x_data,
    int n_trials,
    int n_features,
    int n_params,
    const char* loss,
    double* out_predictions,
    char* out_error_msg,
    int error_msg_len
);

MULTIRL_EXPORT void rnn_backend_free_model(
    void* model_handle
);

}
