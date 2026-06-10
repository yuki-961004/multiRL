// c_backend.cpp
// Flat C ABI wrapper around the LibTorch RNN backend.
// Compiled as a separate SHARED library (multiRL_torch_backend.dll on Windows)
// using MSVC, so R's Rtools/GCC can dynamically load it at runtime.
//
// ALL data crossing the DLL boundary is plain C types only:
//   double*, int, char*, void*
// No STL, no Torch types, no Eigen cross the boundary.

#include <multiRL/c_backend.h>
#include <multiRL/algorithm_torch.hpp>
#include <multiRL/modify_control.hpp>

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef MULTIRL_HAS_TORCH
#include <torch/torch.h>
#include <ATen/Context.h>
#endif

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------
namespace {

void set_error(char* buf, int buf_len, const char* msg) {
    if (buf && buf_len > 0) {
        std::strncpy(buf, msg, static_cast<std::size_t>(buf_len) - 1U);
        buf[buf_len - 1] = '\0';
    }
}

#ifdef MULTIRL_HAS_TORCH

// Helper for uniform sizes computation
std::size_t product(std::initializer_list<int64_t> sizes) {
    std::size_t n = 1;
    for (int64_t s : sizes) { n *= static_cast<std::size_t>(s); }
    return n;
}

// Build a torch::Tensor from a raw double* array.
// The caller owns the source data; we clone to detach from it.
torch::Tensor make_tensor(
    const double* data,
    std::initializer_list<int64_t> sizes
) {
    const std::size_t n = product(sizes);
    std::vector<float> f(n);
    std::transform(
        data,
        data + n,
        f.begin(),
        [](double value) {
            return static_cast<float>(value);
        }
    );
    return torch::from_blob(
        f.data(), sizes, torch::kFloat32
    ).clone();
}

#endif  // MULTIRL_HAS_TORCH

}  // namespace

// ---------------------------------------------------------------------------
// rnn_backend_train
// ---------------------------------------------------------------------------
extern "C" MULTIRL_EXPORT int rnn_backend_train(
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
) {
#ifndef MULTIRL_HAS_TORCH
    set_error(out_error_msg, error_msg_len,
        "multiRL_torch_backend was compiled without LibTorch support.");
    return -1;
#else
    if (!x_data || !y_data || !out_model_handle) {
        set_error(out_error_msg, error_msg_len, "Null pointer passed to rnn_backend_train.");
        return -1;
    }
    *out_model_handle = nullptr;

    try {
        // Build RNNControl from the flat C arguments.
        multiRL::RNNControl ctrl;
        ctrl.architecture   = architecture   ? architecture   : "gru";
        ctrl.loss           = loss           ? loss           : "mse";
        ctrl.regularization = regularization ? regularization : "none";
        ctrl.penalty        = penalty;
        ctrl.epochs         = epochs > 0 ? epochs : 100;
        ctrl.epoch          = ctrl.epochs;
        ctrl.batch_size     = batch_size > 0 ? batch_size : 32;
        ctrl.units          = units > 0 ? units : 32;
        ctrl.layers         = layers > 0 ? layers : 1;
        ctrl.dropout        = dropout;
        ctrl.learning_rate  = learning_rate > 0.0 ? learning_rate : 0.001;
        ctrl.seed           = seed;
        ctrl.threads        = threads;
        ctrl.interop_threads = interop_threads;
        ctrl.device         = device ? device : "cpu";
        ctrl.n_draws        = n_draws;
        ctrl.sample         = n_draws;

        // Apply defaults / normalization.
        ctrl = multiRL::modify_control(ctrl, "rnn");
        multiRL::validate_rnn_control(ctrl);

        // Thread / device setup.
        if (ctrl.threads > 0)          { torch::set_num_threads(ctrl.threads); }
        if (ctrl.interop_threads > 0)  { torch::set_num_interop_threads(ctrl.interop_threads); }
        torch::manual_seed(static_cast<uint64_t>(ctrl.seed));

        torch::Device torch_device(torch::kCPU);
        bool use_cuda = (ctrl.device == "gpu");
        if (use_cuda) {
            if (!torch::cuda::is_available()) {
                set_error(out_error_msg, error_msg_len,
                    "device='gpu' requested but no CUDA device is available.");
                return -2;
            }
            torch_device = torch::Device(torch::kCUDA);
            at::globalContext().setDeterministicAlgorithms(true, true);
        }

        // Copy x and y from double* into float tensors.
        const std::size_t x_size = static_cast<std::size_t>(n_draws * n_trials * n_features);
        std::vector<float> xf(x_size);
        for (std::size_t i = 0; i < x_size; ++i) { xf[i] = static_cast<float>(x_data[i]); }

        const std::size_t y_size = static_cast<std::size_t>(n_draws * n_params);
        std::vector<float> yf(y_size);
        for (std::size_t i = 0; i < y_size; ++i) { yf[i] = static_cast<float>(y_data[i]); }

        torch::Tensor x_train = torch::from_blob(
            xf.data(), {n_draws, n_trials, n_features}, torch::kFloat32
        ).clone().to(torch_device);

        torch::Tensor y_train = torch::from_blob(
            yf.data(), {n_draws, n_params}, torch::kFloat32
        ).clone().to(torch_device);

        // Build model.
        auto* trained = new multiRL::TrainedRnnModel();
        trained->model   = multiRL::TorchSequenceNet(n_features, n_params, ctrl);
        trained->device  = torch_device;
        trained->n_features = n_features;
        trained->n_params   = n_params;
        trained->model->to(torch_device);

        // Optimizer.
        torch::optim::Adam optimizer(
            trained->model->parameters(),
            torch::optim::AdamOptions(ctrl.learning_rate)
        );

        // Training loop.
        const int eff_batch = std::min(ctrl.batch_size, n_draws);
        double final_loss = 0.0;
        for (int ep = 0; ep < ctrl.epochs; ++ep) {
            trained->model->train();
            torch::Tensor order = torch::randperm(n_draws, torch::kLong).to(torch_device);
            for (int start = 0; start < n_draws; start += eff_batch) {
                const int stop = std::min(start + eff_batch, n_draws);
                torch::Tensor idx = order.slice(0, start, stop);
                torch::Tensor xb  = x_train.index_select(0, idx);
                torch::Tensor yb  = y_train.index_select(0, idx);

                optimizer.zero_grad();
                torch::Tensor pred = trained->model->forward(xb);
                torch::Tensor loss_val = multiRL::compute_rnn_loss(pred, yb, ctrl.loss, n_params);
                torch::Tensor reg_val  = trained->model->get_regularization_loss(
                    ctrl.regularization, ctrl.penalty);
                torch::Tensor total = loss_val + reg_val;
                total.backward();
                optimizer.step();
                final_loss = total.item<double>();
            }
        }
        trained->final_loss = final_loss;

        *out_model_handle = static_cast<void*>(trained);
        return 0;

    } catch (const std::exception& ex) {
        set_error(out_error_msg, error_msg_len, ex.what());
        return -1;
    } catch (...) {
        set_error(out_error_msg, error_msg_len, "Unknown error in rnn_backend_train.");
        return -1;
    }
#endif
}

// ---------------------------------------------------------------------------
// rnn_backend_predict
// ---------------------------------------------------------------------------
extern "C" MULTIRL_EXPORT int rnn_backend_predict(
    void* model_handle,
    const double* x_data,
    int n_trials,
    int n_features,
    int n_params,
    const char* loss,
    double* out_predictions,
    char* out_error_msg,
    int error_msg_len
) {
#ifndef MULTIRL_HAS_TORCH
    set_error(out_error_msg, error_msg_len,
        "multiRL_torch_backend was compiled without LibTorch support.");
    return -1;
#else
    if (!model_handle || !x_data || !out_predictions) {
        set_error(out_error_msg, error_msg_len,
            "Null pointer passed to rnn_backend_predict.");
        return -1;
    }

    try {
        auto* trained = static_cast<multiRL::TrainedRnnModel*>(model_handle);
        const std::string loss_str = loss ? loss : "mse";

        const std::size_t x_size = static_cast<std::size_t>(n_trials * n_features);
        std::vector<float> xf(x_size);
        for (std::size_t i = 0; i < x_size; ++i) { xf[i] = static_cast<float>(x_data[i]); }

        torch::Tensor x_obs = torch::from_blob(
            xf.data(), {1, n_trials, n_features}, torch::kFloat32
        ).clone().to(trained->device);

        trained->model->eval();
        torch::NoGradGuard guard;
        torch::Tensor pred = trained->model->forward(x_obs).squeeze(0);

        std::string ll = [](std::string s) {
            for (char& c : s) { c = static_cast<char>(std::tolower(static_cast<unsigned char>(c))); }
            return s;
        }(loss_str);

        if (ll == "qrl") {
            for (int i = 0; i < n_params; ++i) {
                out_predictions[i] = pred[static_cast<long>(n_params + i)].item<double>();
            }
        } else if (ll == "mdn") {
            const int K = 3;
            for (int i = 0; i < n_params; ++i) {
                std::vector<double> logits(K);
                double max_l = -1e308;
                for (int k = 0; k < K; ++k) {
                    logits[k] = pred[static_cast<long>(i * K + k)].item<double>();
                    if (logits[k] > max_l) { max_l = logits[k]; }
                }
                double sum_exp = 0.0;
                std::vector<double> pi(K);
                for (int k = 0; k < K; ++k) { pi[k] = std::exp(logits[k] - max_l); sum_exp += pi[k]; }
                for (int k = 0; k < K; ++k) { pi[k] /= sum_exp; }
                double val = 0.0;
                for (int k = 0; k < K; ++k) {
                    double mu = pred[static_cast<long>(n_params * K + i * K + k)].item<double>();
                    val += pi[k] * mu;
                }
                out_predictions[i] = val;
            }
        } else {
            // mse, mae, huber, nll — first n_params elements are the mean estimates
            for (int i = 0; i < n_params; ++i) {
                out_predictions[i] = pred[static_cast<long>(i)].item<double>();
            }
        }
        return 0;

    } catch (const std::exception& ex) {
        set_error(out_error_msg, error_msg_len, ex.what());
        return -1;
    } catch (...) {
        set_error(out_error_msg, error_msg_len, "Unknown error in rnn_backend_predict.");
        return -1;
    }
#endif
}

// ---------------------------------------------------------------------------
// rnn_backend_free_model
// ---------------------------------------------------------------------------
extern "C" MULTIRL_EXPORT void rnn_backend_free_model(
    void* model_handle
) {
#ifdef MULTIRL_HAS_TORCH
    if (model_handle) {
        delete static_cast<multiRL::TrainedRnnModel*>(model_handle);
    }
#endif
}
