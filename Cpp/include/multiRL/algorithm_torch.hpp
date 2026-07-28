#pragma once

#include <multiRL/modify_control.hpp>
#include <string>

#ifdef MULTIRL_HAS_TORCH
#include <torch/torch.h>

namespace multiRL {

// Dynamic Recurrent Sequence Model
struct TorchSequenceNetImpl : torch::nn::Module {
    torch::nn::RNN rnn{nullptr};
    torch::nn::GRU gru{nullptr};
    torch::nn::LSTM lstm{nullptr};
    torch::nn::Linear hidden{nullptr};
    torch::nn::Dropout dropout{nullptr};
    torch::nn::Linear output{nullptr};
    torch::nn::Embedding subject_embedding{nullptr};

    std::string arch_type;
    bool is_bidirectional;
    int subject_embedding_size = 0;

    TorchSequenceNetImpl(
        int n_features,
        int n_params,
        const RNNControl& control,
        int n_subjects = 1
    );

    torch::Tensor forward(torch::Tensor x);
    torch::Tensor forward(
        torch::Tensor x,
        torch::Tensor lengths,
        torch::Tensor subject_indices
    );
    
    torch::Tensor get_regularization_loss(
        const std::string& reg_type,
        double penalty
    );
};

TORCH_MODULE(TorchSequenceNet);

struct TrainedRnnModel {
    TorchSequenceNet model{nullptr};
    torch::Device device{torch::kCPU};
    double final_loss = 0.0;
    int n_features = 0;
    int n_params = 0;
    int n_subjects = 1;
};

// Custom Loss Function Compute
torch::Tensor compute_rnn_loss(
    torch::Tensor pred,
    torch::Tensor target,
    const std::string& loss_type,
    int n_params
);

// Parameter count helper for output layers
int get_rnn_output_size(
    const std::string& loss_type,
    int n_params
);

// Validate RNN control parameters
void validate_rnn_control(const RNNControl& control);

} // namespace multiRL

#endif // MULTIRL_HAS_TORCH
