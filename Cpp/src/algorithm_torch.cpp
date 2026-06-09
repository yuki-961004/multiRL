#include <multiRL/algorithm_torch.hpp>

#ifdef MULTIRL_HAS_TORCH
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <cctype>

namespace multiRL {

namespace {

const double TORCH_PI = 3.14159265358979323846;

std::string to_lower_str(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](char item) {
        return static_cast<char>(std::tolower(static_cast<unsigned char>(item)));
    });
    return value;
}

} // namespace

int get_rnn_output_size(const std::string& loss_type, int n_params) {
    std::string l = to_lower_str(loss_type);
    if (l == "nll") {
        return n_params * 2;
    } else if (l == "qrl") {
        return n_params * 3;
    } else if (l == "mdn") {
        return n_params * 3 * 3; // K = 3 components: pi, mu, log_var
    } else {
        return n_params; // mse, mae, huber
    }
}

void validate_rnn_control(const RNNControl& control) {
    std::string mt = to_lower_str(control.architecture);
    if (mt != "rnn" && mt != "simple_rnn" && mt != "gru" && mt != "lstm" &&
        mt != "birnn" && mt != "bigru" && mt != "bilstm") {
        throw std::runtime_error(
            "Unknown RNN architecture. Supported values are: rnn, simple_rnn, gru, lstm, birnn, bigru, bilstm."
        );
    }
    std::string loss = to_lower_str(control.loss);
    if (loss != "mse" && loss != "mae" && loss != "huber" && loss != "hbr" &&
        loss != "nll" && loss != "qrl" && loss != "mdn" &&
        loss != "mean_squared_error" && loss != "mean_absolute_error" && loss != "huber_loss") {
        throw std::runtime_error(
            "Unknown RNN loss. Supported values are: mse, mae, huber, nll, qrl, mdn."
        );
    }
    std::string reg = to_lower_str(control.regularization);
    if (reg != "none" && reg != "l1" && reg != "l2" && reg != "l1_l2") {
        throw std::runtime_error(
            "Unknown RNN regularization. Supported values are: none, l1, l2, l1_l2."
        );
    }
}

TorchSequenceNetImpl::TorchSequenceNetImpl(
    int n_features,
    int n_params,
    const RNNControl& control
) {
    arch_type = to_lower_str(control.architecture);
    is_bidirectional = (arch_type.rfind("bi", 0) == 0);

    int recurrent_output_dim = is_bidirectional ? control.units * 2 : control.units;
    int hidden_nodes = std::max(1, control.units / 2);
    int output_nodes = get_rnn_output_size(control.loss, n_params);

    double drop_rate = (control.layers > 1) ? control.dropout : 0.0;

    if (arch_type == "rnn" || arch_type == "simple_rnn" || arch_type == "birnn") {
        rnn = register_module(
            "recurrent",
            torch::nn::RNN(torch::nn::RNNOptions(n_features, control.units)
                .batch_first(true)
                .num_layers(control.layers)
                .bidirectional(is_bidirectional)
                .dropout(drop_rate))
        );
    } else if (arch_type == "gru" || arch_type == "bigru") {
        gru = register_module(
            "recurrent",
            torch::nn::GRU(torch::nn::GRUOptions(n_features, control.units)
                .batch_first(true)
                .num_layers(control.layers)
                .bidirectional(is_bidirectional)
                .dropout(drop_rate))
        );
    } else if (arch_type == "lstm" || arch_type == "bilstm") {
        lstm = register_module(
            "recurrent",
            torch::nn::LSTM(torch::nn::LSTMOptions(n_features, control.units)
                .batch_first(true)
                .num_layers(control.layers)
                .bidirectional(is_bidirectional)
                .dropout(drop_rate))
        );
    }

    hidden = register_module(
        "hidden",
        torch::nn::Linear(recurrent_output_dim, hidden_nodes)
    );

    dropout = register_module(
        "dropout",
        torch::nn::Dropout(torch::nn::DropoutOptions(control.dropout))
    );

    output = register_module(
        "output",
        torch::nn::Linear(hidden_nodes, output_nodes)
    );
}

torch::Tensor TorchSequenceNetImpl::forward(torch::Tensor x) {
    torch::Tensor sequence;
    if (rnn) {
        sequence = std::get<0>(rnn->forward(x));
    } else if (gru) {
        sequence = std::get<0>(gru->forward(x));
    } else if (lstm) {
        sequence = std::get<0>(lstm->forward(x));
    } else {
        throw std::runtime_error("No recurrent module initialized in TorchSequenceNet");
    }

    torch::Tensor last = sequence.select(1, sequence.size(1) - 1);
    torch::Tensor h = torch::relu(hidden->forward(last));
    return output->forward(dropout->forward(h));
}

torch::Tensor TorchSequenceNetImpl::get_regularization_loss(
    const std::string& reg_type,
    double penalty
) {
    torch::Tensor reg_loss = torch::zeros({}, hidden->weight.options());
    if (penalty <= 0.0) {
        return reg_loss;
    }
    std::string rt = to_lower_str(reg_type);
    if (rt == "l1" || rt == "l1_l2") {
        reg_loss = reg_loss + penalty * torch::sum(torch::abs(hidden->weight));
    }
    if (rt == "l2" || rt == "l1_l2") {
        reg_loss = reg_loss + penalty * torch::sum(torch::square(hidden->weight));
    }
    return reg_loss;
}

torch::Tensor compute_rnn_loss(
    torch::Tensor pred,
    torch::Tensor target,
    const std::string& loss_type,
    int n_params
) {
    std::string l = to_lower_str(loss_type);

    if (l == "mse" || l == "mean_squared_error") {
        return torch::mse_loss(pred, target);
    } else if (l == "mae" || l == "mean_absolute_error") {
        return torch::l1_loss(pred, target);
    } else if (l == "huber" || l == "hbr" || l == "huber_loss") {
        return torch::smooth_l1_loss(pred, target);
    } else if (l == "nll") {
        // Gaussian Negative Log-Likelihood
        torch::Tensor mu = pred.slice(1, 0, n_params);
        torch::Tensor log_var = pred.slice(1, n_params, 2 * n_params);
        torch::Tensor precision = torch::exp(-log_var);
        torch::Tensor loss_val = 0.5 * precision * torch::square(target - mu) + 0.5 * log_var;
        return torch::mean(torch::sum(loss_val, -1));
    } else if (l == "qrl") {
        // Quantile Regression Loss (Pinball Loss) for 5%, 50%, 95% quantiles
        double q_values[3] = {0.05, 0.50, 0.95};
        torch::Tensor total_loss = torch::zeros({}, pred.options());
        for (int i = 0; i < 3; ++i) {
            torch::Tensor q_pred = pred.slice(1, i * n_params, (i + 1) * n_params);
            torch::Tensor err = target - q_pred;
            torch::Tensor loss_i = torch::max(q_values[i] * err, (q_values[i] - 1.0) * err);
            total_loss = total_loss + torch::mean(loss_i);
        }
        return total_loss;
    } else if (l == "mdn") {
        // Mixture Density Network (K=3)
        int K = 3;
        torch::Tensor pi_logits = pred.slice(1, 0, n_params * K).reshape({-1, n_params, K});
        torch::Tensor mu = pred.slice(1, n_params * K, 2 * n_params * K).reshape({-1, n_params, K});
        torch::Tensor log_var = pred.slice(1, 2 * n_params * K, 3 * n_params * K).reshape({-1, n_params, K});

        torch::Tensor mix_weights = torch::softmax(pi_logits, -1);
        torch::Tensor y_true_exp = target.unsqueeze(-1); // [batch, n_params, 1]

        double cst = 0.5 * std::log(2.0 * TORCH_PI);
        torch::Tensor log_prob = -cst - 0.5 * log_var - 0.5 * torch::square(y_true_exp - mu) * torch::exp(-log_var);
        torch::Tensor log_mix_weights = torch::log(mix_weights + 1e-8);
        torch::Tensor weighted_log_prob = log_mix_weights + log_prob;

        torch::Tensor log_mix_prob = torch::logsumexp(weighted_log_prob, -1);
        return torch::mean(-torch::sum(log_mix_prob, -1));
    } else {
        throw std::runtime_error("Unsupported loss type: " + loss_type);
    }
}

} // namespace multiRL

#endif // MULTIRL_HAS_TORCH
