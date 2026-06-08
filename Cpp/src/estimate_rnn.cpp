#include <multiRL/estimate_rnn.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#ifdef MULTIRL_HAS_TORCH
#include <ATen/Context.h>
#include <torch/torch.h>
#endif

namespace multiRL {
namespace {

#ifdef MULTIRL_HAS_TORCH

double rnn_safe_value(double value) {
    return std::isfinite(value) ? value : 0.0;
}

double rnn_action_code(
    const std::string& value,
    const std::vector<std::string>& cue
) {
    for (std::size_t index = 0; index < cue.size(); ++index) {
        if (cue[index] == value) {
            return static_cast<double>(index + 1);
        }
    }
    return 0.0;
}

double rnn_chosen_reward(
    const std::vector<std::vector<std::string>>& state,
    const std::vector<double>& reward,
    const std::string& action
) {
    for (std::size_t option = 0; option < state.size(); ++option) {
        for (const std::string& value : state[option]) {
            if (value == action && option < reward.size()) {
                return reward[option];
            }
        }
    }
    return 0.0;
}

std::string rnn_task_subid(const RunTask& task) {
    if (!task.input.idinfo.empty() && !task.input.idinfo[0].empty()) {
        return task.input.idinfo[0][0];
    }
    return "1";
}

std::vector<double> rnn_row_features(
    const TaskSamplerRow& row,
    const std::vector<std::string>& cue
) {
    std::vector<double> out;
    out.reserve(4U + cue.size());
    out.push_back(rnn_action_code(row.simulation, cue));
    out.push_back(rnn_safe_value(row.reward));
    out.push_back(static_cast<double>(row.block));
    out.push_back(static_cast<double>(row.trial));

    for (std::size_t index = 0; index < cue.size(); ++index) {
        if (index < row.probability.size()) {
            out.push_back(rnn_safe_value(row.probability[index]));
        } else {
            out.push_back(0.0);
        }
    }

    return out;
}

std::vector<double> rnn_observed_features(
    const RunTask& task,
    std::size_t row
) {
    std::vector<double> out;
    out.reserve(4U + task.behrule.cue.size());
    out.push_back(rnn_action_code(task.input.action[row], task.behrule.cue));
    out.push_back(rnn_chosen_reward(
        task.input.state[row],
        task.input.reward_table[row],
        task.input.action[row]
    ));
    out.push_back(static_cast<double>(task.input.block[row]));
    out.push_back(static_cast<double>(task.input.trial[row]));

    for (std::size_t index = 0; index < task.behrule.cue.size(); ++index) {
        out.push_back(0.0);
    }

    return out;
}

TaskSamplerControl rnn_sampler_control(const RNNControl& control) {
    TaskSamplerControl out;
    out.n_draws = control.n_draws > 0 ? control.n_draws : control.sample;
    out.seed = static_cast<unsigned int>(control.seed);
    out.threads = control.threads;
    out.lower_bounds = control.lower_bounds;
    out.upper_bounds = control.upper_bounds;
    return out;
}

double rnn_clamp_estimate(
    double value,
    std::size_t index,
    const RNNControl& control
) {
    if (
        index < control.lower_bounds.size() &&
        std::isfinite(control.lower_bounds[index]) &&
        value < control.lower_bounds[index]
    ) {
        return control.lower_bounds[index];
    }

    if (
        index < control.upper_bounds.size() &&
        std::isfinite(control.upper_bounds[index]) &&
        value > control.upper_bounds[index]
    ) {
        return control.upper_bounds[index];
    }

    return value;
}

struct TorchRnnNetImpl : torch::nn::Module {
    torch::nn::GRU recurrent{nullptr};
    torch::nn::Dropout dropout{nullptr};
    torch::nn::Linear output{nullptr};

    TorchRnnNetImpl(
        int n_features,
        int n_params,
        const RNNControl& control
    ) {
        recurrent = register_module(
            "recurrent",
            torch::nn::GRU(torch::nn::GRUOptions(
                n_features,
                control.units
            ).batch_first(true).num_layers(control.layers))
        );
        dropout = register_module(
            "dropout",
            torch::nn::Dropout(torch::nn::DropoutOptions(control.dropout))
        );
        output = register_module(
            "output",
            torch::nn::Linear(control.units, n_params)
        );
    }

    torch::Tensor forward(torch::Tensor x) {
        torch::Tensor sequence = std::get<0>(recurrent->forward(x));
        torch::Tensor last = sequence.select(1, sequence.size(1) - 1);
        return output->forward(dropout->forward(last));
    }
};

TORCH_MODULE(TorchRnnNet);

torch::Tensor rnn_training_x(
    const TaskSamplerResult& sampler,
    int n_draws,
    int n_trials,
    int n_features
) {
    std::vector<float> values(
        static_cast<std::size_t>(n_draws * n_trials * n_features),
        0.0F
    );

    for (const TaskSamplerRow& row : sampler.rows) {
        const int draw = row.draw - 1;
        if (draw < 0 || draw >= n_draws) {
            continue;
        }

        const int trial = row.trial - 1;
        if (trial < 0 || trial >= n_trials) {
            continue;
        }

        const std::vector<double> features = rnn_row_features(
            row,
            sampler.cue_names
        );

        for (int feature = 0; feature < n_features; ++feature) {
            const std::size_t offset =
                static_cast<std::size_t>(
                    (draw * n_trials + trial) * n_features + feature
                );
            values[offset] = static_cast<float>(features[feature]);
        }
    }

    return torch::from_blob(
        values.data(),
        {n_draws, n_trials, n_features},
        torch::kFloat32
    ).clone();
}

torch::Tensor rnn_training_y(
    const TaskSamplerResult& sampler,
    int n_draws,
    int n_params
) {
    std::vector<float> values(
        static_cast<std::size_t>(n_draws * n_params),
        0.0F
    );
    std::vector<bool> seen(static_cast<std::size_t>(n_draws), false);

    for (const TaskSamplerRow& row : sampler.rows) {
        const int draw = row.draw - 1;
        if (draw < 0 || draw >= n_draws || seen[draw]) {
            continue;
        }
        seen[draw] = true;

        for (int param = 0; param < n_params; ++param) {
            if (static_cast<std::size_t>(param) < row.params.size()) {
                values[static_cast<std::size_t>(draw * n_params + param)] =
                    static_cast<float>(row.params[param]);
            }
        }
    }

    return torch::from_blob(
        values.data(),
        {n_draws, n_params},
        torch::kFloat32
    ).clone();
}

torch::Tensor rnn_observed_x(
    const RunTask& task,
    int n_trials,
    int n_features
) {
    std::vector<float> values(
        static_cast<std::size_t>(n_trials * n_features),
        0.0F
    );

    for (std::size_t row = 0; row < task.input.n_rows; ++row) {
        if (row >= static_cast<std::size_t>(n_trials)) {
            break;
        }

        const std::vector<double> features = rnn_observed_features(task, row);
        for (int feature = 0; feature < n_features; ++feature) {
            const std::size_t offset =
                static_cast<std::size_t>(row * n_features + feature);
            values[offset] = static_cast<float>(features[feature]);
        }
    }

    return torch::from_blob(
        values.data(),
        {1, n_trials, n_features},
        torch::kFloat32
    ).clone();
}

EstimateRnnSubjectResult estimate_rnn_subject(
    const RunTask& task,
    const RNNControl& control
) {
    const TaskSamplerResult sampler = task_sampler(
        task,
        rnn_sampler_control(control)
    );

    EstimateRnnSubjectResult out;
    out.subid = rnn_task_subid(task);
    out.parameter_names = task.params.free_names;
    out.n_draws = sampler.control.n_draws;
    out.n_trials = static_cast<int>(task.input.n_rows);
    out.n_features = static_cast<int>(4U + task.behrule.cue.size());
    out.epochs = control.epochs;
    out.backend = "torch";
    out.architecture = control.model_type;

    if (out.parameter_names.empty()) {
        out.status = -1;
        out.message = "estimate_rnn requires at least one free parameter.";
        return out;
    }

    if (control.model_type != "gru") {
        out.message = "Only model_type = 'gru' is currently implemented.";
        out.architecture = "gru";
    }

    if (control.threads > 0) {
        torch::set_num_threads(control.threads);
    }
    torch::manual_seed(static_cast<uint64_t>(control.seed));

    // Determine computation device: cpu or cuda
    torch::Device torch_device(torch::kCPU);
    bool use_cuda = (control.device == "cuda" || control.device == "gpu");
    if (use_cuda) {
        if (torch::cuda::is_available()) {
            torch_device = torch::Device(torch::kCUDA);
        } else {
            throw std::runtime_error(
                "RNN device='cuda' but no CUDA device is available."
            );
        }

        // Prefer deterministic CUDA kernels when they are available.
        at::globalContext().setDeterministicAlgorithms(true, true);
    }

    const int n_params = static_cast<int>(out.parameter_names.size());
    torch::Tensor x_train = rnn_training_x(
        sampler,
        out.n_draws,
        out.n_trials,
        out.n_features
    );
    x_train = x_train.to(torch_device);

    torch::Tensor y_train = rnn_training_y(
        sampler,
        out.n_draws,
        n_params
    );
    y_train = y_train.to(torch_device);

    torch::Tensor x_observed = rnn_observed_x(
        task,
        out.n_trials,
        out.n_features
    );
    x_observed = x_observed.to(torch_device);

    TorchRnnNet model(out.n_features, n_params, control);
    model->to(torch_device);
    torch::optim::Adam optimizer(
        model->parameters(),
        torch::optim::AdamOptions(control.learning_rate)
    );

    const int batch_size = std::min(control.batch_size, out.n_draws);
    for (int epoch = 0; epoch < control.epochs; ++epoch) {
        model->train();
        torch::Tensor order = torch::randperm(out.n_draws, torch::kLong).to(torch_device);

        for (int start = 0; start < out.n_draws; start += batch_size) {
            const int stop = std::min(start + batch_size, out.n_draws);
            torch::Tensor index = order.slice(0, start, stop);
            torch::Tensor xb = x_train.index_select(0, index);
            torch::Tensor yb = y_train.index_select(0, index);

            optimizer.zero_grad();
            torch::Tensor pred = model->forward(xb);
            torch::Tensor loss = torch::mse_loss(pred, yb);
            loss.backward();
            optimizer.step();
            out.loss = loss.item<double>();
        }
    }

    model->eval();
    torch::NoGradGuard guard;
    torch::Tensor prediction = model->forward(x_observed).squeeze(0);

    out.estimates.resize(out.parameter_names.size(), missing_real());
    for (std::size_t index = 0; index < out.estimates.size(); ++index) {
        const double value = prediction[static_cast<long>(index)]
            .item<double>();
        out.estimates[index] = rnn_clamp_estimate(value, index, control);
    }

    out.status = 1;
    if (out.message.empty()) {
        out.message = "ok";
    }
    return out;
}

#endif  // MULTIRL_HAS_TORCH

}  // namespace

std::vector<TaskSamplerResult> estimate_rnn(
    const std::vector<RunTask>& tasks,
    const TaskSamplerControl& control
) {
    return task_sampler(tasks, control);
}

std::vector<EstimateRnnSubjectResult> estimate_rnn(
    const std::vector<RunTask>& tasks,
    const RNNControl& raw_control
) {
    const RNNControl control = modify_control(raw_control, "rnn");

#ifndef MULTIRL_HAS_TORCH
    (void) tasks;
    (void) control;
    throw std::runtime_error(
        "estimate_rnn requires LibTorch support. Rebuild multiRL with "
        "MULTIRL_ENABLE_RNN_TORCH=ON and a valid Torch installation."
    );
#else
    std::vector<EstimateRnnSubjectResult> out;
    out.reserve(tasks.size());

    for (const RunTask& task : tasks) {
        out.push_back(estimate_rnn_subject(task, control));
    }

    return out;
#endif
}

}  // namespace multiRL


