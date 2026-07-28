#include <multiRL/shell_rpl_e.hpp>

#include <multiRL/process_model_free.hpp>

#include <map>
#include <set>
#include <tuple>
#include <unordered_map>

namespace multiRL {
namespace {

std::string replay_task_subid(const RunTask& task) {
    if (!task.input.idinfo.empty() && !task.input.idinfo[0].empty()) {
        return task.input.idinfo[0][0];
    }
    return "1";
}

RunTask replay_task(const RunTask& task, const ReplayFit& fit) {
    RunTask out = task;
    out.settings.generate = true;
    out.settings.mode = "simulating";
    out.settings.name = fit.model;

    for (const auto& item : fit.params) {
        out.params.values[item.first] = item.second;
    }

    return out;
}

using PlotKey = std::tuple<
    std::string,
    std::string,
    std::string,
    std::string,
    int,
    std::string
>;

using DenomKey = std::tuple<
    std::string,
    std::string,
    std::string,
    std::string,
    int
>;

void add_plot_count(
    std::map<PlotKey, int>& counts,
    std::map<DenomKey, int>& totals,
    const std::string& source,
    const std::string& model,
    const std::string& model_id,
    const std::string& subid,
    const int block,
    const std::string& action
) {
    const PlotKey plot_key = std::make_tuple(
        source,
        model,
        model_id,
        subid,
        block,
        action
    );
    const DenomKey denom_key = std::make_tuple(
        source,
        model,
        model_id,
        subid,
        block
    );

    counts[plot_key] += 1;
    totals[denom_key] += 1;
}

std::vector<ReplayPlotRow> make_plot_rows(
    const std::map<PlotKey, int>& counts,
    const std::map<DenomKey, int>& totals
) {
    std::vector<ReplayPlotRow> out;
    out.reserve(counts.size());

    for (const auto& item : counts) {
        ReplayPlotRow row;
        row.source = std::get<0>(item.first);
        row.model = std::get<1>(item.first);
        row.model_id = std::get<2>(item.first);
        row.subid = std::get<3>(item.first);
        row.block = std::get<4>(item.first);
        row.action = std::get<5>(item.first);
        row.count = item.second;

        const DenomKey denom_key = std::make_tuple(
            row.source,
            row.model,
            row.model_id,
            row.subid,
            row.block
        );
        const auto found = totals.find(denom_key);
        row.n = found == totals.end() ? 0 : found->second;
        row.ratio = row.n == 0
            ? missing_real()
            : static_cast<double>(row.count) / static_cast<double>(row.n);

        out.push_back(row);
    }

    return out;
}

}  // namespace

ReplayResult shell_rpl_e(
    const std::vector<RunTask>& tasks,
    const std::vector<ReplayFit>& fits
) {
    ReplayResult out;

    std::unordered_map<std::string, const RunTask*> task_by_subid;
    std::set<std::string> subjects;
    std::set<std::string> models;
    std::map<PlotKey, int> counts;
    std::map<DenomKey, int> totals;

    for (const RunTask& task : tasks) {
        const std::string subid = replay_task_subid(task);
        task_by_subid[subid] = &task;
        subjects.insert(subid);

        for (std::size_t row = 0; row < task.input.n_rows; ++row) {
            add_plot_count(
                counts,
                totals,
                "human",
                "Human",
                "Human",
                subid,
                task.input.block[row],
                task.input.action[row]
            );
        }
    }

    for (const ReplayFit& fit : fits) {
        const auto found = task_by_subid.find(fit.subid);
        if (found == task_by_subid.end()) {
            out.replay_success = false;
            continue;
        }

        RunTask task = replay_task(*found->second, fit);
        const RunResult result = process_model_free(task);
        models.insert(fit.model_id);

        for (std::size_t row = 0; row < task.input.n_rows; ++row) {
            ReplayRow replay;
            replay.source = "model";
            replay.model = fit.model;
            replay.model_id = fit.model_id;
            replay.subid = fit.subid;
            replay.block = task.input.block[row];
            replay.trial = task.input.trial[row];
            replay.action = result.result.simulation[row];
            replay.reward = result.result.reward[row];
            out.replay.push_back(replay);

            add_plot_count(
                counts,
                totals,
                replay.source,
                replay.model,
                replay.model_id,
                replay.subid,
                replay.block,
                replay.action
            );
        }
    }

    out.plot_data = make_plot_rows(counts, totals);
    out.n_models = static_cast<int>(models.size());
    out.n_subjects = static_cast<int>(subjects.size());
    out.generate = true;
    return out;
}

}  // namespace multiRL
