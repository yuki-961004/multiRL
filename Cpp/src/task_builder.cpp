#include <multiRL/task_builder.hpp>

#include <multiRL/modify_funcs.hpp>

#include <unordered_map>

namespace multiRL {

RunTask task_builder(
    const Process1Input& input,
    const Process2Behrule& behrule,
    const Params& params,
    const Settings& settings,
    const PriorGroup& priors
) {
    RunTask task;
    task.input = input;
    task.behrule = behrule;
    task.params = params;
    task.settings = settings;
    task.priors = priors;
    task.funcs = modify_funcs();
    return task;
}

std::vector<RunTask> split_task_by_subject(const RunTask& task) {
    if (task.input.idinfo.empty() || task.input.idinfo[0].empty()) {
        return {task};
    }

    std::vector<std::string> subject_order;
    std::unordered_map<std::string, std::vector<std::size_t>> rows_by_subject;

    for (std::size_t row = 0; row < task.input.n_rows; ++row) {
        const std::string& subid = task.input.idinfo[row][0];
        if (rows_by_subject.find(subid) == rows_by_subject.end()) {
            subject_order.push_back(subid);
        }
        rows_by_subject[subid].push_back(row);
    }

    std::vector<RunTask> out;
    out.reserve(rows_by_subject.size());

    for (const std::string& subid : subject_order) {
        const std::vector<std::size_t>& rows = rows_by_subject[subid];
        RunTask subject_task = task;
        Process1Input input;
        input.n_options = task.input.n_options;
        input.n_rows = rows.size();

        for (const std::size_t row : rows) {
            input.state.push_back(task.input.state[row]);
            input.reward_table.push_back(task.input.reward_table[row]);
            input.action.push_back(task.input.action[row]);
            input.block.push_back(task.input.block[row]);
            input.trial.push_back(task.input.trial[row]);
            input.idinfo.push_back(task.input.idinfo[row]);

            if (!task.input.exinfo.empty()) {
                input.exinfo.push_back(task.input.exinfo[row]);
            }
        }

        subject_task.input = input;
        out.push_back(subject_task);
    }

    return out;
}

}  // namespace multiRL
