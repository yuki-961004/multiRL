#include <multiRL/modify_context.hpp>

namespace multiRL {

TrialContext modify_context(
    const RunTask& task,
    const Process3Loop& output,
    std::size_t row
) {
    TrialContext context;
    context.row = row;
    context.rownum = static_cast<int>(row + 1);
    context.shown = output.shown[row];
    context.count = output.count[row];
    context.idinfo = task.input.idinfo[row];
    context.exinfo = task.input.exinfo[row];
    context.behave = output.behave[row];
    context.cue = task.behrule.cue;
    context.rsp = task.behrule.rsp;
    context.state = task.input.state[row];
    context.systems = task.settings.system;
    return context;
}

void modify_context_choice(
    TrialContext& context,
    const Process3Loop& output,
    std::size_t row
) {
    context.behave = output.behave[row];
}

void modify_context_qvalue(
    TrialContext& context,
    const std::vector<std::vector<double>>& qvalue,
    double exploration,
    const std::vector<std::string>& systems
) {
    context.qvalue = qvalue;
    context.exploration = exploration;
    context.systems = systems;
}

void modify_context_outcome(
    TrialContext& context,
    double reward,
    double utility,
    bool is_nb,
    bool is_fp
) {
    context.reward = reward;
    context.utility = utility;
    context.is_nb = is_nb;
    context.is_fp = is_fp;
}

void modify_context_system(
    TrialContext& context,
    const std::vector<double>& value0,
    const std::vector<double>& values,
    double qi,
    const std::string& system
) {
    context.value0 = value0;
    context.values = values;
    context.qi = qi;
    context.system = system;
}

}  // namespace multiRL
