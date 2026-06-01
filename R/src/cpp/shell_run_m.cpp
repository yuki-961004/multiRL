#include <multiRL/shell_run_m.hpp>

#include <multiRL/criterion_posterior.hpp>
#include <multiRL/process_MDP_free.hpp>

namespace multiRL {

RunResult shell_run_m(const RunTask& task) {
    RunResult out;
    out.result = process_MDP_free(task);
    out.metric = criterion_posterior(task, out.result);
    return out;
}

}  // namespace multiRL
