#include <multiRLcpp/shell_run_m.hpp>

#include <multiRLcpp/criterion_posterior.hpp>
#include <multiRLcpp/process_MDP_free.hpp>

namespace multiRLcpp {

RunResult shell_run_m(const RunTask& task) {
    RunResult out;
    out.result = process_MDP_free(task);
    out.metric = criterion_posterior(task, out.result);
    return out;
}

}  // namespace multiRLcpp
