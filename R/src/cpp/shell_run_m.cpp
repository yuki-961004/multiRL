#include <multiRL/shell_run_m.hpp>

#include <multiRL/process_model_free.hpp>

namespace multiRL {

RunResult shell_run_m(const RunTask& task) {
    return process_model_free(task);
}

}  // namespace multiRL
