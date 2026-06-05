#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <multiRL/estimate_abc.hpp>
#include <multiRL/estimate_mle.hpp>
#include <multiRL/estimate_map.hpp>
#include <multiRL/estimate_rnn.hpp>
#ifdef MULTIRL_HAS_STAN
#include <multiRL/estimate_mcmc.hpp>
#endif
#include <multiRL/modify_priors.hpp>
#include <multiRL/process_model_free.hpp>
#include <multiRL/shell_rcv_d.hpp>
#include <multiRL/shell_rpl_e.hpp>
#include <multiRL/shell_run_m.hpp>
#include <multiRL/task_builder.hpp>
#include <multiRL/task_sampler.hpp>
#include <multiRL/types.hpp>

namespace {

multiRL::Params py_params_to_cpp(
    const std::unordered_map<std::string, double>& params,
    const std::vector<std::string>& free_names
) {
    multiRL::Params out;
    out.values = params;
    out.free_names = free_names;
    return out;
}

pybind11::dict py_wrap_result(
#include "py_wrapper_common.hpp"

#endif

// Auto-generated module registration
void register_py_estimate_mle(pybind11::module&);
void register_py_estimate_map(pybind11::module&);
void register_py_estimate_abc(pybind11::module&);
void register_py_estimate_rnn(pybind11::module&);
void register_py_shell_rcv_d(pybind11::module&);
void register_py_shell_rpl_e(pybind11::module&);
void register_py_estimate_mcmc(pybind11::module&);
void register_py_shell_fit_p(pybind11::module&);
void register_py_shell_run_m(pybind11::module&);

PYBIND11_MODULE(_backend, module) {
    module.doc() = "multiRL C++ backend for Python";
    register_py_estimate_mle(module);
    register_py_estimate_map(module);
    register_py_estimate_abc(module);
    register_py_estimate_rnn(module);
    register_py_shell_rcv_d(module);
    register_py_shell_rpl_e(module);
    register_py_estimate_mcmc(module);
    register_py_shell_fit_p(module);
    register_py_shell_run_m(module);
}