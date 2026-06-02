#include <multiRL/info_nlopt.hpp>

#include <string>

#ifdef MULTIRL_HAS_NLOPT
#include <nlopt.hpp>
#endif

namespace multiRL {
namespace NloptInfo {

#ifdef MULTIRL_HAS_NLOPT

std::string message(nlopt::result code) {
    switch (code) {
        case nlopt::SUCCESS:
            return "Generic success (NLOPT_SUCCESS).";
        case nlopt::STOPVAL_REACHED:
            return "Stop value reached (NLOPT_STOPVAL_REACHED).";
        case nlopt::FTOL_REACHED:
            return "Function tolerance reached (NLOPT_FTOL_REACHED).";
        case nlopt::XTOL_REACHED:
            return "Parameter tolerance reached (NLOPT_XTOL_REACHED).";
        case nlopt::MAXEVAL_REACHED:
            return "Maximum evaluations reached (NLOPT_MAXEVAL_REACHED).";
        case nlopt::MAXTIME_REACHED:
            return "Maximum time reached (NLOPT_MAXTIME_REACHED).";
        case nlopt::FAILURE:
            return "Generic failure (NLOPT_FAILURE).";
        case nlopt::INVALID_ARGS:
            return "Invalid arguments (NLOPT_INVALID_ARGS).";
        case nlopt::OUT_OF_MEMORY:
            return "Out of memory (NLOPT_OUT_OF_MEMORY).";
        case nlopt::ROUNDOFF_LIMITED:
            return "Roundoff limited (NLOPT_ROUNDOFF_LIMITED).";
        case nlopt::FORCED_STOP:
            return "Forced stop (NLOPT_FORCED_STOP).";
        default:
            return "Unknown NLopt result.";
    }
}

std::string stop_reason(nlopt::result code) {
    switch (code) {
        case nlopt::SUCCESS:
            return "converged";
        case nlopt::STOPVAL_REACHED:
            return "stopval";
        case nlopt::FTOL_REACHED:
            return "ftol";
        case nlopt::XTOL_REACHED:
            return "xtol";
        case nlopt::MAXEVAL_REACHED:
            return "maxeval";
        case nlopt::MAXTIME_REACHED:
            return "maxtime";
        case nlopt::FAILURE:
            return "failure";
        case nlopt::INVALID_ARGS:
            return "invalid_args";
        case nlopt::OUT_OF_MEMORY:
            return "out_of_memory";
        case nlopt::ROUNDOFF_LIMITED:
            return "roundoff";
        case nlopt::FORCED_STOP:
            return "forced_stop";
        default:
            return "unknown";
    }
}

std::string code_name(nlopt::result code) {
    switch (code) {
        case nlopt::SUCCESS:
            return "SUCCESS";
        case nlopt::STOPVAL_REACHED:
            return "STOPVAL_REACHED";
        case nlopt::FTOL_REACHED:
            return "FTOL_REACHED";
        case nlopt::XTOL_REACHED:
            return "XTOL_REACHED";
        case nlopt::MAXEVAL_REACHED:
            return "MAXEVAL_REACHED";
        case nlopt::MAXTIME_REACHED:
            return "MAXTIME_REACHED";
        case nlopt::FAILURE:
            return "FAILURE";
        case nlopt::INVALID_ARGS:
            return "INVALID_ARGS";
        case nlopt::OUT_OF_MEMORY:
            return "OUT_OF_MEMORY";
        case nlopt::ROUNDOFF_LIMITED:
            return "ROUNDOFF_LIMITED";
        case nlopt::FORCED_STOP:
            return "FORCED_STOP";
        default:
            return "UNKNOWN";
    }
}

bool is_success(nlopt::result code) {
    return code > 0 && code < 5;
}

bool is_stopping(nlopt::result code) {
    return code == nlopt::STOPVAL_REACHED ||
        code == nlopt::FTOL_REACHED ||
        code == nlopt::XTOL_REACHED ||
        code == nlopt::MAXEVAL_REACHED ||
        code == nlopt::MAXTIME_REACHED;
}

bool is_error(nlopt::result code) {
    return code < 0 && code != nlopt::FORCED_STOP;
}

NLoptStatusInfo status(nlopt::result code) {
    NLoptStatusInfo out;
    out.code = static_cast<int>(code);
    out.code_name = code_name(code);
    out.message = message(code);
    out.stop_reason = stop_reason(code);
    out.is_success = is_success(code);
    out.is_stopping_condition = is_stopping(code);
    out.is_error = is_error(code);
    return out;
}

std::string summary(nlopt::result code) {
    const NLoptStatusInfo info = status(code);
    return info.code_name + ": " + info.message;
}

#endif

}  // namespace NloptInfo
}  // namespace multiRL