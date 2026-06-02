#pragma once

#include <string>

#ifdef MULTIRL_HAS_NLOPT
#include <nlopt.hpp>
#endif

namespace multiRL {

struct NLoptStatusInfo {
    int code = 0;
    std::string code_name = "UNKNOWN";
    std::string message = "Unknown result";
    std::string stop_reason = "failure";
    bool is_success = false;
    bool is_stopping_condition = false;
    bool is_error = false;
};

#ifdef MULTIRL_HAS_NLOPT

std::string nlopt_result_message(nlopt::result code);
std::string nlopt_stop_reason(nlopt::result code);
std::string nlopt_result_code_name(nlopt::result code);
bool nlopt_is_success(nlopt::result code);
bool nlopt_is_stopping_condition(nlopt::result code);
bool nlopt_is_error(nlopt::result code);
NLoptStatusInfo nlopt_status_info(nlopt::result code);
std::string nlopt_status_summary(nlopt::result code);

#endif

}  // namespace multiRL
