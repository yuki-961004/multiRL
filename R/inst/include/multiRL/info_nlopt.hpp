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

namespace NloptInfo {

std::string message(nlopt::result code);
std::string stop_reason(nlopt::result code);
std::string code_name(nlopt::result code);
bool is_success(nlopt::result code);
bool is_stopping(nlopt::result code);
bool is_error(nlopt::result code);
NLoptStatusInfo status(nlopt::result code);
std::string summary(nlopt::result code);

}  // namespace NloptInfo

#endif

}  // namespace multiRL