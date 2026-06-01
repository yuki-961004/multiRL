#pragma once

#include <multiRL/types.hpp>

namespace multiRL {

Process1Input process_1_input(
    const StringMatrix& object,
    const DoubleMatrix& reward,
    const std::vector<std::string>& action,
    const std::vector<int>& block,
    const std::vector<int>& trial,
    const StringMatrix& idinfo,
    const StringMatrix& exinfo
);

Process2Behrule process_2_behrule(
    const std::vector<std::string>& cue,
    const std::vector<std::string>& rsp
);

Process3Record process_3_record(const RunTask& task);

Process3Record process_4_output(
    const RunTask& task,
    Process3Record record
);

CriterionResult process_5_metric(
    const RunTask& task,
    const Process3Record& output
);

RunResult process_MDP_free(const RunTask& task);

}  // namespace multiRL
