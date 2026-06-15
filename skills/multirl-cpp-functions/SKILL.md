---
name: multirl-cpp-functions
description: Create C++ model component functions for the multiRL repository. Use when a user wants to define a new reinforcement-learning model by authoring the six compiled C++ functions instead of R or Python callbacks.
---

# multiRL C++ Function Authoring

This skill helps users of the multiRL GitHub repository create compiled C++
model functions that work with every estimator in multiRL.

Use this skill when the user wants to define a new model, customize an
existing model, or translate a mathematical reinforcement-learning rule into
multiRL C++ source code.

## Design Policy

multiRL models should be defined through compiled C++ functions.

Do not create R callback functions.
Do not create Python callback functions.
Do not depend on Rcpp, pybind11, reticulate, pandas, numpy, or torch inside the
six model component functions.

The reason is estimator compatibility. Runtime R/Python callbacks cannot be
used safely by all compiled backends, especially MCMC and RNN workflows. C++
functions keep the model compatible with:

- `run_m`
- `fit_p`
- `rcv_d`
- `rpl_e`
- `estimate_mle`
- `estimate_map`
- `estimate_mcmc`
- `estimate_abc`
- `estimate_rnn`

## Repository Files To Inspect First

Before creating or editing functions, inspect the current repository version of
these files:

- `Cpp/include/multiRL/types.hpp`
- `Cpp/include/multiRL/funcs.hpp`
- `Cpp/src/modify_funcs.cpp`
- `Cpp/src/process_model_free.cpp`
- `Cpp/src/func_alpha.cpp`
- `Cpp/src/func_beta.cpp`
- `Cpp/src/func_gamma.cpp`
- `Cpp/src/func_delta.cpp`
- `Cpp/src/func_epsilon.cpp`
- `Cpp/src/func_zeta.cpp`

If the repository syncs C++ files into R or Python package folders, do not edit
the synced copies by hand. Update the core `Cpp/` files first, then use the
repository's build or sync workflow.

## The Six Functions

multiRL model-free processes are composed from six functions:

1. `lrng_func`: learning update, often governed by `alpha`.
2. `prob_func`: choice probability, often governed by `beta`.
3. `util_func`: reward utility, often governed by `gamma`.
4. `bias_func`: bias, UCB, or stickiness, often governed by `delta`.
5. `expl_func`: exploration switch, often governed by `epsilon`.
6. `dcay_func`: decay or reset, often governed by `zeta`.

## Required C++ Interface

Every function must live in namespace `multiRL`.

The function inputs must be:

```cpp
const TrialContext& context,
const Params& params
```

The default six functions have these signatures:

```cpp
double func_alpha(
    const TrialContext& context,
    const Params& params
);

std::vector<double> func_beta(
    const TrialContext& context,
    const Params& params
);

double func_gamma(
    const TrialContext& context,
    const Params& params
);

std::vector<double> func_delta(
    const TrialContext& context,
    const Params& params
);

int func_epsilon(
    const TrialContext& context,
    const Params& params
);

std::vector<double> func_zeta(
    const TrialContext& context,
    const Params& params
);
```

For additional built-in functions, keep the same input pattern and return type.
Use clear names such as:

- `func_alpha_rw`
- `func_beta_softmax`
- `func_gamma_power`
- `func_delta_ucb`
- `func_epsilon_greedy`
- `func_zeta_decay`

## TrialContext Fields

Use `context` for trial state and process state.

Important fields include:

- `context.row`: zero-based row index.
- `context.rownum`: one-based row number.
- `context.shown`: shown cue indicator values.
- `context.count`: cue exposure counts.
- `context.idinfo`: subject-level metadata.
- `context.exinfo`: extra trial-level metadata.
- `context.behave`: previous behavior record.
- `context.cue`: cue names.
- `context.rsp`: response names.
- `context.state`: current task state matrix.
- `context.systems`: active systems, such as `RL` and `WM`.
- `context.qvalue`: action values used by probability functions.
- `context.value0`: initial values for the active system.
- `context.values`: current values for the active system.
- `context.exploration`: exploration flag.
- `context.reward`: selected reward.
- `context.utility`: selected utility.
- `context.qi`: selected value for the active system.
- `context.is_nb`: whether the row begins a new block.
- `context.is_fp`: whether the selected cue is first presented.
- `context.system`: current active system.
- `context.features`: online hidden features prepared by the process loop.

If a field is intentionally unused, mark it explicitly:

```cpp
(void) context;
```

or:

```cpp
(void) context.reward;
```

## Hidden Features

Use `context.features` when a model needs online information from the
current or previous trial. These fields replace the old R/Python hidden-state
callback style with compiled C++ state that remains compatible with all
estimators.

Access hidden features with direct dot access:

```cpp
const double pe_prev = context.features.pe_prev;
const double progress = context.features.progress;
```

Do not use string lookup for hidden features. Direct fields are easier to
audit, faster, and safer for compiled MCMC, ABC, and RNN backends.

### Before-Choice Features

These features are available before the current action is generated:

- `context.features.progress`: trial progress within the whole input table.
- `context.features.block_progress`: trial progress within the current block.
- `context.features.session_progress`: same scale as `progress`, reserved for
  session-level designs.
- `context.features.log_trial_block`: `log(1 + trial)` within block.
- `context.features.prev_reward`: reward observed on the previous trial.
- `context.features.prev_repeat_sign`: `1` after a repeat, `-1` after a
  switch, and `0` before a previous choice exists.
- `context.features.prev_switch`: `1` after a switch and `0` otherwise.
- `context.features.choice_streak`: signed repeat streak for the current
  previous action.
- `context.features.pe_prev`: prediction error from the previous trial.
- `context.features.abs_pe_prev`: absolute previous prediction error.
- `context.features.count_imbalance`: difference between the first two shown
  cue counts.
- `context.features.count_imbalance_abs`: absolute count imbalance.
- `context.features.valid_count_total`: number of currently shown cues.
- `context.features.first_trial_in_block`: `1` on the first trial in a block.
- `context.features.alpha_prev`: previous generated or fitted alpha value.
- `context.features.beta_prev`: previous generated or fitted beta value.
- `context.features.kappa_prev`: previous generated or fitted kappa value.

### Q-Value Features

These features become meaningful after the q-value row for the current trial
has been assembled. They are especially useful in `prob_func`:

- `context.features.qvalue1`: first finite q-value.
- `context.features.qvalue2`: second finite q-value.
- `context.features.q_max`: maximum finite q-value.
- `context.features.q_abs_diff`: absolute difference between the first two
  finite q-values.
- `context.features.q_mean`: mean finite q-value.
- `context.features.decision_entropy`: softmax entropy of current q-values.
- `context.features.q_chosen_prev`: previous q-value for the current action.
- `context.features.q_unchosen_prev`: previous mean q-value for unchosen
  actions.

### After-Outcome Features

These features are available after reward and utility are known. They are
most useful in `dcay_func` and `lrng_func`:

- `context.features.reward`: reward selected on the current trial.
- `context.features.utility`: utility selected on the current trial.
- `context.features.pe`: current prediction error.
- `context.features.abs_pe`: absolute current prediction error.
- `context.features.q_chosen`: chosen q-value before learning.

The first HiddenState version intentionally excludes longer trace features
such as `PE_trace`, `reward_trace`, or `switch_trace`. Add those only when the
process loop has an explicit trace state and update rule.

## Params Fields

Use `params` for scalar model parameters.

Read a required parameter with:

```cpp
const double alpha = params.get("alpha");
```

Check an optional parameter with:

```cpp
if (params.has("alphaP")) {
    const double alpha_p = params.get("alphaP");
}
```

Missing numeric values are represented by `NaN`. Check them with:

```cpp
if (std::isnan(value)) {
    // Handle missing value.
}
```

Do not decide free parameter lists inside the component function. Free and
fixed parameter choices belong to the model definition or wrapper layer.

## Determinism Rules

Compiled estimators should be reproducible when a seed is provided.

Follow these rules:

- Do not use global mutable RNG state.
- Do not use `std::random_device`.
- Do not use R or Python RNG.
- If randomness is necessary, use a local RNG seeded from deterministic inputs.
- Prefer deterministic formulas inside the six functions whenever possible.

Example deterministic local RNG:

```cpp
std::mt19937 rng(
    static_cast<std::mt19937::result_type>(context.rownum)
);
```

## Numerical Rules

Keep numerical behavior stable:

- Use `missing_real()` for missing `double` values.
- Preserve vector sizes expected by the process loop.
- For vector outputs, return one value per cue.
- For probability functions, use stable softmax by subtracting the maximum
  finite value before exponentiation.
- Check denominators before division.
- Throw `std::invalid_argument` for invalid model definitions.
- Throw `std::runtime_error` for impossible runtime states.

## Registration Workflow

When adding a new function:

1. Add the function declaration to `Cpp/include/multiRL/funcs.hpp`.
2. Add the function implementation under `Cpp/src/`.
3. Add the implementation file to the relevant CMake source list.
4. Update `Cpp/src/modify_funcs.cpp` to map the user-facing string name to the
   C++ function pointer.
5. Run the repository's CMake configure or sync step if it copies C++ sources
   into R or Python wrapper folders.
6. Run a small `run_m` smoke test before estimator tests.

When replacing default behavior, prefer editing the existing `func_*.cpp` file
and keeping the public function name stable.

## Function-Specific Output Contracts

### `lrng_func`

Returns the updated value for the selected cue in the active system.

Usually reads:

- `context.qi`
- `context.utility`
- `context.reward`
- `context.system`
- `context.is_fp`
- `params.get("alpha")`

### `prob_func`

Returns a probability vector with one value per cue.

Hidden or unavailable options should generally be `missing_real()`. Visible
probabilities should sum to 1 after lapse, mixture, or exploration handling.

### `util_func`

Returns a scalar utility for the selected reward.

Usually reads:

- `context.reward`
- `params.get("gamma")`

### `bias_func`

Returns a bias vector with one value per cue.

Usually reads:

- `context.shown`
- `context.count`
- `context.behave`
- `context.state`
- `context.cue`

### `expl_func`

Returns `1` for exploration and `0` for exploitation.

If it uses randomness, the randomness must be local and deterministic.

### `dcay_func`

Returns a full next-value vector for the active system before the selected cue
is overwritten by `lrng_func`.

Usually reads:

- `context.values`
- `context.value0`
- `context.is_nb`
- `context.reward`
- `params.get("zeta")`

## Minimal Example

```cpp
#include <multiRL/funcs.hpp>

#include <cmath>
#include <stdexcept>

namespace multiRL {

double func_alpha_example(
    const TrialContext& context,
    const Params& params
) {
    const double alpha = params.get("alpha");
    const double qvalue = context.qi;
    const double utility = context.utility;

    if (std::isnan(alpha)) {
        throw std::invalid_argument("alpha is required.");
    }

    return qvalue + alpha * (utility - qvalue);
}

}  // namespace multiRL
```

## Review Checklist

Before finishing, verify:

- The function uses only `TrialContext` and `Params` as inputs.
- There are no R or Python callbacks.
- There are no Rcpp, pybind11, pandas, numpy, or torch dependencies.
- Vector outputs have the expected length.
- Missing values use `missing_real()` or `std::isnan`.
- RNG, if present, is deterministic and local.
- Headers include only what is required.
- The function is declared in `funcs.hpp`.
- The function is registered in `modify_funcs.cpp`.
- New `.cpp` files are included by CMake.
- The model can pass a small `run_m` smoke test.
