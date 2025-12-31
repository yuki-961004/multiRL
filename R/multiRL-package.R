#' @keywords internal
"_PACKAGE"
#' @aliases multiRL
#' 
#' @title multiRL: Reinforcement Learning Tools for Multi-Armed Bandit
#'
#' @section Steps:
#' \itemize{
#'   \item \code{\link[multiRL]{run_m}}: 
#'     Step 1: Building reinforcement learning model
#'   \item \code{\link[multiRL]{rcv_d}}: 
#'     Step 2: Generating fake data for parameter and model recovery
#'   \item \code{\link[multiRL]{fit_p}}: 
#'     Step 3: Optimizing parameters to fit real data
#'   \item \code{\link[multiRL]{rpl_e}}: 
#'     Step 4: Replaying the experiment with optimal parameters
#' }
#' 
#' @section Document:
#' \itemize{
#'   \item \code{\link[multiRL]{data}}: 
#'     What kind of data structure the package actually accepts.
#'   \item \code{\link[multiRL]{colnames}}: 
#'     How to format your column names the right way.
#'   \item \code{\link[multiRL]{behrule}}: 
#'     How to define your latent learning rules.
#'   \item \code{\link[multiRL]{funcs}}: 
#'     These functions are the building blocks of your model.
#'   \item \code{\link[multiRL]{params}}: 
#'     A breakdown of every parameter used in the functions.
#'   \item \code{\link[multiRL]{priors}}: 
#'     Define the prior distributions for each free parameter.
#'   \item \code{\link[multiRL]{settings}}: 
#'     The general configuration and settings for your models.
#'   \item \code{\link[multiRL]{policy}}: 
#'     Decide if the agent chooses for itself (on-policy) or 
#'     simply copies human behavior (off-policy).
#'   \item \code{\link[multiRL]{estimate}}: 
#'     Pick an estimation method (MLE, MAP, ABC, or RNN).
#'   \item \code{\link[multiRL]{algorithm}}: 
#'     The optimization algorithms used for likelihood-based inference.
#'   \item \code{\link[multiRL]{control}}: 
#'     Fine-tune how the estimation methods and algorithms behave.
#' }
#' 
#' @section Models:
#' \itemize{
#'   \item \code{\link[multiRL]{TD}}: Temporal Difference model
#'   \item \code{\link[multiRL]{RSTD}}: Risk-Sensitive Temporal Difference model
#'   \item \code{\link[multiRL]{Utility}}: Utility model
#' }
#' 
#' @section Functions:
#' \itemize{
#'   \item \code{\link[multiRL]{func_alpha}}: Learning Rate
#'   \item \code{\link[multiRL]{func_beta}}: Inverse Temperature
#'   \item \code{\link[multiRL]{func_gamma}}: Utility Function
#'   \item \code{\link[multiRL]{func_delta}}: Upper-Confidence-Bound
#'   \item \code{\link[multiRL]{func_epsilon}}: Exploration Functions
#'   \item \code{\link[multiRL]{func_zeta}}: Working Memory System
#' }
#' 
#' @section Processes:
#' \itemize{
#'   \item \code{\link[multiRL]{process_1_input}}: 
#'     Standardize all inputs into a structured S4 object.
#'   \item \code{\link[multiRL]{process_2_behrule}}: 
#'     Define the specific latent learning rules for the agent.
#'   \item \code{\link[multiRL]{process_3_record}}: 
#'     Initialize an empty container to track the MDP outputs.
#'   \item \code{\link[multiRL]{process_4_output_cpp}}: 
#'     C++ Version: Markov Decision Process. 
#'   \item \code{\link[multiRL]{process_4_output_r}}: 
#'     R Version: Markov Decision Process.
#'   \item \code{\link[multiRL]{process_5_metric}}: 
#'     Compute various statistical metrics for different estimation methods.
#' }
#' 
#' @section Estimation:
#' \itemize{
#'   \item \code{\link[multiRL]{estimate_0_ENV}}: Estimation environment
#'   \item \code{\link[multiRL]{estimate_1_LBI}}: Likelihood-Based Inference
#'   \item \code{\link[multiRL]{estimate_1_MLE}}: Maximum Likelihood
#'   \item \code{\link[multiRL]{estimate_1_MAP}}: Maximum A Posteriori
#'   \item \code{\link[multiRL]{estimate_2_SBI}}: Simulation-Based Inference
#'   \item \code{\link[multiRL]{estimate_2_ABC}}: Approximate Bayesian Computation
#'   \item \code{\link[multiRL]{engine_ABC}}: The engine of ABC
#'   \item \code{\link[multiRL]{estimate_2_RNN}}: Neural network estimation
#'   \item \code{\link[multiRL]{engine_RNN}}: The engine of RNN
#'   \item \code{\link[multiRL]{estimation_methods}}: Shell function of estimate
#' }
#' 
#' @section Datasets:
#' \itemize{
#'   \item \code{\link[multiRL]{TAB}}: 
#'      Two-Armed Bandit data
#'   \item \code{\link[multiRL]{MAB}}: 
#'      Multi-Armed Bandit data
#' }
#' 
#' @section Summary: 
#' \itemize{
#'   \item \code{\link[multiRL]{summary,multiRL.model-method}}: 
#'      S4 method summary
#' }
#' 
#' @section Plot: 
#' \itemize{
#'   \item \code{\link[multiRL]{plot.multiRL.replay}}: 
#'      S3 method plot
#' }
#' 
#' @name multiRL-package
NULL