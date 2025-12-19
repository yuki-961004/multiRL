#' @title Core Functions
#' @name funcs
#' @description 
#'  
#'  The Markov Decision Process (MDP) underlying Reinforcement Learning can be 
#'    decomposed into six fundamental components. By modifying these six 
#'    functions, an immense number of distinct RL models can be created. Users 
#'    only need to grasp the basic MDP process and subsequently tailor these 
#'    six functions to construct a unique reinforcement learning model.
#' 
#' @section Class:  
#' \code{funcs [List]}
#' 
#' @section Details:  
#' \itemize{
#'    \item Action Select
#'    \itemize{
#'        \item Step 1: Agent uses \code{bias_func} 
#'              to apply a bias term to the value of each option.
#'        \item Step 2: Agent uses \code{expl_func} 
#'              to decide whether to make a purely random exploratory choice.
#'        \item Step 3: Agent uses \code{prob_func} 
#'              to compute the selection probability for each action.
#'    }
#'    \item Value Update
#'    \itemize{
#'        \item Step 4: Agent uses \code{util_func} 
#'              to translate the objective reward into subjective utility.
#'        \item Step 5: Agent uses \code{rate_func} 
#'              to update the value of the chosen option.
#'        \item Step 6: Agent uses \code{dcay_func} 
#'              to update the value of the unchosen options.
#'    }
#' }
#' 
#' @section Learning Rate (\eqn{\alpha}):  
#' 
#'  \code{rate_func} is the function that determines the learning rate 
#'    (\eqn{\alpha}). This function governs how the model selects the 
#'    \eqn{\alpha}. For instance, you can set different learning rates for 
#'    different circumstances. Rather than 'learning' in a general sense, the 
#'    learning rate determines whether the agent updates its expected values 
#'    (Q-values) using an aggressive or conservative step size across different 
#'    conditions.
#' 
#'  \deqn{Q_{new} = Q_{old} + \alpha \cdot (R - Q_{old})}
#' 
#' @section Soft-Max (\eqn{\beta}):  
#' 
#'  \code{prob_func} is the function defined by the inverse temperature 
#'    parameter (\eqn{\beta}) and the lapse parameter. 
#'    
#'    The inverse temperature parameter governs the randomness of choice. 
#'    If \eqn{\beta} approaches 0, the agent will choose between different 
#'    actions completely at random. 
#'    As \eqn{\beta} increases, the choice becomes more dependent on the 
#'    expected value (\eqn{Q_{t}}), meaning actions with higher expected values 
#'    have a proportionally higher probability of being chosen. 
#'    
#'    Note: This formula includes a normalization of the (\eqn{Q_{t}}) values.
#' 
#'  \deqn{
#'    P_{t}(a) = 
#'    \frac{
#'      \exp\left( \beta \cdot \left( Q_t(a) - \max_{j} Q_t(a_j) \right) \right)
#'    }{
#'      \sum_{i=1}^{k} \exp\left(
#'        \beta \cdot \left( Q_t(a_i) - \max_{j} Q_t(a_j) \right) \right
#'      )
#'    }
#'  }
#' 
#'  The function below, which incorporates the constant lapse rate, is a 
#'    correction to the standard soft-max rule. This is designed to prevent the 
#'    probability of any action from becoming exactly 0. When the lapse 
#'    parameter is set to 0.01, every action has at least a 1\% probability 
#'    of being executed. If the number of available actions becomes excessively 
#'    large (e.g., greater than 100), it would be more appropriate to set the 
#'    lapse parameter to a much smaller value.
#' 
#'  \deqn{
#'    P_{t}(a) = (1 - lapse \cdot N_{shown}) \cdot P_{t}(a) + lapse
#'  }
#' 
#' @section Utility Function (\eqn{\gamma}):  
#' 
#'  \code{util_func} is defined by the utility exponent parameter (\eqn{\gamma}). 
#'    Its purpose is to account for the fact that the objective reward received 
#'    by human may hold a different subjective value (utility) across 
#'    different subjects. 
#'    
#'    Note: The built-in function is formulated according to Stevens' power law.
#' 
#'  \deqn{U(R) = {R}^{\gamma}}
#' 
#' @section Upper Confidence Bound (\eqn{\delta}):  
#' 
#'  \code{bias_func} is the function defined by the parameter (\eqn{\delta}). 
#'    This function signifies that the expected value of an action is not 
#'    solely determined by the received reward, but is also influenced by the 
#'    number of times the action has been executed. Specifically, an action 
#'    that has been executed fewer times receives a larger exploration bias. 
#'    This mechanism prompts exploration and ensures the agent to execute 
#'    every action at least once.
#' 
#'  \deqn{
#'    \text{Bias} = \delta \cdot \sqrt{\frac{\log(N + e)}{N + 10^{-10}}}
#'  }
#' 
#' @section Epsilon–First, Greedy, Decreasing (\eqn{\epsilon}):  
#' 
#'  \code{expl_func} is the function defined by the parameter (\eqn{\epsilon}) 
#'    and the constant threshold. This function controls the probability with 
#'    which the agent engages in exploration (i.e., making a random choice) 
#'    versus exploitation (i.e., making a value-based choice).
#' 
#'  \eqn{\epsilon–first}: The agent must choose randomly for a fixed number of 
#'    initial trials. Once the number of trials exceeds the threshold, the agent 
#'    must exclusively choose based on value.
#' 
#'  \deqn{
#'  P(x) = 
#'  \begin{cases}
#'    i \le \text{threshold}, & x=1  \\
#'    i > \text{threshold}, & x=0 
#'  \end{cases}
#'  }
#'  
#'  \eqn{\epsilon–greedy}: The agent performs a random choice with probability 
#'    \eqn{\epsilon} and makes a value-based choice with probability 
#'    \eqn{1-\epsilon}.
#'  
#'  \deqn{
#'  P(x) = 
#'  \begin{cases}
#'    \epsilon, & x=1  \\
#'    1-\epsilon, & x=0 
#'  \end{cases}
#'  }
#'  
#'  \eqn{\epsilon–decreasing}: The probability of making a random choice 
#'    gradually decreases as the number of trials increases throughout the 
#'    experiment.
#'  
#'  \deqn{
#'  P(x) = 
#'  \begin{cases}
#'    \frac{1}{1+\epsilon \cdot i}, & x=1  \\
#'    \frac{\epsilon \cdot i}{1+\epsilon \cdot i}, & x=0 
#'  \end{cases}
#'  }
#' 
#' @section Working Memory (\eqn{\zeta}):  
#' 
#'  \code{dcay_func} is the function defined by the decay rate parameter 
#'    (\eqn{\zeta}) and the constant bonus. This function indicates that at the 
#'    end of each trial, not only the value of the chosen option will be changed 
#'    according to the learning rate, but also the values of the unchosen 
#'    options also undergo change. 
#'    
#'  It is due to the limitations of working memory capacity, the values of the 
#'    unchosen options are hypothesized to decay back towards their initial 
#'    value at a rate determined by the decay rate parameter (\eqn{\zeta}). 
#'    
#'  \deqn{W_{new} = W_{old} + \zeta \cdot (W_{0} - W_{old})}
#'    
#'  Furthermore, if the feedback of the chosen option provides information 
#'    relevant to the unchosen options, this decay rate may be enhanced or 
#'    mitigated by the constant bonus.
#' 
#' @section Example: 
#' \preformatted{ # inner functions
#'  funcs = list(
#'    rate_func = multiRL::func_alpha
#'    prob_func = multiRL::func_beta
#'    util_func = multiRL::func_gamma
#'    bias_func = multiRL::func_delta
#'    expl_func = multiRL::func_epsilon
#'    dcay_func = multiRL::func_zeta
#'  )
#' }
#' 
NULL