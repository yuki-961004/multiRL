\name{MAB}
\alias{MAB}
\title{Simulated Multi-Arm Bandit Dataset}
\description{
  A simulated multi-armed bandit (MAB) dataset featuring a complex 
  stimulus-response structure.
  The set of four distinct stimuli (red, blue, yellow, green) is not 
  isomorphic to the set of four available choices (up, down, left, right).
  Crucially, multiple stimuli may map to the same underlying choice 
  (e.g., Red and Blue both map to 'Up').
  This design requires the reinforcement learning model to learn the 
  latent mapping from observable stimuli to the set of potential actions, 
  making it a challenging test case for model fitting.
}
\format{
  A data frame with 9000 rows and 12 columns:
  \describe{
    \item{Subject}{Subject ID, an integer ranging from 1 to 30.
     }
    \item{Block}{Block number, an integer ranging from 1 to 6.
     }
    \item{Trial}{Trial number within each block, an integer (1 to 50).
     }
    \item{Object_1, Object_2, Object_3, Object_4}{
      Stimulus-response combinations (string) for four objects, formatted
      as "Color_Direction" (e.g., "Red_Up"). Each column is independently
      balanced and shuffled.
    }
    \item{Reward_1, Reward_2, Reward_3, Reward_4}{
      Reward values for four choice arms (Decks), following the classic
      Iowa Gambling Task (IGT) structure with adjusted values.
      \itemize{
        \item \code{Reward_1} (Bad): High gain (+100) with high frequency,
          mid-sized fine (-250). Long-term net loss.
        \item \code{Reward_2} (Bad): High gain (+100) with low frequency,
          large fine (-1250). Long-term net loss.
        \item \code{Reward_3} (Good): Low gain (+50) with high frequency,
          small fine (-50). Long-term net gain.
        \item \code{Reward_4} (Good): Low gain (+50) with low frequency,
          mid-sized fine (-250). Long-term net gain.
      }
      Rewards are balanced at the Block level.
    }
    \item{Action}{
      The simulated choice made by the subject on that trial (string),
      randomly sampled from "Up", "Down", "Left", or "Right".
     
    }
  }
}