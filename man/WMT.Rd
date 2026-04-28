\name{WMT}
\alias{WMT}
\title{Data from Collins and Frank (2012)}
\description{
  This dataset originates from Experiment of Collins and Frank (2012),
  titled "How much of reinforcement learning is working memory, not 
  reinforcement learning? A behavioral, computational, and neurogenetic analysis"
  (\doi{10.1111/j.1460-9568.2011.07980.x}).
  The raw data is publicly available on the GitHub
  at \url{https://github.com/AnneCollins/WMH}.
  For the purposes of this package, we've performed
  basic cleaning and preprocessing of the original dataset.
}
\format{
  A data frame with 53089 rows and 15 columns:
  \describe{
    \item{Subject}{Subject ID, an integer (total of 79).}
    \item{Block}{
      Each participant completed 18 blocks, and the set size differed across 
      blocks; therefore, the number of trials also varied across blocks. In 
      addition, different stimuli were used in each block. Although the same 
      letters were used, this did not mean that the stimuli were identical. 
      Overall, participants had to relearn a new set of abstract images 
      whenever they entered a new block.
    }
    \item{Trial}{
      The number of trials was not fixed across blocks. Each stimulus category 
      appeared at least nine times. Therefore, if the set size increased, 
      the number of trials in that block also increased.
    }
    \item{Object_1}{
      Each stimulus category was associated with three possible responses, one 
      of which was correct and assigned a reward value of 1, whereas the other 
      two were assigned a reward value of 0.

      For example, stimulus A was associated with three response keys, J, K, 
      and L, corresponding to three objects: AJ, AK, and AL. If AK was the 
      correct response, then the rewards associated with the other responses 
      were set to 0.
    }
    \item{Object_2}{
      Has the same meaning as Object_1.
    }
    \item{Object_3}{
      Has the same meaning as Object_1.
    }
    \item{Reward_1}{Reward associated with the Object_1.}
    \item{Reward_2}{Reward associated with the Object_2.}
    \item{Reward_3}{Reward associated with the Object_3.}
    \item{Action}{
      The chosen option, 
      either \code{Object_1}, \code{Object_2} or \code{Object_3}.
    }
    \item{SetSize}{
      Participants were required to memorize the correct response (J, K, or L) 
      associated with each abstract image in a set of stimuli. Different blocks 
      imposed different memory loads. For example, Block 1 might require 
      participants to memorize the correct responses for two images, whereas 
      Block 6 might require them to memorize the correct responses for six 
      images. This column indicates how many image–response associations 
      needed to be memorized in a given trial.
    }
    \item{Stim_Count}{
      Indicates how many times the abstract image had appeared up to the 
      current trial.
    }
    \item{RT}{
      The reaction time for the participant’s response.
    }
    \item{PreCorrect}{
      Indicates the number of times the participant had made the correct choice.
    }
    \item{Delay}{
      Indicates how many trials had passed since the last correct choice.
    }
  }
}
