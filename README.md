# RL-Scientific-Discovery
Reinforcement Learning Model for Scientific Discovery on a planetary body; advised in collaboration with NASA

## Introduction

In the pursuit of advancing autonomous space exploration, this project endeavors to
develop an innovative reinforcement learning model tailored for online
learning within the confines of a simulated environment. The specific focus is
on the detection of plumes, such as those found on Enceladus or analogous sci-
entific phenomena on other planetary bodies. The model’s multifaceted objec-
tives encompass not only the identification of these plumes but also the dynamic
guidance of a spacecraft towards their source for comprehensive analysis.

## RL Environment

The reinforcement learning environment was set up using the Gym Python pack-
age to facilitate the setup of state and action spaces and the reward function,
as well as to ease debugging. The state space was set up to include the current
location of the agent as well as a concentration map that signifies the concen-
tration of plumes around the agent.

## Simulation Environment

To accurately model the physics dynamics essential for simulating the satellite
scientific discovery problem, we leveraged the PyBullet library. PyBullet pro-
vided a robust framework for simulating the physical interactions and dynamics
within the simulated environment, enabling a realistic representation of the
challenges associated with satellite navigation and scientific exploration, in this
context around Encladus. The use of PyBullet facilitated the incorporation of
gravitational forces, aerodynamic effects, and other physical phenomena integral
to the accurate portrayal of a satellite’s behavior in space.

## Components of the Reward Function

Proximity to Plume Source: The reward is influenced by the spacecraft’s dis-
tance to the nearest plume source. Closer proximity to an unvisited plume
results in a positive reward. The reward magnitude increases with proximity,
fostering efficient plume detection.
Exploration and Avoidance: The model is incentivized to explore new plumes
intelligently. Upon discovering a plume with a concentration above a threshold
(e.g., 0.8), the spacecraft is rewarded positively. However, revisiting a previously
explored plume incurs a penalty to discourage redundant exploration.

## Quantitative Evaluation Metrics

Average Reward: Measure the average reward achieved by the spacecraft over
multiple episodes. A higher average reward indicates successful learning and
effective navigation toward plume sources.
Exploration vs. Exploitation: Analyze the balance between exploration and
exploitation. Evaluate whether the model efficiently explores the environment
to discover new plumes initially and transitions to exploitation for optimal nav-
igation.
Adaptability to Challenges: Introduce specific challenges in the environment,
such as new obstacles or extreme sensor readings. Assess the model’s adaptabil-
ity by monitoring changes in its behavior and reward accumulation in response
to these challenges.
Training Stability: Examine the stability of the training process. Ensure
that the model converges to a stable policy and does not exhibit erratic behavior
during training.

## Qualitative Evaluation Metrics

Visual Inspection: Visualize the trajectories of the spacecraft over episodes. Ob-
serve whether the model exhibits intelligent exploration, successfully identifies
plumes, and navigates toward their sources.
Adaptive Policy: Evaluate the adaptability of the learned policy. Introduce
novel scenarios or tweak environmental parameters to assess whether the model
updates its policy intelligently and in a manner aligned with scientific objectives.
Handling Unseen Scenarios: Introduce scenarios not encountered during
training to assess the model’s generalization capabilities. Evaluate whether the
spacecraft can handle new plume configurations or environmental conditions.

## Future Work

To try and mitigate other difficulties we are experiencing,
we have been trying to experiment with the Gaussian, and
particularly the alpha, of the plume concentration to make
it better suited for our purposes.
Our drone was also experiencing wide oscillations in the environment during
training episodes. This posed a challenge as it was not immediately obvious if
the issue was due to the reward function and training or due to the simulation
itself. The current belief is that it is due to the simulation, and specifically
due to the ration between the PyBullet frequency and the control frequency.
This is currently being modified still in order to ensure better results on our
agent learning. Furthermore, we tried states that only had one plume (and
generated another on reset by time out or discovery) as well as multi-plume
environments, and the agent would get confused when multiple plumes existed
at once as to which one it should pursue. This issue was amplified by the
oscillations because its sense of plume concentration would vary widely between
each swing it performed movement wise, thus meaning it did not have enough
time to actually learn where to go.
Our issue with the agent only detecting the first plume and not finding
subsequent plumes was interesting because it seemed to have a ”phantom path”
that it followed, every episode gravitating back to the coordinates that used to
contain a plume but no longer did.

Future work involves introducing a penalty prior to episode truncation to fur-
ther discourage early wandering, as well as boosting the magnitude of negative
rewards to counter the effect of diminishing concentrations due to the Gaussian
function. In addition, a dynamic concentration threshold will be introduced
to modify the exploration rate as more plumes are visited and fewer plumes
are left unexplored in the environment, which will help optimize the satellite’s
navigation of the environment.
Another aspect of future work involves reintroducing the multiple plume
state to the environment to better mimic the conditions found in the atmo-
sphere of Enceladus. This essentially would require exploiting the agent’s im-
proved learning on the simple environment to a more complex environment;
however, this element will be held off until the agent achieves sufficient perfor-
mance on the single plume environment, with more consistent trajectories as
well as consistently maximizing reward without over exploration. Ideally, we
will continue training and experimenting with the agent in the single plume en-
vironment until the oscillation issue can be resolved with the underlying physics,
in order to enable isolation of the behavior without the possibility of it being
tied to the agent discerning which goal to choose.

## Installation

Requires pytorch, gymnasium/gym, and stable-baselines3

```bash
pip install torch
pip install gym
pip install stable-baselines3
```
