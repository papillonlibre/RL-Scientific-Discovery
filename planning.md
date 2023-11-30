## Initial Research - parameters

* Plume density
* Plume composition: compounds present in the plumes
* Temperature: could influence if the plumes are ice or water
* Altitude of plume: relative to the spacecraft

## State Space

* Sensor readings: Data from various sensors measuring plume density, composition, temperature, and radiation levels.
* Positional encoding: maybe spherical coordinates for spacecraft (would include altitude then)
* Spacecraft velocity: at the current time step
* Spacecraft health: whether we have been previously been damaged
* Spacecraft energy level: measure of ability to conduct an action

## Action Space
* Thruster control: dictates movement; would abstract this out
* Rotation control: depending on movement characterization
* Sensor control: continuous or maybe at discrete time steps?
* Communication: boolean dependent on how much data we have in the buffer
* Data analysis: reach the target plume to analyze for scientific discovery
  * assume successful each time; switch that flips data analysis on

## Reward Systems
* Successful plume detection: high positive reward
* Proximity to plume: negative reward along trajectory; want to minimize to find shortest path;  
  * If we have a cluster of plumes in our detectable range, we could do a reciprocal reward to ensure we are always getting closer to all of them
* System health: Negative reward for any damage to spacecraft systems due to high radiation levels.
* Learning progress: reward for adapting to new scenarios
  * good thing to have if we can implement it
  * would go hand in hand with TL approach
  * prioritize other rewards first
* Efficiency: Reward for completing the mission with minimal fuel or resource usage.
	* Assumes we have information about the least amount of fuel possible
	* Could structure this as period reward proportional to amount of fuel remaining


## Additional Considerations

a. Penalties: Negative rewards for collisions with obstacles or incorrect analysis.  
b. Exploration: provide rewards for visiting new areas.  
* big consideration
* computer will find a way to cheat system if it can
* definitely want to explore  
* Gym might have an exploration method easy to implement  
c. Time Efficiency: Encourage reaching the plume in a timely manner.  
d. Adaptability: Reward for successfully adapting to changes in the environment.

## Continuous Learning

a. Model Update: Implement mechanisms for the model to learn from each mission and update its policy.  
b. Feedback Loop: Allow the spacecraft to send feedback on unexpected scenarios to improve future missions.

* Dynamic Adaptation  
	* Implement mechanisms for the model to dynamically adapt its policy.
	* Address scenarios such as encountering new obstacles, false positives/negatives in sensors, extreme temperature variations, and radiation.
* Continuous Learning
	* Allow the model to continually learn and update its policy during the mission.
	* Implement a mechanism for the spacecraft to send feedback to the model based on its experiences.


## Testing and Evaluation

* Simulated Testing
  * Test the model extensively in the simulated environment to ensure it performs well under various conditions.
* Evaluation Metrics
  * Define metrics for evaluating the model's performance (e.g., time to reach the plume, accuracy of plume detection).


## RL Algorithm

Deep Q Networks (DQN) could be a suitable choice for addressing this problem, given its ability to handle complex state spaces and its success in reinforcement learning tasks.

a. Input Layer: Use the sensor readings, altitude information, and system health as input features.  
b. Neural Network Architecture: Design a neural network with multiple layers to process the complex state information.


### Q-Value Estimation:  
a. Q-Network: Train the neural network to estimate Q-values for each state-action pair.  
b. Target Q-Network: Implement a target Q-network to stabilize training by using a separate set of parameters.  
### Experience Replay:
a. Memory Buffer: Implement experience replay to store and randomly sample batches of previous experiences.  

b. Epsilon-Greedy Strategy: Implement an epsilon-greedy strategy to balance exploration and exploitation during action selection.  
### Reward Shaping:

Design a reward function based on successful plume detection, proximity to the plume, analysis success, and other criteria mentioned earlier.  
### Training:
a. Loss Function: Use the temporal difference error between predicted and target Q-values as the loss function.  
b. Optimization: Train the model using optimization algorithms like RMSprop or Adam.  

c. Model Update: Periodically update the model based on feedback from each mission to adapt to the environment.  
d. Fine-Tuning: Implement mechanisms for fine-tuning the model's parameters during continuous learning.  

### Model Evaluation

Set up the simulation environment using Pybullet for physics and Gym for reinforcement learning integration.  

Evaluate the model's performance in the simulated environment under various scenarios.

Define metrics such as time to reach the plume, accuracy of plume detection, and resource efficiency for evaluation.  