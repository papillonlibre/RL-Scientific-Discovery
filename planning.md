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

## Reward Systems
* Successful plume detection: high positive reward
* Proximity to plume: negative reward along trajectory; want to minimize to find shortest path;  
  * If we have a cluster of plumes in our detectable range, we could do a reciprocal reward to ensure we are always getting closer to all of them
* System health: Negative reward for any damage to spacecraft systems due to high radiation levels.
* Learning progress: reward for adapting to new scenarios
* Efficiency: Reward for completing the mission with minimal fuel or resource usage.
	* Assumes we have information about the least amount of fuel possible
	* Could structure this as period reward proportional to amount of fuel remaining
