import gymnasium
import numpy as np

class PlumeDroneBulletEnv(gymnasium.Env):
    def __init__(
            self,
            num_drones: int = 1,
            num_plume_sources: int = 1,
            initial_plume_positions: list = None,
            max_steps: int = 1000,
            size: int = 100,
            background_concentration: int = 5,
            incrementer: float = 0.5,
            initial_xyzs=None,
            initial_rpys=None,
            gui=True,
            record=False,
        ):

        """Plume reinforcement learning environment built off of PyBullet.

        This class inherits from BaseRLAviary, which inherits from BaseAviary.

        I've wrote it such that you should not need to touch any of the drone/physics
        specific commands. However, if you need to for whatever reason, feel free to
        reach out to me if you find it too unnecessarily complicated.

        Parameters
        ----------
        num_drones: int, optional
            The desired number of drones in the aviary.
            Keep this at 1 unless you're feeling spicy.
        num_plume_sources: int, optional
            The desired number of plume sources.
            Keep this at 1 unless you're feeling spicy.
        plume_positions: list, optional
            Positions of plume sources
            Keep this at 1 unless you're feeling spicy.
            Leave as None to randomly generate plume sources (recommended)
            List size of plume positions MUST == num_plume_sources.
        max_steps: int, optional
            Number of maximum steps in an episode/epoch.
            Useful for figuring out when to truncate an episode (e.g. if a drone is just taking way too long to find source)
        size: int, optional
            Grid size of the PyBullet environment.
            Change to your liking, I prefer the default.
        background_concentration: int, optional
            Some background concentration that acts as an alpha multiplier to the gaussian distribution function.
            Change to your liking, I prefer the default.
        incrementer: float, optional
            This sets the steps of each movement of the drone. Too much and the drone will swing everywhere,
            too little and the drone will move quite slowly.
            Play around with this one, this default I've had the most luck with
        initial_xyzs: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial XYZ position of the drones.
            Keeping default will work fine, but feel free to change if needed.
        initial_rpys: ndarray | None, optional
            (NUM_DRONES, 3)-shaped array containing the initial orientations of the drones (in radians).
            Keep default.
        physics : Physics, optional
            The desired implementation of PyBullet physics/custom dynamics.
            Keep default.
        pyb_freq : int, optional
            The frequency at which PyBullet steps (a multiple of ctrl_freq).
            Keep default.
        ctrl_freq : int, optional
            The frequency at which the environment steps.
            Keep default.
        gui : bool, optional
            Whether to use PyBullet's GUI.
            Useful for debugging, but training will run faster if off.
        record : bool, optional
            Whether to save a video of the simulation.
            Will be cool to have for demo purposes once you've got the training pipeline down.
        obs : ObservationType, optional
            The type of observation space (kinematic information or vision).
            Keep default.
        act : ActionType, optional
            The type of action space (1 or 3D; RPMS, thurst and torques, waypoint or velocity with PID control; etc.).
            Keep default.
        """

        self.num_plume_sources = num_plume_sources

        if initial_plume_positions:
            self.plume_positions = initial_plume_positions
            self.initial_plume_positions = initial_plume_positions
        else:
            self.plume_positions = [tuple(i) for i in np.random.randint(0, 4, size=(self.num_plume_sources,2))]
            self.initial_plume_positions = None

        self.max_steps = max_steps
        self.incrementer = incrementer
        self.size = self.calculate_size(size)

        self.background_concentration = background_concentration

        # mapping actions to directions
        self._action_to_direction = {
            0: np.array([0, (0 + self.incrementer), 0]),  # right
            1: np.array([(0 + self.incrementer), (0 + self.incrementer), 0]),  # down-right
            2: np.array([(0 + self.incrementer), 0, 0]),  # down
            3: np.array([(0 + self.incrementer), -(0 + self.incrementer), 0]),  # down-left
            4: np.array([0, -(0 + self.incrementer), 0]),  # left
            5: np.array([-(0 + self.incrementer), -(0 + self.incrementer), 0]),  # up-left
            6: np.array([-(0 + self.incrementer), 0, 0]),  # up
            7: np.array([-(0 + self.incrementer), (0 + self.incrementer), 0]),  # up-right
        }

        # initializing concentration map, which will be observation space
        self.concentrations = np.zeros((self.size, self.size))

        # initializing previous_position and next_position for access in reward function
        self.previous_positions = None
        self.next_positions = None

        # initializing visited set. can be useful to know whether drone is backtracking or not
        self.visited = set()

    def reset(self, seed=None):
        """
        Function to reset environment.

        Parameters
        ----------
        seed : int, optional
            seed for reproducibility of training simulation

        Returns
        -------
        gym.spaces obs, dict info
        """

        super().reset()
        if seed is not None:
            np.random.seed(seed)
        self.concentrations = np.zeros((self.size, self.size))
        self.visited = set()

        if self.initial_plume_positions:
            self.plume_positions = self.initial_plume_positions
        else:
            self.plume_positions = [tuple(i) for i in np.random.randint(0, 4, size=(self.num_plume_sources,2))]

        for plume_position in self.plume_positions:
            print(f'loading in plume at position: {plume_position}')
            p.loadURDF('/gym_pybullet_drones/assets/box.urdf', [plume_position[0], plume_position[1], 0], useFixedBase=True)

        return self._computeObs(), {}

    def step(self, action):
        """
        Function to step through environment.

        Parameters
        ----------
        action : int
            index of action

        Returns
        -------
        gym.spaces obs, int reward, bool terminated, bool truncated, dict info
        """

        next_position = self.get_next_positions(action)
        self.previous_positions = self.get_current_positions()
        self.next_positions = next_position

        # NOTE for myself (mark): for some reason, just calling super().step and returning those values seems to work
        # so does passing in False, False for _computeTerminated, _computeTruncated
        # but returning self._computeObs(), self._computeReward(), self._computeTerminated(), self._computeTruncated, self._computeInfo()
        # resets the environment over and over again. can't figure out why, but shouldn't affect anything for you guys

        # return self._computeObs(), self._computeReward(), self._computeTerminated(), self._computeTruncated, self._computeInfo()
        return super().step(next_position)

    # Here you should implement your functions for getting concentrations, calculating rewards and checking termination
    def _computeObs(self):
        """
        Function to give observation space at each step.

        Parameters
        ----------
        None

        Returns
        -------
        state object == gym _observationSpace type (in this case, a dict)
        """

        state = {
            # "agent_positions": np.array(self.get_current_positions()),
            "agent_deltas": np.array(self.get_current_deltas()),
            "concentrations": np.array(self.update_concentration_matrix())
        }
        # print(state["agent_deltas"])
        return state

    def _computeReward(self):
        """
        Function to calculate reward at each step.

        Parameters
        ----------
        None

        Returns
        -------
        int reward
        """

        reward = 0
        concentration = self.get_concentration_value(self.get_current_positions()[0])

        if concentration > 0.9:
            reward += 1000
            print("Reached plume")
        else:
            reward -= (1 - concentration)

        print(f'Concentration: {concentration:.8f}\t\tReward: {reward:.8f}', end='\r')

        """
        min_distance = np.inf
        closest_plume = None
        for drone_position in self.get_current_positions():
            for plume_position in self.plume_positions:
                distance = np.linalg.norm(drone_position[:2] - plume_position)
                if distance < min_distance:
                    min_distance = distance
                    closest_plume = plume_position
        
        concentration = self.get_concentration_value(self.get_current_positions()[0])
        if concentration > 0.8:
            if closest_plume in self.visited:
                print('Already visited this plume')
                reward -= 1_000
            else:
            # Generate a positive reward for finding the goal
                reward += 1_000
                self.visited.add(closest_plume)
                print(f'found plume at position: {closest_plume}')
        else:
            # Generate a negative reward for not finding the goal
            reward -= 1 - concentration
        """

        return reward

    def _computeTerminated(self):
        """
        Function to find out if needs to be terminated at each step.
        Terminated is when an epoch/episode ends because the goal
        state is reached. e.g., the drone has reached the target.

        Parameters
        ----------
        None

        Returns
        -------
        bool of is truncated (true) or not (false)
        """

        return len(self.visited) == self.num_plume_sources

    def _computeTruncated(self):
        """
        Function to find out if needs to be truncated at each step.
        Truncated is when an epoch/episode ends for a reason besides
        reaching the goal state. e.g., when max_steps are reached

        Parameters
        ----------
        None

        Returns
        -------
        bool of is truncated (true) or not (false)
        """

        return self.step_counter >= self.max_steps

    def _computeInfo(self):
        """
        Gather info needed at each step.
        Optional, only for your knowledge purposes of what's going on

        Parameters
        ----------
        None

        Returns
        -------
        dictionary of info values
        """

        # TODO: do this (optional)
        return {
            "previous_positions": self.previous_positions,
            "next_positions": self.next_positions,
            "visited": self.visited,
            "concentrations": self.concentrations,
            "plume_positions": self.plume_positions
        }

    def _actionSpace(self):
        """
        Action space and nature of it for Gym environments

        Parameters
        ----------
        None

        Returns
        -------
        gym.spaces object of what the observation space should look like
        """

        # TODO: change this only if needed
        return spaces.Discrete(8)

    def _observationSpace(self):
        """
        Observation space and nature of it for Gym environments

        Parameters
        ----------
        None

        Returns
        -------
        gym.spaces object of what the observation space should look like
        """

        # TODO: change this only if needed
        return spaces.Dict({
            # "agent_positions": spaces.Box(low=0, high=self.size, shape=(3,), dtype=np.float32),
            "agent_deltas": spaces.Box(low=0, high=self.size, shape=(2,), dtype=np.float32),
            "concentrations": spaces.Box(low=0, high=self.size, shape=(self.size, self.size), dtype=np.float32)
        })