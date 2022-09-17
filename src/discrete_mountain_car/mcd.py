from numpy import arange
import gym


class MCD:
    #Parent class of all discrete mountain car environment implementations.

    def __init__(self, n_splits:int, max_steps:int, max_iterations:int, results_path:str='', stop_at_first_flag:bool=False) -> None:
        """ Initialize the class.

        Args:
            n_splits (int): Number of equal splits of velocity and position.
            max_steps (int): Maximum number of steps per iteration.
            max_iterations (int): Maximum number of iterations.
            results_path (str, optional): Where to store the training results. Defaults to ''.
            stop_at_first_flag (bool, optional): Stop the first time is reached at training, or continue training. Defaults to False.

        Raises:
            TypeError: If n_splits is not an integer.
            ValueError: If n_splits is not greater than 1.
            TypeError: If max_steps is not an integer.
            ValueError: If max_steps is not greater than 1.
            TypeError: If max_iterations is not an integer.
            ValueError: If max_iterations is not greater than 1.
            TypeError: If results_path is not a string.
            TypeError: If stop_at_first_flag is not a boolean.
        
        Returns:
            None
        """        
        if not isinstance(n_splits, int): raise TypeError('n_splits must be an integer')
        if not 1 <= n_splits <= 30: raise ValueError('n_splits must be between 1 and 30')
        if not isinstance(max_steps, int): raise TypeError('max_steps must be an integer')
        if max_steps <= 0: raise ValueError('max_steps must be greater than 0')
        if not isinstance(max_iterations, int): raise TypeError('max_iterations must be an integer')
        if max_iterations <= 0: raise ValueError('max_iterations must be greater than 0')
        if not isinstance(results_path, str): raise TypeError('results_path must be a string')
        if not isinstance(stop_at_first_flag, bool): raise TypeError('stop_at_first_flag must be a boolean')

        self.n_splits = n_splits
        self.max_steps = max_steps
        self.max_iterations = max_iterations
        self.results_path = results_path
        self.stop_at_first_flag = stop_at_first_flag

        env_id = 'MountainCar-v0'
        self.env = gym.make(env_id)
        self.env_high = self.env.observation_space.high
        self.env_low = self.env.observation_space.low


    def define_states(self) -> None:
        """ Divide the space into equal n_splits of position on the x-axis and velocity.

        Returns:
            None
        """        
        self.positions_margin = round((abs(self.env_low[0] - self.env_high[0]))/self.n_splits, 3)
        self.velocities_margin = round((abs(self.env_low[1] - self.env_high[1]))/self.n_splits, 3)
        positions_splits = arange(start=self.env_low[0], stop=self.env_high[0], step=self.positions_margin)
        positions_splits = [round(pos, 2) for pos in positions_splits]
        velocities_splits = arange(start=self.env_low[1], stop=self.env_high[1], step=self.velocities_margin)
        velocities_splits = [round(vel, 2) for vel in velocities_splits]
        self.states = [(pos, vel) for pos in positions_splits for vel in velocities_splits]
        self.rewards = {state : {0:0, 1:0, 2:0} for state in self.states}


    def get_state_for_position_and_velocity(self, position:float, velocity:float) -> tuple:
        """ Get the state for a given position and velocity.

        Args:
            position (float): Position on the x-axis.
            velocity (float): Velocity of the car.

        Returns:
            tuple: State.
        """        
        for state_pos, state_vel in self.states:
            if state_pos <= position <= state_pos + self.positions_margin and state_vel <= velocity <= state_vel + self.velocities_margin:
                return state_pos, state_vel
