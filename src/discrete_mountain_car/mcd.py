from numpy import arange


class MCD:
    #Parent class of all discrete mountain car environment implementations.

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
