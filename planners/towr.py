from planners.simple import *
import subprocess as sub

import lcm
from lcm_types.trunklcm import trunk_state_t

import time

class TowrTrunkPlanner(BasicTrunkPlanner):
    """
    Trunk planner which uses TOWR (https://github.com/ethz-adrl/towr/) to generate
    target motions of the base and feet. 
    """
    def __init__(self):
        BasicTrunkPlanner.__init__(self)

        # Set up LCM subscriber to read optimal trajectory from TOWR
        self.lc = lcm.LCM()
        subscription = self.lc.subscribe("trunk_state", self.lcm_handler)

        # Set up storage of optimal trajectory
        self.traj_finished = False
        self.towr_timestamps = []
        self.towr_data = []
       
        # Call TOWR to generate a nominal trunk trajectory
        self.GenerateTrunkTrajectory()

    def lcm_handler(self, channel, data):
        """
        Handle an incoming LCM message. Essentially, we save the data
        to self.towr_data and self.timestamps. 
        """
        msg = trunk_state_t.decode(data)
       
        self.towr_timestamps.append(msg.timestamp)
        self.towr_data.append(msg)
        
        self.traj_finished = msg.finished   # indicate when the trajectory is over so 
                                            # we can stop listening to LCM

    def GenerateTrunkTrajectory(self):
        """
        Call a TOWR cpp script to generate a trunk model trajectory. 
        Read in the resulting trajectory over LCM. 
        """
        # TODO: pass parameters to TOWR, like goal, initial position, 
        # gait, total trajectory time, etc
        
        # Run the trajectory optimization (TOWR)
        sub.call(["build/test"],stdout=sub.DEVNULL)

        # Read the result over LCM
        self.traj_finished = False  # clear out any stored data
        self.towr_timestamps = []        # from previous trunk trajectories
        self.towr_data = []

        while not self.traj_finished:
            self.lc.handle() 

    def DoSetTrunkOutputs(self, context, output):
        output_dict = output.get_mutable_value()
        t = context.get_time()

        # Find the timestamp in the (stored) TOWR trajectory that is closest 
        # to the curren time
        closest_towr_t = self.towr_timestamps[np.abs(np.array(self.towr_timestamps)-t).argmin()]
        print(closest_towr_t)

        if context.get_time() < 5:
            self.OrientationTest(output_dict, context.get_time())
        else:
            self.RaiseFoot(output_dict)

