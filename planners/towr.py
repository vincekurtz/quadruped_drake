from planners.simple import *
import subprocess as sub

import lcm
from lcm_types.trunklcm import trunk_state_t

class TowrTrunkPlanner(BasicTrunkPlanner):
    """
    Trunk planner which uses TOWR (https://github.com/ethz-adrl/towr/) to generate
    target motions of the base and feet. 
    """
    def __init__(self):
        BasicTrunkPlanner.__init__(self)

        # Set up LCM subscriber to read optimal trajectory from TOWR
        self.lc = lcm.LCM()
        subscription = self.lc.subscribe("trunk_trajectory", self.lcm_handler)

        self.GenerateTrunkTrajectory()

    def lcm_handler(self, channel, data):
        msg = trunk_state_t.decode(data)
        print("hello world")

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
        self.lc.handle()

    def DoSetTrunkOutputs(self, context, output):
        output_dict = output.get_mutable_value()

        if context.get_time() < 5:
            self.OrientationTest(output_dict, context.get_time())
        else:
            self.RaiseFoot(output_dict)

