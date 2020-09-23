import numpy as np
from pydrake.all import *

class BasicTrunkPlanner(LeafSystem):
    """
    Implements the simplest possible trunk-model planner, which generates
    desired positions, velocities, and accelerations for the feet, center-of-mass,
    and body frame orientation. 
    """
    def __init__(self):
        LeafSystem.__init__(self)

        # We'll use an abstract output port so we can send all the
        # data we'd like to include in a dictionary format
        self.DeclareAbstractOutputPort(
                "trunk_trajectory",
                lambda: AbstractValue.Make({}),
                self.DoSetTrunkOutputs)

    def SimpleStanding(self, output_dict):
        """
        Set output values corresponing to simply
        standing on all four feet.
        """
        # Foot positions
        output_dict["p_lf"] = np.array([ 0.175, 0.11, 0.0])
        output_dict["p_rf"] = np.array([ 0.175,-0.11, 0.0])
        output_dict["p_lh"] = np.array([-0.2,   0.11, 0.0])
        output_dict["p_rh"] = np.array([-0.2,  -0.11, 0.0])

        # Foot velocities
        output_dict["pd_lf"] = np.zeros(3)
        output_dict["pd_rf"] = np.zeros(3)
        output_dict["pd_lh"] = np.zeros(3)
        output_dict["pd_rh"] = np.zeros(3)
        
        # Foot accelerations
        output_dict["pdd_lf"] = np.zeros(3)
        output_dict["pdd_rf"] = np.zeros(3)
        output_dict["pdd_lh"] = np.zeros(3)
        output_dict["pdd_rh"] = np.zeros(3)

        # Foot contact states: [lf,rf,lh,rh], True indicates being in contact.
        output_dict["contact_states"] = [True,True,True,True]

        # Foot contact forces, where each row corresponds to a foot [lf,rf,lh,rh].
        output_dict["f_cj"] = np.zeros((3,4))

        # Body pose
        output_dict["rpy_body"] = np.array([0.0, 0.0, 0.0])
        output_dict["p_body"] = np.array([0.0, 0.0, 0.30])

        # Body velocities
        output_dict["rpyd_body"] = np.zeros(3)
        output_dict["pd_body"] = np.zeros(3)

        # Body accelerations
        output_dict["rpydd_body"] = np.zeros(3)
        output_dict["pdd_body"] = np.zeros(3)

    def OrientationTest(self, output_dict, t):
        """
        Given the current time t, generate output values for
        for a simple orientation test.
        """
        self.SimpleStanding(output_dict)
        output_dict["rpy_body"] = np.array([0.0, 0.4*np.sin(t), 0.4*np.cos(t)])

    def RaiseFoot(self, output_dict):
        """
        Modify the simple standing output values to lift one foot
        off the ground.
        """
        self.SimpleStanding(output_dict)
        output_dict["contact_states"] = [True,False,True,True]
        output_dict["p_rf"] = np.array([ 0.175,-0.11, 0.1])

    def DoSetTrunkOutputs(self, context, output):
        output_dict = output.get_mutable_value()

        if context.get_time() < 5:
            self.OrientationTest(output_dict, context.get_time())
        else:
            self.RaiseFoot(output_dict)

