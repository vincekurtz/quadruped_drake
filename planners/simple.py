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

    def DoSetTrunkOutputs(self, context, output):
        output_dict = output.get_mutable_value()

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
        output_dict["rpy_body"] = np.array([0.0, 0.0, np.pi])
        output_dict["p_body"] = np.array([0.0, 0.0, 0.25])

        # Body velocities
        output_dict["w_body"] = np.zeros(3)
        output_dict["pd_body"] = np.zeros(3)

        # Body accelerations
        output_dict["wd_body"] = np.zeros(3)
        output_dict["pdd_body"] = np.zeros(3)





