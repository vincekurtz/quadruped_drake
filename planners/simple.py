import numpy as np
from pydrake.all import *

class BasicTrunkPlanner(LeafSystem):
    """
    Implements the simplest possible trunk-model planner, which generates
    desired positions, velocities, and accelerations for the feet, center-of-mass,
    and body frame orientation. 
    """
    def __init__(self, frame_ids):
        LeafSystem.__init__(self)

        # Dictionary of geometry frame ids {"trunk": trunk_frame_id, "lf": lf_foot_frame_id, ...}
        self.frame_ids = frame_ids

        # We'll use an abstract output port so we can send all the
        # data we'd like to include in a dictionary format
        self.DeclareAbstractOutputPort(
                "trunk_trajectory",
                lambda: AbstractValue.Make({}),
                self.SetTrunkOutputs)

        # Another output port is used to send geometry data regarding the
        # trunk model to the scene graph for visualization
        fpv = FramePoseVector()
        for frame in self.frame_ids:
            fpv.set_value(frame_ids[frame], RigidTransform())

        self.DeclareAbstractOutputPort(
                "trunk_geometry",
                lambda: AbstractValue.Make(fpv),
                self.SetGeometryOutputs)

        # The output data is a class-level object so we can be sure we're sending
        # the same info to the controller as to the scene graph
        self.output_dict = {}
        self.SimpleStanding()  # set initial values to self.output_dict

    def SimpleStanding(self):
        """
        Set output values corresponing to simply
        standing on all four feet.
        """
        # Foot positions
        self.output_dict["p_lf"] = np.array([ 0.175, 0.11, 0.0])   # mini cheetah
        self.output_dict["p_rf"] = np.array([ 0.175,-0.11, 0.0])
        self.output_dict["p_lh"] = np.array([-0.2,   0.11, 0.0])
        self.output_dict["p_rh"] = np.array([-0.2,  -0.11, 0.0])
        #self.output_dict["p_lf"] = np.array([ 0.34, 0.19, 0.0])    # anymal
        #self.output_dict["p_rf"] = np.array([ 0.34,-0.19, 0.0])
        #self.output_dict["p_lh"] = np.array([-0.34, 0.19, 0.0])
        #self.output_dict["p_rh"] = np.array([-0.34,-0.19, 0.0])

        # Foot velocities
        self.output_dict["pd_lf"] = np.zeros(3)
        self.output_dict["pd_rf"] = np.zeros(3)
        self.output_dict["pd_lh"] = np.zeros(3)
        self.output_dict["pd_rh"] = np.zeros(3)
        
        # Foot accelerations
        self.output_dict["pdd_lf"] = np.zeros(3)
        self.output_dict["pdd_rf"] = np.zeros(3)
        self.output_dict["pdd_lh"] = np.zeros(3)
        self.output_dict["pdd_rh"] = np.zeros(3)

        # Foot contact states: [lf,rf,lh,rh], True indicates being in contact.
        self.output_dict["contact_states"] = [True,True,True,True]

        # Foot contact forces, where each row corresponds to a foot [lf,rf,lh,rh].
        self.output_dict["f_cj"] = np.zeros((3,4))

        # Body pose
        self.output_dict["rpy_body"] = np.array([0.0, 0.0, 0.0])
        self.output_dict["p_body"] = np.array([0.0, 0.0, 0.3])

        # Body velocities
        self.output_dict["rpyd_body"] = np.zeros(3)
        self.output_dict["pd_body"] = np.zeros(3)

        # Body accelerations
        self.output_dict["rpydd_body"] = np.zeros(3)
        self.output_dict["pdd_body"] = np.zeros(3)

        # Max control input (accelerations)
        self.output_dict["u2_max"] = 0.0

    def OrientationTest(self, t):
        """
        Given the current time t, generate output values for
        for a simple orientation test.
        """
        self.SimpleStanding()
        self.output_dict["rpy_body"] = np.array([0.0, 0.4*np.sin(t), 0.4*np.cos(t)])
        self.output_dict["rpyd_body"] = np.array([0.0, 0.4*np.cos(t), -0.4*np.sin(t)])
        self.output_dict["rpydd_body"] = np.array([0.0, -0.4*np.sin(t), -0.4*np.cos(t)])

    def RaiseFoot(self, t):
        """
        Modify the simple standing output values to lift one foot
        off the ground.
        """
        self.SimpleStanding()
        self.output_dict["p_body"] += np.array([-0.1, 0.05, 0.0])

        if t>1:
            self.output_dict["contact_states"] = [True,False,True,True]
            self.output_dict["p_rf"] += np.array([ 0.0, 0.0, 0.1])

    def EdgeTest(self):
        """
        Move the trunk right to the edge of feasibility, ensuring that
        friction constraints become active (may require a smaller timestep)
        """
        self.SimpleStanding()
        self.output_dict["p_body"] += np.array([-0.1, 0.63, 0.0])

    def SetTrunkOutputs(self, context, output):
        self.output_dict = output.get_mutable_value()

        self.SimpleStanding()
        #self.output_dict["p_body"] += np.array([0,0,0.05])
        #self.OrientationTest(context.get_time())
        #self.EdgeTest()
        #self.RaiseFoot(context.get_time())

    def SetGeometryOutputs(self, context, output):
        fpv = output.get_mutable_value()
        fpv.clear()
       
        X_trunk = RigidTransform()
        X_trunk.set_rotation(RollPitchYaw(self.output_dict["rpy_body"]))
        X_trunk.set_translation(self.output_dict["p_body"])
        
        fpv.set_value(self.frame_ids["trunk"], X_trunk)

        for foot in ["lf","rf","lh","rh"]:
            X_foot = RigidTransform()
            X_foot.set_translation(self.output_dict["p_%s" % foot])
            fpv.set_value(self.frame_ids[foot],X_foot)
