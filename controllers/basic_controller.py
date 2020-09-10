import numpy as np
from pydrake.all import *

class BasicController(LeafSystem):
    """
    A simple PD controller for a quadruped robot. 

                  -----------------------
                  |                     |
      [q,qd] ---> |   BasicController   | ---> tau
                  |                     |
                  -----------------------

    Includes some basic dynamics computations, so other
    more complex controllers can inherit from this class.
    """
    def __init__(self, plant, dt):
        LeafSystem.__init__(self)

        self.dt = dt
        self.plant = plant
        self.context = self.plant.CreateDefaultContext()  # stores q,qd

        # AutoDiff plant and context for values that require automatic differentiation
        self.plant_autodiff = plant.ToAutoDiffXd()
        self.context_autodiff = self.plant_autodiff.CreateDefaultContext()

        # Declare input and output ports
        self.DeclareVectorInputPort(
                "quad_state",
                BasicVector(self.plant.num_positions() + self.plant.num_velocities()))

        self.DeclareVectorOutputPort(
                "quad_torques",
                BasicVector(self.plant.num_actuators()),
                self.DoSetControlTorques)

        # Relevant frames for the CoM and each foot
        self.world_frame = self.plant.world_frame()
        self.lf_foot_frame = self.plant.GetFrameByName("LF_FOOT")  # left front
        self.rf_foot_frame = self.plant.GetFrameByName("RF_FOOT")  # right front
        self.lh_foot_frame = self.plant.GetFrameByName("LH_FOOT")  # left hind
        self.rh_foot_frame = self.plant.GetFrameByName("RH_FOOT")  # right hind
        
        self.world_frame_autodiff = self.plant_autodiff.world_frame()
        self.lf_foot_frame_autodiff = self.plant_autodiff.GetFrameByName("LF_FOOT")
        self.rf_foot_frame_autodiff = self.plant_autodiff.GetFrameByName("RF_FOOT")
        self.lh_foot_frame_autodiff = self.plant_autodiff.GetFrameByName("LH_FOOT")
        self.rh_foot_frame_autodiff = self.plant_autodiff.GetFrameByName("RH_FOOT")

    def UpdateStoredContext(self, context):
        """
        Use the data in the given input context to update self.context.
        This should be called at the beginning of each timestep.
        """
        state = self.EvalVectorInput(context, 0).get_value()
        q = state[:self.plant.num_positions()]
        v = state[-self.plant.num_velocities():]

        self.plant.SetPositions(self.context, q)
        self.plant.SetVelocities(self.context, v)

    def CalcDynamics(self):
        """
        Compute dynamics quantities, M, Cv, tau_g, and S such that the
        robot's dynamics are given by 

            M(q)vd + C(q,v)v + tau_g = S'u + tau_ext.

        Assumes that self.context has been set properly. 
        """
        M = self.plant.CalcMassMatrixViaInverseDynamics(self.context)
        Cv = self.plant.CalcBiasTerm(self.context)
        tau_g = -self.plant.CalcGravityGeneralizedForces(self.context)
        S = self.plant.MakeActuationMatrix().T

        return M, Cv, tau_g, S

    def CalcCoriolisMatrix(self):
        """
        Compute the coriolis matrix C(q,qd) using autodiff.
        
        Assumes that self.context has been set properly.
        """
        q = self.plant.GetPositions(self.context)
        v = self.plant.GetVelocities(self.context)

        def Cv_fcn(v):
            self.plant_autodiff.SetPositions(self.context_autodiff, q)
            self.plant_autodiff.SetVelocities(self.context_autodiff, v)
            return self.plant_autodiff.CalcBiasTerm(self.context_autodiff)

        C = 0.5*jacobian(Cv_fcn,v)
        return C

    def CalcComQuantities(self):
        """
        Compute the position (p), jacobian (J) and 
        jacobian-time-derivative-times-v (Jdv) for the center-of-mass.
        
        Assumes that self.context has been set properly. 
        """
        p = self.plant.CalcCenterOfMassPosition(self.context)
        J = self.plant.CalcJacobianCenterOfMassTranslationalVelocity(self.context,
                                                                     JacobianWrtVariable.kV,
                                                                     self.world_frame,
                                                                     self.world_frame)
        Jdv = self.plant.CalcBiasCenterOfMassTranslationalAcceleration(self.context,
                                                                      JacobianWrtVariable.kV,
                                                                      self.world_frame,
                                                                      self.world_frame)
        return p, J, Jdv

    def CalcComJacobianDot(self):
        """
        Compute the time derivative of the center-of-mass Jacobian (Jd)
        directly using autodiff. 

        Assumes that self.context has been set properly. 
        """
        q = self.plant.GetPositions(self.context)
        v = self.plant.GetVelocities(self.context)

        def J_fcn(q):
            self.plant_autodiff.SetPositions(self.context_autodiff, q)
            self.plant_autodiff.SetVelocities(self.context_autodiff, v)
            return self.plant_autodiff.CalcJacobianCenterOfMassTranslationalVelocity(self.context_autodiff,
                                                                                     JacobianWrtVariable.kV,
                                                                                     self.world_frame_autodiff,
                                                                                     self.world_frame_autodiff)
        Jd = jacobian(J_fcn,q)@self.plant.MapVelocityToQDot(self.context,v)
        
        return Jd

    def CalcFramePositionQuantities(self, frame):
        """
        Compute the position (p), jacobian (J) and 
        jacobian-time-derivative-times-v (Jdv) for the given frame
        
        Assumes that self.context has been set properly. 
        """
        p = self.plant.CalcPointsPositions(self.context,
                                           frame,
                                           np.array([0,0,0]),
                                           self.world_frame)
        J = self.plant.CalcJacobianTranslationalVelocity(self.context,
                                                         JacobianWrtVariable.kV,
                                                         frame,
                                                         np.array([0,0,0]),
                                                         self.world_frame,
                                                         self.world_frame)
        Jdv = self.plant.CalcBiasTranslationalAcceleration(self.context,
                                                           JacobianWrtVariable.kV,
                                                           frame,
                                                           np.array([0,0,0]),
                                                           self.world_frame,
                                                           self.world_frame)
        return p, J, Jdv

    def CalcFrameJacobianDot(self, frame):
        """
        Compute the time derivative of the given frame's Jacobian (Jd)
        directly using autodiff. 

        Note that `frame` must be an autodiff type frame. 

        Assumes that self.context has been set properly. 
        """
        q = self.plant.GetPositions(self.context)
        v = self.plant.GetVelocities(self.context)

        def J_fcn(q):
            self.plant_autodiff.SetPositions(self.context_autodiff, q)
            self.plant_autodiff.SetVelocities(self.context_autodiff, v)
            return self.plant_autodiff.CalcJacobianTranslationalVelocity(self.context_autodiff,
                                                                         JacobianWrtVariable.kV,
                                                                         frame,
                                                                         np.array([0,0,0]),
                                                                         self.world_frame_autodiff,
                                                                         self.world_frame_autodiff)
        # TODO: there seems to be a bug here with the jacobian computation
        Jd = jacobian(J_fcn,q)@self.plant.MapVelocityToQDot(self.context,v)
        return Jd

    def DoSetControlTorques(self, context, output):
        """
        This function gets called at every timestep and sets output torques. 
        """
        self.UpdateStoredContext(context)
        q = self.plant.GetPositions(self.context)
        v = self.plant.GetVelocities(self.context)
        
        # Tuning parameters
        Kp = 100*np.eye(self.plant.num_velocities())
        Kd = 10*np.eye(self.plant.num_velocities())

        # Fun with dynamics
        M, Cv, tau_g, S = self.CalcDynamics()
        #C = self.CalcCoriolisMatrix()    # note that computations based on autodiff show things way down
        p_com, J_com, Jdv_com = self.CalcComQuantities()
        #Jd_com = self.CalcComJacobianDot()

        p_lf, J_lf, Jdv_lf = self.CalcFramePositionQuantities(self.lf_foot_frame)
        #Jd_lf = self.CalcFrameJacobianDot(self.lf_foot_frame_autodiff)

        # Nominal joint angles
        q_nom = np.asarray([ 0.0, 0.0, 0.0, 1.0,     # base orientation
                             0.0, 0.0, 0.5,          # base position
                             0.0, 0.0, 0.0, 0.0,     # ad/ab
                             0.5, 0.5,-0.5,-0.5,     # hip
                            -0.8,-0.8, 0.8, 0.8])    # knee

        # Compute desired generalized forces
        q_err = self.plant.MapQDotToVelocity(self.context, q-q_nom)  # Need to use qd=N(q)*v here,
                                                                     # since q and v have different sizes
        qd_err = v - np.zeros(self.plant.num_velocities())
        tau = - Kp@q_err - Kd@qd_err

        # Use actuation matrix to map generalized forces to control inputs
        u = S@tau
        output.SetFromVector(u)
