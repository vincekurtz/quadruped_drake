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

        # Use actuation matrix to map to control inputs
        S = self.plant.MakeActuationMatrix().T
        u = S@tau

        output.SetFromVector(u)
