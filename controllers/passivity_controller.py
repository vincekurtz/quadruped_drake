from controllers.basic_controller import *

class PassivityController(BasicController):
    """
    A passivity/approximate simulation-based whole-body controller. 
    Takes as input desired positions/velocities/accelerations of the 
    feet, center-of-mass, and base frame orientation and computes
    corresponding joint torques. 
    """
    def __init__(self, plant, dt):
        BasicController.__init__(self, plant, dt)

        # inputs from the trunk model are sent in a dictionary
        self.DeclareAbstractInputPort(
                "trunk_input",
                AbstractValue.Make({}))  

        # Set the friction coefficient
        self.mu = 0.7

        # Choose a solver
        self.solver = GurobiSolver()

        # Storage function for numerically computing Jbar
        self.last_Jbar = None
        self.last_contact_feet = None

    def AddJacobianTypeCost(self, J, qdd, Jd_qd, xdd_des, weight=1.0):
        """
        Add a quadratic cost of the form

            weight * || J*qdd + Jd_qd - xdd_des ||^2

        to the whole-body controller QP.
        """
        # Put in the form 1/2*qdd'*Q*qdd + c'*qdd for fast formulation
        Q = weight*np.dot(J.T,J)
        c = weight*(np.dot(Jd_qd.T,J) - np.dot(xdd_des.T,J)).T

        return self.mp.AddQuadraticCost(Q,c,qdd)

    def AddGeneralizedForceCost(self, tau_nom, S, tau, J_c, f_c, weight=1.0):
        """
        Add a cost which attempts to track the given generalized force
        tau_nom with applied control torques tau and contact forces f_c,

            weight * || tau_nom - S'*tau + J_c'*f_c ||^2

        to the whole-body QP.
        """
        # Stack J_c, f_c 
        if len(f_c) > 0:
            f = np.vstack(f_c)
            J = np.vstack(J_c)
        else:
            f = np.zeros((0,1))
            J = np.zeros((0,self.plant.num_velocities()))

        # Put in the form 1/2*x'*Q*x + c'*x for fast formulation
        x = np.vstack([tau,f])
        A = np.hstack([S.T,J.T])
        Q = weight*A.T@A
        c = -weight*A.T@tau_nom

        return self.mp.AddQuadraticCost(Q,c,x)

    def AddJacobianTypeConstraint(self, J, qdd, Jd_qd, xdd_des):
        """
        Add a linear constraint of the form
            J*qdd + Jd_qd == xdd_des
        to the whole-body controller QP.
        """
        A_eq = J     # A_eq*qdd == b_eq
        b_eq = xdd_des-Jd_qd

        return self.mp.AddLinearEqualityConstraint(A_eq, b_eq, qdd)

    def AddDynamicsConstraint(self, M, qdd, C, tau_g, S, tau, J_c, f_c):
        """
        Add a dynamics constraint of the form
            M*qdd + Cv + tau_g == S'*tau + sum(J_c[i]'*f_c[i])
        to the whole-body controller QP.
        """
        # We'll rewrite the constraints in the form A_eq*x == b_eq for speed
        A_eq = np.hstack([M, -S.T])
        x = np.vstack([qdd,tau])

        for j in range(len(J_c)):
            A_eq = np.hstack([A_eq, -J_c[j].T])
            x = np.vstack([x,f_c[j]])

        b_eq = -C-tau_g

        return self.mp.AddLinearEqualityConstraint(A_eq,b_eq,x)

    def AddFrictionPyramidConstraint(self, f_c):
        """
        Add a friction pyramid constraint for the given set of contact forces
        to the whole-body controller QP.
        """
        num_contacts = len(f_c)

        A_i = np.asarray([[ 1, 0, -self.mu],   # pyramid approximation of CWC for one
                          [-1, 0, -self.mu],   # contact force f \in R^3
                          [ 0, 1, -self.mu],
                          [ 0,-1, -self.mu]])

        # We'll formulate as lb <= Ax <= ub, where x=[f_1',f_2',...]'
        A = np.kron(np.eye(num_contacts),A_i)

        ub = np.zeros((4*num_contacts,1))
        lb = -np.inf*np.ones((4*num_contacts,1))

        x = np.vstack([f_c[j] for j in range(num_contacts)])

        return self.mp.AddLinearConstraint(A=A,lb=lb,ub=ub,vars=x)

    def AddContactConstraint(self, J_c, vd, Jdv_c, v):
        """
        Add contact constraints with velocity damping

            J_c[j]*vd + Jdv_c[j] == -Kd*J_c[j]*v
        """
        Kd = 150*np.eye(3)

        num_contacts = len(J_c)
        for j in range(num_contacts):
            pd = (J_c[j]@v).reshape(3,1)
            pdd_des = -Kd@pd

            constraint = self.AddJacobianTypeConstraint(J_c[j], vd, Jdv_c[j], pdd_des)

    def AddVdotConstraint(self, tau, f_c, delta, qd_tilde, S, J_c, M, Cv, tau_g, qdd_des, p_tilde, v_tilde, Kp, C):
        """
        Add a constraint

            Vdot <= delta
        
        which ensures that the simulation fuction

            V = 1/2qd_tilde'*M*qd_tilde + p_tilde'*Kp*p_tilde

        is decreasing. 
        """
        # Stack J_c, f_c 
        if len(f_c) > 0:
            f = np.vstack(f_c)
            J = np.vstack(J_c)
        else:
            f = np.zeros((0,1))
            J = np.zeros((0,self.plant.num_velocities()))

        # We'll formulate as lb <= Ax <= ub, where x=[tau, f_c, delta]'
        x = np.vstack([tau,f,delta])
        A = (qd_tilde.T @ np.hstack([S.T, J.T]))[np.newaxis]
        A = np.hstack([A, -np.eye(1)])
        
        ub = qd_tilde.T @ (M@qdd_des + Cv + tau_g) - p_tilde.T@Kp@v_tilde \
                - qd_tilde.T@C@qd_tilde
        ub *= np.ones(1)
        
        lb = -np.inf*np.ones(1)

        return self.mp.AddLinearConstraint(A=A,lb=lb,ub=ub,vars=x)

    def DoSetControlTorques(self, context, output):
        self.UpdateStoredContext(context)
        q = self.plant.GetPositions(self.context)
        v = self.plant.GetVelocities(self.context)
        
        # Compute Dynamics Quantities
        M, Cv, tau_g, S = self.CalcDynamics()
        C = self.CalcCoriolisMatrix()
        
        # Get setpoint data from the trunk model
        trunk_data = self.EvalAbstractInput(context,1).get_value()

        # Compute desired generalized forces
        #
        #     tau_nom = M*qdd_des + C*qd_des + tau_g - J'*(Kp*p_tilde + Kd*v_tilde)
        #
        # where
        #    
        #     p_tilde = p - p_des
        #     v_tilde = pd - pd_des
        #     Jbar = J'*inv(J*J')
        #     qd_des = Jbar*pd_des
        #     qdd_des = Jbar_dot*pd_des + Jbar*pdd_des


        # Compute body pose, jacobians, etc
        X_body, J_body, Jdv_body = self.CalcFramePoseQuantities(self.body_frame)
        p_body = X_body.translation()
        rpy_body = RollPitchYaw(X_body.rotation())

        # Note current foot positions, Jacobians, etc
        p_lf, J_lf, Jdv_lf = self.CalcFramePositionQuantities(self.lf_foot_frame)
        p_rf, J_rf, Jdv_rf = self.CalcFramePositionQuantities(self.rf_foot_frame)
        p_lh, J_lh, Jdv_lh = self.CalcFramePositionQuantities(self.lh_foot_frame)
        p_rh, J_rh, Jdv_rh = self.CalcFramePositionQuantities(self.rh_foot_frame)

        p_feet = np.array([p_lf.flatten(), p_rf.flatten(), p_lh.flatten(), p_rh.flatten()])
        J_feet = np.array([J_lf, J_rf, J_lh, J_rh])
        Jdv_feet = np.array([Jdv_lf, Jdv_rf, Jdv_lh, Jdv_rh])

        # Unpack desired positions, velocities, accelerations of feet
        p_des_feet = np.array([trunk_data["p_lf"],trunk_data["p_rf"],trunk_data["p_lh"],trunk_data["p_rh"]])
        pd_des_feet = np.array([trunk_data["pd_lf"],trunk_data["pd_rf"],trunk_data["pd_lh"],trunk_data["pd_rh"]])
        pdd_des_feet = np.array([trunk_data["pdd_lf"],trunk_data["pdd_rf"],trunk_data["pdd_lh"],trunk_data["pdd_rh"]])

        # Note which feet are in contact (_c) and which feet are in swing (_s)
        contact_feet = trunk_data["contact_states"]
        swing_feet = [not foot for foot in contact_feet]
        num_contact = sum(contact_feet)
        num_swing = sum(swing_feet)

        p_c = p_feet[contact_feet]
        J_c = J_feet[contact_feet]
        Jdv_c = Jdv_feet[contact_feet]

        # Obtain p_tilde, v_tilde, pd_des, pdd_des, J for the body
        p_body_tilde = p_body - trunk_data["p_body"]
        rpy_body_tilde = rpy_body.CalcAngularVelocityInParentFromRpyDt(rpy_body.vector() - trunk_data["rpy_body"])
        p_body_tilde = np.hstack([rpy_body_tilde, p_body_tilde])

        pd_body = J_body@v
        pd_body_des = np.hstack([
                            rpy_body.CalcAngularVelocityInParentFromRpyDt(trunk_data["rpyd_body"]),
                            trunk_data["pd_body"]
                      ])
        pdd_body_des = np.hstack([
                            rpy_body.CalcAngularVelocityInParentFromRpyDt(trunk_data["rpydd_body"]),
                            trunk_data["pdd_body"]
                       ])
        v_body_tilde = pd_body - pd_body_des

        # Obtain p_tilde, v_tilde, pd_des, pdd_des, J for the swing feet
        if any(swing_feet):
            p_s = np.hstack(p_feet[swing_feet])
            p_s_des = np.hstack(p_des_feet[swing_feet])
            p_s_tilde = p_s - p_s_des

            J_s = np.vstack(J_feet[swing_feet])
            pd_s = J_s@v
            pd_s_des = np.hstack(pd_des_feet[swing_feet])
            v_s_tilde = pd_s - pd_s_des

            pdd_s_des = np.hstack(pdd_des_feet[swing_feet])
        else:
            p_s_tilde = np.zeros((0,))
            v_s_tilde = np.zeros((0,))
            J_s = np.zeros((0,self.plant.num_velocities()))
            pd_s_des = np.zeros((0,))
            pdd_s_des = np.zeros((0,))

        # Compute p_tilde, v_tilde, qd_des, qdd_des for all output variables
        J = np.vstack([J_body, J_s])
        p_tilde = np.hstack([p_body_tilde, p_s_tilde])
        v_tilde = np.hstack([v_body_tilde, v_s_tilde])
        pd_des = np.hstack([pd_body_des, pd_s_des])
        pdd_des = np.hstack([pdd_body_des, pdd_s_des])

        # Compute Jbar and Jbar_dot (numerically)
        Jbar = J.T@np.linalg.inv(J@J.T)

        if context.get_time() > 0 and (self.last_contact_feet == contact_feet):
            Jbar_dot = (Jbar - self.last_Jbar)/self.dt
        else:
            Jbar_dot = np.zeros(Jbar.shape)
        self.last_Jbar = Jbar
        self.last_contact_feet = contact_feet

        # Use Jbar and Jbar_dot to project task-space velocities and accelerations
        # to joint-space velocities and accelerations
        qd_des = Jbar@pd_des
        qdd_des = Jbar_dot@pd_des + Jbar@pdd_des

        # Compute joint velocity error
        qd_tilde = v - qd_des

        # Tuning parameters
        Kp_body = 1000
        Kp_feet = 500

        Kd_body = 250
        Kd_feet = 30
        
        nf = 3*sum(swing_feet)   # there are 3 foot-related variables (x,y,z positions) for each swing foot
        Kp = np.block([[ Kp_body*np.eye(6),  np.zeros((6,nf))   ],
                       [  np.zeros((nf,6)),  Kp_feet*np.eye(nf) ]])
        Kd = np.block([[ Kd_body*np.eye(6),  np.zeros((6,nf))   ],
                       [  np.zeros((nf,6)),  Kd_feet*np.eye(nf) ]])

        # Compute tau_nom (interface)
        tau_nom = M@qdd_des + C@qd_des + tau_g - J.T@(Kp@p_tilde + Kd@v_tilde)

        # Set up the QP
        #   minimize:
        #     w1*|| tau_nom - S'*tau + sum(J'*f) ||^2 +
        #     w2*|| tau ||^2
        #   subject to:
        #        M*vd + Cv + tau_g = S'*tau + sum(J'*f)
        #        f \in friction cones
        #        J_cj*vd + Jd_cj*v == 0

        self.mp = MathematicalProgram()
        
        vd = self.mp.NewContinuousVariables(self.plant.num_velocities(), 1, 'vd')
        tau = self.mp.NewContinuousVariables(self.plant.num_actuators(), 1, 'tau')
        f_c = [self.mp.NewContinuousVariables(3,1,'f_%s'%j) for j in range(num_contact)]
        delta = self.mp.NewContinuousVariables(1,1,'delta')

        # min || tau_nom - S'*tau + sum(J'*f) ||^2
        self.AddGeneralizedForceCost(tau_nom, S, tau, J_c, f_c, weight=1.0)

        # min w*|| tau ||^2
        #self.mp.AddQuadraticErrorCost(Q=0.1*np.eye(self.plant.num_actuators()),
        #                              x_desired = np.zeros(self.plant.num_actuators()),
        #                              vars=tau)

        # min delta
        #self.mp.AddCost(1.0*delta[0,0])

        # s.t. Vdot <= delta
        self.AddVdotConstraint(tau, f_c, delta, qd_tilde, S, J_c, M, Cv, tau_g, 
                                qdd_des, p_tilde, v_tilde, Kp, C)

        # s.t. delta <= gamma(\|u_2\|_inf)
        vdot_max = 0.5*trunk_data["u2_max"]
        vdot_min = -np.inf
        self.mp.AddLinearConstraint(A=np.eye(1),lb=vdot_min*np.eye(1),ub=vdot_max*np.eye(1),vars=delta)

        # s.t. tau_min <= tau <= tau_max
        #tau_min = -100*np.ones((self.plant.num_actuators(),1))
        #tau_max = 100*np.ones((self.plant.num_actuators(),1))
        #self.mp.AddLinearConstraint(A=np.eye(self.plant.num_actuators()),lb=tau_min,ub=tau_max,vars=tau)

        # s.t.  M*vd + Cv + tau_g = S'*tau + sum(J_c[j]'*f_c[j])
        self.AddDynamicsConstraint(M, vd, Cv, tau_g, S, tau, J_c, f_c)

        if any(contact_feet):
            # s.t. f_c[j] in friction cones
            self.AddFrictionPyramidConstraint(f_c)

            # s.t. J_cj*vd + Jd_cj*v == 0 (+ some daming)
            self.AddContactConstraint(J_c, vd, Jdv_c, v)

        result = self.solver.Solve(self.mp)
        assert result.is_success()
        tau = result.GetSolution(tau)
        output.SetFromVector(tau)
        
        # Fallback PD controller
        #BasicController.DoSetControlTorques(self, context, output)  

        # Set quantities for logging
        self.V = 0.5*qd_tilde.T@M@qd_tilde + p_tilde.T@Kp@p_tilde 
        self.V *= 1/np.min(np.linalg.eigvals(Kp))      # scale by minimum eigenvalue of Kp
        self.err = p_tilde.T@p_tilde
