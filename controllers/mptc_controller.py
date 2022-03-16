from controllers.basic_controller import *

class MPTCController(BasicController):
    """
    A task-space passivity-based whole-body controller, via 
    http://www.roboticsproceedings.org/rss16/p077.pdf.

    Takes as input desired positions/velocities/accelerations of the 
    feet and floating base and computes corresponding joint torques. 
    """
    def __init__(self, plant, dt, use_lcm=False):
        BasicController.__init__(self, plant, dt, use_lcm=use_lcm)

        # inputs from the trunk model are sent in a dictionary
        self.DeclareAbstractInputPort(
                "trunk_input",
                AbstractValue.Make({}))  

        # Set the friction coefficient
        self.mu = 0.7

        # Choose a solver
        #self.solver = GurobiSolver()
        self.solver = OsqpSolver()

        # Storage function for numerically computing Jbar
        self.last_Jbar = None
        self.last_contact_feet = None

    def AddTaskForceCost(self, Jbar, f_des, S, tau, J_c, f_c, W):
        """
        Add a quadratic cost of the form

            1/2 * (f - f_des)' * W * (f - f_des)

        to the whole-body QP, where

            f = Jbar'(S'tau + J_c'f_c) are task-space forces
            f_nom are nominal task-space forces
            W >= 0 is a weighting matrix
        """
        # Stack J_c, f_c 
        if len(f_c) > 0:
            fc = np.vstack(f_c)
            Jc = np.vstack(J_c)
        else:
            fc = np.zeros((0,1))
            Jc = np.zeros((0,self.plant.num_velocities()))

        # Put in the form 1/2*x'*Q*x + c'*x for fast formulation
        U = np.hstack([S.T,Jc.T])
        x = np.vstack([tau, fc])
        
        Q = U.T@Jbar@W@Jbar.T@U
        c = -f_des@W@Jbar.T@U

        return self.mp.AddQuadraticCost(Q, c, x, is_convex=True)

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
        Kd = 100*np.eye(3)

        num_contacts = len(J_c)
        for j in range(num_contacts):
            pd = (J_c[j]@v).reshape(3,1)
            pdd_des = -Kd@pd

            constraint = self.AddJacobianTypeConstraint(J_c[j], vd, Jdv_c[j], pdd_des)

    def ControlLaw(self, context, q, v):
        """
        A MPTC (Modular Passive Tracking Controller) whole-body QP-based controller.

           minimize:
               w_body* || f_body - f_body_des ||^2 +
               w_foot* || f_foot - f_foot_des ||^2 +
           subject to:
                M*vd + Cv + tau_g = S'*tau + sum(J'*f)
                f \in friction cones
                J_cj*vd + Jd_cj*v == 0

        where desired task-space forces are given by 

          f_des = Lambda*xdd_nom + Lambda*Q*(v - Jbar*x_tilde) + f_g - Kp*x_tilde - Kd*x_tilde

        """
        ######### Tuning Parameters #########
        Kp_body_p = 100.0
        Kd_body_p = 10.0

        Kp_body_rpy = Kp_body_p
        Kd_body_rpy = Kd_body_p

        Kp_foot = 200.0
        Kd_foot = 20.0

        w_body = 10.0
        w_foot = 1.0
        #####################################
       
        # Compute Dynamics Quantities
        M, Cv, tau_g, S = self.CalcDynamics()
        C = self.CalcCoriolisMatrix()

        # Get setpoint data from the trunk model
        trunk_data = self.EvalAbstractInput(context,1).get_value()
        
        contact_feet = trunk_data["contact_states"]       # Note: it may be better to determine
        swing_feet = [not foot for foot in contact_feet]  # contact states from the actual robot rather than
        num_contact = sum(contact_feet)                   # the planned trunk trajectory.
        num_swing = sum(swing_feet)

        p_body_nom = trunk_data["p_body"]
        pd_body_nom = trunk_data["pd_body"]
        pdd_body_nom = trunk_data["pdd_body"]

        rpy_body_nom = trunk_data["rpy_body"]
        rpyd_body_nom = trunk_data["rpyd_body"]
        rpydd_body_nom = trunk_data["rpydd_body"]
        
        p_feet_nom = np.array([trunk_data["p_lf"],trunk_data["p_rf"],trunk_data["p_lh"],trunk_data["p_rh"]])
        pd_feet_nom = np.array([trunk_data["pd_lf"],trunk_data["pd_rf"],trunk_data["pd_lh"],trunk_data["pd_rh"]])
        pdd_feet_nom = np.array([trunk_data["pdd_lf"],trunk_data["pdd_rf"],trunk_data["pdd_lh"],trunk_data["pdd_rh"]])

        p_s_nom = p_feet_nom[swing_feet]
        pd_s_nom = pd_feet_nom[swing_feet]
        pdd_s_nom = pdd_feet_nom[swing_feet]

        # Get robot's actual task-space (body pose + foot positions) data
        X_body, J_body, Jdv_body = self.CalcFramePoseQuantities(self.body_frame)
        Jd_body = np.zeros(J_body.shape)  # This is true because the body frame is
                                          # the same as the floating base in q, v
        
        p_body = X_body.translation()
        pd_body = (J_body@v)[3:]

        RPY_body = RollPitchYaw(X_body.rotation())  # RPY object helps convert between angular velocity and rpyd
        rpy_body = RPY_body.vector()
        omega_body = (J_body@v)[:3]   # angular velocity of the body
        rpyd_body = RPY_body.CalcRpyDtFromAngularVelocityInParent(omega_body)

        p_lf, J_lf, Jdv_lf = self.CalcFramePositionQuantities(self.lf_foot_frame)
        p_rf, J_rf, Jdv_rf = self.CalcFramePositionQuantities(self.rf_foot_frame)
        p_lh, J_lh, Jdv_lh = self.CalcFramePositionQuantities(self.lh_foot_frame)
        p_rh, J_rh, Jdv_rh = self.CalcFramePositionQuantities(self.rh_foot_frame)
       
        p_feet = np.array([p_lf, p_rf, p_lh, p_rh]).reshape(4,3)
        J_feet = np.array([J_lf, J_rf, J_lh, J_rh])
        Jdv_feet = np.array([Jdv_lf, Jdv_rf, Jdv_lh, Jdv_rh])
        pd_feet = J_feet@v

        p_s = p_feet[swing_feet]
        pd_s = pd_feet[swing_feet]

        J_c = J_feet[contact_feet]
        Jdv_c = Jdv_feet[contact_feet]
        
        J_s = J_feet[swing_feet]
        Jdv_s = Jdv_feet[swing_feet]

        Jd_s = []         # Only compute JacobianDot for swing feet, 
                          # since this is slow (b/c autodiff)
        if swing_feet[0]:
            Jd_s.append(self.CalcFrameJacobianDot(self.lf_foot_frame_autodiff))
        if swing_feet[1]:
            Jd_s.append(self.CalcFrameJacobianDot(self.rf_foot_frame_autodiff))
        if swing_feet[2]:
            Jd_s.append(self.CalcFrameJacobianDot(self.lh_foot_frame_autodiff))
        if swing_feet[3]:
            Jd_s.append(self.CalcFrameJacobianDot(self.rh_foot_frame_autodiff))

        # Additional task-space dynamics terms
        if any(swing_feet):
            J = np.vstack([J_body, np.vstack(J_s)])
            Jd = np.vstack([Jd_body, np.vstack(Jd_s)])
            Jdv = np.hstack([Jdv_body, np.vstack(Jdv_s).flatten()])
        else:
            J = J_body
            Jd = Jd_body
            Jdv = Jdv_body

        Minv = np.linalg.inv(M)
        Lambda = np.linalg.inv(J@Minv@J.T)
        Jbar = Minv@J.T@Lambda
        Q = J@Minv@C - Jd
        
        # Task-space states and errors
        x = np.hstack([rpy_body, p_body, p_s.flatten()])
        xd = np.hstack([RPY_body.CalcAngularVelocityInParentFromRpyDt(rpyd_body),
                        pd_body,
                        pd_s.flatten()])

        x_nom = np.hstack([rpy_body_nom, p_body_nom, p_s_nom.flatten()])
        xd_nom = np.hstack([RPY_body.CalcAngularVelocityInParentFromRpyDt(rpyd_body_nom),
                            pd_body_nom,
                            pd_s_nom.flatten()])
        xdd_nom = np.hstack([RPY_body.CalcAngularVelocityInParentFromRpyDt(rpydd_body_nom),
                             pdd_body_nom, 
                             pdd_s_nom.flatten()])

        x_tilde = x - x_nom
        xd_tilde = xd - xd_nom

        # Feedback gain and weighting matrices
        nf = 3*sum(swing_feet)   # there are 3 foot-related variables (x,y,z positions) for each swing foot
       
        Kp = np.block([[ np.kron(np.diag([Kp_body_rpy, Kp_body_p]),np.eye(3)), np.zeros((6,nf))   ],
                       [ np.zeros((nf,6)),                                     Kp_foot*np.eye(nf) ]])
        
        Kd = np.block([[ np.kron(np.diag([Kd_body_rpy, Kd_body_p]),np.eye(3)), np.zeros((6,nf))   ],
                       [ np.zeros((nf,6)),                                     Kd_foot*np.eye(nf) ]])

        W = np.diag(np.hstack([w_body*np.ones(6),w_foot*np.ones(nf)]))  # Note: premultiplying by Lambda^{-1} ensures
                                                                        # passivity despite conflicting tasks

        # Desired task-space forces
        f_des = Lambda@xdd_nom + Lambda@Q@(v-Jbar@xd_tilde) + Jbar.T@tau_g - Kp@x_tilde - Kd@xd_tilde

        # Set up and solve the QP
        self.mp = MathematicalProgram()
        
        vd = self.mp.NewContinuousVariables(self.plant.num_velocities(), 1, 'vd')
        tau = self.mp.NewContinuousVariables(self.plant.num_actuators(), 1, 'tau')
        f_c = [self.mp.NewContinuousVariables(3,1,'f_%s'%j) for j in range(num_contact)]

        # min 1/2*(f-f_des)'*W*(f-f_des)
        self.AddTaskForceCost(Jbar, f_des, S, tau, J_c, f_c, W)

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

        # Set quantities for logging
        self.V = 0.5*xd_tilde.T@Lambda@xd_tilde + 0.5*x_tilde.T@Kp@x_tilde 
        self.err = x_tilde.T@x_tilde
        #self.res = result.get_solver_details().primal_res   # OSQP only
        
        fc = np.hstack([result.GetSolution(f) for f in f_c])
        Jc = np.vstack(J_c)
        u = S.T@tau + Jc.T@fc
        f = Jbar.T@u
        f_g = Jbar.T@tau_g
        self.Vdot = xd_tilde.T@(f - f_g + Lambda@Q@(Jbar@xd_tilde - v) - Lambda@xdd_nom + Kp@x_tilde)

        return tau
