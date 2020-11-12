from controllers.basic_controller import *

class PassivityController(BasicController):
    """
    A passivity/approximate simulation-based whole-body controller. 
    Takes as input desired positions/velocities/accelerations of the 
    feet, center-of-mass, and base frame orientation and computes
    corresponding joint torques. 
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

    def ControlLaw(self, context, q, v):
        ######### Tuning Parameters #########
        Kp_body_p = 500.0
        Kd_body_p = 50.0

        Kp_body_rpy = Kp_body_p
        Kd_body_rpy = Kd_body_p

        Kp_foot = 100.0
        Kd_foot = 20.0

        w_body = 10.0
        w_foot = 1.0
        w_Vdot = 1.0
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
        J_s = J_feet[swing_feet]
        Jdv_c = Jdv_feet[contact_feet]
        Jdv_s = Jdv_feet[swing_feet]

        # Task-space (base frame plus any swing feet) Jacobian and related
        if any(swing_feet):
            J = np.vstack([J_body, np.vstack(J_s)])
            Jdv = np.hstack([Jdv_body, np.vstack(Jdv_s).flatten()])
        else:
            J = J_body
            Jdv = Jdv_body
        
        Jbar = J.T@np.linalg.inv(J@J.T)   # Jacobian pseudoinverse

        if context.get_time() > 0 and (self.last_contact_feet == contact_feet):
            # Time derivative of jacobian pseudoinverse
            Jbar_dot = (Jbar - self.last_Jbar)/self.dt
        else:
            Jbar_dot = np.zeros(Jbar.shape)
        self.last_Jbar = Jbar
        self.last_contact_feet = contact_feet

        # Error terms: p_tilde, v_tilde, qd_tilde
        p_tilde = np.hstack([ rpy_body - rpy_body_nom,
                              p_body - p_body_nom,
                              p_s.flatten() - p_s_nom.flatten()
                            ])
        
        v_tilde = np.hstack([ RPY_body.CalcAngularVelocityInParentFromRpyDt(rpyd_body - rpyd_body_nom),
                              pd_body - pd_body_nom,
                              pd_s.flatten() - pd_s_nom.flatten()
                            ])

        qd_tilde = Jbar@v_tilde 

        # Feed-forward terms: pd_nom, pdd_nom, qd_des, qdd_des
        pd_nom = np.hstack([ RPY_body.CalcAngularVelocityInParentFromRpyDt(rpyd_body_nom),
                             pd_body_nom, 
                             pd_s_nom.flatten()
                           ])
        pdd_nom = np.hstack([ RPY_body.CalcAngularVelocityInParentFromRpyDt(rpydd_body_nom),
                              pdd_body_nom, 
                              pdd_s_nom.flatten()
                           ])
       
        qd_des = Jbar@pd_nom
        qdd_des = Jbar_dot@pd_nom + Jbar@pdd_nom
        
        # Construct matrix version of task-space PD gains
        nf = 3*sum(swing_feet)   # there are 3 foot-related variables (x,y,z positions) for each swing foot
        Kp = np.block([[ np.kron(np.diag([Kp_body_rpy, Kp_body_p]),np.eye(3)), np.zeros((6,nf))   ],
                       [ np.zeros((nf,6)),                                     Kp_foot*np.eye(nf) ]])
        Kd = np.block([[ np.kron(np.diag([Kd_body_rpy, Kd_body_p]),np.eye(3)), np.zeros((6,nf))   ],
                       [ np.zeros((nf,6)),                                     Kd_foot*np.eye(nf) ]])

        # Compute interface
        tau_nom = M@qdd_des + C@qd_des + tau_g - J.T@(Kp@p_tilde + Kd@v_tilde)

        # Set up and solve the MP
        #   minimize:
        #       w_body* || J_body*vd + Jd_body*v - vd_body_des ||^2 +
        #       w_foot* || J_s*vd+ Jd_s*v - pdd_s_des ||^2 +
        #       w_V * delta
        #   subject to:
        #        M*vd + Cv + tau_g = S'*tau + sum(J'*f)
        #        f \in friction cones
        #        J_cj*vd + Jd_cj*v == 0
        #        Vdot <= delta
        #        delta <= 0

        self.mp = MathematicalProgram()
        
        vd = self.mp.NewContinuousVariables(self.plant.num_velocities(), 1, 'vd')
        tau = self.mp.NewContinuousVariables(self.plant.num_actuators(), 1, 'tau')
        f_c = [self.mp.NewContinuousVariables(3,1,'f_%s'%j) for j in range(num_contact)]
        delta = self.mp.NewContinuousVariables(1,1,'delta')

        # min || J_body*vd + Jd_body*v - pdd_body_des \|^2
        pdd_body_des = pdd_body_nom - Kp_body_p*(p_body - p_body_nom) - Kd_body_p*(pd_body - pd_body_nom)
        rpydd_body_des = rpydd_body_nom - Kp_body_rpy*(rpy_body - rpy_body_nom) - Kd_body_rpy*(rpyd_body - rpyd_body_nom)
        omegad_body_des = RPY_body.CalcAngularVelocityInParentFromRpyDt(rpydd_body_des)
        vd_body_des = np.hstack([omegad_body_des,pdd_body_des])   # desired spatial acceleration of the body
        self.AddJacobianTypeCost(J_body, vd, Jdv_body, vd_body_des, weight=w_body)

        # min || J_s*vd+ Jd_s*v - pdd_s_des ||^2
        pdd_s_des = pdd_s_nom  - Kp_foot*(p_s - p_s_nom) - Kd_foot*(pd_s - pd_s_nom)
        for i in range(num_swing):
            self.AddJacobianTypeCost(J_s[i], vd, Jdv_s[i], pdd_s_des[i], weight=w_foot)
        
        # min delta
        self.mp.AddCost(w_Vdot*delta[0,0])
        
        # s.t. Vdot <= delta
        self.AddVdotConstraint(tau, f_c, delta, qd_tilde, S, J_c, M, Cv, tau_g, 
                                qdd_des, p_tilde, v_tilde, Kp, C)

        # s.t. delta <= 0
        vdot_max = 0
        vdot_min = -np.inf
        self.mp.AddLinearConstraint(A=np.eye(1),lb=vdot_min*np.eye(1),ub=vdot_max*np.eye(1),vars=delta)

        # s.t.  M*vd + Cv + tau_g = S'*tau + sum(J_c[j]'*f_c[j])
        self.AddDynamicsConstraint(M, vd, Cv, tau_g, S, tau, J_c, f_c)

        if any(contact_feet):
            # s.t. f_c[j] in friction cones
            self.AddFrictionPyramidConstraint(f_c)

            # s.t. J_cj*vd + Jd_cj*v == 0 (+ some daming)
            self.AddContactConstraint(J_c, vd, Jdv_c, v)

        # Set quantities for logging
        self.V = 0.5*qd_tilde.T@M@qd_tilde + p_tilde.T@Kp@p_tilde 
        self.V *= 1/20  # scale V for easier visualization
        self.err = p_tilde.T@p_tilde

        result = self.solver.Solve(self.mp)
        assert result.is_success()
        tau = result.GetSolution(tau)
    
        return tau
