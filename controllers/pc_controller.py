from controllers.mptc_controller import *

class PCController(MPTCController):
    """
    A task-space passivity-based whole-body controller with passivity
    constraints. 

    Takes as input desired positions/velocities/accelerations of the 
    feet and floating base and computes corresponding joint torques. 
    """
    def __init__(self, plant, dt, use_lcm=False):
        MPTCController.__init__(self, plant, dt, use_lcm=use_lcm)

    def AddVdotConstraint(self, Jbar, S, J_c, tau, f_c, tau_g, Lambda, \
                          Q, v, Kp, xdd_nom, xd_tilde, x_tilde, delta):
        """
        Add a constraint Vdot <= delta to the whole-body QP, where

        Vdot = xd_tilde'*(f_task - f_g + Lambda*Q*(Jbar*xd_tilde - v) - Lambda*xdd_nom + Kp*x_tilde)
        """
        # Stack J_c, f_c 
        if len(f_c) > 0:
            fc = np.vstack(f_c)
            Jc = np.vstack(J_c)
        else:
            fc = np.zeros((0,1))
            Jc = np.zeros((0,self.plant.num_velocities()))

        # Put in the form lb <= A*x <= ub
        x = np.vstack([tau, fc, delta])
        U = np.hstack([S.T, Jc.T])

        A = np.hstack([xd_tilde.T@Jbar.T@U, -1])[np.newaxis]
        ub = xd_tilde.T@(Jbar.T@tau_g - Lambda@Q@(Jbar@xd_tilde - v) + Lambda@xdd_nom - Kp@x_tilde)
        lb = np.asarray(-np.inf)

        ub = ub.reshape(1,1)
        lb = lb.reshape(1,1)

        return self.mp.AddLinearConstraint(A=A,lb=lb,ub=ub,vars=x)


    def ControlLaw(self, context, q, v):
        """
        A passivity-constrained whole-body QP-based controller.

           minimize:
               w_body* || f_body - f_body_des ||^2 +
               w_foot* || f_foot - f_foot_des ||^2 +
               w_Vdot* delta
           subject to:
                M*vd + Cv + tau_g = S'*tau + sum(J'*f)
                f \in friction cones
                J_cj*vd + Jd_cj*v == 0
                Vdot <= delta
                delta <= 0

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
        w_Vdot = 0.00
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
        delta = self.mp.NewContinuousVariables(1,'delta')

        # min 1/2*(f-f_des)'*W*(f-f_des)
        self.AddTaskForceCost(Jbar, f_des, S, tau, J_c, f_c, W)

        # min w_Vdot*delta
        #self.mp.AddCost(w_Vdot*delta[0])

        # min w_Vdot* || delta - Vdot_nom ||^2
        #Vdot_nom = (-xd_tilde.T@Kd@xd_tilde)*np.eye(1)
        #self.mp.AddQuadraticErrorCost(Q=w_Vdot*np.eye(1),
        #                              x_desired = Vdot_nom,
        #                              vars=delta)

        # min w_Vdot* || delta ||^2
        #self.mp.AddCost(w_Vdot*delta.T@delta)

        # s.t.  M*vd + Cv + tau_g = S'*tau + sum(J_c[j]'*f_c[j])
        self.AddDynamicsConstraint(M, vd, Cv, tau_g, S, tau, J_c, f_c)

        if any(contact_feet):
            # s.t. f_c[j] in friction cones
            self.AddFrictionPyramidConstraint(f_c)

            # s.t. J_cj*vd + Jd_cj*v == 0 (+ some daming)
            self.AddContactConstraint(J_c, vd, Jdv_c, v)
    
        # s.t. Vdot <= delta
        self.AddVdotConstraint(Jbar, S, J_c, tau, f_c, tau_g, Lambda,
                               Q, v, Kp, xdd_nom, xd_tilde, x_tilde, delta)

        # s.t. delta <= 0
        self.mp.AddLinearConstraint(A=np.eye(1),
                                    lb=-np.inf*np.eye(1),
                                    ub=0*np.eye(1),
                                    vars=delta)

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
