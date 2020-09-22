#include <cmath>
#include <iostream>

#include <towr/terrain/examples/height_map_examples.h>
#include <towr/nlp_formulation.h>
#include <ifopt/ipopt_solver.h>

#include <towr/initialization/gait_generator.h>
#include <Eigen/Dense>


using namespace towr;

// A simple example of generating a trunk-model trajectory for a quadruped, and
// sending the results to Drake.
int main() {

    NlpFormulation formulation;

    // terrain
    formulation.terrain_ = std::make_shared<FlatGround>(0.0);

    // Kinematic limits and dynamic parameters
    formulation.model_ = RobotModel(RobotModel::Anymal);   // TODO: use mini cheetah

    // initial position
    auto nominal_stance_B = formulation.model_.kinematic_model_->GetNominalStanceInBase(); 
    double z_ground = 0.0;
    formulation.initial_ee_W_ = nominal_stance_B;
    std::for_each(formulation.initial_ee_W_.begin(), formulation.initial_ee_W_.end(),
            [&](Eigen::Vector3d& p){ p.z() = z_ground; } // feet at 0 height
    );
    formulation.initial_base_.lin.at(kPos).z() = - nominal_stance_B.front().z() + z_ground;


    //formulation.initial_base_.lin.at(kPos).z() = 0.5;
    //formulation.initial_ee_W_.push_back(Eigen::Vector3d::Zero());

    // desired goal state
    formulation.final_base_.lin.at(towr::kPos) << 1.0, 0.0, 0.5;

    // Total duration of the movement
    float total_duration = 2.0;

    // Parameters defining contact sequence and default durations. We use
    // a GaitGenerator with some predifined gaits
    auto gait_gen_ = GaitGenerator::MakeGaitGenerator(4);
    auto id_gait   = static_cast<GaitGenerator::Combos>(0); // TODO: figure out what different gaits are
    gait_gen_->SetCombo(id_gait);
    for (int ee=0; ee<4; ++ee) {
        formulation.params_.ee_phase_durations_.push_back(gait_gen_->GetPhaseDurations(total_duration, ee));
        formulation.params_.ee_in_contact_at_start_.push_back(gait_gen_->IsInContactAtStart(ee));
    }

    // Indicate whether to optimize over gaits as well
    //formulation.params_.OptimizePhaseDurations();

    // Initialize the nonlinear-programming problem with the variables,
    // constraints and costs.
    ifopt::Problem nlp;
    SplineHolder solution;
    for (auto c : formulation.GetVariableSets(solution))
        nlp.AddVariableSet(c);
    for (auto c : formulation.GetConstraints(solution))
        nlp.AddConstraintSet(c);
    for (auto c : formulation.GetCosts())
        nlp.AddCostSet(c);

    // Choose ifopt solver (IPOPT or SNOPT), set some parameters and solve.
    // solver->SetOption("derivative_test", "first-order");
    auto solver = std::make_shared<ifopt::IpoptSolver>();
    solver->SetOption("jacobian_approximation", "exact"); // "finite difference-values"
    solver->SetOption("max_cpu_time", 20.0);
    solver->Solve(nlp);

    // Can directly view the optimization variables through:
    // Eigen::VectorXd x = nlp.GetVariableValues()
    // However, it's more convenient to access the splines constructed from these
    // variables and query their values at specific times:
    using namespace std;
    cout.precision(2);
    nlp.PrintCurrent(); // view variable-set, constraint violations, indices,...
    cout << fixed;
    cout << "\n====================\nQuadruped trajectory:\n====================\n";

    double t = 0.0;
    while (t<=solution.base_linear_->GetTotalTime() + 1e-5) {
        cout << "t=" << t << "\n";
        cout << "Base linear position x,y,z:   \t";
        cout << solution.base_linear_->GetPoint(t).p().transpose() << "\t[m]" << endl;

        cout << "Base Euler roll, pitch, yaw:  \t";
        Eigen::Vector3d rad = solution.base_angular_->GetPoint(t).p();
        cout << (rad/M_PI*180).transpose() << "\t[deg]" << endl;

        //cout << "Foot position x,y,z:          \t";
        //cout << solution.ee_motion_.at(0)->GetPoint(t).p().transpose() << "\t[m]" << endl;

        //cout << "Contact force x,y,z:          \t";
        //cout << solution.ee_force_.at(0)->GetPoint(t).p().transpose() << "\t[N]" << endl;

        //bool contact = solution.phase_durations_.at(0)->IsContactPhase(t);
        //std::string foot_in_contact = contact? "yes" : "no";
        //cout << "Foot in contact:              \t" + foot_in_contact << endl;

        cout << endl;

        t += 0.2;
    }
}

