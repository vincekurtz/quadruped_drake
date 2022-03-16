#!/usr/bin/env python

from pydrake.all import *
from controllers import *
from planners import BasicTrunkPlanner, TowrTrunkPlanner
import os
import sys

############### Common Parameters ###################
show_trunk_model = True
use_lcm = False

planning_method = "towr"   # "towr" or "basic"
control_method = "ID"      # ID = Inverse Dynamics (standard QP), 
                           # B = Basic (simple joint-space PD), 
                           # MPTC = task-space passivity
                           # PC = passivity-constrained
                           # CLF = control-lyapunov-function based

sim_time = 6.0
dt = 5e-3
target_realtime_rate = 1.0

show_diagram = False
make_plots = False

#####################################################

# Drake only loads things relative to the drake path, so we have to do some hacking
# to load an arbitrary file
robot_description_path = "./models/mini_cheetah/mini_cheetah_mesh.urdf"
drake_path = getDrakePath()
robot_description_file = "drake/" + os.path.relpath(robot_description_path, start=drake_path)

robot_urdf  = FindResourceOrThrow(robot_description_file)
builder = DiagramBuilder()
scene_graph = builder.AddSystem(SceneGraph())
plant = builder.AddSystem(MultibodyPlant(time_step=dt))
plant.RegisterAsSourceForSceneGraph(scene_graph) 
quad = Parser(plant=plant).AddModelFromFile(robot_urdf,"quad")

# Add a flat ground with friction
X_BG = RigidTransform()
surface_friction = CoulombFriction(
        static_friction = 1.0,
        dynamic_friction = 1.0)
plant.RegisterCollisionGeometry(
        plant.world_body(),      # the body for which this object is registered
        X_BG,                    # The fixed pose of the geometry frame G in the body frame B
        HalfSpace(),             # Defines the geometry of the object
        "ground_collision",      # A name
        surface_friction)        # Coulomb friction coefficients
plant.RegisterVisualGeometry(
        plant.world_body(),
        X_BG,
        HalfSpace(),
        "ground_visual",
        np.array([0.5,0.5,0.5,0.0]))    # Color set to be completely transparent

# Turn off gravity
#g = plant.mutable_gravity_field()
#g.set_gravity_vector([0,0,0])

plant.Finalize()
assert plant.geometry_source_is_registered()

# Add custom visualizations for the trunk model
trunk_source = scene_graph.RegisterSource("trunk")
trunk_frame = GeometryFrame("trunk")
scene_graph.RegisterFrame(trunk_source, trunk_frame)

trunk_shape = Box(0.4,0.2,0.1)
trunk_color = np.array([0.1,0.1,0.1,0.4])
X_trunk = RigidTransform()
X_trunk.set_translation(np.array([0.0,0.0,0.0]))

trunk_geometry = GeometryInstance(X_trunk,trunk_shape,"trunk")
if show_trunk_model:
    trunk_geometry.set_illustration_properties(MakePhongIllustrationProperties(trunk_color))
scene_graph.RegisterGeometry(trunk_source, trunk_frame.id(), trunk_geometry)

trunk_frame_ids = {"trunk":trunk_frame.id()}

for foot in ["lf","rf","lh","rh"]:
    foot_frame = GeometryFrame(foot)
    scene_graph.RegisterFrame(trunk_source, foot_frame)
   
    foot_shape = Sphere(0.02)
    X_foot = RigidTransform()
    foot_geometry = GeometryInstance(X_foot,foot_shape,foot)
    if show_trunk_model:
        foot_geometry.set_illustration_properties(MakePhongIllustrationProperties(trunk_color))

    scene_graph.RegisterGeometry(trunk_source, foot_frame.id(), foot_geometry)
    trunk_frame_ids[foot] = foot_frame.id()

# Create high-level trunk-model planner and low-level whole-body controller
if planning_method == "basic":
    planner = builder.AddSystem(BasicTrunkPlanner(trunk_frame_ids))
elif planning_method == "towr":
    planner = builder.AddSystem(TowrTrunkPlanner(trunk_frame_ids))
else:
    print("Invalid planning method %s" % planning_method)
    sys.exit(1)

if control_method == "B":
    controller = builder.AddSystem(BasicController(plant,dt,use_lcm=use_lcm))
elif control_method == "ID":
    controller = builder.AddSystem(IDController(plant,dt,use_lcm=use_lcm))
elif control_method == "MPTC":
    controller = builder.AddSystem(MPTCController(plant,dt,use_lcm=use_lcm))
elif control_method == "PC":
    controller = builder.AddSystem(PCController(plant,dt,use_lcm=use_lcm))
elif control_method == "CLF":
    controller = builder.AddSystem(CLFController(plant,dt,use_lcm=use_lcm))
else:
    print("Invalid control method %s" % control_method)
    sys.exit(1)

# Set up the Scene Graph
builder.Connect(
        scene_graph.get_query_output_port(),
        plant.get_geometry_query_input_port())
builder.Connect(
        plant.get_geometry_poses_output_port(),
        scene_graph.get_source_pose_port(plant.get_source_id()))
builder.Connect(
        planner.GetOutputPort("trunk_geometry"),
        scene_graph.get_source_pose_port(trunk_source))

# Connect the trunk-model planner to the controller
if not control_method == "B":
    builder.Connect(planner.GetOutputPort("trunk_trajectory"), controller.get_input_port(1))

# Connect the controller to the simulated plant
builder.Connect(controller.GetOutputPort("quad_torques"),
                plant.get_actuation_input_port(quad))
builder.Connect(plant.get_state_output_port(),
                controller.GetInputPort("quad_state"))

# Add loggers
logger = LogVectorOutput(controller.GetOutputPort("output_metrics"),builder)

# Set up the Visualizer
DrakeVisualizer().AddToBuilder(builder, scene_graph)
ConnectContactResultsToDrakeVisualizer(builder, plant, scene_graph)

# Compile the diagram: no adding control blocks from here on out
diagram = builder.Build()
diagram.set_name("diagram")
diagram_context = diagram.CreateDefaultContext()

# Visualize the diagram
if show_diagram:
    plt.figure()
    plot_system_graphviz(diagram,max_depth=2)
    plt.show()

# Simulator setup
simulator = Simulator(diagram, diagram_context)
if use_lcm:
    # If we're using LCM to send messages to another simulator or a real
    # robot, we don't want Drake to slow things down, so we'll publish
    # new messages as fast as possible
    simulator.set_target_realtime_rate(0.0)
else:
    simulator.set_target_realtime_rate(target_realtime_rate)

# Set initial states
plant_context = diagram.GetMutableSubsystemContext(plant, diagram_context)
q0 = np.asarray([ 1.0, 0.0, 0.0, 0.0,     # base orientation
                  0.0, 0.0, 0.3,          # base position
                  0.0,-0.8, 1.6, 
                  0.0,-0.8, 1.6, 
                  0.0,-0.8, 1.6, 
                  0.0,-0.8, 1.6])
qd0 = np.zeros(plant.num_velocities())
plant.SetPositions(plant_context,q0)
plant.SetVelocities(plant_context,qd0)

# Run the simulation!
simulator.AdvanceTo(sim_time)

if make_plots:
    log = logger.FindLog(diagram_context)

    # Plot stuff
    t = log.sample_times()[10:]
    V = log.data()[0,10:]   
    err = log.data()[1,10:]
    res = log.data()[2,10:]
    Vdot = log.data()[3,10:]

    plt.figure()
    #plt.subplot(4,1,1)
    #plt.plot(t, res, linewidth='2')
    #plt.ylabel("Residual")

    plt.subplot(3,1,1)
    plt.plot(t, Vdot, linewidth='2')
    plt.axhline(0,linestyle='dashed', color='grey')
    plt.ylabel("$\dot{V}$")

    plt.subplot(3,1,2)
    plt.plot(t, V, linewidth='2')
    plt.ylabel("$V$")

    plt.subplot(3,1,3)
    plt.plot(t, err, linewidth='2')
    plt.ylabel("$\|y_1-y_2\|^2$")
    plt.xlabel("time (s)")

    plt.show()
