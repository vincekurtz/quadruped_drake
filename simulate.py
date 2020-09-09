#!/usr/bin/env python

from pydrake.all import *
import os

# Drake only loads things relative to the drake path, so we have to do some hacking
# to load an arbitrary file
robot_description_path = "./models/anymal_b_simple_description/urdf/anymal_drake.urdf" # relative to this file
drake_path = getDrakePath()
robot_description_file = "drake/" + os.path.relpath(robot_description_path, start=drake_path)

robot_urdf  = FindResourceOrThrow(robot_description_file)
builder = DiagramBuilder()
scene_graph = builder.AddSystem(SceneGraph())
dt = 5e-3
plant = builder.AddSystem(MultibodyPlant(time_step=dt))
plant.RegisterAsSourceForSceneGraph(scene_graph)
quad = Parser(plant=plant).AddModelFromFile(robot_urdf,"quad")

# Add a flat ground with friction
X_BG = RigidTransform()
surface_friction = CoulombFriction(
        static_friction = 0.7,
        dynamic_friction = 0.1)
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

plant.Finalize()
assert plant.geometry_source_is_registered()

# Set up the Scene Graph
builder.Connect(
        scene_graph.get_query_output_port(),
        plant.get_geometry_query_input_port())
builder.Connect(
        plant.get_geometry_poses_output_port(),
        scene_graph.get_source_pose_port(plant.get_source_id()))

# Hook up a controller
controller = builder.AddSystem(ConstantVectorSource(np.zeros(plant.num_actuators())))
builder.Connect(controller.get_output_port(),
                plant.get_actuation_input_port(quad))

# Set up the Visualizer
ConnectDrakeVisualizer(builder=builder, scene_graph=scene_graph)
ConnectContactResultsToDrakeVisualizer(builder, plant)

# Compile the diagram: no adding control blocks from here on out
diagram = builder.Build()
diagram.set_name("diagram")
diagram_context = diagram.CreateDefaultContext()

# Visualize the diagram
#plt.figure()
#plot_system_graphviz(diagram,max_depth=2)
#plt.show()

# Simulator setup
simulator = Simulator(diagram, diagram_context)
simulator.set_target_realtime_rate(1.0)
simulator.set_publish_every_time_step(False)

# Set initial states
plant_context = diagram.GetMutableSubsystemContext(plant, diagram_context)
q0 = np.zeros(plant.num_positions())
q0[3] = 1.0  # make a valid quaternion
q0[6] = 1.0  # z position of base frame
qd0 = np.zeros(plant.num_velocities())
plant.SetPositions(plant_context,q0)
plant.SetVelocities(plant_context,qd0)

print(plant.num_actuators())

simulator.AdvanceTo(5.00)
