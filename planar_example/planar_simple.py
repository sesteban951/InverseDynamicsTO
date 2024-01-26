#!/usr/bin/env python

import numpy as np
import time

from pydrake.all import (StartMeshcat, DiagramBuilder,
        AddMultibodyPlantSceneGraph, AddDefaultVisualization, Parser)

from pyidto.trajectory_optimizer import TrajectoryOptimizer
from pyidto.trajectory_optimizer import TrajectoryOptimizer
from pyidto.problem_definition import ProblemDefinition
from pyidto.solver_parameters import SolverParameters
from pyidto.trajectory_optimizer_solution import TrajectoryOptimizerSolution
from pyidto.trajectory_optimizer_stats import TrajectoryOptimizerStats

def define_spinner_optimization_problem():
    """
    Create a problem definition for the spinner.
    """
    problem = ProblemDefinition()
    problem.num_steps = 50
    problem.q_init = np.array([0.53, 0.53,            # z,x position
                               0.0,                  # theta
                               0.0, 0.0, 0.0, 0.0,   # right leg: hip, knee, ankle, toe
                               0.0, 0.0, 0.0, 0.0])  # left leg: hip, knee, ankle, toe
    problem.v_init = np.array([0.0, 0.0,            
                               0.0, 
                               0.0, 0.0, 0.0, 0.0, 
                               0.0, 0.0, 0.0, 0.0])

    r = 0
    l = 0
    #                 0  1  2  3  4      5  6  7  8      9 10
    Qq  =  np.array([[100, 0, 0, 0, 0,     0, 0, 0, 0,     0, 0], # q0
                     [0, 100, 0, 0, 0,     0, 0, 0, 0,     0, 0],  # q1
                     [0, 0, 10, 0, 0,     0, 0, 0, 0,     0, 0],  # q2
                     [0, 0, 0, 10, 0,     0, 0, 0, 0,     0, 0],  # q3
                     [0, 0, 0, 0, 10+r,   r, 0, 0, 0,     0, 0],  # q4
                     [0, 0, 0, 0, r,   10+r, 0, 0, 0,     0, 0],  # q5
                     [0, 0, 0, 0, 0,     0, 10, 0, 0,     0, 0],  # q6
                     [0, 0, 0, 0, 0,     0, 0, 10, 0,     0, 0],  # q7
                     [0, 0, 0, 0, 0,     0, 0, 0, 10+l,   l, 0],  # q8
                     [0, 0, 0, 0, 0,     0, 0, 0, l,   10+l, 0],  # q9
                     [0, 0, 0, 0, 0,     0, 0, 0, 0,     0, 10]]) # q10
    
    assert np.all(np.linalg.eigvals(Qq) > 0), "Qq must be positive definite"

    problem.Qq = Qq
    problem.Qv = 1 * np.eye(11)
    problem.R = np.diag([1e9,1e9,1e9,1,1,1,1,1,1,1,1])
    problem.Qf_q = 10 * np.eye(11)
    problem.Qf_v = 1 * np.eye(11)

    q_nom = []   # Can't use list comprehension here because of Eigen conversion
    v_nom = []
    for i in range(problem.num_steps + 1):
        q_nom_ = np.array([0.53, 0.53,           # z,x position
                           0.0,                  # theta
                           0.0, 0.0, 0.0, 0.0,   # right leg: hip, knee, ankle, toe
                           0.0, 0.0, 0.0, 0.0])  # left leg: hip, knee, ankle, toe 
        v_nom_ = np.zeros(11)
        
        q_nom.append(q_nom_)
        v_nom.append(v_nom_)

    problem.q_nom = q_nom
    problem.v_nom = v_nom

    return problem

def define_spinner_solver_parameters():
    """
    Create a set of solver parameters for the spinner.
    """
    params = SolverParameters()

    params.max_iterations = 200
    params.scaling = True
    params.equality_constraints = True
    params.Delta0 = 1e1
    params.Delta_max = 1e5
    params.num_threads = 4

    params.contact_stiffness = 200
    params.dissipation_velocity = 0.1
    params.smoothing_factor = 0.01
    params.friction_coefficient = 0.5
    params.stiction_velocity = 0.05

    params.verbose = True

    return params

def define_spinner_initial_guess(num_steps, q0):
    """
    Create an initial guess for the spinner
    """
    q_guess = []
    for i in range(num_steps + 1):
        q_ = np.array([0.53, 0.53,            # z,x position
                               0.0,                  # theta
                               0.0, 0.0, 0.0, 0.0,   # right leg: hip, knee, ankle, toe
                               0.0, 0.0, 0.0, 0.0]) 
        q_guess.append(q_)

    q_guess[0] = q0

    return q_guess

def visualize_trajectory(q, time_step, model_file, meshcat=None):
    """
    Display the given trajectory (list of configurations) on meshcat
    """
    # Create a simple Drake diagram with a plant model
    builder = DiagramBuilder()
    plant, scene_graph = AddMultibodyPlantSceneGraph(builder, 1e-3)
    Parser(plant).AddModels(model_file)
    plant.Finalize()

    # Connect to the meshcat visualizer
    AddDefaultVisualization(builder, meshcat)

    # Build the system diagram
    diagram = builder.Build()
    diagram_context = diagram.CreateDefaultContext()
    plant_context = diagram.GetMutableSubsystemContext(plant, diagram_context)
    plant.get_actuation_input_port().FixValue(plant_context,
            np.zeros(plant.num_actuators()))
    
    # Step through q, setting the plant positions at each step
    meshcat.StartRecording()
    for k in range(len(q)):
        diagram_context.SetTime(k * time_step)
        plant.SetPositions(plant_context, q[k])
        diagram.ForcedPublish(diagram_context)
        time.sleep(time_step)
    meshcat.StopRecording()
    meshcat.PublishRecording()
    
if __name__=="__main__":
    # Start up meshcat (for viewing the result)
    meshcat = StartMeshcat()

    # Relative path to the model file that we'll use
    model_file = "./urdf/harpy_planar.urdf"

    # Specify a cost function and target trajectory
    problem = define_spinner_optimization_problem()

    # Specify solver parameters, including contact modeling parameters
    params = define_spinner_solver_parameters()

    # Specify the timestep we'll use to discretize the trajectory
    time_step = 0.05

    # Specify an initial guess
    q_guess = define_spinner_initial_guess(problem.num_steps, problem.q_init)

    # Create the optimizer object
    opt = TrajectoryOptimizer(model_file, problem, params, time_step)

    # Allocate some structs that will hold the solution
    solution = TrajectoryOptimizerSolution()
    stats = TrajectoryOptimizerStats()

    # Solve the optimization problem
    opt.Solve(q_guess, solution, stats)

    solve_time = np.sum(stats.iteration_times)
    print(f"Solved in {solve_time:.4f} seconds")
   
    # Play back the solution on meshcat
    visualize_trajectory(solution.q, time_step, model_file, meshcat)
