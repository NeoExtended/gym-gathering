[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# Gym Gathering

Python package providing gym environments for the particle gathering task.

# Getting started

### Particle Gathering
Particle gathering is an algorithmic problem where particles - which are randomly distributed in a maze-like environment - should be gathered at a single position using only global control inputs.
This means that particles cannot be moved individually and instead all particles are moved into the same direction at the same time. 
Just think of the particles as magnetic dust in a maze, surrounded by powerful electronic magnets.

This problem becomes interesting in scenarios where a certain payload should be brought to a target by very small agents that do not have enough volume to store the energy for their movements. 
An example would be the transport of particles inside the human body (e.g. to combat a tumor).

## Installation

# Simulation Environment
This package provides a simulation of the particle gathering problem as a [gym](https://github.com/openai/gym) - compatible reinforcement learning problem.

## Parameters
The simulation is integrated as a series of named gym-environments that exhibit different behavior.

Each environment can be further customized using parameters. 
Especially the reward function and the observations can be custom-built. 
The included reward function as well as observation generation is already parameterizable. 

### Mazes
This package comes with four fixed mazes:
TODO: Add image

Additionally mazes with vessel-like structure can be randomly generated. 

### Particle Physics
The simulation can be run using different modes depending on the desired behavior of the particles, the number of particles

### Goal Positions

### Number of particles

# Environments


# Benchmarks