# SuperMarioBros-NEAT
## Agent learns to play Super Mario Bros with NeuroEvolution of Augmenting Topologies (NEAT) algorithm
### Introduction
The goal of this project is to develop an AL based agent that can learn how to play the popular game Super Mario Bros using the NeuroEvolution of Augmenting Topologies (NEAT) algorithm. For building the environment, Gym Retro library is used, that lets us turn classic video games into Gym environments. Initially the agent has no information about the game and it developes a strategy to figure out how to play the game and hence maximize the reward.

### Requirements
This project uses python 3.7  
To install retro library and use it to make the environment:
  1. Install the library using pip
```
  pip install gym-retro
```
  2. Import the game using the command (The .nes file should be in your working directory)
```
  python -m retro.import /path/to/your/working/directory
```

To install neat library using pip:
```
pip install neat-python
```

### Run
