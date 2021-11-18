# Reacher Continuous Control

## How to run this project
1. Clone this repository `git clone https://github.com/kirbs-/rl-continuous-reacher`
2. Download the environment from one of the links below.  You need only select the environment that matches your operating system:
      - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
   - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
   - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
   - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)
3. Place the file in the rl-continuous-reacher GitHub repository and unzip (or decompress) the file. 
4. Install dependencies with `pip install -r requirements.txt` **Note Python 3.7+ is required.**
5. Execute `Continuous_Control.ipynb` notebook to train an agent.

## Environment
Reacher environment conists of a single double-jointed arm that an agent controls. The agent's goal is to keep its hand in a moving target location. A reward of 0.1 is given at each time step if the hand is in the correct location.

The agent's action space consist of a vector of four elements between -1 and 1. Each element corresponds to position of an arm accuator that controls one of the torque applied to and arm joint. 

At each step, the agent recieves a state vector of 33 elements containing the position, rotation, speed and angular velocities of the arm. 

The task is considered solved when the agent collects an average of score of 30 over 100 episodes.

