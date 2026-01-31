# Single Intersection Traffic Light Control using MDP
This project models a single signalized intersection as a Markov Decision Process (MDP)
and computes an optimal traffic signal control policy using policy iteration.

## Model
- State: (q_NS, q_EW, phase)
- Actions: KEEP, SWITCH
- Arrival process: Poisson
- Reward: negative total queue length

## Method
- Policy iteration implemented from scratch in Python
- No reinforcement learning libraries used

## How to Run
python traffic_mdp.py

> *All code was implemented in Python and is available on GitHub for reproducibility.*
