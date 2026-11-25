# Reading an RL model's weights with an LLM

A few recent posts have discussed "Weight-Space Programming", a term coined by Rikiya Takehi, Jayden Teoh, Alex Zhang and Omar Khattab in ["Sparks of Weight-Space Programming: LLMs can natively read & write neural network weights"](https://rikiyat.github.io/blog/weight-space-programming/).

The authors make several claims about LLMs' ability to read and write neural network weights. I wanted to see whether I could reproduce some of those claims on a problem I control.

## Why this problem?

I've been working with one of my clients on a multi-agent warehouse routing algorithm. Think Amazon warehouse robots. I know the problem, the environment and the reinforcement learning algorithms used to solve it well.

I chose this problem for a few reasons:

* It is far from niche, but RL policies are probably not the most common models an LLM would have encountered during training.
* The environment has complex dynamics: real-time clocks, movement simulations and various constraints. Not your average gridworld.
* The training recipe and architecture are non-standard, using a custom variation of GRPO with Monte Carlo tree search (MCTS).
* The input observation is a 3D tensor of handcrafted features: some continuous, some discrete and some categorical.

I fed only the model weights to the LLM and asked, in short: **"Tell me what this model does."**

## What it found

* **"The model is a tile-based control agent."** True.
* **An actor-critic reinforcement learning setup.** True: a policy gradient method, with a critic to estimate the value function.
* **"Output actions indicate 4 cardinal directions and a stay action. The last one is STAY."** True. The action space is discrete, with five actions, and STAY is indeed the last logit.
* **"Input seems to be a centered 9x17 grid."** True. To be honest, the dimensions are not surprising given the input shape. The "centered" part is.
* **"The ring projections retain 64, 32, 16, 8, 4, then 2 features per cell."** This surprised me. There is indeed a ring projection in the model's input processing.

It also inferred:

> "... the readout preserves considerably more local detail than distant detail—consistent with an agent-centered view used for movement, collision avoidance, or nearby interactions."

Its conclusion:

> "Supaplex-like navigation-and-hazard game | Compatible with terrain, moving entities, local spatial detail, and potentially five movement/wait actions. No weights specifically identify rocks, diamonds, digging, or falling-object rules."

## My conclusions

Some of these observations would be obvious to a human ML expert. Still, the fact that an LLM can infer them is impressive.

It used no tools. It didn't run a forward pass to see what the model does. It just looked at the weights and inferred what the model might be doing.