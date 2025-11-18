# Agent-Based SIR Simulation with Reinforcement Learning for Intervention Optimization

This project implements an agent-based SIR (Susceptible-Infectious-Recovered) epidemic model with reinforcement learning (RL) optimization for intervention strategies. The simulation uses a well-mixed population where infectious agents make random contacts, and RL agents learn optimal intervention policies to minimize both epidemiological and social costs.

## Project Overview

The project is divided into four main parts:

- **Part A**: Building an Agent-Based SIR simulator (well-mixed population)
- **Part B**: Designing intervention mechanisms (contact reduction and transmission reduction)
- **Part C**: RL-based intervention optimization using tabular Q-learning
- **Part D**: Experiments including empirical R₀ estimation and policy comparisons

## Features

- **Well-mixed agent-based SIR model**: Each infectious agent makes `C` contacts chosen uniformly without replacement
- **Two intervention types**:
  - Contact reduction: scales the number of contacts `C_eff = (1 - u) * C`
  - Transmission reduction: scales per-contact transmission probability
- **Tabular Q-learning agent**: Learns optimal intervention policies to minimize combined epidemiological and social costs
- **Empirical R₀ estimation**: Index-case method to estimate basic reproduction number
- **Policy evaluation**: Compares learned policies against baseline strategies

## Requirements

### Python Version
- Python 3.10 or higher

### Dependencies
```bash
pip install numpy matplotlib tqdm
```

Required packages:
- `numpy` - Numerical computations
- `matplotlib` - Plotting and visualization
- `tqdm` - Progress bars (notebook version)
- `typing` - Type hints (standard library)

Optional packages:
- `numba` - For performance optimization (if needed)
- `seaborn` - Enhanced plotting (optional)

## Project Structure

```
.
├── agentic-sir-simulation.ipynb  # Main Jupyter notebook with all code
├── Guidelines.md                  # Detailed implementation guidelines
├── README.md                      # This file
└── Seminar 4 - Agent-Based SIR Simulation with Intervention Optimization via RL.pdf
```

## Usage

### Running the Notebook

1. Open `agentic-sir-simulation.ipynb` in Jupyter Notebook or JupyterLab
2. Ensure you have the required dependencies installed
3. Run cells sequentially from top to bottom

**Important**: The notebook must be executed in order, as later cells depend on earlier ones:
- Cells 0-3: Setup and imports
- Cells 4-8: Part A (SIR simulator)
- Cells 9-12: Plot 1 (SIR dynamics)
- Cells 13-16: Part B (Interventions)
- Cells 17-21: Part C (RL training)
- Cells 22-23: Plot 3 (Training curve)
- Cells 24-26: Part D (R₀ estimation and Plot 2)
- Cells 27-31: Policy evaluation and comparisons

### Key Parameters

#### SIR Model Parameters
- `N`: Population size (default: 1000-5000)
- `I0`: Initial number of infectious agents (default: 5)
- `beta`: Transmission rate per day (default: 0.15)
- `gamma`: Recovery rate per day (default: 1/7 ≈ 0.143)
- `C`: Number of contacts per infectious agent per step (default: 8)
- `dt`: Time step size in days (default: 1.0)
- `T`: Total simulation time steps (default: 150-200)

#### RL Parameters
- `actions`: Discrete action set `[0.0, 0.25, 0.5, 0.75, 1.0]`
- `n_episodes`: Number of training episodes (default: 150-400)
- `alpha`: Learning rate (default: 0.1)
- `gamma_q`: Discount factor (default: 0.99)
- `eps_start`: Initial exploration rate (default: 0.2)
- `eps_end`: Final exploration rate (default: 0.01)
- `n_bins`: State discretization bins for S/I fractions (default: 6-8)
- `t_bins`: Time discretization bins (default: 8-10)

#### Cost Function Parameters
- `lambda_epi`: Epidemiological cost weight (default: 1.0)
- `lambda_soc`: Social cost weight (default: 0.1)
- Cost formula: `cost_t = lambda_epi * new_infections_t + lambda_soc * u_t²`

## Reproducibility

The project uses fixed random seeds for reproducibility:
- **Global seed**: `SEED = 42`
- **Episode-specific seeds**: `SEED + episode_number` for training
- **Evaluation seed**: `SEED + 999` for policy evaluation

All stochastic operations use `set_seed()` to ensure reproducible results.

## Outputs

The notebook generates three main plots:

1. **Plot 1 - SIR Dynamics**: Time series of Susceptible, Infectious, and Recovered populations
2. **Plot 2 - Beta vs Empirical R₀**: Comparison of empirical R₀ estimates with theoretical R₀ = β/γ
3. **Plot 3 - RL Training Curve**: Episode returns with moving average showing learning progress

Additionally, the notebook prints:
- Policy evaluation results comparing no intervention, constant intervention, and learned policy
- Empirical R₀ estimates with standard errors

## Model Details

### SIR State Transitions
- **Susceptible (S) → Infectious (I)**: Occurs when a susceptible agent contacts an infectious agent with probability `p_trans = 1 - exp(-β * dt)`
- **Infectious (I) → Recovered (R)**: Occurs with probability `p_rec = 1 - exp(-γ * dt)`

### Intervention Mechanisms
- **Contact reduction**: Reduces the effective number of contacts per infectious agent
- **Transmission reduction**: Reduces the per-contact transmission probability (equivalent to masking)

### RL State Space
The state is discretized as `(S/N, I/N, t/T)` where:
- `S/N`: Fraction of susceptible population
- `I/N`: Fraction of infectious population  
- `t/T`: Normalized time step

### Reward Function
Reward = `-cost`, where cost combines:
- Epidemiological cost: proportional to new infections
- Social cost: quadratic in intervention intensity

## Extensions and Future Work

Potential enhancements:
- **Spatial/Network models**: Replace well-mixed contacts with grid-based or network-based contact structures
- **Advanced RL**: Replace tabular Q-learning with actor-critic or policy gradient methods (PyTorch)
- **Performance optimization**: Use Numba JIT compilation for large population simulations
- **Continuous actions**: Extend to continuous action spaces using function approximation

## References

See `Guidelines.md` for detailed implementation guidelines, design decisions, and code structure.

## License

This project is part of a seminar on "State-of-the-Art in AI Research" and is intended for educational purposes.

## Contact

For questions or issues, please refer to the course materials or guidelines document.

