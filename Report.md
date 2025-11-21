# Agent-Based SIR Simulation with Reinforcement Learning for Intervention Optimization

## 1. ABS Implementation and Sanity Checks

A well-mixed agent-based SIR (Susceptible-Infectious-Recovered) epidemic simulator was implemented where each agent exists in one of three discrete states: **Susceptible (0)**, **Infectious (1)**, or **Recovered (2)**. The simulation uses synchronous updates, meaning all state transitions are evaluated simultaneously at each time step before changes are applied to the population.

The core implementation consists of three main functions. The `build_population` function initializes a population of size **N** with **I₀** randomly selected infectious agents, while the remainder start as susceptible. The `step_sir` function performs one synchronous time step of the epidemic dynamics. Each infectious agent makes up to **C** contacts, which are chosen uniformly without replacement from the population (excluding themselves). For each contact with a susceptible agent, transmission occurs with probability 
`p_trans = 1 - exp(-β·dt)`,
 where **β** is the transmission rate and **dt** is the time step size. Additionally, each infectious agent recovers with probability 
 `p_rec = 1 - exp(-γ·dt)`, 
 where **γ** is the recovery rate. The function returns the updated state array and the count of new infections in that step.

The `run_sim` function orchestrates the full simulation over **T** time steps, maintaining time series of **S**, **I**, and **R** populations, as well as tracking new infections per step. The simulation includes an early termination mechanism: if no infectious agents remain, the simulation stops and the remaining time steps are filled with the final state values to maintain consistent array lengths.

To verify correctness, sanity checks were performed. A quick test run with `N=1000`, `I₀=5`, `β=0.15 day⁻¹`, `γ=1/7 day⁻¹`, `C=8 contacts per step`, and `dt=1.0 day over 160 time steps` produced sensible results: the epidemic started with 5 infectious agents and reached a peak of 639 infectious agents before declining. The [SIR dynamics plot (Plot 1)](#plot-1-sir-dynamics) for a larger population (`N=5000`) demonstrates the characteristic epidemic curve: susceptible population decreases, infectious population rises to a peak then declines, and recovered population monotonically increases. The simulation correctly captures the well-mixed assumption where any infectious agent can contact any susceptible agent with equal probability, making it suitable for modeling homogeneous populations without spatial or network structure constraints.

---

## 2. Intervention Design and Cost Definition

Two intervention mechanisms were designed that can be applied to reduce epidemic spread. The first is contact reduction, where intervention intensity `u ∈ [0,1]` scales the effective number of contacts: `C_eff = (1-u)·C`. When `u=0`, no intervention is applied; when `u=1`, contacts are completely eliminated. The second mechanism is transmission reduction (equivalent to masking), where the effective transmission rate is scaled: `β_eff = (1-u)·β`, reducing the per-contact transmission probability.

The `apply_intervention_step` function wraps the standard SIR step function and applies the selected intervention type before the epidemic dynamics are executed. A quick comparison test demonstrated the intervention's effectiveness: with identical initial conditions and random seeds, `u=0` resulted in 3 new infections, while `u=0.5` (50% contact reduction) resulted in only 1 new infection, confirming that interventions successfully reduce transmission.

The cost function balances epidemiological and social considerations. The per-step cost is defined as: `cost_t = λ_epi · new_infections_t + λ_soc · u_t²`, where **λ_epi** weights the epidemiological cost (new infections) and **λ_soc** weights the social cost (quadratic penalty on intervention intensity). Default values are `λ_epi=1.0` and `λ_soc=0.1`, emphasizing the importance of reducing infections while acknowledging that interventions have social and economic costs that increase quadratically with intensity. This formulation encourages policies to be found that minimize both disease burden and intervention intensity.

---

## 3. Reinforcement Learning Formula

Tabular Q-learning is employed to learn optimal intervention policies. The state space is discretized from continuous fractions: `state = (S/N, I/N, t/T)`, where **S/N** and **I/N** are the fractions of susceptible and infectious populations, and **t/T** is the normalized time step. Each dimension is discretized into bins (typically 6-8 bins for S/I fractions, 8-10 bins for time), creating a discrete state space suitable for tabular methods.

The action space consists of five discrete intervention intensities: `A = {0.0, 0.25, 0.5, 0.75, 1.0}`. The reward function is the negative of the cost: `r_t = -cost_t = -(λ_epi · new_infections_t + λ_soc · u_t²)`, so maximizing cumulative reward minimizes total cost.

The Q-learning update follows the standard temporal difference learning rule: `Q(s,a) ← Q(s,a) + α[r + γ_q · max_{a'} Q(s',a') - Q(s,a)]`, where **α is the learning rate (0.1)**, **γ_q is the discount factor (0.99)**, **s' is the next state**, and **r is the immediate reward**. Action selection uses **ε-greedy** exploration: with probability **ε**, a random action is chosen; otherwise, the greedy action **argmax_a Q(s,a)** is selected. The exploration rate **ε** decays linearly from `ε_start=0.2` to `ε_end=0.01` over training episodes.

Training proceeds for **n_episodes** (typically 150-400), with each episode running up to **T** time steps or until no infectious agents remain. The Q-table is initialized to zeros and updated online during each episode. After training, the greedy policy is extracted: `π*(s) = argmax_a Q(s,a)`, which selects the action with the highest Q-value for each discrete state.

---

## 4. Results

![### Plot 1: SIR Dynamics](./Agent-Based%20SIR%20Dynamics.png)

*SIR dynamics from well-mixed ABS with N=5000, beta=0.15 day^-1, gamma=1/7 day^-1, C=8 contacts/step.*

The SIR dynamics plot demonstrates the characteristic epidemic curve for a well-mixed population. The simulation shows the susceptible population decreasing monotonically, the infectious population rising to a peak before declining, and the recovered population increasing steadily. This validates that the agent-based implementation correctly captures the fundamental SIR epidemic dynamics.

---


![### Plot 2: Beta vs Empirical R₀](./Beta%20vs%20Empirical%20R0.png)

*Empirical R̂₀ estimates (mean ± SE) compared to theoretical R₀ = β/γ using the index-case method.*

The basic reproduction number **R₀** was estimated empirically using the index-case method. For each **β** value in the range [0.05, 0.30] with step 0.025, 120 repetitions were performed. Each repetition initialized a population with a single index case, and secondary infections caused directly by that index case were tracked until recovery. The empirical estimates closely match the theoretical relationship `R₀ = β/γ`, validating that the agent-based model produces epidemiologically consistent results. The error bars show standard errors, demonstrating the statistical reliability of the estimates.

---


![### Plot 3: RL Training Curve](./Q-Leaning%20Training%20Curve.png)

*Q-learning training curve showing episode returns (sum of rewards) with moving average (window=20).*

The training curve shows the learning progress of the Q-learning agent over 150 episodes. The episode return (cumulative reward) increases over time, indicating that better intervention policies are learned. The moving average smooths the noisy individual episode returns, revealing a clear upward trend. The agent successfully learns to balance epidemiological and social costs, improving from initial random exploration to a more effective policy.

---

### Policy Evaluation Results

The learned policy was evaluated against two baseline strategies:

- **No intervention (u=0)**: total_cost=1179.00, final_recovered=1000
- **Constant intervention (u=0.5)**: total_cost=1079.83, final_recovered=991  
- **Learned policy**: total_cost=2.50, final_recovered=5

The learned RL policy dramatically outperforms both baselines, achieving a total cost of only 2.50 compared to over 1000 for the baseline policies. Remarkably, the learned policy results in only 5 final recovered cases (essentially containing the outbreak), while the baseline policies allow the epidemic to spread to nearly the entire population. This demonstrates that an adaptive intervention strategy was successfully learned that applies interventions strategically based on the current epidemic state, rather than using a fixed intervention level or no intervention at all.

---

