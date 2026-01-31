"""
Single-intersection traffic light control as an MDP
Policy Iteration (from scratch, no RL libraries)

State: (qNS, qEW, phase)
  qNS, qEW in [0..Q_MAX]
  phase in {0(NS green), 1(EW green)}

Action: 0=KEEP, 1=SWITCH
- SWITCH triggers a "lost time" for LOST_STEPS steps where no discharge happens,
  then phase becomes the other direction.

To keep the MDP finite while still modeling lost time, we augment phase with a
small timer t in [0..LOST_STEPS], where t>0 means we are currently in lost time.
So internal state actually: (qNS, qEW, phase, t)
But the interface still matches the assignment: traffic light control MDP.

Reward per step: -(qNS + qEW)   (minimize queue => delay proxy)

Transitions:
- Arrivals are Poisson with mean lambda * dt; truncated to [0..A_MAX]
- Discharge: if green and t==0:
    d = min(queue, floor(mu * dt))
  else d = 0

You can:
1) Set lambdas from real counts: veh/hour -> lambda = (veh/hour)/3600
2) Run policy iteration
3) Simulate optimal vs baselines

Author: you (with ChatGPT help)
"""

from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import Dict, Tuple, List

# -----------------------------
# Config (edit these)
# -----------------------------
DT = 5.0                   # seconds per step
GAMMA = 0.98
Q_MAX = 20                 # queue cap (finite states)
A_MAX = 8                  # arrival truncation per step (per approach)
SAT_FLOW_PER_LANE = 1900.0 # veh/hour/lane (typical default)
LANES_NS = 1
LANES_EW = 1
LOST_STEPS = 2             # e.g., 2 steps * 5s = 10s lost time
EVAL_EPS = 1e-8            # policy evaluation convergence
EVAL_MAX_ITERS = 50_000
PI_MAX_ITERS = 200

# Example: from real data (veh/hour). Replace with your chosen intersection/time.
VOL_NS_PER_HOUR = 900.0
VOL_EW_PER_HOUR = 600.0

# For reproducibility of simulation
RNG_SEED = 7


# -----------------------------
# Helpers: Poisson PMF with truncation
# -----------------------------
# -----------------------------
# Poisson arrival model
# -----------------------------
def poisson_pmf(k: int, lam: float) -> float:
    """
    Poisson probability mass function.

    FORMULA:
        P(K = k) = (λ^k * e^{-λ}) / k!

    where:
        λ = expected number of arrivals in one time step
        k = number of arriving vehicles
    """
    return math.exp(-lam) * (lam ** k) / math.factorial(k)


def truncated_poisson_probs(lam: float, k_max: int) -> List[float]:
    """
    Truncated Poisson distribution.

    FORMULA:
        P_trunc(k) = P(k),  k = 0, 1, ..., k_max-1
        P_trunc(k_max) = P(K >= k_max)

    This ensures a finite state space for the MDP.
    """
    probs = [poisson_pmf(k, lam) for k in range(k_max + 1)]
    s = sum(probs)
    if s < 1.0:
        probs[-1] += (1.0 - s)
    else:
        probs = [p / s for p in probs]
    return probs


def sample_truncated_poisson(probs: List[float], rng: random.Random) -> int:
    r = rng.random()
    c = 0.0
    for k, p in enumerate(probs):
        c += p
        if r <= c:
            return k
    return len(probs) - 1


# -----------------------------
# MDP state definition
# -----------------------------
# State s = (q_NS, q_EW, p, t)
#
# FORMULA:
#   q_NS(t+1) = min(Q_max, q_NS(t) - D_NS(t) + A_NS(t))
#   q_EW(t+1) = min(Q_max, q_EW(t) - D_EW(t) + A_EW(t))
#
# where:
#   A_NS, A_EW ~ Poisson(λ * Δt)
#   D_NS, D_EW are discharge flows
#
# p ∈ {0,1} is the current green phase
# t is the remaining lost-time counter
#
State = Tuple[int, int, int, int]

Action = int                       # 0 KEEP, 1 SWITCH

@dataclass(frozen=True)
class MDPParams:
    dt: float
    gamma: float
    q_max: int
    a_max: int
    lost_steps: int
    mu_ns: int  # discharge capacity per step (veh/step)
    mu_ew: int
    lam_ns_step: float  # mean arrivals per step
    lam_ew_step: float

def build_params() -> MDPParams:
    lam_ns_step = (VOL_NS_PER_HOUR / 3600.0) * DT
    lam_ew_step = (VOL_EW_PER_HOUR / 3600.0) * DT
    # discharge per step: mu = sat_flow * lanes / 3600 * dt
    mu_ns = int((SAT_FLOW_PER_LANE * LANES_NS / 3600.0) * DT)
    mu_ew = int((SAT_FLOW_PER_LANE * LANES_EW / 3600.0) * DT)
    mu_ns = max(mu_ns, 1)
    mu_ew = max(mu_ew, 1)
    return MDPParams(
        dt=DT, gamma=GAMMA, q_max=Q_MAX, a_max=A_MAX, lost_steps=LOST_STEPS,
        mu_ns=mu_ns, mu_ew=mu_ew, lam_ns_step=lam_ns_step, lam_ew_step=lam_ew_step
    )

def all_states(p: MDPParams) -> List[State]:
    states: List[State] = []
    for qns in range(p.q_max + 1):
        for qew in range(p.q_max + 1):
            for phase in (0, 1):
                for t in range(p.lost_steps + 1):
                    states.append((qns, qew, phase, t))
    return states

def reward(s: State) -> float:
    """
    Reward function.

    FORMULA:
        R(s, a) = - (q_NS + q_EW)

    This is a proxy for minimizing total waiting time
    (Little's Law: average queue length ∝ average delay).
    """
    qns, qew, _, _ = s
    return -(qns + qew)


def next_phase_and_timer(phase: int, t: int, a: Action, p: MDPParams) -> Tuple[int, int]:
    """
    Traffic signal phase transition.

    FORMULA:
        if t > 0:
            t' = t - 1
            p' = p
        else if a == SWITCH:
            p' = 1 - p
            t' = T_lost
        else:
            p' = p
            t' = 0
    """
    if t > 0:
        return phase, t - 1
    if a == 0:  # KEEP
        return phase, 0
    else:       # SWITCH
        return 1 - phase, p.lost_steps

def discharge_amount(qns: int, qew: int, phase: int, t: int, p: MDPParams) -> Tuple[int, int]:
    """
    Vehicle discharge model.

    FORMULA:
        D_NS = min(q_NS, μ_NS * Δt)   if NS green
        D_EW = min(q_EW, μ_EW * Δt)   if EW green

        D = 0 during lost time

    μ is computed from saturation flow:
        μ = (saturation_flow * lanes / 3600) * Δt
    """
    if t > 0:
        return 0, 0
    if phase == 0:  # NS green
        dns = min(qns, p.mu_ns)
        return dns, 0
    else:           # EW green
        dew = min(qew, p.mu_ew)
        return 0, dew


def clip_queue(x: int, q_max: int) -> int:
    if x < 0: return 0
    if x > q_max: return q_max
    return x

def build_transition_model(p: MDPParams) -> Dict[Tuple[State, Action], List[Tuple[State, float]]]:
    """
    Build sparse transitions:
      T[(s,a)] = [(s', prob), ...]
    Arrival distributions are truncated Poisson for NS and EW, independent => product probs.
    """
    probs_ns = truncated_poisson_probs(p.lam_ns_step, p.a_max)
    probs_ew = truncated_poisson_probs(p.lam_ew_step, p.a_max)

    T: Dict[Tuple[State, Action], List[Tuple[State, float]]] = {}
    states = all_states(p)

    for s in states:
        qns, qew, phase, t = s
        for a in (0, 1):
            nphase, nt = next_phase_and_timer(phase, t, a, p)
            dns, dew = discharge_amount(qns, qew, phase, t, p)

            outcomes: Dict[State, float] = {}
            for ans, p_ans in enumerate(probs_ns):
                for aew, p_aew in enumerate(probs_ew):
                    qns2 = clip_queue(qns - dns + ans, p.q_max)
                    qew2 = clip_queue(qew - dew + aew, p.q_max)
                    s2 = (qns2, qew2, nphase, nt)
                    outcomes[s2] = outcomes.get(s2, 0.0) + p_ans * p_aew

            # Normalize safety
            total = sum(outcomes.values())
            if total <= 0:
                T[(s, a)] = [(s, 1.0)]
            else:
                T[(s, a)] = [(sp, pr / total) for sp, pr in outcomes.items()]
    return T


# -----------------------------
# Policy Iteration
# -----------------------------
def policy_evaluation(
    states: List[State],
    T: Dict[Tuple[State, Action], List[Tuple[State, float]]],
    pi: Dict[State, Action],
    gamma: float,
    eps: float = EVAL_EPS,
    max_iters: int = EVAL_MAX_ITERS
) -> Dict[State, float]:
    
    V = {s: 0.0 for s in states}
    for _ in range(max_iters):
        delta = 0.0
        for s in states:
            a = pi[s]
            v_new = 0.0
            r = reward(s)
            for sp, pr in T[(s, a)]:
                v_new += pr * (r + gamma * V[sp])
            delta = max(delta, abs(v_new - V[s]))
            V[s] = v_new
        if delta < eps:
            break
    return V

def policy_improvement(
    states: List[State],
    T: Dict[Tuple[State, Action], List[Tuple[State, float]]],
    V: Dict[State, float],
    gamma: float
) -> Tuple[Dict[State, Action], bool]:
    stable = True
    new_pi: Dict[State, Action] = {}
    for s in states:
        best_a = None
        best_q = -1e18
        r = reward(s)
        for a in (0, 1):
            q = 0.0
            for sp, pr in T[(s, a)]:
                q += pr * (r + gamma * V[sp])
            if q > best_q + 1e-12:
                best_q = q
                best_a = a
        new_pi[s] = int(best_a)
        # if changed
        if "pi_old" in locals():
            pass
    return new_pi, stable  # stable computed outside for simplicity

def policy_iteration(p: MDPParams) -> Tuple[Dict[State, Action], Dict[State, float]]:
    states = all_states(p)
    T = build_transition_model(p)

    # init random-ish policy: KEEP by default
    pi = {s: 0 for s in states}

    for it in range(PI_MAX_ITERS):
        V = policy_evaluation(states, T, pi, p.gamma)
        policy_stable = True

        for s in states:
            old_a = pi[s]
            # argmax over actions
            r = reward(s)
            q_keep = sum(pr * (r + p.gamma * V[sp]) for sp, pr in T[(s, 0)])
            q_switch = sum(pr * (r + p.gamma * V[sp]) for sp, pr in T[(s, 1)])
            new_a = 0 if q_keep >= q_switch else 1
            pi[s] = new_a
            if new_a != old_a:
                policy_stable = False

        if policy_stable:
            # converged
            return pi, V

    return pi, V


# -----------------------------
# Baselines + Simulation
# -----------------------------
def fixed_time_policy(step_idx: int, cycle_keep_steps: int = 6) -> Action:
    # keep for cycle_keep_steps, then switch, repeating
    # (6 steps * 5s = 30s green chunks)
    return 1 if (step_idx % cycle_keep_steps) == (cycle_keep_steps - 1) else 0

def threshold_policy(s: State, margin: int = 2) -> Action:
    qns, qew, phase, t = s
    if t > 0:
        return 0  # cannot do much; keep
    if phase == 0:
        # NS green; if EW much larger, switch
        return 1 if (qew - qns) >= margin else 0
    else:
        return 1 if (qns - qew) >= margin else 0

def step_env(s: State, a: Action, p: MDPParams, rng: random.Random,
             probs_ns: List[float], probs_ew: List[float]) -> Tuple[State, float]:
    qns, qew, phase, t = s
    # phase/t update
    nphase, nt = next_phase_and_timer(phase, t, a, p)

    # discharge uses CURRENT phase and t (before update) - consistent with model build
    dns, dew = discharge_amount(qns, qew, phase, t, p)

    ans = sample_truncated_poisson(probs_ns, rng)
    aew = sample_truncated_poisson(probs_ew, rng)

    qns2 = clip_queue(qns - dns + ans, p.q_max)
    qew2 = clip_queue(qew - dew + aew, p.q_max)

    s2 = (qns2, qew2, nphase, nt)
    return s2, reward(s)

def simulate(policy_name: str,
             policy_fn,
             p: MDPParams,
             steps: int,
             episodes: int = 30,
             start_state: State = (0, 0, 0, 0)) -> Dict[str, float]:
    rng = random.Random(RNG_SEED)
    probs_ns = truncated_poisson_probs(p.lam_ns_step, p.a_max)
    probs_ew = truncated_poisson_probs(p.lam_ew_step, p.a_max)

    total_queue_means = []
    delay_per_vehicle_means = []
    max_queue_means = []

    for ep in range(episodes):
        s = start_state
        total_queue = 0.0
        max_queue = 0
        # delay proxy: sum of queue over time * dt (veh*sec)
        total_wait_veh_sec = 0.0
        # count served vehicles (approx): discharged each step
        served = 0

        for t_idx in range(steps):
            qns, qew, phase, tt = s
            if policy_name == "optimal":
                a = policy_fn[s]
            elif policy_name == "fixed":
                a = policy_fn(t_idx)
            else:
                a = policy_fn(s)

            # compute discharge for served count (same rule)
            dns, dew = discharge_amount(qns, qew, phase, tt, p)
            served += (dns + dew)

            s, r = step_env(s, a, p, rng, probs_ns, probs_ew)
            q = s[0] + s[1]
            total_queue += q
            max_queue = max(max_queue, q)
            total_wait_veh_sec += q * p.dt

        avg_total_queue = total_queue / steps
        # average delay per vehicle: (total waiting veh-sec) / (vehicles served)
        avg_delay = (total_wait_veh_sec / served) if served > 0 else float("inf")

        total_queue_means.append(avg_total_queue)
        delay_per_vehicle_means.append(avg_delay)
        max_queue_means.append(max_queue)

    def mean(xs): return sum(xs) / len(xs)

    return {
        "avg_total_queue": mean(total_queue_means),
        "avg_delay_s_per_veh": mean(delay_per_vehicle_means),
        "avg_max_queue": mean(max_queue_means),
        "episodes": episodes
    }


# -----------------------------
# Main
# -----------------------------
def main():
    p = build_params()
    print("=== Parameters ===")
    print(f"DT={p.dt}s, gamma={p.gamma}, Q_MAX={p.q_max}, A_MAX={p.a_max}, LOST_STEPS={p.lost_steps}")
    print(f"Arrivals per step: lamNS={p.lam_ns_step:.3f}, lamEW={p.lam_ew_step:.3f} (mean veh/step)")
    print(f"Discharge per step: muNS={p.mu_ns}, muEW={p.mu_ew} (veh/step)")

    print("\n=== Running Policy Iteration ===")
    pi, V = policy_iteration(p)
    print("Policy iteration done.")

    # quick policy sanity: show what it does for a few states (t=0)
    samples = [
        (0, 0, 0, 0),
        (10, 2, 0, 0),
        (2, 10, 0, 0),
        (15, 15, 0, 0),
        (5, 12, 1, 0),
        (12, 5, 1, 0),
    ]
    print("\n=== Sample actions (0=KEEP, 1=SWITCH) ===")
    for s in samples:
        print(f"state={s} -> action={pi[s]}")

    # Simulation horizon: 1 hour
    steps = int(3600 / p.dt)
    print(f"\n=== Simulating {steps} steps (~1 hour), {30} episodes ===")
    res_opt = simulate("optimal", pi, p, steps=steps, episodes=30)
    res_fix = simulate("fixed", fixed_time_policy, p, steps=steps, episodes=30)
    res_thr = simulate("threshold", threshold_policy, p, steps=steps, episodes=30)

    print("\n=== Results (mean over episodes) ===")
    def fmt(r): return f"avg_queue={r['avg_total_queue']:.3f}, delay={r['avg_delay_s_per_veh']:.2f}s/veh, maxQ={r['avg_max_queue']:.2f}"
    print(f"Fixed-time : {fmt(res_fix)}")
    print(f"Threshold  : {fmt(res_thr)}")
    print(f"Optimal PI : {fmt(res_opt)}")

    print("\nCopy these numbers into Slide 2 table.")

if __name__ == "__main__":
    main()
