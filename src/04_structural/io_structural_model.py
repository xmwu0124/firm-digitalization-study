"""
Structural Estimation: Dynamic Discrete Choice Model with Entry/Exit
================================================================================
IO-Style Estimation using JAX for Firm Digital Adoption Decisions

Model:
- Firms face dynamic decision: adopt digital technology or wait
- State variables: firm size, industry competition, year
- Payoffs: adoption gives persistent productivity gain but has fixed cost
- Solution: Backward induction on Bellman equation
- Estimation: Maximum likelihood via Conditional Choice Probability (CCP) inversion

References:
- Rust (1987): Optimal replacement of GMC bus engines
- Aguirregabiria & Mira (2007): Sequential estimation of dynamic discrete games
- Arcidiacono & Miller (2011): CCP estimation of dynamic models

Author: Research Team
Date: October 2025
"""

import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax.scipy.special import logsumexp
from scipy.optimize import minimize
from typing import Tuple, Dict, NamedTuple
from pathlib import Path
import sys
import time

sys.path.append(str(Path(__file__).resolve().parent.parent))
from config_loader import CONFIG, PATHS, setup_logger, set_random_seed

logger = setup_logger("io_estimation")

# JAX configuration
jax.config.update("jax_enable_x64", True)

# ============================================================================
# Model Primitives
# ============================================================================

class ModelParams(NamedTuple):
    """Structural parameters"""
    # Flow payoffs
    alpha_size: float       # Benefit of size if adopted
    alpha_comp: float       # Benefit of low competition if adopted
    alpha_tech: float       # Benefit for tech industry if adopted
    
    # Fixed costs
    fc_adopt: float         # One-time adoption cost
    fc_maintain: float      # Per-period maintenance cost if adopted
    
    # Dynamics
    beta: float            # Discount factor
    delta: float           # Depreciation rate of digital capital
    
    # Transition probabilities
    gamma_size: float      # AR(1) coefficient for size evolution
    sigma_size: float      # Std dev of size shocks
    
    # Extreme value scale
    scale: float           # Scale of Type I EV errors

class StateSpace(NamedTuple):
    """Discretized state space"""
    size_grid: jnp.ndarray      # Firm size grid (log scale)
    comp_grid: jnp.ndarray      # Competition index grid
    tech_dummy: jnp.ndarray     # Tech industry indicator
    adopted: jnp.ndarray        # Already adopted indicator
    
    n_size: int
    n_comp: int
    n_tech: int
    n_adopt: int

# ============================================================================
# Flow Payoffs
# ============================================================================

@jit
def flow_payoff(
    size: float,
    comp: float, 
    tech: float,
    adopted: float,
    action: float,  # 1 = adopt this period, 0 = don't
    already_adopted: float,  # State: already adopted before
    params: ModelParams
) -> float:
    """
    Flow payoff in current period
    
    Args:
        size: Log firm size
        comp: Competition index (higher = more competitive)
        tech: Tech industry dummy
        adopted: Current period adoption decision
        already_adopted: Indicator if adopted in past
        params: Structural parameters
    
    Returns:
        Flow payoff
    """
    # If already adopted, can't adopt again (action ignored)
    effective_adoption = jnp.maximum(already_adopted, action)
    
    # Benefits of digital technology (only if adopted)
    benefit = effective_adoption * (
        params.alpha_size * size +
        params.alpha_comp * (-comp) +  # Negative: high competition is bad
        params.alpha_tech * tech
    )
    
    # Costs
    # One-time fixed cost if adopting this period
    adoption_cost = action * (1.0 - already_adopted) * params.fc_adopt
    
    # Per-period maintenance cost if technology is active
    maintenance_cost = effective_adoption * params.fc_maintain
    
    return benefit - adoption_cost - maintenance_cost

# ============================================================================
# State Transitions
# ============================================================================

@jit
def transition_prob(
    size_today: float,
    size_tomorrow: float,
    params: ModelParams
) -> float:
    """
    Probability of transitioning to size_tomorrow given size_today
    
    Assumes: log(size_t) = gamma * log(size_{t-1}) + N(0, sigma^2)
    """
    mean_tomorrow = params.gamma_size * size_today
    std_tomorrow = params.sigma_size
    
    # Normal density
    z = (size_tomorrow - mean_tomorrow) / std_tomorrow
    log_prob = -0.5 * jnp.log(2 * jnp.pi) - jnp.log(std_tomorrow) - 0.5 * z**2
    
    return jnp.exp(log_prob)

def build_transition_matrix(
    state_space: StateSpace,
    params: ModelParams
) -> jnp.ndarray:
    """
    Build state transition matrix
    
    Returns:
        P[s,a,s']: Transition probability from state s to s' given action a
        Shape: (n_states, n_actions, n_states)
    """
    n_size = state_space.n_size
    n_comp = state_space.n_comp
    n_tech = state_space.n_tech
    n_adopt = state_space.n_adopt
    n_states = n_size * n_comp * n_tech * n_adopt
    n_actions = 2
    
    # For now, assume competition and tech industry are fixed
    # Only size evolves stochastically, adoption is deterministic
    
    P = np.zeros((n_states, n_actions, n_states))
    
    for s_idx in range(n_states):
        # Decode state
        size_idx = s_idx % n_size
        comp_idx = (s_idx // n_size) % n_comp
        tech_idx = (s_idx // (n_size * n_comp)) % n_tech
        adopt_idx = (s_idx // (n_size * n_comp * n_tech)) % n_adopt
        
        size_today = state_space.size_grid[size_idx]
        
        for a in range(n_actions):
            # Next period adoption status
            if adopt_idx == 1:  # Already adopted
                next_adopt_idx = 1
            else:  # Not yet adopted
                next_adopt_idx = a  # 0 if wait, 1 if adopt now
            
            # Compute transition probabilities over size
            for next_size_idx in range(n_size):
                size_tomorrow = state_space.size_grid[next_size_idx]
                
                # State transition probability
                trans_prob = float(transition_prob(size_today, size_tomorrow, params))
                
                # Encode next state
                next_s_idx = (
                    next_size_idx +
                    comp_idx * n_size +
                    tech_idx * n_size * n_comp +
                    next_adopt_idx * n_size * n_comp * n_tech
                )
                
                P[s_idx, a, next_s_idx] = trans_prob
    
    # Normalize (in case of numerical errors)
    P = P / P.sum(axis=2, keepdims=True)
    
    return jnp.array(P)

# ============================================================================
# Value Function Iteration
# ============================================================================

@jit
def bellman_operator(
    V: jnp.ndarray,
    P: jnp.ndarray,
    flow_payoffs: jnp.ndarray,
    params: ModelParams
) -> jnp.ndarray:
    """
    Apply Bellman operator
    
    Args:
        V: Current value function [n_states]
        P: Transition matrix [n_states, n_actions, n_states]
        flow_payoffs: Flow payoffs [n_states, n_actions]
        params: Model parameters
    
    Returns:
        New value function [n_states]
    """
    # Expected continuation value for each (s, a)
    # Shape: [n_states, n_actions]
    EV = jnp.einsum('ijk,k->ij', P, V)
    
    # Total value for each (s, a)
    total_value = flow_payoffs + params.beta * EV
    
    # Logit inclusive value (accounts for discrete choice shocks)
    # V(s) = scale * log(sum_a exp(Q(s,a) / scale))
    V_new = params.scale * logsumexp(total_value / params.scale, axis=1)
    
    return V_new

def solve_dynamic_program(
    state_space: StateSpace,
    params: ModelParams,
    P: jnp.ndarray,
    flow_payoffs: jnp.ndarray,
    tol: float = 1e-6,
    max_iter: int = 1000
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Solve dynamic program via value function iteration
    
    Returns:
        V: Value function
        policy: Choice probabilities [n_states, n_actions]
    """
    n_states = len(flow_payoffs)
    V = jnp.zeros(n_states)
    
    logger.info(f"Solving DP with {n_states} states...")
    start = time.time()
    
    for iteration in range(max_iter):
        V_new = bellman_operator(V, P, flow_payoffs, params)
        
        error = jnp.max(jnp.abs(V_new - V))
        V = V_new
        
        if iteration % 100 == 0:
            logger.info(f"  Iteration {iteration}: error = {error:.6f}")
        
        if error < tol:
            logger.info(f"  Converged in {iteration+1} iterations ({time.time()-start:.2f}s)")
            break
    else:
        logger.warning(f"  Did not converge after {max_iter} iterations")
    
    # Compute choice probabilities
    EV = jnp.einsum('ijk,k->ij', P, V)
    Q = flow_payoffs + params.beta * EV
    
    # Logit probabilities
    policy = jax.nn.softmax(Q / params.scale, axis=1)
    
    return V, policy

# ============================================================================
# Likelihood Construction
# ============================================================================

def encode_state(
    size_val: float,
    comp_val: float,
    tech_val: float,
    adopt_val: float,
    state_space: StateSpace
) -> int:
    """Map continuous state values to discrete state index"""
    # Find nearest grid points
    size_idx = jnp.argmin(jnp.abs(state_space.size_grid - size_val))
    comp_idx = jnp.argmin(jnp.abs(state_space.comp_grid - comp_val))
    tech_idx = int(tech_val)
    adopt_idx = int(adopt_val)
    
    state_idx = (
        size_idx +
        comp_idx * state_space.n_size +
        tech_idx * state_space.n_size * state_space.n_comp +
        adopt_idx * state_space.n_size * state_space.n_comp * state_space.n_tech
    )
    
    return int(state_idx)

def compute_likelihood(
    data: pd.DataFrame,
    state_space: StateSpace,
    params: ModelParams,
    P: jnp.ndarray,
    flow_payoffs: jnp.ndarray
) -> float:
    """
    Compute log-likelihood of observed choices
    
    Args:
        data: Panel data with columns [gvkey, year, size, comp, tech, adopted, action]
        state_space: Discretized state space
        params: Structural parameters
        P: Transition matrix
        flow_payoffs: Flow payoff matrix
    
    Returns:
        Log-likelihood
    """
    # Solve for value function and policy
    _, policy = solve_dynamic_program(state_space, params, P, flow_payoffs)
    
    # Compute log-likelihood
    log_lik = 0.0
    
    for _, row in data.iterrows():
        # Map to state index
        state_idx = encode_state(
            row['log_size'],
            row['competition'],
            row['tech_industry'],
            row['ever_adopted_before'],
            state_space
        )
        
        # Observed action (0 or 1)
        action = int(row['adopt_this_period'])
        
        # Choice probability
        prob = float(policy[state_idx, action])
        
        # Add to log-likelihood (avoid log(0))
        log_lik += np.log(max(prob, 1e-10))
    
    return log_lik

# ============================================================================
# Estimation
# ============================================================================

def create_state_space(
    size_min: float = 3.0,
    size_max: float = 8.0,
    n_size: int = 10,
    comp_min: float = 0.0,
    comp_max: float = 1.0,
    n_comp: int = 5
) -> StateSpace:
    """Create discretized state space"""
    size_grid = jnp.linspace(size_min, size_max, n_size)
    comp_grid = jnp.linspace(comp_min, comp_max, n_comp)
    tech_dummy = jnp.array([0.0, 1.0])
    adopted = jnp.array([0.0, 1.0])
    
    return StateSpace(
        size_grid=size_grid,
        comp_grid=comp_grid,
        tech_dummy=tech_dummy,
        adopted=adopted,
        n_size=n_size,
        n_comp=n_comp,
        n_tech=2,
        n_adopt=2
    )

def compute_flow_payoffs(
    state_space: StateSpace,
    params: ModelParams
) -> jnp.ndarray:
    """
    Precompute flow payoffs for all (state, action) pairs
    
    Returns:
        Array of shape [n_states, n_actions]
    """
    n_states = (
        state_space.n_size * 
        state_space.n_comp * 
        state_space.n_tech * 
        state_space.n_adopt
    )
    n_actions = 2
    
    payoffs = np.zeros((n_states, n_actions))
    
    for s_idx in range(n_states):
        # Decode state
        size_idx = s_idx % state_space.n_size
        comp_idx = (s_idx // state_space.n_size) % state_space.n_comp
        tech_idx = (s_idx // (state_space.n_size * state_space.n_comp)) % state_space.n_tech
        adopt_idx = (s_idx // (state_space.n_size * state_space.n_comp * state_space.n_tech))
        
        size = float(state_space.size_grid[size_idx])
        comp = float(state_space.comp_grid[comp_idx])
        tech = float(state_space.tech_dummy[tech_idx])
        already_adopted = float(state_space.adopted[adopt_idx])
        
        for action in range(n_actions):
            payoffs[s_idx, action] = float(flow_payoff(
                size, comp, tech, float(action), float(action), already_adopted, params
            ))
    
    return jnp.array(payoffs)

def prepare_estimation_data(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for estimation
    
    Required columns in panel:
    - gvkey, year, emp, industry, treated, digital_year
    """
    logger.info("Preparing estimation data...")
    
    df = panel.copy()
    
    # Construct variables
    df['log_size'] = np.log(df['emp'].fillna(df['emp'].median()))
    
    # Competition index (inverse of mean size in industry-year)
    comp = df.groupby(['industry', 'year'])['emp'].transform('mean')
    df['competition'] = 1.0 / (1.0 + comp)  # Normalize to [0,1]
    
    # Tech industry dummy
    df['tech_industry'] = (df['industry'] == 'Technology').astype(int)
    
    # Ever adopted before this year
    df['ever_adopted_before'] = (
        (df['digital_year'].notna()) & 
        (df['year'] > df['digital_year'])
    ).astype(int)
    
    # Adopt this period
    df['adopt_this_period'] = (
        (df['digital_year'].notna()) &
        (df['year'] == df['digital_year'])
    ).astype(int)
    
    # Keep only firms that can make a decision
    # (already adopted don't choose again, never adopters can still choose)
    df = df[
        (df['ever_adopted_before'] == 0) |  # Haven't adopted yet
        (df['adopt_this_period'] == 1)      # Adopting this period
    ].copy()
    
    logger.info(f"  Estimation sample: {len(df):,} observations")
    logger.info(f"  Unique firms: {df['gvkey'].nunique():,}")
    logger.info(f"  Adoption events: {df['adopt_this_period'].sum():,}")
    
    return df

def estimate_structural_model(
    data: pd.DataFrame,
    initial_params: ModelParams,
    state_space: StateSpace
) -> Dict:
    """
    Estimate structural parameters via MLE
    
    Args:
        data: Prepared panel data
        initial_params: Starting values
        state_space: Discretized state space
    
    Returns:
        Dictionary with results
    """
    logger.info("="*70)
    logger.info("STRUCTURAL ESTIMATION (IO MODEL)")
    logger.info("="*70)
    
    # Build transition matrix (fixed during estimation)
    logger.info("\nBuilding transition matrix...")
    P = build_transition_matrix(state_space, initial_params)
    logger.info(f"  Transition matrix shape: {P.shape}")
    
    # Parameter bounds
    bounds = [
        (0.0, 2.0),     # alpha_size
        (0.0, 2.0),     # alpha_comp
        (0.0, 2.0),     # alpha_tech
        (0.0, 10.0),    # fc_adopt
        (0.0, 2.0),     # fc_maintain
        (0.90, 0.99),   # beta
        (0.0, 0.5),     # delta
        (0.5, 0.99),    # gamma_size
        (0.01, 1.0),    # sigma_size
        (0.1, 1.0),     # scale
    ]
    
    # Objective function
    def neg_log_likelihood(theta):
        """Negative log-likelihood for minimization"""
        params = ModelParams(*theta)
        
        # Compute flow payoffs for current parameters
        flow_payoffs = compute_flow_payoffs(state_space, params)
        
        # Compute likelihood
        log_lik = compute_likelihood(data, state_space, params, P, flow_payoffs)
        
        return -log_lik
    
    # Starting values
    x0 = [
        initial_params.alpha_size,
        initial_params.alpha_comp,
        initial_params.alpha_tech,
        initial_params.fc_adopt,
        initial_params.fc_maintain,
        initial_params.beta,
        initial_params.delta,
        initial_params.gamma_size,
        initial_params.sigma_size,
        initial_params.scale,
    ]
    
    logger.info("\nStarting optimization...")
    logger.info(f"  Initial parameters: {x0}")
    logger.info(f"  Initial log-lik: {-neg_log_likelihood(x0):.2f}")
    
    # Optimize
    start_time = time.time()
    result = minimize(
        neg_log_likelihood,
        x0=x0,
        method='L-BFGS-B',
        bounds=bounds,
        options={
            'maxiter': 100,
            'ftol': 1e-6,
            'gtol': 1e-5,
            'disp': True
        }
    )
    elapsed = time.time() - start_time
    
    # Extract results
    theta_hat = result.x
    params_hat = ModelParams(*theta_hat)
    log_lik_hat = -result.fun
    
    logger.info("\n" + "="*70)
    logger.info("ESTIMATION RESULTS")
    logger.info("="*70)
    logger.info(f"\nLog-likelihood: {log_lik_hat:.2f}")
    logger.info(f"Optimization time: {elapsed:.1f}s")
    logger.info(f"Converged: {result.success}")
    logger.info(f"Message: {result.message}")
    
    logger.info("\nParameter Estimates:")
    logger.info("-"*70)
    param_names = [
        'alpha_size (benefit of size)',
        'alpha_comp (benefit of low competition)',
        'alpha_tech (tech industry premium)',
        'fc_adopt (fixed cost adoption)',
        'fc_maintain (maintenance cost)',
        'beta (discount factor)',
        'delta (depreciation)',
        'gamma_size (size persistence)',
        'sigma_size (size shock std)',
        'scale (logit scale)'
    ]
    
    for name, val_init, val_hat in zip(param_names, x0, theta_hat):
        logger.info(f"  {name:40s}: {val_hat:8.4f}  (initial: {val_init:.4f})")
    
    # Compute standard errors (numerical Hessian)
    logger.info("\nComputing standard errors...")
    try:
        from scipy.optimize import approx_fprime
        
        eps = 1e-5
        gradient = approx_fprime(theta_hat, neg_log_likelihood, eps)
        
        # Finite difference Hessian
        hessian = np.zeros((len(theta_hat), len(theta_hat)))
        for i in range(len(theta_hat)):
            theta_plus = theta_hat.copy()
            theta_plus[i] += eps
            grad_plus = approx_fprime(theta_plus, neg_log_likelihood, eps)
            
            theta_minus = theta_hat.copy()
            theta_minus[i] -= eps
            grad_minus = approx_fprime(theta_minus, neg_log_likelihood, eps)
            
            hessian[i, :] = (grad_plus - grad_minus) / (2 * eps)
        
        # Standard errors
        cov_matrix = np.linalg.inv(hessian)
        se = np.sqrt(np.diag(cov_matrix))
        
        logger.info("\nStandard Errors:")
        logger.info("-"*70)
        for name, est, stderr in zip(param_names, theta_hat, se):
            t_stat = est / stderr if stderr > 0 else np.nan
            logger.info(f"  {name:40s}: {stderr:8.4f}  (t = {t_stat:6.2f})")
    
    except Exception as e:
        logger.warning(f"Could not compute standard errors: {e}")
        se = np.full_like(theta_hat, np.nan)
        cov_matrix = np.full((len(theta_hat), len(theta_hat)), np.nan)
    
    # Save results
    results = {
        'params_hat': params_hat,
        'theta_hat': theta_hat,
        'se': se,
        'cov_matrix': cov_matrix,
        'log_likelihood': log_lik_hat,
        'optimization_result': result,
        'elapsed_time': elapsed,
        'param_names': param_names
    }
    
    return results

# ============================================================================
# Counterfactual Analysis
# ============================================================================

def counterfactual_subsidy(
    state_space: StateSpace,
    params: ModelParams,
    P: jnp.ndarray,
    subsidy_amount: float
) -> Dict:
    """
    Compute counterfactual adoption rates under subsidy policy
    
    Args:
        state_space: State space
        params: Estimated parameters
        P: Transition matrix
        subsidy_amount: Reduction in fixed cost
    
    Returns:
        Dictionary with counterfactual results
    """
    logger.info(f"\nComputing counterfactual: subsidy = {subsidy_amount:.2f}")
    
    # Baseline
    flow_payoffs_base = compute_flow_payoffs(state_space, params)
    _, policy_base = solve_dynamic_program(state_space, params, P, flow_payoffs_base)
    
    # With subsidy
    params_subsidy = params._replace(fc_adopt=params.fc_adopt - subsidy_amount)
    flow_payoffs_sub = compute_flow_payoffs(state_space, params_subsidy)
    _, policy_sub = solve_dynamic_program(state_space, params_subsidy, P, flow_payoffs_sub)
    
    # Average adoption probability across states
    # (weight by steady-state distribution, or just simple average)
    avg_adopt_base = float(policy_base[:, 1].mean())
    avg_adopt_sub = float(policy_sub[:, 1].mean())
    
    increase = avg_adopt_sub - avg_adopt_base
    pct_increase = increase / avg_adopt_base if avg_adopt_base > 0 else np.nan
    
    logger.info(f"  Baseline adoption rate: {avg_adopt_base:.3f}")
    logger.info(f"  With subsidy: {avg_adopt_sub:.3f}")
    logger.info(f"  Increase: {increase:.3f} ({pct_increase:.1%})")
    
    return {
        'subsidy': subsidy_amount,
        'baseline_adoption': avg_adopt_base,
        'subsidy_adoption': avg_adopt_sub,
        'increase': increase,
        'pct_increase': pct_increase
    }

# ============================================================================
# Main Execution
# ============================================================================

def run_io_estimation():
    """Main function to run structural estimation"""
    logger.info("="*70)
    logger.info("IO STRUCTURAL ESTIMATION - DIGITAL ADOPTION")
    logger.info("="*70)
    
    set_random_seed(CONFIG['analysis']['seed'])
    
    # Load data
    panel_path = PATHS['data_processed'] / 'firm_panel.csv'
    if not panel_path.exists():
        logger.error(f"Data file not found: {panel_path}")
        logger.error("Run 02_synthetic_data/generate_panel.py first")
        return
    
    panel = pd.read_csv(panel_path)
    logger.info(f"Loaded panel: {len(panel):,} observations")
    
    # Prepare estimation data
    data = prepare_estimation_data(panel)
    
    # Create state space
    state_space = create_state_space(
        size_min=data['log_size'].min(),
        size_max=data['log_size'].max(),
        n_size=8,  # Keep small for speed
        n_comp=5
    )
    logger.info(f"\nState space size: {state_space.n_size * state_space.n_comp * state_space.n_tech * state_space.n_adopt}")
    
    # Initial parameter guess
    initial_params = ModelParams(
        alpha_size=0.5,
        alpha_comp=0.3,
        alpha_tech=0.4,
        fc_adopt=2.0,
        fc_maintain=0.2,
        beta=0.95,
        delta=0.15,
        gamma_size=0.9,
        sigma_size=0.2,
        scale=0.5
    )
    
    # Estimate
    results = estimate_structural_model(data, initial_params, state_space)
    
    # Save estimates
    estimates_df = pd.DataFrame({
        'Parameter': results['param_names'],
        'Estimate': results['theta_hat'],
        'Std_Error': results['se'],
        'T_Stat': results['theta_hat'] / results['se']
    })
    
    out_path = PATHS['tables'] / 'io_structural_estimates.csv'
    estimates_df.to_csv(out_path, index=False, float_format='%.4f')
    logger.info(f"\n✓ Saved estimates to: {out_path}")
    
    # Counterfactual: What if adoption subsidy = 1.0?
    P = build_transition_matrix(state_space, results['params_hat'])
    cf_results = counterfactual_subsidy(
        state_space,
        results['params_hat'],
        P,
        subsidy_amount=1.0
    )
    
    # Save counterfactual
    cf_df = pd.DataFrame([cf_results])
    cf_path = PATHS['tables'] / 'io_counterfactual_subsidy.csv'
    cf_df.to_csv(cf_path, index=False, float_format='%.4f')
    logger.info(f"✓ Saved counterfactual to: {cf_path}")
    
    logger.info("\n" + "="*70)
    logger.info("IO ESTIMATION COMPLETE")
    logger.info("="*70)
    
    return results

if __name__ == "__main__":
    results = run_io_estimation()
    
    print("\n" + "="*70)
    print("IO STRUCTURAL ESTIMATION COMPLETE")
    print("="*70)
    print("\nKey outputs:")
    print("  - Parameter estimates: output/tables/io_structural_estimates.csv")
    print("  - Counterfactual results: output/tables/io_counterfactual_subsidy.csv")
