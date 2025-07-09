import jax
from jax import vmap
import jax.numpy as np
import equinox as eqx
from jax.lax import stop_gradient
from jax.scipy.special import logsumexp
from src.id import PID
from src.trainers.util import loss_step
from typing import Tuple
from src.base import (Target,
                      PIDCarry,
                      PIDOpt,
                      PIDParameters)
from jaxtyping import PyTree
from jax.lax import map


def get_weights(log_weights):
    """Convert log weights to normalized weights via softmax"""
    return jax.nn.softmax(log_weights)


def de_particle_grad(key: jax.random.PRNGKey,
                     pid : PID,
                     target: Target,
                     particles: jax.Array,
                     y: jax.Array,
                     mc_n_samples: int):
    '''
    Compute the gradient of the first variation
    using pathwise monte carlo gradient estimation
    with number of samples set to `mc_n_samples`
    '''
    def ediff_score(particle, eps):
        '''
        Compute the expectation of the difference
        of scores using the reparameterization trick.
        '''
        vf = vmap(pid.conditional.f, (None, None, 0))
        samples = vf(particle, y, eps)
        assert samples.shape == (mc_n_samples, target.dim)
        logq = vmap(pid.log_prob, (0, None))(samples, y)
        logp = vmap(target.log_prob, (0, None))(samples, y)
        assert logp.shape == (mc_n_samples,)
        assert logq.shape == (mc_n_samples,)
        logp = np.mean(logp, 0)
        logq = np.mean(logq, 0)
        return logq - logp
    eps = pid.conditional.base_sample(key, mc_n_samples)
    grad = vmap(jax.grad(lambda p: ediff_score(p, eps)))(particles)
    return grad


def de_weight_grad(key: jax.random.PRNGKey,
                   pid: PID,
                   target: Target,
                   log_weights: jax.Array,
                   y: jax.Array,
                   mc_n_samples: int):
    '''
    Compute gradient w.r.t. log_weights for weighted mixture
    '''
    def weight_loss(lw):
        # Create temporary PID with updated log_weights
        temp_pid = eqx.tree_at(lambda p: p.log_weights, pid, lw)
        # Get the proper static part from the temp_pid
        params, static = eqx.partition(temp_pid, temp_pid.get_filter_spec())
        return de_loss(key, params, static, target, y, 
                      PIDParameters(mc_n_samples=mc_n_samples))
    
    return jax.grad(weight_loss)(log_weights)


def de_loss(key: jax.random.PRNGKey,
            params: PyTree,
            static: PyTree,
            target: Target,
            y: jax.Array,
            hyperparams: PIDParameters):
    '''
    Density Estimation Loss with path-wise gradients and weighted mixture
    '''
    pid = eqx.combine(params, static)
    _samples = pid.sample(key, hyperparams.mc_n_samples, None)
    
    # Compute weighted log probability
    weights = get_weights(pid.log_weights)
    
    # Compute log probabilities for each particle
    logq_per_particle = vmap(lambda particle: 
        vmap(eqx.combine(stop_gradient(params), static).log_prob, (0, None))(_samples, None)
    )(pid.particles)
    
    # Weighted mixture log probability
    logq_weighted = logsumexp(
        vmap(lambda w, logq: np.log(w) + logq)(weights, logq_per_particle),
        axis=0
    )
    logq = np.mean(logq_weighted, axis=0)
    
    logp = vmap(target.log_prob, (0, None))(_samples, y)
    logp = np.mean(logp, axis=0)
    
    # Add entropy regularization to prevent weight collapse
    entropy = -np.sum(weights * np.log(weights + 1e-8))
    lambda_entropy = getattr(hyperparams, 'lambda_entropy', 1e-4) #0.01)
    
    return logq - logp - lambda_entropy * entropy


def de_particle_step(key: jax.random.PRNGKey,
                     pid: PID,
                     target: Target,
                     y: jax.Array,
                     optim: PIDOpt,
                     carry: PIDCarry,
                     hyperparams: PIDParameters):
    '''
    Particle Step for Density Estimation with learnable weights.
    '''
    r_key, w_key = jax.random.split(key, 2)
    
    # Update particle positions
    grad_fn = lambda particles: de_particle_grad(
        r_key,
        pid,
        target,
        particles,
        y,
        hyperparams.mc_n_samples)
    g_grad, r_precon_state = optim.r_precon.update(
        pid.particles,
        grad_fn,
        carry.r_precon_state,)
    update, r_opt_state = optim.r_optim.update(
        g_grad,
        carry.r_opt_state,
        params=pid.particles,
        index=y)
    pid = eqx.tree_at(lambda tree : tree.particles,
                      pid,
                      pid.particles + update)
    
    # Update log_weights if optimizer exists
    if hasattr(optim, 'w_optim') and hasattr(carry, 'w_opt_state'):
        weight_grad = de_weight_grad(
            w_key,
            pid,
            target,
            pid.log_weights,
            y,
            hyperparams.mc_n_samples)
        
        weight_update, w_opt_state = optim.w_optim.update(
            weight_grad,
            carry.w_opt_state,
            params=pid.log_weights)
        
        pid = eqx.tree_at(lambda tree: tree.log_weights,
                          pid,
                          pid.log_weights + weight_update)
        
        carry = PIDCarry(
            id=pid,
            theta_opt_state=carry.theta_opt_state,
            r_opt_state=r_opt_state,
            r_precon_state=r_precon_state,
            w_opt_state=w_opt_state)
    else:
        carry = PIDCarry(
            id=pid,
            theta_opt_state=carry.theta_opt_state,
            r_opt_state=r_opt_state,
            r_precon_state=r_precon_state)
    
    return pid, carry


def de_step(key: jax.random.PRNGKey,
            carry: PIDCarry,
            target: Target,
            y: jax.Array,
            optim: PIDOpt,
            hyperparams: PIDParameters) -> Tuple[float, PIDCarry]:
    '''
    Density Estimation Step.
    '''
    theta_key, r_key = jax.random.split(key, 2)
    def loss(key, params, static):
        return de_loss(key,
                       params,
                       static,
                       target,
                       y,
                       hyperparams)
    lval, pid, theta_opt_state = loss_step(
        theta_key,
        loss,
        carry.id,
        optim.theta_optim,
        carry.theta_opt_state,
    )

    pid, carry = de_particle_step(
        r_key,
        pid,
        target,
        y,
        optim,
        carry,
        hyperparams)

    carry = PIDCarry(
        id=pid,
        theta_opt_state=theta_opt_state,
        r_opt_state=carry.r_opt_state,
        r_precon_state=carry.r_precon_state,
        w_opt_state=getattr(carry, 'w_opt_state', None))
    
    # Log effective sample size for monitoring
    if hasattr(pid, 'log_weights'):
        weights = get_weights(pid.log_weights)
        ess = 1.0 / np.sum(weights ** 2)
        # Could add logging here: print(f"ESS: {ess:.2f} / {len(weights)}")
    
    return lval, carry