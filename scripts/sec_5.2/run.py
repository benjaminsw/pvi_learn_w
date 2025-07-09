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
    Compute gradient for log_weights using soft mixture (fully differentiable).
    '''
    def weight_loss_fn(logw):
        # Use soft mixture throughout - NO categorical sampling
        weights = jax.nn.softmax(logw)  # Convert to normalized weights
        
        # Sample from base distribution (not from mixture)
        base_samples = jax.random.normal(key, (mc_n_samples, pid.conditional.d_z))
        
        # Transform base samples through each particle's conditional
        samples_per_particle = vmap(lambda particle: 
            vmap(pid.conditional.f, (None, None, 0))(particle, y, 
                jax.random.normal(key, (mc_n_samples, pid.conditional.d_x)))
        )(pid.particles)  # shape: (N_particles, mc_n_samples, d_x)
        
        # Compute log probabilities for each particle-sample combination
        logq_per_particle = vmap(lambda particle_samples:
            vmap(pid.conditional.log_prob, (0, None, None))(particle_samples, 
                                                           np.broadcast_to(pid.particles[0], particle_samples.shape[:-1] + (pid.particles.shape[-1],)), 
                                                           y)
        )(samples_per_particle)  # shape: (N_particles, mc_n_samples)
        
        # Create weighted mixture using log-sum-exp (soft, differentiable)
        weighted_logq = logsumexp(
            logq_per_particle + np.log(weights)[:, None],
            axis=0
        )  # shape: (mc_n_samples,)
        
        logq = np.mean(weighted_logq)
        
        # Compute target log probability on mixture samples
        # Use weighted sampling for target evaluation
        mixture_samples = np.sum(
            samples_per_particle * weights[:, None, None],
            axis=0
        )  # shape: (mc_n_samples, d_x)
        
        logp = np.mean(vmap(target.log_prob, (0, None))(mixture_samples, y))
        
        # Add entropy regularization to prevent weight collapse
        entropy = -np.sum(weights * np.log(weights + 1e-8))
        lambda_entropy = 0.01
        
        return logq - logp - lambda_entropy * entropy
    
    # Compute gradient and log its norm for debugging
    grad = jax.grad(weight_loss_fn)(log_weights)
    grad_norm = np.linalg.norm(grad)
    print(f"Weight gradient norm: {grad_norm:.6f}")
    
    return grad


def de_loss(key: jax.random.PRNGKey,
            params: PyTree,
            static: PyTree,
            target: Target,
            y: jax.Array,
            hyperparams: PIDParameters):
    '''
    Density Estimation Loss using soft mixture (fully differentiable)
    '''
    pid = eqx.combine(params, static)
    
    # Use soft mixture sampling (no categorical sampling)
    weights = jax.nn.softmax(pid.log_weights)  # Remove stop_gradient
    
    # Sample from base distribution
    base_key, sample_key = jax.random.split(key)
    base_samples = jax.random.normal(base_key, (hyperparams.mc_n_samples, pid.conditional.d_z))
    
    # Transform through each particle
    samples_per_particle = vmap(lambda particle:
        vmap(pid.conditional.f, (None, None, 0))(particle, y, 
            jax.random.normal(sample_key, (hyperparams.mc_n_samples, pid.conditional.d_x)))
    )(pid.particles)
    
    # Compute weighted mixture samples
    mixture_samples = np.sum(
        samples_per_particle * weights[:, None, None],
        axis=0
    )
    
    # Compute log probabilities using soft mixture
    logq_per_particle = vmap(lambda particle_samples:
        vmap(pid.conditional.log_prob, (0, None, None))(particle_samples, 
                                                       np.broadcast_to(pid.particles[0], particle_samples.shape[:-1] + (pid.particles.shape[-1],)), 
                                                       y)
    )(samples_per_particle)
    
    # Weighted log-sum-exp for mixture log probability
    logq = np.mean(logsumexp(
        logq_per_particle + np.log(weights)[:, None],
        axis=0
    ))
    
    # Target log probability on mixture samples
    logp = np.mean(vmap(target.log_prob, (0, None))(mixture_samples, y))
    
    # Add entropy regularization
    entropy = -np.sum(weights * np.log(weights + 1e-8))
    lambda_entropy = 0.01
    
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
    if hasattr(optim, 'w_optim') and optim.w_optim is not None and hasattr(carry, 'w_opt_state'):
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
            r_precon_state=r_precon_state,
            w_opt_state=getattr(carry, 'w_opt_state', None))
    
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
        weights = jax.nn.softmax(pid.log_weights)  # Remove stop_gradient
        ess = 1.0 / np.sum(weights ** 2)
        print(f"ESS: {ess:.1f}/{len(weights)}, Max weight: {np.max(weights):.3f}")
    
    return lval, carry