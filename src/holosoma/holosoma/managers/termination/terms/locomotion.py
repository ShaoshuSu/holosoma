"""Locomotion-specific termination terms."""

from __future__ import annotations

from holosoma.managers.observation.terms.locomotion import get_projected_gravity
from holosoma.utils.safe_torch_import import torch


def _apply_probability(mask: torch.Tensor, probability: float, device: torch.device) -> torch.Tensor:
    """Optionally apply probabilistic gating to a mask."""
    if probability >= 1.0:
        return mask
    if probability <= 0.0:
        return torch.zeros_like(mask, dtype=torch.bool)
    sample = torch.rand(1, device=device)
    return mask & (sample < probability)


def contact_forces_exceeded(
    env, force_threshold: float = 1.0, contact_indices_attr: str = "termination_contact_indices"
) -> torch.Tensor:
    """Terminate if contact forces exceed threshold.

    Note: If you want to disable contact termination, simply don't add this term to your
    termination config instead of using a flag.
    """
    indices = getattr(env, contact_indices_attr)
    contact_forces = env.simulator.contact_forces[:, indices, :]
    return torch.any(torch.norm(contact_forces, dim=-1) > force_threshold, dim=1)


def gravity_tilt_exceeded(
    env, threshold_x: float, threshold_y: float
) -> torch.Tensor:
    """Terminate if projected gravity exceeds roll/pitch thresholds."""
    grav = get_projected_gravity(env)
    tilt_x = torch.abs(grav[:, 0]) > threshold_x
    tilt_y = torch.abs(grav[:, 1]) > threshold_y
    return tilt_x | tilt_y


def base_height_below_threshold(env, min_height: float) -> torch.Tensor:
    """Terminate if the pelvis sits less than ``min_height`` m above the terrain.

    Uses the terrain-aware pelvis-above-ground height from the locomotion terrain
    term when available (``TerrainLocomotion.base_heights`` — computed by a
    downward ray cast at the pelvis), matching FAR-FALCON's
    ``_get_base_heights`` semantics. On terrains where no terrain state exists
    (``plane`` / ``none``) or the state doesn't expose ``base_heights``, falls
    back to comparing the pelvis world-Z directly against ``min_height``.

    Without this fix, spawning onto a lower stair of a high-difficulty
    ``stairs_up`` tile (world-Z of the local step goes negative) would
    immediately trip a world-Z-based threshold even though the robot is
    standing normally on the stair.
    """
    terrain_state = None
    manager = getattr(env, "terrain_manager", None)
    if manager is not None:
        try:
            terrain_state = manager.get_state("locomotion_terrain")
        except Exception:
            terrain_state = None
    base_heights = getattr(terrain_state, "base_heights", None)
    if base_heights is None or not hasattr(base_heights, "numel") or base_heights.numel() == 0:
        base_heights = env.simulator.robot_root_states[:, 2]
    return base_heights < min_height


def dof_position_limit_exceeded(env, probability: float = 1.0) -> torch.Tensor:
    """Terminate when DOF position limits are exceeded."""
    lower_violation = -(env.simulator.dof_pos - env.simulator.dof_pos_limits_termination[:, 0]).clip(max=0.0)
    upper_violation = (env.simulator.dof_pos - env.simulator.dof_pos_limits_termination[:, 1]).clip(min=0.0)
    violation = torch.sum(lower_violation + upper_violation, dim=1) > 0.0
    return _apply_probability(violation, probability, env.device)


def dof_velocity_limit_exceeded(env, probability: float = 1.0) -> torch.Tensor:
    """Terminate when DOF velocity limits are exceeded."""
    delta = (
        torch.abs(env.simulator.dof_vel)
        - env.dof_vel_limits * env.config.termination_scales.termination_close_to_dof_vel_limit
    ).clip(min=0.0, max=1.0)
    violation = torch.sum(delta, dim=1) > 0.0
    return _apply_probability(violation, probability, env.device)


def torque_limit_exceeded(env, probability: float = 1.0) -> torch.Tensor:
    """Terminate when actuator torques exceed limits."""
    torques = env.action_manager.get_term("joint_control").torques
    delta = (
        torch.abs(torques) - env.torque_limits * env.config.termination_scales.termination_close_to_torque_limit
    ).clip(min=0.0, max=1.0)
    violation = torch.sum(delta, dim=1) > 0.0
    return _apply_probability(violation, probability, env.device)
