# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Surrogate hook-attachment model for the hook-grasping task.

This module is the *novel core* of the task and is intentionally free of any Isaac Sim / pxr
dependency so it can be unit-tested on plain PyTorch (see ``tests/test_attachment.py``).

Physical story it encodes (user requirements 2-4):

  2. The hook is **not** guaranteed to attach on contact -> a valid contact triggers a Bernoulli
     draw with success probability ``p_catch``.
  3. The **relative velocity of the hook w.r.t. the bar** governs ``p_catch``: motion *parallel-up*
     along the (vertical) wall gives the highest probability, *parallel-down* the lowest. Some
     into-wall approach is required (you have to actually hit the wall), and the hook must be within
     a capture radius of a horizontal bar.
  4. Once attached, if the wire goes **slack** (the drone drifts closer to the anchor than the wire's
     natural length, so tension vanishes) for longer than a short dwell, the hook **falls off**.

All functions are batched over ``(num_envs,)`` and, where relevant, over candidate bars.

Frame conventions
-----------------
The basket walls are vertical, the catchable bars are horizontal. For a bar we therefore define, in
world frame:
  * ``wall_up``     - unit tangent pointing up the wall. For vertical walls this is world +z.
  * ``wall_normal`` - unit outward normal of the wall (points away from the basket interior).
  * ``bar_axis``    - unit horizontal direction along the bar.
These three are orthonormal. ``compute_catch_probability`` decomposes the hook's relative velocity
onto them: the ``wall_up`` component drives the up>down probability, the ``wall_normal`` component
gates the "did it actually approach the wall" condition.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class HookAttachmentCfg:
    """Tunable parameters of the surrogate catch/slack model."""

    # --- capture geometry ---------------------------------------------------------------
    capture_radius: float = 0.03
    """Max hook-to-bar distance (m) for a contact to be considered. ~hook radius (papers: 0.02-0.023 m),
    widened a little for the surrogate so the learning signal is not razor-thin."""

    # --- probability shaping ------------------------------------------------------------
    p_max: float = 0.9
    """Catch probability for an ideal fast parallel-UP sweep into the bar (per contact step). Raised back to
    0.9 (from 0.7) so a good sweep almost always latches -- reward the correct approach instead of the
    catch (retry-after-failure). NOTE: this is a PER-STEP probability, so it still compounds over the steps
    the hook dwells in the capture zone (k steps -> 1-(1-0.9)^k); a lingering sweep catches almost certainly."""
    p_min: float = 0.02
    """Catch probability for a parallel-DOWN sweep (near-zero, but not exactly 0)."""
    v_ref: float = 1.0
    """Reference speed (m/s) that saturates the up/down direction factor."""
    catch_speed_max: float = 2.5
    """Total relative hook speed (m/s) above which the catch fails: a fast FLING-THROUGH bounces/skips off the
    bar rather than hooking. p_catch is multiplied by a factor that is 1 below v_ref, 0 at/above catch_speed_max,
    linear between -- so only a CONTROLLED engagement latches, not a violent swing-through (also safer for the
    multi-drone extension: no benefit to slamming near the other drones)."""
    require_approach: bool = True
    """If True, a contact only counts when the hook has a non-trivial INTO-wall velocity component
    (``v_normal < -approach_eps``); models that the hook must be swung against the wall to engage."""
    approach_eps: float = 0.02
    """Into-wall speed (m/s) below which we do not treat the touch as an engagement attempt."""
    require_upsweep: bool = True
    """PHYSICAL J-HOOK MODEL: a hook only seats on a bar when its throat is RISING onto the bar, i.e. the
    hook must be sweeping UP at contact. When True, (a) ``direction_factor`` gives ~``p_min`` for any
    ``v_up <= 0`` and ramps to ``p_max`` at ``v_up >= v_ref`` (no free midpoint for a horizontal drift), and
    (b) a contact additionally requires ``v_up > upsweep_v_min``. Together these REMOVE the unphysical
    'creep the dangling hook into the zone at ~0 velocity' optimum -> the drone must PUMP the pendulum so
    the hook up-sweeps onto the bar (real cable-suspended aerial-manipulation maneuver). Set False for the
    legacy symmetric model."""
    upsweep_v_min: float = 0.12
    """Minimum upward hook speed (m/s, along wall_up) for a contact to count as an engagement when
    ``require_upsweep`` -- a genuine up-stroke, not incidental drift."""
    min_chord_angle_deg: float = 0.0
    """PHYSICAL SWING GATE: a catch is only allowed when the wire (chord = drone-anchor -> hook) makes at least
    this angle FROM VERTICAL. 0 = off (any angle, incl. a hook hanging straight below the drone = dead-hang creep).
    Set > 0 (e.g. 10 deg) to FORBID catching a near-vertical dangling hook -> the hook must be SWUNG OUT to latch,
    which makes the perpendicular pendulum up-swing physically necessary and removes the creep/hang catch optimum.
    The env can anneal this from 0 -> target over training (chord-angle curriculum)."""
    chord_angle_perp_only: bool = False
    """When True, the ``min_chord_angle_deg`` swing-out is measured as the SIGNED, POSITIVE (OUTWARD) component in
    the WALL-NORMAL direction: sin(angle) = (chord . wall_normal)/|chord|, requiring chord.n > sin(angle). Since
    wall_normal points OUTWARD, this demands the drone be OUTSIDE the wall with the hook swung UP onto the bar from
    outside (the correct J-hook catch). It forbids BOTH the sideways along-bar flick AND the drone-on-top /
    into-wall accidental catch (drone hovering over the basket bobbing the hook down)."""

    # --- slack / detach -----------------------------------------------------------------
    wire_length: float = 0.55
    """Natural (taut) length of the wire from the drone anchor to the hook tip (m)."""
    slack_margin: float = 0.03
    """The wire is considered slack when drone-anchor-to-anchor distance < wire_length - slack_margin."""
    slack_dwell_steps: int = 5
    """Grace period: consecutive slack steps TOLERATED before a detach can occur (brief tension dips ride out
    on momentum / the hook's curve). Once slack persists past this, the angle-dependent detach roll fires."""
    detach_p_vertical: float = 1.0
    """Detach probability once slack outlasts the dwell and the pull is VERTICAL: a J-hook pulled straight along
    gravity slides off the bar (guaranteed)."""
    detach_p_horizontal: float = 0.2
    """Detach probability once slack outlasts the dwell and the pull is HORIZONTAL: the curved throat of the
    J-hook can still rest on the bar, so it usually stays. Interpolated by |cos(wire angle from vertical)|."""


def _project(v: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    """Scalar projection of ``v`` (..., 3) onto unit ``axis`` (..., 3) -> (...,)."""
    return (v * axis).sum(dim=-1)


def direction_factor(v_up: torch.Tensor, cfg: HookAttachmentCfg) -> torch.Tensor:
    """Map the up-component of relative velocity to a probability in ``[p_min, p_max]``.

    ``require_upsweep`` (physical J-hook): only an UP-stroke seats the throat on the bar, so ``v_up <= 0``
    -> ``p_min`` and it ramps LINEARLY to ``p_max`` at ``v_up >= v_ref``. There is NO free midpoint at
    ``v_up == 0`` -> a hook drifting horizontally into the zone cannot latch, which removes the creep optimum.

    Legacy symmetric model (``require_upsweep=False``): ``v_up > 0`` -> toward ``p_max``; ``v_up < 0`` -> toward
    ``p_min``; ``v_up == 0`` -> midpoint. Saturates at ``+/- v_ref``.
    """
    if cfg.require_upsweep:
        u = torch.clamp(v_up / cfg.v_ref, 0.0, 1.0)  # in [0, 1]; v_up<=0 -> 0 -> p_min
        return cfg.p_min + (cfg.p_max - cfg.p_min) * u
    u = torch.clamp(v_up / cfg.v_ref, -1.0, 1.0)  # in [-1, 1]
    return cfg.p_min + (cfg.p_max - cfg.p_min) * (u + 1.0) * 0.5


def compute_catch_probability(
    hook_pos: torch.Tensor,      # (N, 3) world
    hook_vel: torch.Tensor,      # (N, 3) world
    bar_pos: torch.Tensor,       # (N, 3) closest point on the target bar, world
    bar_axis: torch.Tensor,      # (N, 3) unit, horizontal along the bar
    wall_normal: torch.Tensor,   # (N, 3) unit, outward wall normal
    wall_up: torch.Tensor,       # (N, 3) unit, up the wall (world +z for vertical walls)
    cfg: HookAttachmentCfg,
    bar_vel: torch.Tensor | None = None,   # (N, 3) world, 0 if the basket is static this episode
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(p_catch, contact_mask)`` for a single candidate bar per env.

    ``p_catch`` is the per-step Bernoulli success probability; it is 0 where there is no valid contact.
    ``contact_mask`` marks envs whose hook is within the capture radius (and, if required, approaching).
    """
    if bar_vel is None:
        bar_vel = torch.zeros_like(hook_vel)

    dist = torch.norm(hook_pos - bar_pos, dim=-1)
    within = dist < cfg.capture_radius

    v_rel = hook_vel - bar_vel
    v_up = _project(v_rel, wall_up)
    v_normal = _project(v_rel, wall_normal)  # >0 away from wall, <0 into wall

    contact = within
    if cfg.require_approach:
        contact = contact & (v_normal < -cfg.approach_eps)
    if cfg.require_upsweep:
        # a genuine up-stroke is required to seat the J-hook throat on the bar (not incidental drift)
        contact = contact & (v_up > cfg.upsweep_v_min)

    p_dir = direction_factor(v_up, cfg)
    # controlled-engagement gate: scale the catch prob down as the TOTAL relative speed grows past v_ref, to 0 at
    # catch_speed_max -> a fast fling-through can't latch (only a settled, moderate-speed hook-over engages).
    speed = torch.norm(v_rel, dim=-1)
    speed_factor = ((cfg.catch_speed_max - speed) / (cfg.catch_speed_max - cfg.v_ref)).clamp(0.0, 1.0)
    p_catch = torch.where(contact, p_dir * speed_factor, torch.zeros_like(p_dir))
    return p_catch, contact


def sample_catches(p_catch: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    """Bernoulli draw per env -> bool tensor of newly-caught envs."""
    r = torch.rand(p_catch.shape, device=p_catch.device, generator=generator)
    return r < p_catch


def is_slack(anchor_pos: torch.Tensor, attach_point: torch.Tensor, cfg: HookAttachmentCfg) -> torch.Tensor:
    """True where the wire is slack: drone-anchor-to-attach-point distance < wire_length - margin."""
    d = torch.norm(anchor_pos - attach_point, dim=-1)
    return d < (cfg.wire_length - cfg.slack_margin)


def update_attachment_state(
    attached: torch.Tensor,        # (N,) bool, current state (modified functionally -> returned)
    slack_timer: torch.Tensor,     # (N,) long, consecutive slack-step counter
    attach_point: torch.Tensor,    # (N, 3) world, valid where attached
    *,
    hook_pos: torch.Tensor,        # (N, 3)
    hook_vel: torch.Tensor,        # (N, 3)
    anchor_pos: torch.Tensor,      # (N, 3) drone-side wire anchor (world)
    bar_pos: torch.Tensor,         # (N, 3) closest point on target bar
    bar_axis: torch.Tensor,        # (N, 3)
    wall_normal: torch.Tensor,     # (N, 3)
    wall_up: torch.Tensor,         # (N, 3)
    cfg: HookAttachmentCfg,
    generator: torch.Generator | None = None,
    bar_vel: torch.Tensor | None = None,
    catch_latch: torch.Tensor | None = None,   # (N,) bool, "already drew for the current dwell in the zone"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Advance the per-env attachment state by one control step.

    Returns ``(attached, slack_timer, attach_point, newly_caught, just_detached, catch_latch)``.

    - For **unattached** envs: draw the catch **once per approach-attempt**, not every step. The
      ``catch_latch`` marks envs that have already had their Bernoulli draw while continuously inside the
      capture radius; further steps in the same dwell do NOT re-draw. The latch resets when the hook leaves
      the capture radius, so a MISS forces the controller to back out and RE-APPROACH for a fresh attempt
      (retry-after-failure). Without this, the per-step draw would compound over the dwell and ``p_max`` would
      be nearly a guaranteed catch regardless of its value.
    - For **attached** envs: brief slack is TOLERATED for ``slack_dwell_steps`` (grace: quick tension dips ride
      out on momentum / the hook's curve). Only when slack PERSISTS past the dwell do we roll an angle-dependent
      detach probability (vertical pull -> ~``detach_p_vertical``, horizontal -> ~``detach_p_horizontal``),
      rolled ONCE as the timer crosses the dwell. This combines the dwell grace WITH the angle model.
    """
    attached = attached.clone()
    slack_timer = slack_timer.clone()
    attach_point = attach_point.clone()
    if catch_latch is None:
        catch_latch = torch.zeros_like(attached)
    else:
        catch_latch = catch_latch.clone()

    # --- try to catch (only envs not yet attached, one draw per dwell) -----------------
    within = torch.norm(hook_pos - bar_pos, dim=-1) < cfg.capture_radius
    p_catch, contact = compute_catch_probability(
        hook_pos, hook_vel, bar_pos, bar_axis, wall_normal, wall_up, cfg, bar_vel=bar_vel
    )
    # eligible = a valid contact this step AND we have not already used this dwell's single attempt
    eligible = contact & (~catch_latch) & (~attached)
    # CHORD-ANGLE gate: forbid latching a near-vertical dangling hook (dead-hang creep). The wire chord
    # (anchor -> hook) must make at least ``min_chord_angle_deg`` from vertical, i.e. the hook is swung OUT.
    if cfg.min_chord_angle_deg > 0.0:
        chord = anchor_pos - hook_pos
        clen = torch.norm(chord, dim=-1).clamp_min(1e-6)
        if getattr(cfg, "chord_angle_perp_only", False):
            # require swing-out at a POSITIVE (OUTWARD) wall angle: sin(perp angle) = (chord . wall_normal)/|chord|,
            # SIGNED. wall_normal points OUTWARD, so chord.n > 0 means the drone/anchor is OUTSIDE the wall relative
            # to the hook -> the hook is swung UP onto the bar FROM OUTSIDE (correct J-hook catch). This forbids BOTH
            # the sideways along-bar flick (small normal component) AND the drone-on-top / into-wall catch
            # (chord.n <= 0, drone hovering over the basket bobbing the hook down = accidental latch).
            perp_signed = (chord * wall_normal).sum(-1) / clen
            swung_out = perp_signed > math.sin(math.radians(cfg.min_chord_angle_deg))
        else:
            vert = chord[..., 2].abs() / clen                                    # |cos(angle from vertical)|
            swung_out = vert < math.cos(math.radians(cfg.min_chord_angle_deg))
        eligible = eligible & swung_out
    p_eff = torch.where(eligible, p_catch, torch.zeros_like(p_catch))
    newly_caught = sample_catches(p_eff, generator=generator) & (~attached)
    attached = attached | newly_caught
    attach_point = torch.where(newly_caught.unsqueeze(-1), bar_pos, attach_point)
    slack_timer = torch.where(newly_caught, torch.zeros_like(slack_timer), slack_timer)
    # consume the attempt where we drew; hold the latch while still inside the zone; reset on leaving it
    catch_latch = torch.where(within, catch_latch | eligible, torch.zeros_like(catch_latch))

    # --- slack / detach: dwell grace, THEN an angle-dependent roll -----------------------
    # When the wire goes slack (drone too close -> attachment force ~0), the hook may slip off the bar. Brief
    # slack is TOLERATED for slack_dwell_steps (momentum / the hook's curve rides out a quick dip). Only when
    # slack persists PAST the dwell do we decide: a J-hook pulled VERTICALLY slides straight off
    # (p ~ detach_p_vertical); at an ANGLE the curved throat can still rest on the bar (p ~ detach_p_horizontal).
    # p is interpolated by |cos(angle of the drone->attach wire from vertical)|. The roll fires ONCE, on the step
    # the slack timer first exceeds the dwell; slack_timer resets to 0 when the wire goes taut again (grace resets).
    slack_now = is_slack(anchor_pos, attach_point, cfg) & attached
    slack_timer = torch.where(slack_now, slack_timer + 1, torch.zeros_like(slack_timer))
    crossed = slack_now & (slack_timer == cfg.slack_dwell_steps + 1)   # first step slack outlasts the grace
    wire = anchor_pos - attach_point
    verticalness = wire[..., 2].abs() / torch.norm(wire, dim=-1).clamp_min(1e-6)   # |cos(angle from vertical)|
    p_detach = cfg.detach_p_horizontal + (cfg.detach_p_vertical - cfg.detach_p_horizontal) * verticalness
    r = torch.rand(p_detach.shape, device=p_detach.device, generator=generator)
    just_detached = crossed & (r < p_detach)
    attached = attached & (~just_detached)
    slack_timer = torch.where(just_detached, torch.zeros_like(slack_timer), slack_timer)

    return attached, slack_timer, attach_point, newly_caught, just_detached, catch_latch
