# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the surrogate hook-attachment model (M3 down-payment).

Runs on plain PyTorch, no Isaac Sim needed -- execute the file directly (it prints a summary and asserts):
    python exts/MARL_mav_carry_ext/MARL_mav_carry_ext/tasks/directMARL/hook_grasp/tests/test_attachment.py
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from attachment import (  # noqa: E402
    HookAttachmentCfg,
    compute_catch_probability,
    update_attachment_state,
)

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def _bar_frame(n):
    """A single +x wall: normal=+x, up=+z, bar axis=+y. Bar at x=0.3."""
    bar_pos = torch.tensor([[0.30, 0.0, 1.0]], device=DEV).repeat(n, 1)
    bar_axis = torch.tensor([[0.0, 1.0, 0.0]], device=DEV).repeat(n, 1)
    wall_normal = torch.tensor([[1.0, 0.0, 0.0]], device=DEV).repeat(n, 1)
    wall_up = torch.tensor([[0.0, 0.0, 1.0]], device=DEV).repeat(n, 1)
    return bar_pos, bar_axis, wall_normal, wall_up


def test_direction_ordering_up_gt_side_gt_down():
    """Legacy SYMMETRIC model (require_upsweep=False): parallel-up > sideways > parallel-down."""
    cfg = HookAttachmentCfg(require_upsweep=False)
    bar_pos, bar_axis, wall_normal, wall_up = _bar_frame(1)
    hook_pos = bar_pos.clone()  # in contact

    # All velocities include a small into-wall (-x) component so `require_approach` passes.
    approach = torch.tensor([[-0.1, 0.0, 0.0]], device=DEV)
    v_up = approach + torch.tensor([[0.0, 0.0, 1.0]], device=DEV)     # sweep up
    v_down = approach + torch.tensor([[0.0, 0.0, -1.0]], device=DEV)  # sweep down
    v_side = approach + torch.tensor([[0.0, 1.0, 0.0]], device=DEV)   # along the bar

    p_up = compute_catch_probability(hook_pos, v_up, bar_pos, bar_axis, wall_normal, wall_up, cfg)[0]
    p_down = compute_catch_probability(hook_pos, v_down, bar_pos, bar_axis, wall_normal, wall_up, cfg)[0]
    p_side = compute_catch_probability(hook_pos, v_side, bar_pos, bar_axis, wall_normal, wall_up, cfg)[0]

    assert p_up.item() > p_side.item() > p_down.item(), (p_up, p_side, p_down)
    assert p_up.item() > 0.6 and p_down.item() < 0.1
    print(f"[ordering-legacy] p_up={p_up.item():.3f} > p_side={p_side.item():.3f} > p_down={p_down.item():.3f}  OK")


def test_upsweep_required_kills_creep():
    """PHYSICAL J-hook model (require_upsweep=True, default): only a genuine UP-sweep into the wall seats the
    hook. A horizontal 'creep' (v_up==0) and a down-sweep MUST NOT catch, even in perfect contact -- this is
    what removes the unphysical creep-into-the-zone optimum."""
    cfg = HookAttachmentCfg()  # require_upsweep True by default
    bar_pos, bar_axis, wall_normal, wall_up = _bar_frame(1)
    hook_pos = bar_pos.clone()
    approach = torch.tensor([[-0.3, 0.0, 0.0]], device=DEV)  # gentle into-wall so require_approach passes
    p_creep, c_creep = compute_catch_probability(hook_pos, approach + torch.tensor([[0.0, 1.0, 0.0]], device=DEV),
                                                 bar_pos, bar_axis, wall_normal, wall_up, cfg)   # v_up == 0
    p_down, _ = compute_catch_probability(hook_pos, approach + torch.tensor([[0.0, 0.0, -0.5]], device=DEV),
                                          bar_pos, bar_axis, wall_normal, wall_up, cfg)
    p_up, c_up = compute_catch_probability(hook_pos, approach + torch.tensor([[0.0, 0.0, 0.6]], device=DEV),
                                           bar_pos, bar_axis, wall_normal, wall_up, cfg)
    assert p_creep.item() == 0.0 and not c_creep.item(), p_creep      # horizontal creep cannot latch
    assert p_down.item() == 0.0, p_down                                # down-sweep cannot latch
    assert p_up.item() > 0.4 and c_up.item(), p_up                     # a real up-sweep does
    print(f"[upsweep] creep={p_creep.item():.3f} down={p_down.item():.3f} up={p_up.item():.3f}  OK")


def test_proximity_gate():
    """Requirement: no catch when the hook is outside the capture radius."""
    cfg = HookAttachmentCfg()
    bar_pos, bar_axis, wall_normal, wall_up = _bar_frame(1)
    far_hook = bar_pos + torch.tensor([[0.5, 0.0, 0.0]], device=DEV)  # 0.5 m away
    v_up = torch.tensor([[-0.1, 0.0, 1.0]], device=DEV)
    p, contact = compute_catch_probability(far_hook, v_up, bar_pos, bar_axis, wall_normal, wall_up, cfg)
    assert p.item() == 0.0 and not contact.item()
    print("[proximity] far hook -> p=0  OK")


def test_require_approach_gate():
    """A hook grazing the bar without any into-wall motion should not catch."""
    cfg = HookAttachmentCfg()
    bar_pos, bar_axis, wall_normal, wall_up = _bar_frame(1)
    hook_pos = bar_pos.clone()
    v_up_no_approach = torch.tensor([[0.0, 0.0, 1.0]], device=DEV)  # pure up, no into-wall component
    p, contact = compute_catch_probability(hook_pos, v_up_no_approach, bar_pos, bar_axis, wall_normal, wall_up, cfg)
    assert p.item() == 0.0 and not contact.item()
    print("[approach-gate] up but no into-wall motion -> p=0  OK")


def test_monte_carlo_catch_rate():
    """Empirical catch rate over many envs matches the analytic probability ordering."""
    cfg = HookAttachmentCfg()
    n = 20000
    bar_pos, bar_axis, wall_normal, wall_up = _bar_frame(n)
    hook_pos = bar_pos.clone()
    anchor = hook_pos + torch.tensor([[0.0, 0.0, cfg.wire_length]], device=DEV)  # taut, above hook

    def rate(vel_vec):
        vel = torch.tensor([vel_vec], device=DEV).repeat(n, 1)
        attached = torch.zeros(n, dtype=torch.bool, device=DEV)
        slack_timer = torch.zeros(n, dtype=torch.long, device=DEV)
        attach_point = torch.zeros(n, 3, device=DEV)
        attached, *_ = update_attachment_state(
            attached, slack_timer, attach_point,
            hook_pos=hook_pos, hook_vel=vel, anchor_pos=anchor,
            bar_pos=bar_pos, bar_axis=bar_axis, wall_normal=wall_normal, wall_up=wall_up, cfg=cfg,
        )
        return attached.float().mean().item()

    r_up = rate([-0.1, 0.0, 1.0])
    r_down = rate([-0.1, 0.0, -1.0])
    assert r_up > 0.6 and r_down < 0.1 and r_up > r_down   # single-step up-catch rate ~= p_max (0.7)
    print(f"[monte-carlo] catch rate up={r_up:.3f}  down={r_down:.3f}  OK")


def test_slack_detach_dwell_then_angle():
    """Requirement 4 (dwell + angle model): brief slack is TOLERATED for slack_dwell_steps, then, once slack
    outlasts the dwell, the hook may slip off with an angle-dependent probability rolled ONCE at the crossing:
    vertical pull ~ detach_p_vertical, horizontal ~ detach_p_horizontal (a J-hook slides off vertically, but
    its curve can rest on the bar at an angle). During the grace, no detach occurs."""
    cfg = HookAttachmentCfg()
    n = 20000
    bar_pos, bar_axis, wall_normal, wall_up = _bar_frame(n)
    hook_pos = bar_pos.clone()
    zero_vel = torch.zeros(n, 3, device=DEV)

    def run_slack(offset, steps):
        """Hold the wire slack for `steps` control steps; return (detached-by-now fraction, per-step detach)."""
        attached = torch.ones(n, dtype=torch.bool, device=DEV)
        slack_timer = torch.zeros(n, dtype=torch.long, device=DEV)
        attach_point = bar_pos.clone()
        anchor = attach_point + torch.tensor([offset], device=DEV)   # close -> slack; direction sets the angle
        detached_any = torch.zeros(n, dtype=torch.bool, device=DEV)
        per_step = []
        for _ in range(steps):
            attached, slack_timer, attach_point, _, just_detached, _ = update_attachment_state(
                attached, slack_timer, attach_point,
                hook_pos=hook_pos, hook_vel=zero_vel, anchor_pos=anchor,
                bar_pos=bar_pos, bar_axis=bar_axis, wall_normal=wall_normal, wall_up=wall_up, cfg=cfg,
            )
            detached_any = detached_any | just_detached
            per_step.append(just_detached.float().mean().item())
        return detached_any.float().mean().item(), per_step

    # During the dwell grace (first slack_dwell_steps slack steps) NOTHING detaches, even vertical.
    _, per_step_vert = run_slack([0.0, 0.0, 0.1], cfg.slack_dwell_steps + 1)
    grace_detach = sum(per_step_vert[: cfg.slack_dwell_steps])
    assert grace_detach == 0.0, per_step_vert
    # The roll fires exactly on the step slack outlasts the dwell (index slack_dwell_steps, i.e. timer==dwell+1).
    assert per_step_vert[cfg.slack_dwell_steps] > 0.95, per_step_vert

    r_vert, _ = run_slack([0.0, 0.0, 0.1], cfg.slack_dwell_steps + 1)   # wire vertical -> ~detach_p_vertical
    r_horiz, _ = run_slack([0.1, 0.0, 0.0], cfg.slack_dwell_steps + 1)  # wire horizontal -> ~detach_p_horizontal
    assert r_vert > 0.95, r_vert
    assert abs(r_horiz - cfg.detach_p_horizontal) < 0.03, r_horiz
    print(f"[slack-detach] grace={cfg.slack_dwell_steps} steps no detach; post-dwell rate: "
          f"vertical={r_vert:.3f} (~{cfg.detach_p_vertical}) horizontal={r_horiz:.3f} "
          f"(~{cfg.detach_p_horizontal})  OK")


def test_taut_stays_attached():
    """A taut wire keeps the hook attached indefinitely."""
    cfg = HookAttachmentCfg()
    bar_pos, bar_axis, wall_normal, wall_up = _bar_frame(1)
    hook_pos = bar_pos.clone()
    attached = torch.ones(1, dtype=torch.bool, device=DEV)
    slack_timer = torch.zeros(1, dtype=torch.long, device=DEV)
    attach_point = bar_pos.clone()
    taut_anchor = attach_point + torch.tensor([[0.0, 0.0, cfg.wire_length + 0.05]], device=DEV)  # beyond length
    zero_vel = torch.zeros(1, 3, device=DEV)
    for _ in range(20):
        attached, slack_timer, attach_point, _, just_detached, _ = update_attachment_state(
            attached, slack_timer, attach_point,
            hook_pos=hook_pos, hook_vel=zero_vel, anchor_pos=taut_anchor,
            bar_pos=bar_pos, bar_axis=bar_axis, wall_normal=wall_normal, wall_up=wall_up, cfg=cfg,
        )
        assert not just_detached.item()
    assert attached.item()
    print("[taut] taut wire stays attached for 20 steps  OK")


if __name__ == "__main__":
    tests = [
        test_direction_ordering_up_gt_side_gt_down,
        test_proximity_gate,
        test_require_approach_gate,
        test_monte_carlo_catch_rate,
        test_slack_detach_dwell_then_angle,
        test_taut_stays_attached,
    ]
    print(f"device={DEV}")
    for t in tests:
        t()
    print(f"\nAll {len(tests)} attachment tests passed.")
