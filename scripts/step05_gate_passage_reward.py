# scripts/step05_gate_passage_reward.py
#
# Robust gate crossing counting using segment-plane intersection in GateEntrance frame.
#
# Per robot:
#   - score[p] (net: +1 correct, -1 wrong)
# Global:
#   - global_net (sum of per-robot net)
#   - global_pos_count (number of correct-direction crossings)
#   - global_neg_count (number of wrong-direction crossings)
# Per step:
#   - step_reward: instantaneous shared reward at each frame
#
# Crossing detection:
#   If segment from (x_prev,y_prev) to (x_cur,y_cur) intersects x=0,
#   we compute y_cross via linear interpolation and check corridor bounds.
#
# Anti-jitter latch:
#   After a counted crossing, robot must move away from plane by REARM_DIST
#   before it can count another crossing.
#
# Run (spawn + monitor):
#   .\run.bat scripts\step05_gate_passage_reward.py --spawn --n 10 --seed 1
#
# Run (monitor existing /World/Robots/*):
#   .\run.bat scripts\step05_gate_passage_reward.py

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import argparse
import math
import random
import numpy as np

import omni.usd
from pxr import Usd, UsdGeom, Gf

from omni.isaac.core.utils.stage import add_reference_to_stage

# ---- Scene contract ----
USD_PATH = "usd/dirgate_arena_base.usd"
GATE_ENTRANCE = "/World/GateEntrance"
ROBOT_SPAWN = "/World/RobotSpawn"

ARENA_WALL_PREFIX = "/World/arena/wall"   # wall1..wall12
GATE_WALL_PREFIX = "/World/gate/wall"     # wall1..wall2

# ---- Robot asset ----
NOVA_CARTER_USD = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
    "Assets/Isaac/5.1/Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd"
)

CLEARANCE_SCALE = 1.10

# Robustness knobs:
ARM_EPS = 1e-6        # treat x close to 0 as 0 for numeric stability
REARM_DIST = 0.20     # meters in gate frame: must be this far from plane to re-arm
BINARY_STEP_REWARD = False  # if True: step_reward in {-1,0,+1}, else sum events in step


# ---------------- helpers ----------------
def get_world_xform(stage, prim_path: str) -> Gf.Matrix4d:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Prim not found: {prim_path}")
    cache = UsdGeom.XformCache()
    return cache.GetLocalToWorldTransform(prim)


def compute_aabb(stage, prim_path: str):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Prim not found: {prim_path}")
    bbox_cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        includedPurposes=[UsdGeom.Tokens.default_],
    )
    bbox = bbox_cache.ComputeWorldBound(prim)
    aabb = bbox.ComputeAlignedBox()
    mn = aabb.GetMin()
    mx = aabb.GetMax()
    return (
        np.array([float(mn[0]), float(mn[1]), float(mn[2])]),
        np.array([float(mx[0]), float(mx[1]), float(mx[2])]),
    )


def aabb_xy_to_circle(min_xyz, max_xyz):
    min_xy = min_xyz[:2]
    max_xy = max_xyz[:2]
    c = (min_xy + max_xy) * 0.5
    half = (max_xy - min_xy) * 0.5
    r = float(np.linalg.norm(half))
    return c, r


def list_wall_paths(stage, prefix: str, max_idx: int = 64):
    out = []
    for i in range(1, max_idx + 1):
        p = f"{prefix}{i}"
        if stage.GetPrimAtPath(p).IsValid():
            out.append(p)
    return out


def estimate_arena_center_radius_from_walls(stage):
    wall_paths = list_wall_paths(stage, ARENA_WALL_PREFIX, 64)
    if len(wall_paths) < 6:
        raise RuntimeError(f"Not enough arena walls found under {ARENA_WALL_PREFIX}. Found={len(wall_paths)}")

    centers = []
    for p in wall_paths:
        xf = get_world_xform(stage, p)
        t = xf.ExtractTranslation()
        centers.append(np.array([float(t[0]), float(t[1])], dtype=float))

    centers = np.stack(centers, axis=0)
    center = centers.mean(axis=0)
    d = np.linalg.norm(centers - center[None, :], axis=1)
    radius = float(d.min())  # conservative
    return center, radius, wall_paths


def gather_wall_obstacles(stage, arena_walls, gate_walls):
    circles = []
    for p in arena_walls + gate_walls:
        mn, mx = compute_aabb(stage, p)
        c, r = aabb_xy_to_circle(mn, mx)
        circles.append((c, r))
    return circles


def set_prim_pose_xyz_yaw(stage, prim_path: str, xyz: np.ndarray, yaw: float):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Prim not found: {prim_path}")
    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(float(xyz[0]), float(xyz[1]), float(xyz[2])))
    xform.AddRotateZOp().Set(float(math.degrees(yaw)))


def sample_pose(center_xy, radius, rng: random.Random):
    r = radius * math.sqrt(rng.random())
    theta = 2.0 * math.pi * rng.random()
    x = center_xy[0] + r * math.cos(theta)
    y = center_xy[1] + r * math.sin(theta)
    yaw = 2.0 * math.pi * rng.random()
    return np.array([x, y], dtype=float), yaw


def circle_collides(c_xy, c_r, obstacles, other_robots):
    for oc_xy, oc_r in obstacles:
        if np.linalg.norm(c_xy - oc_xy) <= (c_r + oc_r):
            return True
    for rc_xy, rc_r in other_robots:
        if np.linalg.norm(c_xy - rc_xy) <= (c_r + rc_r):
            return True
    return False


def world_to_gate_xy(stage, gate_xf_inv: Gf.Matrix4d, robot_path: str):
    xf = get_world_xform(stage, robot_path)
    t = xf.ExtractTranslation()
    p_w = Gf.Vec3d(float(t[0]), float(t[1]), float(t[2]))
    p_g = gate_xf_inv.Transform(p_w)  # Vec3d supported in your build
    return float(p_g[0]), float(p_g[1])


def corridor_half_width_from_gate_walls(stage, gate_xf_inv: Gf.Matrix4d):
    gate_walls = list_wall_paths(stage, GATE_WALL_PREFIX, 16)
    if len(gate_walls) < 2:
        return 1e9

    ys = []
    for p in gate_walls[:2]:
        mn, mx = compute_aabb(stage, p)
        c = 0.5 * (mn + mx)  # world AABB center
        p_w = Gf.Vec3d(float(c[0]), float(c[1]), float(c[2]))
        p_g = gate_xf_inv.Transform(p_w)
        ys.append(float(p_g[1]))

    half = 0.5 * abs(ys[0] - ys[1])

    if half < 1e-4:
        print("[WARN] corridor_half_width collapsed (~0). Disabling corridor filter.")
        return 1e9

    return 0.90 * half  # margin


def segment_crosses_plane_x0(x0, y0, x1, y1):
    """
    Returns (did_cross, alpha, y_cross).
    alpha in [0,1] where point is P = (1-alpha)*P0 + alpha*P1, and x(alpha)=0.
    """
    # If both endpoints are (almost) on the same side, no crossing.
    # We allow exact 0 at an endpoint, but avoid counting "stuck on plane" as crossing.
    if abs(x0) < ARM_EPS and abs(x1) < ARM_EPS:
        return False, None, None

    # If x0 and x1 have same sign and neither is ~0 -> no crossing
    if (x0 > ARM_EPS and x1 > ARM_EPS) or (x0 < -ARM_EPS and x1 < -ARM_EPS):
        return False, None, None

    denom = (x1 - x0)
    if abs(denom) < 1e-12:
        return False, None, None

    # Solve x0 + alpha*(x1-x0) = 0 -> alpha = -x0/(x1-x0)
    alpha = -x0 / denom
    if alpha < 0.0 or alpha > 1.0:
        return False, None, None

    y_cross = y0 + alpha * (y1 - y0)
    return True, alpha, y_cross


# ---------------- main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--spawn", action="store_true")
    ap.add_argument("--max_tries", type=int, default=20000)
    args, _ = ap.parse_known_args()

    ctx = omni.usd.get_context()
    print(f"[STEP05] Opening stage: {USD_PATH}")
    ctx.open_stage(USD_PATH)
    for _ in range(8):
        simulation_app.update()
    stage = ctx.get_stage()
    if stage is None:
        raise RuntimeError("Stage failed to load.")

    gate_xf_inv = get_world_xform(stage, GATE_ENTRANCE).GetInverse()
    corridor_half_w = corridor_half_width_from_gate_walls(stage, gate_xf_inv)

    print(f"[STEP05] corridor_half_width ~ {corridor_half_w:.3f} m")
    print(f"[STEP05] REARM_DIST={REARM_DIST:.3f}  BINARY_STEP_REWARD={BINARY_STEP_REWARD}")

    # --- robots ---
    robot_paths = []
    if args.spawn:
        arena_center, arena_radius, arena_walls = estimate_arena_center_radius_from_walls(stage)
        gate_walls = list_wall_paths(stage, GATE_WALL_PREFIX, 16)
        obstacles = gather_wall_obstacles(stage, arena_walls, gate_walls)

        # spawn center
        spawn_xy = np.array([0.0, 0.0], dtype=float)
        if stage.GetPrimAtPath(ROBOT_SPAWN).IsValid():
            t = get_world_xform(stage, ROBOT_SPAWN).ExtractTranslation()
            spawn_xy = np.array([float(t[0]), float(t[1])], dtype=float)

        # measure robot footprint and spawn_z
        tmp_path = "/World/__tmp_robot_measure"
        add_reference_to_stage(NOVA_CARTER_USD, tmp_path)
        for _ in range(6):
            simulation_app.update()

        mn, mx = compute_aabb(stage, tmp_path)
        _, robot_r = aabb_xy_to_circle(mn, mx)
        robot_r *= CLEARANCE_SCALE
        spawn_z = -float(mn[2])

        stage.RemovePrim(tmp_path)
        for _ in range(2):
            simulation_app.update()

        # create robots
        for i in range(args.n):
            p = f"/World/Robots/NovaCarter_{i:03d}"
            add_reference_to_stage(NOVA_CARTER_USD, p)
            robot_paths.append(p)

        for _ in range(10):
            simulation_app.update()

        rng = random.Random(args.seed)
        placed = []
        tries = 0
        sample_radius = max(0.0, arena_radius - robot_r)

        for i, p in enumerate(robot_paths):
            ok = False
            while tries < args.max_tries:
                tries += 1
                xy, yaw = sample_pose(spawn_xy, sample_radius, rng)

                if np.linalg.norm(xy - arena_center) > (arena_radius - robot_r):
                    continue
                if circle_collides(xy, robot_r, obstacles, placed):
                    continue

                set_prim_pose_xyz_yaw(stage, p, np.array([xy[0], xy[1], spawn_z], dtype=float), yaw)
                placed.append((xy, robot_r))
                ok = True
                break
            if not ok:
                raise RuntimeError(f"Failed to place robot {i} after {tries} samples.")

        print(f"[STEP05] Spawned {len(robot_paths)} robots.")
    else:
        robots_root = stage.GetPrimAtPath("/World/Robots")
        if not robots_root.IsValid():
            raise RuntimeError("No /World/Robots found. Run with --spawn or create robots first.")
        for prim in Usd.PrimRange(robots_root):
            p = prim.GetPath().pathString
            if p.startswith("/World/Robots/NovaCarter_"):
                robot_paths.append(p)
        robot_paths = sorted(set(robot_paths))
        print(f"[STEP05] Tracking {len(robot_paths)} existing robots.")

    # --- reward state ---
    score = {p: 0 for p in robot_paths}
    global_net = 0
    global_pos_count = 0
    global_neg_count = 0

    # Previous gate-frame positions
    prev_xy = {}
    # Per-robot latch: True means this robot is allowed to register a crossing
    armed = {}

    # Initialize state (so first true crossing counts immediately)
    for p in robot_paths:
        xg, yg = world_to_gate_xy(stage, gate_xf_inv, p)
        prev_xy[p] = (xg, yg)
        armed[p] = True  # start armed

    print("[STEP05] Robust crossing: segment intersection with x=0 + re-arm latch.")
    print("  +1 for -x -> +x, -1 for +x -> -x (within corridor).")

    while simulation_app.is_running():
        simulation_app.update()

        step_reward = 0

        for p in robot_paths:
            x1, y1 = world_to_gate_xy(stage, gate_xf_inv, p)
            x0, y0 = prev_xy[p]

            # re-arm if far from plane
            if not armed[p] and abs(x1) > REARM_DIST:
                armed[p] = True

            # Only attempt to count if armed
            if armed[p]:
                did_cross, alpha, y_cross = segment_crosses_plane_x0(x0, y0, x1, y1)
                if did_cross:
                    # corridor filter using y at the actual crossing location
                    if abs(y_cross) <= corridor_half_w:
                        # direction
                        if x0 < 0.0 and x1 > 0.0:
                            score[p] += 1
                            global_net += 1
                            global_pos_count += 1
                            if BINARY_STEP_REWARD:
                                step_reward = 1
                            else:
                                step_reward += 1
                            print(
                                f"[PASS +1] {p} score={score[p]}  "
                                f"GLOBAL_NET={global_net} POS={global_pos_count} NEG={global_neg_count}  "
                                f"x:{x0:.3f}->{x1:.3f} y_cross={y_cross:.3f}"
                            )

                        elif x0 > 0.0 and x1 < 0.0:
                            score[p] -= 1
                            global_net -= 1
                            global_neg_count += 1
                            if BINARY_STEP_REWARD:
                                step_reward = -1
                            else:
                                step_reward -= 1
                            print(
                                f"[PASS -1] {p} score={score[p]}  "
                                f"GLOBAL_NET={global_net} POS={global_pos_count} NEG={global_neg_count}  "
                                f"x:{x0:.3f}->{x1:.3f} y_cross={y_cross:.3f}"
                            )

                        # disarm after a counted crossing to avoid jitter double counts
                        armed[p] = False

            # update previous position
            prev_xy[p] = (x1, y1)

        if step_reward != 0:
            print(
                f"[STEP_REWARD] r_t={step_reward}  "
                f"GLOBAL_POS={global_pos_count} GLOBAL_NEG={global_neg_count} GLOBAL_NET={global_net}"
            )

    simulation_app.close()


if __name__ == "__main__":
    main()
