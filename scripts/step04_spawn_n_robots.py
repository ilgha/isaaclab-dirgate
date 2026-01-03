from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import argparse
import math
import random
import numpy as np

import omni.usd
from pxr import Usd, UsdGeom, Gf

from omni.isaac.core.utils.stage import add_reference_to_stage

USD_PATH = "usd/dirgate_arena_base.usd"

# Scene contract (your prims)
ROBOT_SPAWN = "/World/RobotSpawn"
ARENA_WALL_PREFIX = "/World/arena/wall"   # wall1..wall12
GATE_WALL_PREFIX = "/World/gate/wall"     # wall1..wall2

# Robot asset
NOVA_CARTER_USD = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/"
    "Assets/Isaac/5.1/Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd"
)

CLEARANCE_SCALE = 1.10  # inflate robot radius slightly


def get_world_transform(stage, prim_path: str) -> Gf.Matrix4d:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Prim not found: {prim_path}")
    cache = UsdGeom.XformCache()
    return cache.GetLocalToWorldTransform(prim)


def get_world_xy_yaw(stage, prim_path: str):
    xf = get_world_transform(stage, prim_path)
    t = xf.ExtractTranslation()
    r = xf.ExtractRotationMatrix()
    yaw = float(math.atan2(r[1][0], r[0][0]))
    return np.array([float(t[0]), float(t[1])], dtype=float), yaw


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
    return np.array([float(mn[0]), float(mn[1]), float(mn[2])]), np.array([float(mx[0]), float(mx[1]), float(mx[2])])


def aabb_xy_to_circle(min_xyz, max_xyz):
    min_xy = min_xyz[:2]
    max_xy = max_xyz[:2]
    c = (min_xy + max_xy) * 0.5
    half = (max_xy - min_xy) * 0.5
    r = float(np.linalg.norm(half))
    return c, r


def list_wall_paths(stage, prefix: str, max_idx: int):
    paths = []
    for i in range(1, max_idx + 1):
        p = f"{prefix}{i}"
        if stage.GetPrimAtPath(p).IsValid():
            paths.append(p)
    return paths


def estimate_arena_center_radius_from_walls(stage):
    # You have 12 walls under /World/arena
    wall_paths = list_wall_paths(stage, ARENA_WALL_PREFIX, 32)
    if len(wall_paths) < 6:
        raise RuntimeError(f"Not enough arena walls found under {ARENA_WALL_PREFIX} (found {len(wall_paths)})")

    centers = []
    for p in wall_paths:
        xf = get_world_transform(stage, p)
        t = xf.ExtractTranslation()
        centers.append(np.array([float(t[0]), float(t[1])], dtype=float))

    centers = np.stack(centers, axis=0)
    center = centers.mean(axis=0)

    # radius ≈ min distance from center to any wall center (conservative)
    d = np.linalg.norm(centers - center[None, :], axis=1)
    radius = float(d.min())
    return center, radius, wall_paths


def gather_obstacle_circles(stage, arena_wall_paths, gate_wall_paths):
    circles = []
    for p in arena_wall_paths + gate_wall_paths:
        mn, mx = compute_aabb(stage, p)
        c, r = aabb_xy_to_circle(mn, mx)
        circles.append((c, r))
    return circles


def sample_pose(center_xy, radius, rng: random.Random):
    # uniform in disk
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


def set_prim_pose_xyz_yaw(stage, prim_path: str, xyz: np.ndarray, yaw: float):
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        raise RuntimeError(f"Prim not found: {prim_path}")

    xform = UsdGeom.Xformable(prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(float(xyz[0]), float(xyz[1]), float(xyz[2])))
    xform.AddRotateZOp().Set(float(math.degrees(yaw)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max_tries", type=int, default=20000)
    args, _ = ap.parse_known_args()

    ctx = omni.usd.get_context()
    print(f"[STEP04] Opening stage: {USD_PATH}")
    ctx.open_stage(USD_PATH)
    for _ in range(8):
        simulation_app.update()
    stage = ctx.get_stage()
    if stage is None:
        raise RuntimeError("Stage failed to load.")

    spawn_xy, _ = get_world_xy_yaw(stage, ROBOT_SPAWN)

    arena_center, arena_radius, arena_walls = estimate_arena_center_radius_from_walls(stage)
    gate_walls = list_wall_paths(stage, GATE_WALL_PREFIX, 32)

    print(f"[STEP04] spawn center: {spawn_xy.tolist()}")
    print(f"[STEP04] arena center: {arena_center.tolist()}  radius(center->wallCenters)~{arena_radius:.3f}")
    print(f"[STEP04] arena walls: {len(arena_walls)}, gate walls: {len(gate_walls)}")

    # --- Measure robot footprint (radius) and required spawn Z so it sits on the floor ---
    tmp_path = "/World/__tmp_robot_measure"
    add_reference_to_stage(NOVA_CARTER_USD, tmp_path)
    for _ in range(6):
        simulation_app.update()

    mn, mx = compute_aabb(stage, tmp_path)
    _, robot_r = aabb_xy_to_circle(mn, mx)
    robot_r *= CLEARANCE_SCALE
    spawn_z = -float(mn[2])  # lift so bottom touches z=0

    # remove temp
    stage.RemovePrim(tmp_path)
    for _ in range(2):
        simulation_app.update()

    print(f"[STEP04] robot radius (inflated): {robot_r:.3f} m")
    print(f"[STEP04] robot spawn_z to sit on floor: {spawn_z:.3f} m")

    # Obstacles: ONLY walls (no floor/patch)
    obstacles = gather_obstacle_circles(stage, arena_walls, gate_walls)
    print(f"[STEP04] obstacle circles (walls only): {len(obstacles)}")

    # sampling disk: centered at RobotSpawn, radius = (min distance to wall centers) - margin
    sample_center = spawn_xy
    sample_radius = arena_radius - robot_r
    if sample_radius <= 0.0:
        raise RuntimeError("Arena too small vs robot footprint after margin.")

    rng = random.Random(args.seed)

    # Create robot prims
    for i in range(args.n):
        prim_path = f"/World/Robots/NovaCarter_{i:03d}"
        add_reference_to_stage(NOVA_CARTER_USD, prim_path)

    for _ in range(10):
        simulation_app.update()

    placed = []  # list of (xy, r)
    tries = 0

    for i in range(args.n):
        prim_path = f"/World/Robots/NovaCarter_{i:03d}"
        ok = False

        while tries < args.max_tries:
            tries += 1
            xy, yaw = sample_pose(sample_center, sample_radius, rng)

            # Keep inside arena based on arena_center estimate (conservative)
            if np.linalg.norm(xy - arena_center) > (arena_radius - robot_r):
                continue

            if circle_collides(xy, robot_r, obstacles, placed):
                continue

            set_prim_pose_xyz_yaw(stage, prim_path, np.array([xy[0], xy[1], spawn_z], dtype=float), yaw)
            placed.append((xy, robot_r))
            print(f"[STEP04] placed {prim_path} at xy={xy.tolist()} yaw={yaw:.2f}")
            ok = True
            break

        if not ok:
            raise RuntimeError(
                f"Failed to place robot {i} after {tries} samples. "
                f"Try smaller --n, increase --max_tries, or reduce CLEARANCE_SCALE."
            )

    print(f"[STEP04] Done. Placed {len(placed)}/{args.n} robots in {tries} samples.")

    while simulation_app.is_running():
        simulation_app.update()

    simulation_app.close()


if __name__ == "__main__":
    main()
