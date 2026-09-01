"""Query AirSim scene objects - fixed API for cosysairsim."""
import sys
import json
import collections
import time

try:
    import cosysairsim as airsim
except ImportError:
    import airsim

PORT = 41451

def main():
    client = airsim.VehicleClient()
    client.confirmConnection()

    objects = client.simListSceneObjects()
    print(f"Total objects: {len(objects)}")

    # Categorize
    categories = collections.defaultdict(list)
    for obj in objects:
        name = str(obj) if isinstance(obj, str) else str(obj.name)
        if '_UAID' in name:
            prefix = name.split('_UAID')[0]
        elif '_C_' in name:
            prefix = name.split('_C_')[0] + '_C'
        else:
            parts = name.split('_')
            prefix = '_'.join(parts[:3]) if len(parts) > 2 else name
        categories[prefix].append(name)

    # Query specific object types
    target_types = ['SM_Pylon', 'SM_PGS_Pole', 'SM_fire_extinguisher',
                    'SM_Bench', 'BP_Worker_C', 'SM_TunCol1_Tunnel_base']

    print("\n=== Key Object Details ===")
    results = {}
    for target in target_types:
        names = categories.get(target, [])
        if not names:
            continue

        positions = []
        for name in names:
            try:
                pose = client.simGetObjectPose(name)
                pos = pose.position
                positions.append({
                    'name': name,
                    'x': pos.x_val, 'y': pos.y_val, 'z': pos.z_val
                })
            except Exception as e:
                pass

        if positions:
            xs = [p['x'] for p in positions]
            ys = [p['y'] for p in positions]
            zs = [p['z'] for p in positions]
            print(f"\n{target} ({len(positions)} objects):")
            print(f"  X range: [{min(xs):.1f}, {max(xs):.1f}] m")
            print(f"  Y range: [{min(ys):.1f}, {max(ys):.1f}] m")
            print(f"  Z range: [{min(zs):.1f}, {max(zs):.1f}] m")
            for p in positions[:3]:
                print(f"  Example: ({p['x']:.1f}, {p['y']:.1f}, {p['z']:.1f})")
            if len(positions) > 3:
                print(f"  ... and {len(positions)-3} more")
            results[target] = positions

    # Track BP_Worker movement
    print("\n=== BP_Worker Movement Tracking ===")
    worker_names = categories.get('BP_Worker_C', [])
    if worker_names:
        # Sample positions at two time points
        poses1 = {}
        for name in worker_names:
            try:
                pose = client.simGetObjectPose(name)
                poses1[name] = (pose.position.x_val, pose.position.y_val, pose.position.z_val)
            except:
                pass

        print(f"  T0 positions:")
        for name, pos in poses1.items():
            print(f"    {name}: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")

        time.sleep(2.0)

        poses2 = {}
        for name in worker_names:
            try:
                pose = client.simGetObjectPose(name)
                poses2[name] = (pose.position.x_val, pose.position.y_val, pose.position.z_val)
            except:
                pass

        print(f"  T1 positions (after 2s):")
        for name, pos in poses2.items():
            print(f"    {name}: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})")

        print(f"\n  Estimated velocities (over 2s):")
        for name in worker_names:
            if name in poses1 and name in poses2:
                dx = poses2[name][0] - poses1[name][0]
                dy = poses2[name][1] - poses1[name][1]
                dz = poses2[name][2] - poses1[name][2]
                dist = (dx**2 + dy**2 + dz**2)**0.5
                speed = dist / 2.0  # m/s
                print(f"    {name}: dx={dx:.2f}, dy={dy:.2f}, dz={dz:.2f}, speed={speed:.2f} m/s")

    # Summary for paper
    print("\n=== PAPER TABLE SUMMARY (ObsTunnel / ENV3) ===")
    print(f"Static obstacles:")
    for t in ['SM_Pylon', 'SM_PGS_Pole', 'SM_fire_extinguisher', 'SM_Bench']:
        count = len(categories.get(t, []))
        if count > 0:
            print(f"  {t}: {count}")
    print(f"Dynamic obstacles:")
    print(f"  BP_Worker_C: {len(worker_names)}")

    # Save
    with open("scene_objects_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            'scene': 'ObsTunnel (ENV3)',
            'total_objects': len(objects),
            'categories': {k: len(v) for k, v in categories.items()},
            'static_obstacle_counts': {
                'SM_Pylon': len(categories.get('SM_Pylon', [])),
                'SM_PGS_Pole': len(categories.get('SM_PGS_Pole', [])),
                'SM_fire_extinguisher': len(categories.get('SM_fire_extinguisher', [])),
                'SM_Bench': len([k for k in categories if k.startswith('SM_Bench')]),
            },
            'dynamic_obstacle_count': len(worker_names),
            'worker_positions_t0': poses1 if worker_names else {},
            'worker_positions_t1': poses2 if worker_names else {},
        }, f, indent=2, ensure_ascii=False)
    print("\nSaved to scene_objects_summary.json")

if __name__ == "__main__":
    main()
