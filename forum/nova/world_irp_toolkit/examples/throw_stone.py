from world_irp import WorldIRP

def main():
    irp = WorldIRP()
    # Example: throw a stone upward and forward
    state = {"pos": [0,0,1], "vel": [3,1,8], "mass": 1.0}
    rollout = irp.refine(state, steps=60, dt=0.05)
    print("Final state:", rollout[-1])
    irp.plot_rollout(rollout)

if __name__ == "__main__":
    main()
