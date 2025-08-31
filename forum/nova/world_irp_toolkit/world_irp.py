import numpy as np
import matplotlib.pyplot as plt

class WorldIRP:
    """
    Minimal 'world IRP' prototype: simulates projectile motion.
    Input: dict with {pos, vel, mass}
    Output: rollout trajectory over time (list of states)
    """
    def __init__(self, gravity=np.array([0,0,-9.81])):
        self.gravity = gravity

    def refine(self, state, steps=50, dt=0.05):
        pos = np.array(state["pos"], dtype=float)
        vel = np.array(state["vel"], dtype=float)
        mass = state.get("mass", 1.0)

        traj = []
        for i in range(steps):
            traj.append({"t": i*dt, "pos": pos.copy(), "vel": vel.copy()})
            vel = vel + self.gravity*dt
            pos = pos + vel*dt
            if pos[2] < 0:  # ground contact
                pos[2] = 0
                vel[2] = -0.5 * vel[2]  # dampened bounce
        return traj

    def plot_rollout(self, rollout):
        xs = [s["pos"][0] for s in rollout]
        ys = [s["pos"][1] for s in rollout]
        zs = [s["pos"][2] for s in rollout]
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")
        ax.plot(xs, ys, zs)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        plt.show()
