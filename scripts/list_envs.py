"""List the project's Gymnasium task IDs without launching Isaac Sim."""

import gymnasium as gym

from humanoid_dreamwaq import locomotion  # noqa: F401


if __name__ == "__main__":
    for spec in gym.registry.values():
        if spec.id.startswith("Unitree-G1-29dof-Velocity"):
            print(spec.id)
