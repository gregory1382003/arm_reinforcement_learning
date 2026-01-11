import time

import robosuite as suite
from robosuite.controllers import load_controller_config
from robosuite.wrappers import GymWrapper

from td3_torch import Agent
import tube_pick_place_env  # noqa: F401


def main():
    env = suite.make(
        "TubePickPlace",
        robots=["UR5e"],
        controller_configs=load_controller_config(default_controller="JOINT_VELOCITY"),
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        horizon=300,
        reward_shaping=True,
        control_freq=20,
    )
    env = GymWrapper(env)

    agent = Agent(
        actor_learning_rate=0.001,
        critic_learning_rate=0.001,
        tau=0.005,
        input_dims=env.observation_space.shape,
        env=env,
        n_actions=env.action_space.shape[0],
        layer1_size=256,
        layer2_size=128,
        batch_size=128,
    )
    agent.load_module()

    obs = env.reset()
    done = False
    while True:
        action = agent.choose_action(obs, validation=True)
        obs, _, done, _ = env.step(action)
        env.render()
        if done:
            obs = env.reset()
            done = False
        time.sleep(1.0 / 60.0)


if __name__ == "__main__":
    main()
