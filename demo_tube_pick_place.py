import time

import numpy as np

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
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=("agentview", "robot0_eye_in_hand"),
        horizon=300,
        reward_shaping=True,
        control_freq=20,
    )
    env = GymWrapper(env, keys=["object-state", "robot0_proprio-state"])

    use_cv2 = False
    cv2 = None
    plt = None
    img_handle = None
    try:
        import cv2  # type: ignore

        use_cv2 = True
    except Exception:
        try:
            import matplotlib.pyplot as plt  # type: ignore

            plt.ion()
            fig, ax = plt.subplots()
            img_handle = ax.imshow(np.zeros((256, 256, 3), dtype=np.uint8))
            ax.set_title("robot0_eye_in_hand")
            ax.axis("off")
            plt.show(block=False)
        except Exception:
            pass

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
        obs_dict = env.env._get_observations()
        eye_img = obs_dict.get("robot0_eye_in_hand_image")
        if eye_img is not None:
            if eye_img.dtype != np.uint8:
                eye_img = np.clip(eye_img, 0.0, 1.0)
                eye_img = (eye_img * 255).astype(np.uint8)
            if use_cv2:
                cv2.imshow("robot0_eye_in_hand", eye_img[:, :, ::-1])
                cv2.waitKey(1)
            elif img_handle is not None:
                img_handle.set_data(eye_img)
                plt.pause(0.001)
        if done:
            obs = env.reset()
            done = False
        time.sleep(1.0 / 60.0)


if __name__ == "__main__":
    main()
