import os
from torch.utils.tensorboard import SummaryWriter
import robosuite as suite
from robosuite.wrappers import GymWrapper
from td3_torch import Agent
import tube_pick_place_env  # noqa: F401

if __name__ == '__main__':

    if not os.path.exists("tmp/265_proj"):
        os.makedirs("tmp/265_proj")

    env_name = "TubePickPlace"

    env = suite.make(
        env_name,
        robots=["UR5e"],
        controller_configs=suite.load_controller_config(default_controller="JOINT_VELOCITY"),
        has_renderer=False,
        use_camera_obs=True,
        camera_names=("agentview", "robot0_eye_in_hand"),
        camera_heights=64,
        camera_widths=64,
        horizon=300,
        reward_shaping=True,
        control_freq=20,
    )

    env = GymWrapper(
        env,
        keys=[
            "object-state",
            "robot0_proprio-state",
            "robot0_eye_in_hand_image",
        ],
    )

    actor_learning_rate = 0.001
    critic_learning_rate = 0.001
    batch_size = 128
    layer1_size = 256
    layer2_size = 128

    agent = Agent(
        actor_learning_rate=actor_learning_rate,
        critic_learning_rate=critic_learning_rate,
        tau=0.005,
        input_dims=env.observation_space.shape,
        env=env,
        n_actions=env.action_space.shape[0],
        layer1_size=layer1_size,
        layer2_size=layer2_size,
        batch_size=batch_size,
        max_size=10000,
    )

    writer = SummaryWriter('logs')
    n_games = 100
    best_score = 0
    episode_identifier = f"0 - actor_learning_rate={actor_learning_rate} critic_learning_rate={critic_learning_rate} " \
                         f"layer_1_size={layer1_size} layer_2_size={layer2_size}"
    agent.load_module()

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0

        while not done:
            action = agent.choose_action(observation)
            next_observation, reward, done, info = env.step(action)
            score += reward
            agent.remember(observation, action, reward, next_observation, done)
            agent.learn()
            observation = next_observation

        writer.add_scalar(f"Score - {episode_identifier}", score, global_step=i)

        if i % 10 == 0:
            agent.save_models()

        print(f"Episode:{i} Score: {score}")
