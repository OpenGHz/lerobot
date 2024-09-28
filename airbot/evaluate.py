from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.common.utils.utils import init_hydra_config
from lerobot.common.robot_devices.robots.factory import make_robot
from lerobot.common.robot_devices.utils import busy_wait
import time
import torch


inference_time_s = 40
fps = 15
device = "cuda"

ckpt_path = "outputs/train/03-24-03_airbot_play_gym_diffusion_default/checkpoints/025000/pretrained_model"
policy = DiffusionPolicy.from_pretrained(ckpt_path)
policy.to(device)

robot_path = "airbot/play/configs/test_task/robot/airbot_play.yaml"
robot_overrides = None
robot_cfg = init_hydra_config(robot_path, robot_overrides)
robot = make_robot(robot_cfg)
robot.connect()

try:
    for _ in range(inference_time_s * fps):
        start_time = time.perf_counter()

        # Read the follower state and access the frames from the cameras
        observation = robot.capture_observation()

        # Convert to pytorch format: channel first and float32 in [0,1]
        # with batch dimension
        for name in observation:
            if "image" in name:
                observation[name] = observation[name].type(torch.float32) / 255
                observation[name] = observation[name].permute(2, 0, 1).contiguous()
            observation[name] = observation[name].unsqueeze(0)
            observation[name] = observation[name].to(device)

        # Compute the next action with the policy
        # based on the current observation
        action = policy.select_action(observation)
        # Remove batch dimension
        action = action.squeeze(0)
        # Move to cpu, if not already the case
        action = action.to("cpu")
        # Order the robot to move
        # print(action)
        robot.send_action(action)

        dt_s = time.perf_counter() - start_time
        busy_wait(1 / fps - dt_s)
except Exception as e:
    print(e)
    robot.disconnect()
else:
    robot.disconnect()