#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Process zarr files formatted similar to: https://github.com/real-stanford/diffusion_policy"""

import shutil
from pathlib import Path

import torch
import tqdm
import zarr
from datasets import Dataset, Features, Image, Sequence, Value
from PIL import Image as PILImage

from lerobot.common.datasets.lerobot_dataset import CODEBASE_VERSION
from lerobot.common.datasets.push_dataset_to_hub.utils import (
    concatenate_episodes,
    get_default_encoding,
    save_images_concurrently,
)
from lerobot.common.datasets.utils import (
    calculate_episode_data_index,
    hf_transform_to_torch,
)
from lerobot.common.datasets.video_utils import VideoFrame, encode_video_frames
from lerobot.common.datasets.push_dataset_to_hub._diffusion_policy_replay_buffer import (
    ReplayBuffer as DiffusionPolicyReplayBuffer,
)
from lerobot.scripts.push_dataset_to_hub import init_parser, push_dataset_to_hub

DATASET_NAME = None

def check_format(zarr_path: str):
    zarr_data = zarr.open(zarr_path, mode="r")
    # print(zarr_data.tree())
    required_datasets = {
        "data/action",
        "data/state",
        "data/img",
        "meta/episode_ends",
        # "data/n_contacts",
    }
    for dataset in required_datasets:
        assert dataset in zarr_data, f"Missing dataset: {dataset}"
    nb_frames = zarr_data["data/img"].shape[0]

    required_datasets.remove("meta/episode_ends")

    assert all(
        nb_frames == zarr_data[dataset].shape[0] for dataset in required_datasets
    )


def load_from_raw(
    data_path: Path,
    videos_dir: Path,
    fps: int,
    video: bool,
    episodes: list[int] | None = None,
    encoding: dict | None = None,
):
    zarr_path = data_path
    print("copy_from_path:", zarr_path)
    zarr_data = DiffusionPolicyReplayBuffer.copy_from_path(zarr_path)
    print("convert data to torch tensors")
    episode_ids = torch.from_numpy(zarr_data.get_episode_idxs())
    assert len(
        {zarr_data[key].shape[0] for key in zarr_data.keys()}  # noqa: SIM118
    ), "Some data type dont have the same number of total frames."

    imgs = torch.from_numpy(zarr_data["img"])  # b h w c
    states = torch.from_numpy(zarr_data["state"])
    actions = torch.from_numpy(zarr_data["action"])

    # load data indices from which each episode starts and ends
    from_ids, to_ids = [], []
    from_idx = 0
    for to_idx in zarr_data.meta["episode_ends"]:
        from_ids.append(from_idx)
        to_ids.append(to_idx)
        from_idx = to_idx

    num_episodes = len(from_ids)

    ep_dicts = []
    ep_ids = episodes if episodes else range(num_episodes)
    for ep_idx, selected_ep_idx in tqdm.tqdm(enumerate(ep_ids)):
        from_idx = from_ids[selected_ep_idx]
        to_idx = to_ids[selected_ep_idx]
        num_frames = to_idx - from_idx

        # sanity check
        assert (episode_ids[from_idx:to_idx] == ep_idx).all()

        # get image
        image = imgs[from_idx:to_idx]
        assert image.min() >= 0.0
        assert image.max() <= 255.0
        image = image.type(torch.uint8)

        # get done (last step of demonstration is considered done)
        done = torch.zeros(num_frames, dtype=torch.bool)
        done[-1] = True

        ep_dict = {}
        imgs_array = [x.numpy() for x in image]
        img_key = "observation.image"
        if video:
            # save png images in temporary directory
            tmp_imgs_dir = videos_dir / "tmp_images"
            print("Saving images concurrently to", tmp_imgs_dir)
            save_images_concurrently(imgs_array, tmp_imgs_dir)

            # encode images to a mp4 video
            fname = f"{img_key}_episode_{ep_idx:06d}.mp4"
            video_path = videos_dir / fname
            print("Encoding video to", video_path)
            if encoding is None:
                encoding = {
                    "vcodec": "libx264",
                }
            encode_video_frames(tmp_imgs_dir, video_path, fps, **encoding)

            # clean temporary images directory
            shutil.rmtree(tmp_imgs_dir)

            # store the reference to the video frame
            ep_dict[img_key] = [
                {"path": f"videos/{fname}", "timestamp": i / fps}
                for i in range(num_frames)
            ]
        else:
            ep_dict[img_key] = [PILImage.fromarray(x) for x in imgs_array]

        ep_dict["observation.state"] = states[from_idx:to_idx]
        ep_dict["action"] = actions[from_idx:to_idx]
        ep_dict["episode_index"] = torch.tensor(
            [ep_idx] * num_frames, dtype=torch.int64
        )
        ep_dict["frame_index"] = torch.arange(0, num_frames, 1)
        ep_dict["timestamp"] = torch.arange(0, num_frames, 1) / fps
        # ep_dict["next.observation.image"] = image[1:],
        # ep_dict["next.observation.state"] = agent_pos[1:],
        # TODO(rcadene)] = verify that done are aligned with image and agent_pos
        ep_dict["next.done"] = torch.cat([done[1:], done[[-1]]])
        ep_dicts.append(ep_dict)

    data_dict = concatenate_episodes(ep_dicts)

    total_frames = data_dict["frame_index"].shape[0]
    data_dict["index"] = torch.arange(0, total_frames, 1)
    return data_dict


def to_hf_dataset(data_dict, video):
    features = {}
    if video:
        features["observation.image"] = VideoFrame()
    else:
        features["observation.image"] = Image()

    features["observation.state"] = Sequence(
        length=data_dict["observation.state"].shape[1],
        feature=Value(dtype="float32", id=None),
    )

    features["action"] = Sequence(
        length=data_dict["action"].shape[1], feature=Value(dtype="float32", id=None)
    )
    features["episode_index"] = Value(dtype="int64", id=None)
    features["frame_index"] = Value(dtype="int64", id=None)
    features["timestamp"] = Value(dtype="float32", id=None)
    features["next.done"] = Value(dtype="bool", id=None)
    features["index"] = Value(dtype="int64", id=None)

    hf_dataset = Dataset.from_dict(data_dict, features=Features(features))
    hf_dataset.set_transform(hf_transform_to_torch)
    return hf_dataset


def from_raw_to_lerobot_format(
    raw_dir: Path,
    videos_dir: Path,
    fps: int,
    video: bool = True,
    episodes: list[int] | None = None,
    encoding: dict | None = None,
):
    # sanity check
    data_path = raw_dir / Path(DATASET_NAME + ".zarr")
    # print(data_path)
    check_format(data_path)
    print("Loading data from", data_path)
    print("1", encoding)  # None
    data_dict = load_from_raw(data_path, videos_dir, fps, video, episodes, encoding)
    print("Converting data to HF dataset")
    hf_dataset = to_hf_dataset(data_dict, video)
    print("Calculating episode data index")
    episode_data_index = calculate_episode_data_index(hf_dataset)
    info = {
        "codebase_version": CODEBASE_VERSION,
        "fps": fps,
        "video": video,
    }
    if video:
        info["encoding"] = get_default_encoding()

    return hf_dataset, episode_data_index, info


def main():
    global DATASET_NAME
    parser = init_parser()
    parser.add_argument(
        "--file-name",
        type=str,
        required=True,
        help="Name of the zarr file",
    )
    args = parser.parse_args()
    args.push_to_hub = 0
    DATASET_NAME = args.file_name
    params = vars(args)
    params.pop("file_name")
    push_dataset_to_hub(
        **params, from_raw_to_lerobot_format=from_raw_to_lerobot_format
    )


if __name__ == "__main__":
    main()
