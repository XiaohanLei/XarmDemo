# Copyright (c) 2022-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

import os
import time
import tqdm
import random
import yaml
import argparse

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from collections import defaultdict
from contextlib import redirect_stdout

import torch


os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["BITSANDBYTES_NOWELCOME"] = "1"

import config as exp_cfg_mod
import model.rvt_agent as rvt_agent
import model.mvt.config as mvt_cfg_mod

from utils.dataset_utils import get_dataset

from model.mvt.mvt import MVT
from utils.rvt_utils import (
    TensorboardManager,
    short_name,
    get_num_feat,
    RLBENCH_TASKS,
)
from utils.peract_utils import (
    CAMERAS,
    SCENE_BOUNDS,
    IMAGE_SIZE,
)


# new train takes the dataset as input
def train(agent, dataset):
    agent.train()
    log = defaultdict(list)
    steps = 0
    for raw_batch in dataset:
        steps += 1
        update_args = {
            "replay_sample": raw_batch,
            "backprop": True,
            "reset_log": (steps == 1),
        }
        agent.update(**update_args)
        if steps > 2:
            break
    return log


def save_agent(agent, path, epoch):
    model = agent._network
    optimizer = agent._optimizer
    lr_sched = agent._lr_sched


    model_state = model.state_dict()

    torch.save(
        {
            "epoch": epoch,
            "model_state": model_state,
            "optimizer_state": optimizer.state_dict(),
            "lr_sched_state": lr_sched.state_dict(),
        },
        path,
    )


def get_tasks(exp_cfg):
    parsed_tasks = exp_cfg.tasks.split(",")
    if parsed_tasks[0] == "all":
        tasks = RLBENCH_TASKS
    else:
        tasks = parsed_tasks
    return tasks


def get_logdir(cmd_args, exp_cfg):
    log_dir = os.path.join(cmd_args.log_dir, exp_cfg.exp_id)
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def dump_log(exp_cfg, mvt_cfg, cmd_args, log_dir):
    with open(f"{log_dir}/exp_cfg.yaml", "w") as yaml_file:
        with redirect_stdout(yaml_file):
            print(exp_cfg.dump())

    with open(f"{log_dir}/mvt_cfg.yaml", "w") as yaml_file:
        with redirect_stdout(yaml_file):
            print(mvt_cfg.dump())

    args = cmd_args.__dict__
    with open(f"{log_dir}/args.yaml", "w") as yaml_file:
        yaml.dump(args, yaml_file)


def experiment(cmd_args, devices):
    """experiment.

    :param rank:
    :param cmd_args:
    :param devices: list or int. if list
    """
    rank = 0
    device = devices[rank]
    device = f"cuda:{device}"

    exp_cfg = exp_cfg_mod.get_cfg_defaults()
    if cmd_args.exp_cfg_path != "":
        exp_cfg.merge_from_file(cmd_args.exp_cfg_path)
    if cmd_args.exp_cfg_opts != "":
        exp_cfg.merge_from_list(cmd_args.exp_cfg_opts.split(" "))

    old_exp_cfg_peract_lr = exp_cfg.peract.lr
    old_exp_cfg_exp_id = exp_cfg.exp_id

    exp_cfg.peract.lr *= len(devices) * exp_cfg.bs
    if cmd_args.exp_cfg_opts != "":
        exp_cfg.exp_id += f"_{short_name(cmd_args.exp_cfg_opts)}"
    if cmd_args.mvt_cfg_opts != "":
        exp_cfg.exp_id += f"_{short_name(cmd_args.mvt_cfg_opts)}"

    if rank == 0:
        print(f"dict(exp_cfg)={dict(exp_cfg)}")
    exp_cfg.freeze()

    # Things to change
    BATCH_SIZE_TRAIN = exp_cfg.bs
    # to match peract, iterations per epoch
    TRAINING_ITERATIONS = int(exp_cfg.train_iter // (exp_cfg.bs * len(devices)))
    EPOCHS = exp_cfg.epochs
    log_dir = get_logdir(cmd_args, exp_cfg)
    tasks = get_tasks(exp_cfg)
    print("Training on {} tasks: {}".format(len(tasks), tasks))

    t_start = time.time()

    train_dataset = get_dataset(BATCH_SIZE_TRAIN)
    t_end = time.time()
    print("Created Dataset. Time Cost: {} minutes".format((t_end - t_start) / 60.0))

    if exp_cfg.agent == "our":
        mvt_cfg = mvt_cfg_mod.get_cfg_defaults()
        if cmd_args.mvt_cfg_path != "":
            mvt_cfg.merge_from_file(cmd_args.mvt_cfg_path)
        if cmd_args.mvt_cfg_opts != "":
            mvt_cfg.merge_from_list(cmd_args.mvt_cfg_opts.split(" "))

        mvt_cfg.feat_dim = get_num_feat(exp_cfg.peract)
        mvt_cfg.freeze()

        # for maintaining backward compatibility
        assert mvt_cfg.num_rot == exp_cfg.peract.num_rotation_classes, print(
            mvt_cfg.num_rot, exp_cfg.peract.num_rotation_classes
        )

        torch.cuda.set_device(device)
        torch.cuda.empty_cache()
        rvt = MVT(
            renderer_device=device,
            **mvt_cfg,
        ).to(device)

        agent = rvt_agent.RVTAgent(
            network=rvt,
            image_resolution=[IMAGE_SIZE, IMAGE_SIZE],
            add_lang=mvt_cfg.add_lang,
            stage_two=mvt_cfg.stage_two,
            rot_ver=mvt_cfg.rot_ver,
            scene_bounds=SCENE_BOUNDS,
            cameras=CAMERAS,
            log_dir=f"{log_dir}/test_run/",
            cos_dec_max_step=EPOCHS * TRAINING_ITERATIONS,
            **exp_cfg.peract,
            **exp_cfg.rvt,
        )
        agent.build(training=True, device=device)
    else:
        assert False, "Incorrect agent"

    start_epoch = 0
    end_epoch = EPOCHS

    if rank == 0:
        ## logging unchanged values to reproduce the same setting
        temp1 = exp_cfg.peract.lr
        temp2 = exp_cfg.exp_id
        exp_cfg.defrost()
        exp_cfg.peract.lr = old_exp_cfg_peract_lr
        exp_cfg.exp_id = old_exp_cfg_exp_id
        dump_log(exp_cfg, mvt_cfg, cmd_args, log_dir)
        exp_cfg.peract.lr = temp1
        exp_cfg.exp_id = temp2
        exp_cfg.freeze()
        tb = TensorboardManager(log_dir)

    print("Start training ...", flush=True)
    i = start_epoch
    while True:
        if i == end_epoch:
            break

        print(f"Rank [{rank}], Epoch [{i}]: Training on train dataset")
        out = train(agent, train_dataset)

        if rank == 0:
            tb.update("train", i, out)

        if rank == 0:
            # TODO: add logic to only save some models
            save_agent(agent, f"{log_dir}/model_{i}.pth", i)
            save_agent(agent, f"{log_dir}/model_last.pth", i)
        i += 1

    if rank == 0:
        tb.close()
        print("[Finish]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.set_defaults(entry=lambda cmd_args: parser.print_help())

    parser.add_argument("--refresh_replay", action="store_true", default=False)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--mvt_cfg_path", type=str, default="model/mvt/configs/rvt2.yaml")
    parser.add_argument("--exp_cfg_path", type=str, default="configs/rvt2.yaml")

    parser.add_argument("--mvt_cfg_opts", type=str, default="")
    parser.add_argument("--exp_cfg_opts", type=str, default="")

    parser.add_argument("--log-dir", type=str, default="runs")
    parser.add_argument("--with-eval", action="store_true", default=False)

    cmd_args = parser.parse_args()
    del (
        cmd_args.entry
    )  # hack for multi processing -- removes an argument called entry which is not picklable

    devices = cmd_args.device.split(",")
    devices = [int(x) for x in devices]

    experiment(cmd_args, devices)

