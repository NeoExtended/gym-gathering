import argparse
import sys
import time
from typing import Dict

import cv2
import keyboard
import numpy as np
from stable_baselines3.common.env_util import make_vec_env

import gym_gathering


def upscale(img, scale=2):
    w, h = img.shape[:2]
    return cv2.resize(img, (h * scale, w * scale), interpolation=cv2.INTER_NEAREST)


def imshow(img, scale=1, delay=25):
    if scale > 1:
        img = upscale(img, scale)

    rgb_image = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imshow("image", rgb_image)
    cv2.waitKey(delay)


def interact(env, keymap: Dict[str, int], scale: int, delay: int):
    obs = env.reset()
    done = False

    while not done:
        imshow(env.render(mode="rgb_array"), scale, delay)
        time.sleep(0.01)
        # env.render(mode="human")
        action = 0
        while True:
            key_pressed = False
            for key in keymap:
                if keyboard.is_pressed(key):
                    action = keymap[key]
                    key_pressed = True
                    break
            if key_pressed:
                break
            time.sleep(0.001)

        obs, rewards, dones, info = env.step([action])
        if np.sum(dones) > 0:
            done = True
        # print(rewards)


def main(args=None):
    # parse arguments
    if args is None:
        args = sys.argv[1:]
    parser = argparse.ArgumentParser("Play the environments yourself.")
    parser.add_argument("--env", type=str)
    parser.add_argument("--upscale", type=int, default=3)
    parser.add_argument("--delay", type=int, default=25)
    args = parser.parse_args(args=args)

    problem = None
    keymap = gym_gathering.KEYMAP

    env = make_vec_env(args.env, n_envs=1)
    env.seed(44)
    interact(env, keymap, args.upscale, args.delay)


if __name__ == "__main__":
    main()
