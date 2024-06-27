#
# Copyright (c) Since 2023 Ogata Laboratory, Waseda University
#
# Released under the AGPL license.
# see https://www.gnu.org/licenses/agpl-3.0.txt
#

import sys
import os
import glob
import argparse
import numpy as np
from multiprocessing import Pool
from pathlib import Path
from eipl.utils import resize_img, calc_minmax, list_to_numpy

parser = argparse.ArgumentParser()
parser.add_argument("--in_dir", type=str, default="./data/")
parser.add_argument("--out_dir", type=str, default="./data/")
parser.add_argument("--train_keywords", nargs="*", required=False)
parser.add_argument("--test_keywords", nargs="*", required=False)
parser.add_argument("--skip", type=int, default=1)
parser.add_argument("--cropped_img_size", type=int, required=False)
parser.add_argument("--resized_img_size", type=int, required=False)
parser.add_argument("-j", "--nproc", type=int, default=1)
parser.add_argument("-q", "--quiet", action="store_true")
args = parser.parse_args()
print("arguments:")
for k, v in vars(args).items():
    print(" " * 4 + f"{k}: {v}")
print()


def load_skip_resize_data(file_info):
    skip, resized_img_size, filename = file_info
    print(" " * 4 + filename)
    npz_data = np.load(filename)
    try:
        _front_images = npz_data["front_image"][::skip]
        _side_images = npz_data["side_image"][::skip]
        if args.cropped_img_size is not None:
            [fro_lef, fro_top, sid_lef, sid_top] = [
                (
                    images_shape[ax] - args.cropped_img_size
                ) // 2 for images_shape in [
                    _front_images.shape,
                    _side_images.shape
                ] for ax in [1, 2]
            ]
            [fro_rig, fro_bot, sid_rig, sid_bot] = [
                (p + args.cropped_img_size) for p in [
                    fro_lef, fro_top, sid_lef, sid_top
                ]
            ]
            _front_images = _front_images[:, fro_lef:fro_rig, fro_top:fro_bot, :]
            _side_images = _side_images[:, sid_lef:sid_rig, sid_top:sid_bot, :]
        if resized_img_size is not None:
            _front_images = resize_img(_front_images, (resized_img_size, resized_img_size))
            _side_images = resize_img(_side_images, (resized_img_size, resized_img_size))
        _wrenches = npz_data["wrench"][::skip]
        _joints = npz_data["joint"][::skip]
    except KeyError as e:
        print(f"{e.__class__.__name__}: filename={filename}")
        raise
    assert len(_front_images) == len(_side_images)
    assert len(_front_images) == len(_wrenches)
    assert len(_front_images) == len(_joints)
    return (_front_images, _side_images, _wrenches, _joints)


def save_arr(out_base_name, out_subpath_name, arr_data, quiet):
    """ almost an alias for np.save() """
    out_path = Path(out_base_name, out_subpath_name)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_path, arr_data)
    if not quiet:
        print(" " * 4 + f"{out_path}:\t{arr_data.shape}")


def load_data(in_dir, skip, resized_img_size, nproc):
    front_images = []
    side_images = []
    wrenches = []
    joints = []
    masks = []

    in_file_names = glob.glob(os.path.join(in_dir, "**/*.npz"), recursive=True)
    in_file_names.sort()
    try:
        assert len(in_file_names) >= 1, f"{len(in_file_names)=}"
    except AssertionError as e:
        sys.stderr.write(f"{sys.stderr.name} {in_dir=}\n")
        raise
    pool = Pool(nproc)
    loaded_data = pool.map(
        load_skip_resize_data, [(skip, resized_img_size, in_file_name) for in_file_name in in_file_names]
    )

    seq_length = []
    for (_front_images, _side_images, _wrenches, _joints) in loaded_data:
        seq_length.append(len(_joints))
    max_seq = max(seq_length)

    for (_front_images, _side_images, _wrenches, _joints) in loaded_data:
        front_images.append(_front_images)
        side_images.append(_side_images)
        wrenches.append(_wrenches)
        joints.append(_joints)
        masks.append(np.concatenate((np.ones(len(_joints)), np.zeros(max_seq - len(_joints)))))

    front_images = list_to_numpy(front_images, max_seq)
    side_images = list_to_numpy(side_images, max_seq)
    wrenches = list_to_numpy(wrenches, max_seq)
    joints = list_to_numpy(joints, max_seq)
    masks = np.stack(masks)

    return front_images, side_images, wrenches, joints, masks, in_file_names


if __name__ == "__main__":
    # load data
    if not args.quiet:
        print("input files:")
    front_images, side_images, wrenches, joints, masks, in_file_names = load_data(
        args.in_dir, args.skip, args.resized_img_size, args.nproc
    )

    # dataset keywords
    if args.train_keywords is not None:
        # use arguments to train files
        train_keywords = args.train_keywords
    else:
        # no arguments to train files
        # set train keywords excluding middle file
        i_pivot = (len(in_file_names) - 1) // 2
        train_keywords = [
            Path(
                in_file_name
            ).stem for i, in_file_name in enumerate(
                in_file_names
            ) if i != i_pivot
        ]
    if args.test_keywords is not None:
        # use arguments to test files
        test_keywords = args.test_keywords
    else:
        # no arguments to test files
        # set test keywords excluding train keywords
        test_keywords = [
            Path(in_file_name).stem for in_file_name in in_file_names if all([
                (w not in in_file_name) for w in train_keywords
            ])
        ]
    if not args.quiet:
        print()
        print("train keywords:\t", train_keywords)
        print("test keywords:\t", test_keywords)

    # dataset index
    train_list, test_list = list(), list()
    for i, in_file_name in enumerate(in_file_names):
        if any([(w in in_file_name) for w in train_keywords]):
            train_list.append(i)
        if any([(w in in_file_name) for w in test_keywords]):
            test_list.append(i)
    if not args.quiet:
        print()
        print("\n".join(["train files:"] + [(" " * 4 + in_file_names[i]) for i in train_list]))
        print("\n".join(["test files:"] + [(" " * 4 + in_file_names[i]) for i in test_list]))

    # save
    if not args.quiet:
        print()
        print("output files:")
    save_arr(args.out_dir, "train/masks.npy", masks[train_list].astype(np.float32), args.quiet)
    save_arr(args.out_dir, "train/front_images.npy", front_images[train_list].astype(np.uint8), args.quiet)
    save_arr(args.out_dir, "train/side_images.npy", side_images[train_list].astype(np.uint8), args.quiet)
    save_arr(args.out_dir, "train/wrenches.npy", wrenches[train_list].astype(np.float32), args.quiet)
    save_arr(args.out_dir, "train/joints.npy", joints[train_list].astype(np.float32), args.quiet)
    save_arr(args.out_dir, "test/masks.npy", masks[test_list].astype(np.float32), args.quiet)
    save_arr(args.out_dir, "test/front_images.npy", front_images[test_list].astype(np.uint8), args.quiet)
    save_arr(args.out_dir, "test/side_images.npy", side_images[test_list].astype(np.uint8), args.quiet)
    save_arr(args.out_dir, "test/wrenches.npy", wrenches[test_list].astype(np.float32), args.quiet)
    save_arr(args.out_dir, "test/joints.npy", joints[test_list].astype(np.float32), args.quiet)

    # save joint bounds
    joint_bounds = calc_minmax(joints)
    save_arr(args.out_dir, "joint_bounds.npy", joint_bounds, args.quiet)

    # save wrench bounds
    wrench_bounds = calc_minmax(wrenches)
    save_arr(args.out_dir, "wrench_bounds.npy", wrench_bounds, args.quiet)
