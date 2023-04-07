import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import ast


def parse_args():
    parser = argparse.ArgumentParser(description="scatter the raw pcd or multi scale voxel (of lidar).")
    parser.add_argument("--path", help="abs path of .npy file.")
    parser.add_argument("--dot_size", help="float val specify scattered point size.", type=float)
    parser.add_argument("--scale_ratio", help="xyz order scaling ratio, tuple.", type=str)

    return parser.parse_args()

def prepare_xyz(_input: np.ndarray, _is_pcd=True)->tuple:
    if _is_pcd:
        x = _input[:, 0]
        y = _input[:, 1]
        z = _input[:, 2]
    else:
        if "coors" in _input.keys():
            first_batch_xyz = _input["coors"][:, 1:].cpu()
            x = first_batch_xyz[:, 0]
            y = first_batch_xyz[:, 1]
            z = first_batch_xyz[:, 2]
    
    return x, y, z


def scatter_plot(x, y, z, _this_shape, _is_pcd=True, _dot_size=1, _scale_s=str):
    print("before plot:")

    fig = plt.figure()
    # This warning is matplotlib version specific.
    # ax = Axes3D(fig)
    ax = Axes3D(fig, auto_add_to_figure=False)
    fig.add_axes(ax)

    if is_pcd is not True:
        ax.scatter(x, y, z, c=((x - _this_shape[0] / 2) ** 2 + (y - _this_shape[1] / 2) ** 2), s=_dot_size)
    else:
        ax.scatter(x, y, z, c=((x) ** 2 + (y) ** 2), s=_dot_size)

    _scale_r = ast.literal_eval(_scale_s)

    plt.gca().set_box_aspect(_scale_r)
    plt.xlabel("left 2 right.")
    plt.ylabel("back 2 front.")
    plt.show()

    print("end plot.")



if __name__ == "__main__":
    args = parse_args()

    this_voxel = np.array
    is_pcd = True
    this_shape = [1, 1, 1]

    if "pcd" in args.path:
        this_voxel = np.load(args.path)
    else:    
        this_voxel = np.load(args.path, allow_pickle=True).item()
        this_shape = this_voxel["voxel_shape"]
        is_pcd = False
    
    x, y, z = prepare_xyz(this_voxel, is_pcd)
    scatter_plot(x, y, z, this_shape, is_pcd, args.dot_size, args.scale_ratio)
