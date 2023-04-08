import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse


cat_indice = ["car", "trunk", "c_v", "bus", "trailer", "barrier", "motorcycle", "bicycle", "pedestrian", "cone"]
path = "/home/qyh/Archives/bevfusion/sample_case/pred_meta/pred_meta.npy"
data_pred = np.load(path, allow_pickle=True).item()
data_gt = np.load("/home/qyh/Archives/bevfusion/BEV_gaussian_heatmap.npy")
# keys: "center", "hight", "dim", "rot", "vel", "heatmap", "query_heatmap_score", "dense_heatmap". vals: tensor.
# first dim of each value is batch_size.
# `center`:             [batch_size, 2, prop_cnt]
# `heatmap`:            [batch_size, 10, prop_cnt]
# `query_heatmap_score`:*
# `dense_heatmap`:      [batch_size, 10, 180, 180], ∈R.


def parse_args():
    # 0: car
    # 1: trunk
    # 2: c_v
    # 3: bus
    # 4: trailer
    # 5: barrier
    # 6: motorcycle
    # 7: bicycle
    # 8: pedestrian
    # 9: traffic cone
    # `NOTE:` it exist some problem when you want to pass a bool type parameter in terminal, which will always be true.
    # with `action="store_true"`, if we want the parameter to be `True`, we just type the operation like "--if_contrast",
    # and don't type it when we want it to be `False`.
    parser = argparse.ArgumentParser(description="category-specific dense heatmap.")
    parser.add_argument("--type", help="0-9 category indicator. 10 indicates all.", type=int)
    parser.add_argument("--if_contrast", help="whether stack pred/gt heatmap layers with alpha.", action="store_true")
    parser.add_argument("--is_pred", help="when --if_contrast is False, whether visual pred or gt.", action="store_true")

    return parser.parse_args()


def cat_heatmap_show(_map_gt: np.array, _map_pred: np.array, if_contrast: bool=True, _type: int=0, is_pred: bool=True)->any:
    if if_contrast:
        plt.imshow(_map_gt, cmap="Reds", origin="lower", alpha=1, vmin=0, vmax=1)
        plt.imshow(_map_pred, cmap="Blues", origin="lower", alpha=0.7, vmin=0, vmax=1)
        plt.colorbar()
        plt.title(cat_indice[_type] + " category, GT(red) versus pred(blue). max confid: " + str(_map_pred.max()))
    else:
        if is_pred:
            plt.imshow(_map_pred, cmap="Reds", origin="lower", vmin=0, vmax=1)
            plt.colorbar()
            plt.title(cat_indice[_type] + " category, pred. max confid: " + str(_map_pred.max()))
        else:
            plt.imshow(_map_gt, cmap="Reds", origin="lower", vmin=0, vmax=1)
            plt.colorbar()
            plt.title(cat_indice[_type] + " category, gt.")

    plt.show()


def all_heatmaps_show(_map_gt: np.array, _map_pred: np.array, if_contrast: bool=True, is_pred: bool=True)->any:
    if if_contrast:
        plt.imshow(_map_gt, cmap="Reds", origin="lower", alpha=1, vmin=0, vmax=1)
        plt.imshow(_map_pred, cmap="Blues", origin="lower", alpha=0.7, vmin=0, vmax=1)
        plt.colorbar()
        plt.title("all cats GT (red) versus pred (blue).")
    else:
        if is_pred:
            plt.imshow(_map_pred, cmap="Reds", origin="lower", vmin=0, vmax=1)
            plt.colorbar()
            plt.title("all cat, pred. max confid: " + str(_map_pred.max()))
        else:
            plt.imshow(_map_gt, cmap="Reds", origin="lower", vmin=0, vmax=1)
            plt.colorbar()
            plt.title("all cat, GT.")

    plt.show()


if __name__ == "__main__":
    arg = parse_args()
    heatmap = np.array
    heatmap_gt = data_gt[0, ...].transpose((0, 2, 1))
    heatmap_pred = np.array(data_pred["dense_heatmap"][0].cpu()).transpose((0, 2, 1)).astype(heatmap_gt.dtype)
    heatmap_pred = 1 / (1 + np.exp(-1 * heatmap_pred))

    print("begin:")
    if 0 <= arg.type and arg.type < 10:
        heatmap_gt = heatmap_gt[arg.type, ...]
        heatmap_pred = heatmap_pred[arg.type, ...]
        cat_heatmap_show(heatmap_gt, heatmap_pred, _type=arg.type, if_contrast=arg.if_contrast, is_pred=arg.is_pred)
    else:
        heatmap_gt = heatmap_gt.max(axis=0)
        heatmap_pred = heatmap_pred.max(axis=0)
        all_heatmaps_show(heatmap_gt, heatmap_pred, if_contrast=arg.if_contrast, is_pred=arg.is_pred)

    print("end.")
