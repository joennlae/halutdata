import json
from math import ceil, sqrt
import re
from typing import Any, Literal
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import glob

all_layers = [
    "layer1.0.conv1",
    "layer1.0.conv2",
    "layer1.0.conv3",
    "layer1.0.downsample.0",
    "layer1.1.conv1",
    "layer1.1.conv2",
    "layer1.1.conv3",
    "layer1.2.conv1",
    "layer1.2.conv2",
    "layer1.2.conv3",
    "layer2.0.conv1",
    "layer2.0.conv2",
    "layer2.0.conv3",
    "layer2.0.downsample.0",
    "layer2.1.conv1",
    "layer2.1.conv2",
    "layer2.1.conv3",
    "layer2.2.conv1",
    "layer2.2.conv2",
    "layer2.2.conv3",
    "layer2.3.conv1",
    "layer2.3.conv2",
    "layer2.3.conv3",
    "layer3.0.conv1",
    "layer3.0.conv2",
    "layer3.0.conv3",
    "layer3.0.downsample.0",
    "layer3.1.conv1",
    "layer3.1.conv2",
    "layer3.1.conv3",
    "layer3.2.conv1",
    "layer3.2.conv2",
    "layer3.2.conv3",
    "layer3.3.conv1",
    "layer3.3.conv2",
    "layer3.3.conv3",
    "layer3.4.conv1",
    "layer3.4.conv2",
    "layer3.4.conv3",
    "layer3.5.conv1",
    "layer3.5.conv2",
    "layer3.5.conv3",
    "layer4.0.conv1",
    "layer4.0.conv2",
    "layer4.0.conv3",
    "layer4.0.downsample.0",
    "layer4.1.conv1",
    "layer4.1.conv2",
    "layer4.1.conv3",
    "layer4.2.conv1",
    "layer4.2.conv2",
    "layer4.2.conv3",
]

downsampled = [
    "layer1.0.downsample.0",
    "layer2.0.downsample.0",
    "layer3.0.downsample.0",
    "layer4.0.downsample.0",
]

conv3x3 = [
    "layer1.0.conv2",
    "layer1.1.conv2",
    "layer1.2.conv2",
    "layer2.0.conv2",
    "layer2.1.conv2",
    "layer2.2.conv2",
    "layer2.3.conv2",
    "layer3.0.conv2",
    "layer3.1.conv2",
    "layer3.2.conv2",
    "layer3.3.conv2",
    "layer3.4.conv2",
    "layer3.5.conv2",
    "layer4.0.conv2",
    "layer4.1.conv2",
    "layer4.2.conv2",
]

layers_interesting = [
    "layer1.1.conv3",
    "layer2.0.conv1",
    "layer2.3.conv3",
    "layer3.0.conv1",
    "layer3.0.conv3",
    "layer3.3.conv3",
    "layer4.0.conv1",
    "layer4.0.conv3",
    "layer4.1.conv1",
    "layer4.2.conv3",
]

ds_cnn_layers = [
    "conv1",
    "conv2",
    "conv3",
    "conv4",
    "conv5",
    "conv6",
    "conv7",
    "conv8",
    "conv9",
]

layers_levit = [
    "blocks.0.m.qkv",
    "blocks.0.m.proj.1",
    "blocks.1.m.0",
    "blocks.1.m.2",
    "blocks.2.m.qkv",
    "blocks.2.m.proj.1",
    "blocks.3.m.0",
    "blocks.3.m.2",
    "blocks.4.kv",
    "blocks.4.q.1",
    "blocks.4.proj.1",
    "blocks.5.m.0",
    "blocks.5.m.2",
    "blocks.6.m.qkv",
    "blocks.6.m.proj.1",
    "blocks.7.m.0",
    "blocks.7.m.2",
    "blocks.8.m.qkv",
    "blocks.8.m.proj.1",
    "blocks.9.m.0",
    "blocks.9.m.2",
    "blocks.10.m.qkv",
    "blocks.10.m.proj.1",
    "blocks.11.m.0",
    "blocks.11.m.2",
    "blocks.12.kv",
    "blocks.12.q.1",
    "blocks.12.proj.1",
    "blocks.13.m.0",
    "blocks.13.m.2",
    "blocks.14.m.qkv",
    "blocks.14.m.proj.1",
    "blocks.15.m.0",
    "blocks.15.m.2",
    "blocks.16.m.qkv",
    "blocks.16.m.proj.1",
    "blocks.17.m.0",
    "blocks.17.m.2",
    "blocks.18.m.qkv",
    "blocks.18.m.proj.1",
    "blocks.19.m.0",
    "blocks.19.m.2",
    "blocks.20.m.qkv",
    "blocks.20.m.proj.1",
    "blocks.21.m.0",
    "blocks.21.m.2",
    "head",
    "head_dist",
]


def json_to_dataframe(
    path: str,
    layer_name: str,
    max_C: int = 128,
    prefix: str = "",
    model: str = "resnet-50",
) -> pd.DataFrame:
    files = glob.glob(path + "/*.json")
    regex = rf"{layer_name}_\d.+\.json"
    pattern = re.compile(regex)
    files_res = [x for x in files if pattern.search(x)]

    dfs = []  # an empty list to store the data frames
    if len(files_res) == 0:
        return pd.DataFrame()
    for file in files_res:
        data = pd.read_json(file)  # read data frame from json file
        if layer_name + ".learned_n" not in data.columns:
            data[layer_name + ".learned_n"] = data.iloc[0][
                layer_name + ".learned_a_shape"
            ]
            data[layer_name + ".learned_d"] = data.iloc[1][
                layer_name + ".learned_a_shape"
            ]
            K = data.iloc[0][layer_name + ".K"]
            C = data.iloc[0][layer_name + ".C"]
            data[layer_name + ".learned_m"] = int(
                data.iloc[0][layer_name + ".L_size"] / (4 * K * C)
            )
        C = data.iloc[0][layer_name + ".C"]
        if C > max_C:
            continue
        if layer_name + ".learned_a_shape" in data.columns:
            data = data.drop([1])
            data = data.drop(
                columns=[
                    layer_name + ".learned_a_shape",
                    layer_name + ".learned_b_shape",
                ]
            )

        data["hue_string"] = prefix + str(C)

        data["test_name"] = layer_name + "-" + str(data.iloc[0][layer_name + ".C"])
        data["layer_name_canonical"] = layer_name
        data["layer_name"] = layer_name + (
            " (3x3)" if "conv2" in layer_name else " (1x1)"
        )
        data["row_name"] = layer_name.split(".")[0]
        data["col_name"] = layer_name[len(layer_name.split(".")[0]) + 1 :]
        dfs.append(data)  # append the data frame to the list

    df = pd.concat(
        dfs, ignore_index=True
    )  # concatenate all the data frames in the list.

    df = df.drop(columns="halut_layers")
    if model == "resnet-50":
        df["top_1_accuracy_100"] = df["top_1_accuracy"]  # * 100
    else:
        df["top_1_accuracy_100"] = df["top_1_accuracy"]
        df["top_1_accuracy"] = df["top_1_accuracy"] / 100

    layer_info = {}
    if model == "resnet-50":
        layer_info = layer_info_resnet
    elif model == "levit":
        layer_info = layer_info_levit
    elif model == "ds-cnn":
        layer_info = layer_info_ds_cnn
    df["table_info"] = str(layer_info[layer_name][:2]) + (
        ""
        if len(layer_info[layer_name]) == 2
        else (
            " ("
            + str(layer_info[layer_name][2])
            + "x"
            + str(layer_info[layer_name][3])
            + ")"
        )
    )
    df["in_channels"] = layer_info[layer_name][0]
    df["out_channels"] = layer_info[layer_name][1]
    df.columns = df.columns.str.replace(layer_name + ".", "")
    df.sort_values(
        by=["rows", "test_name"], inplace=True, ignore_index=True, ascending=False
    )
    # df = df.reindex(sorted(df.columns), axis=1)
    return df


layer_info_resnet = {
    "conv1": [64, 3, 7, 7],
    "layer1.0.conv1": [64, 64, 1, 1],
    "layer1.0.conv2": [64, 64, 3, 3],
    "layer1.0.conv3": [256, 64, 1, 1],
    "layer1.0.downsample.0": [256, 64, 1, 1],
    "layer1.1.conv1": [64, 256, 1, 1],
    "layer1.1.conv2": [64, 64, 3, 3],
    "layer1.1.conv3": [256, 64, 1, 1],
    "layer1.2.conv1": [64, 256, 1, 1],
    "layer1.2.conv2": [64, 64, 3, 3],
    "layer1.2.conv3": [256, 64, 1, 1],
    "layer2.0.conv1": [128, 256, 1, 1],
    "layer2.0.conv2": [128, 128, 3, 3],
    "layer2.0.conv3": [512, 128, 1, 1],
    "layer2.0.downsample.0": [512, 256, 1, 1],
    "layer2.1.conv1": [128, 512, 1, 1],
    "layer2.1.conv2": [128, 128, 3, 3],
    "layer2.1.conv1": [128, 512, 1, 1],
    "layer2.1.conv2": [128, 128, 3, 3],
    "layer2.1.conv3": [512, 128, 1, 1],
    "layer2.2.conv1": [128, 512, 1, 1],
    "layer2.2.conv2": [128, 128, 3, 3],
    "layer2.2.conv3": [512, 128, 1, 1],
    "layer2.3.conv1": [128, 512, 1, 1],
    "layer2.3.conv2": [128, 128, 3, 3],
    "layer2.3.conv3": [512, 128, 1, 1],
    "layer3.0.conv1": [256, 512, 1, 1],
    "layer3.0.conv2": [256, 256, 3, 3],
    "layer3.0.conv3": [1024, 256, 1, 1],
    "layer3.0.downsample.0": [1024, 512, 1, 1],
    "layer3.1.conv1": [256, 1024, 1, 1],
    "layer3.1.conv2": [256, 256, 3, 3],
    "layer3.1.conv3": [1024, 256, 1, 1],
    "layer3.2.conv1": [256, 1024, 1, 1],
    "layer3.2.conv2": [256, 256, 3, 3],
    "layer3.2.conv3": [1024, 256, 1, 1],
    "layer3.3.conv1": [256, 1024, 1, 1],
    "layer3.3.conv2": [256, 256, 3, 3],
    "layer3.3.conv3": [1024, 256, 1, 1],
    "layer3.4.conv1": [256, 1024, 1, 1],
    "layer3.4.conv2": [256, 256, 3, 3],
    "layer3.4.conv3": [1024, 256, 1, 1],
    "layer3.5.conv1": [256, 1024, 1, 1],
    "layer3.5.conv2": [256, 256, 3, 3],
    "layer3.5.conv3": [1024, 256, 1, 1],
    "layer4.0.conv1": [512, 1024, 1, 1],
    "layer4.0.conv2": [512, 512, 3, 3],
    "layer4.0.conv3": [2048, 512, 1, 1],
    "layer4.0.downsample.0": [2048, 1024, 1, 1],
    "layer4.1.conv1": [512, 2048, 1, 1],
    "layer4.1.conv2": [512, 512, 3, 3],
    "layer4.1.conv3": [2048, 512, 1, 1],
    "layer4.2.conv1": [512, 2048, 1, 1],
    "layer4.2.conv2": [512, 512, 3, 3],
    "layer4.2.conv1": [512, 2048, 1, 1],
    "layer4.2.conv2": [512, 512, 3, 3],
    "layer4.2.conv3": [2048, 512, 1, 1],
}

layer_info_levit = {
    "blocks.0.m.qkv": [256, 128],
    "blocks.0.m.proj.1": [128, 128],
    "blocks.1.m.0": [256, 128],
    "blocks.1.m.2": [128, 256],
    "blocks.2.m.qkv": [256, 128],
    "blocks.2.m.proj.1": [128, 128],
    "blocks.3.m.0": [256, 128],
    "blocks.3.m.2": [128, 256],
    "blocks.4.kv": [640, 128],
    "blocks.4.q.1": [128, 128],
    "blocks.4.proj.1": [256, 512],
    "blocks.5.m.0": [512, 256],
    "blocks.5.m.2": [256, 512],
    "blocks.6.m.qkv": [384, 256],
    "blocks.6.m.proj.1": [256, 192],
    "blocks.7.m.0": [512, 256],
    "blocks.7.m.2": [256, 512],
    "blocks.8.m.qkv": [384, 256],
    "blocks.8.m.proj.1": [256, 192],
    "blocks.9.m.0": [512, 256],
    "blocks.9.m.2": [256, 512],
    "blocks.10.m.qkv": [384, 256],
    "blocks.10.m.proj.1": [256, 192],
    "blocks.11.m.0": [512, 256],
    "blocks.11.m.2": [256, 512],
    "blocks.12.kv": [1280, 256],
    "blocks.12.q.1": [256, 256],
    "blocks.12.proj.1": [384, 1024],
    "blocks.13.m.0": [768, 384],
    "blocks.13.m.2": [384, 768],
    "blocks.14.m.qkv": [512, 384],
    "blocks.14.m.proj.1": [384, 256],
    "blocks.15.m.0": [768, 384],
    "blocks.15.m.2": [384, 768],
    "blocks.16.m.qkv": [512, 384],
    "blocks.16.m.proj.1": [384, 256],
    "blocks.17.m.0": [768, 384],
    "blocks.17.m.2": [384, 768],
    "blocks.18.m.qkv": [512, 384],
    "blocks.18.m.proj.1": [384, 256],
    "blocks.19.m.0": [768, 384],
    "blocks.19.m.2": [384, 768],
    "blocks.20.m.qkv": [512, 384],
    "blocks.20.m.proj.1": [384, 256],
    "blocks.21.m.0": [768, 384],
    "blocks.21.m.2": [384, 768],
    "head": [1000, 384],
    "head_dist": [1000, 384],
}

layer_info_ds_cnn = {
    "conv1": [64, 1, 10, 4],
    "conv2": [64, 1, 3, 3],
    "conv3": [64, 64, 1, 1],
    "conv4": [64, 1, 3, 3],
    "conv5": [64, 64, 1, 1],
    "conv6": [64, 1, 3, 3],
    "conv7": [64, 64, 1, 1],
    "conv8": [64, 1, 3, 3],
    "conv9": [64, 64, 1, 1],
}


layer_info_resnet_18_layers = {
    # out, in , kernel_size
    "layer1.0.conv1": [64, 64, 3, 3],
    "layer1.0.conv2": [64, 64, 3, 3],
    "layer1.1.conv1": [64, 64, 3, 3],
    "layer1.1.conv2": [64, 64, 3, 3],
    "layer2.0.conv1": [128, 64, 3, 3],
    "layer2.0.conv2": [128, 128, 3, 3],
    "layer2.0.downsample.0": [128, 64, 1, 1],
    "layer2.1.conv1": [128, 128, 3, 3],
    "layer2.1.conv2": [128, 128, 3, 3],
    "layer3.0.conv1": [256, 128, 3, 3],
    "layer3.0.conv2": [256, 256, 3, 3],
    "layer3.0.downsample.0": [256, 128, 1, 1],
    "layer3.1.conv1": [256, 256, 3, 3],
    "layer3.1.conv2": [256, 256, 3, 3],
    "layer4.0.conv1": [512, 256, 3, 3],
    "layer4.0.conv2": [512, 512, 3, 3],
    "layer4.0.downsample.0": [512, 256, 1, 1],
    "layer4.1.conv1": [512, 512, 3, 3],
    "layer4.1.conv2": [512, 512, 3, 3],
    "fc": [100, 512],  # 100 for cifar-100
}


def resnet18_layer_info_plot() -> None:
    all_data = []
    # pylint: disable=consider-using-dict-items
    for layer in layer_info_resnet_18_layers.keys():
        values = layer_info_resnet_18_layers[layer]
        kernel = "-"
        in_ = "-"
        out_ = "-"
        if len(values) > 2:
            type_ = "Conv2d"
            D = values[2] * values[3] * values[1]
            M = values[0]
            kernel = f"[{values[2]}, {values[3]}]"
            in_ = values[1]  # type: ignore[assignment]
            out_ = values[0]  # type: ignore[assignment]
        else:
            type_ = "Linear"
            D = values[1]
            M = values[0]

        c_width = D / 64
        row = {
            "name": layer,
            "type": type_,
            "D": D,
            # "M": M,
            "in": in_,
            "out": out_,
            "kernel": kernel,
            "c_width": c_width,
        }
        all_data.append(row)

    print(all_data)
    df = pd.DataFrame(all_data)
    print(df)
    cidx = pd.Index(
        ["Layer Name", "Type", "D", "In", "Out", "Kernel", "Codebook width"]
    )
    df.columns = cidx
    styler = df.style
    styler.format(subset="Codebook width", precision=0).format_index(
        escape="latex", axis=1
    ).format_index(escape="latex", axis=0).hide(level=0, axis=0)
    styler.to_latex(
        f"table_resnet18.tex",
        clines="skip-last;data",
        convert_css=True,
        position_float="centering",
        multicol_align="|c|",
        hrules=True,
        # float_format="%.2f",
    )


def calculate_MACs(layer_name: str, df: pd.DataFrame) -> int:
    N = df[layer_name + ".learned_n"] / df[layer_name + ".rows"]
    D = df[layer_name + ".learned_d"]
    M = df[layer_name + ".learned_m"]

    return N * D * M


def calculate_MACs_resnet(layer_name: str, df: pd.DataFrame, N: int = 256) -> int:
    layer_info = layer_info_resnet_18_layers[layer_name]
    if len(layer_info) > 2:
        # conv
        D = layer_info[2] * layer_info[3] * layer_info[1]
        M = layer_info[0]
    else:
        # linear
        D = layer_info[1]
        M = layer_info[0]
    return N * D * M


def json_to_dataframe_macs(path: str, max_C: int = 16) -> pd.DataFrame:
    files = glob.glob(path + "/*.json")
    dfs = []  # an empty list to store the data frames
    total_macs = 0
    for layer_name in all_layers:
        regex = rf"{layer_name}_.+\.json"
        pattern = re.compile(regex)
        files_res = [x for x in files if pattern.search(x)]
        file = files_res[0]
        print(file)
        data = pd.read_json(file)  # read data frame from json file
        if layer_name + ".learned_n" not in data.columns:
            data[layer_name + ".learned_n"] = data.iloc[0][
                layer_name + ".learned_a_shape"
            ]
            data[layer_name + ".learned_d"] = data.iloc[1][
                layer_name + ".learned_a_shape"
            ]
            K = data.iloc[0][layer_name + ".K"]
            C = data.iloc[0][layer_name + ".C"]
            data[layer_name + ".learned_m"] = int(
                data.iloc[0][layer_name + ".L_size"] / (4 * K * C)
            )
        C = data.iloc[0][layer_name + ".C"]
        if C > max_C:
            continue
        if layer_name + ".learned_a_shape" in data.columns:
            data = data.drop([1])
            data = data.drop(
                columns=[
                    layer_name + ".learned_a_shape",
                    layer_name + ".learned_b_shape",
                ]
            )
        macs_layer = calculate_MACs(layer_name, data)
        total_macs += macs_layer
        data[layer_name + ".macs"] = macs_layer
        data[layer_name + ".layer_name"] = layer_name
        data.columns = data.columns.str.replace(layer_name + ".", "")
        dfs.append(data)  # append the data frame to the list

    df = pd.concat(
        dfs, ignore_index=True
    )  # concatenate all the data frames in the list.

    df = df.drop(columns="halut_layers")
    df["top_1_accuracy_100"] = df["top_1_accuracy"]  #  * 100
    df.sort_values(by=["macs"], inplace=True, ignore_index=True, ascending=False)
    print("TOTAL MACS", total_macs)
    return df


def json_to_multi_layer(path: str, max_C: int = 128, prefix: str = "") -> pd.DataFrame:
    files = glob.glob(path + "/*.json")

    dfs = []  # an empty list to store the data frames
    for file in files:
        data = pd.read_json(file)  # read data frame from json file
        data = data.drop([1])
        layers = json.loads(data.iloc[0]["halut_layers"])
        C = data.iloc[0][list(layers.keys())[0] + ".C"]
        if C > max_C:
            continue
        saved_macs = 0
        print(data)
        for layer_name in layers.keys():
            if layer_name + ".learned_a_shape" in data.columns:
                data = data.drop(
                    columns=[
                        layer_name + ".learned_a_shape",
                        layer_name + ".learned_b_shape",
                    ]
                )
            macs_layer = calculate_MACs_resnet(
                layer_name, data, N=256
            )  # n is not correct but would need more infos
            saved_macs += macs_layer
            data[layer_name + ".macs"] = macs_layer
        data["num_replaced_layers"] = len(list(layers.keys()))
        data["last_replaced"] = list(layers.keys())[0]
        data["hue_string"] = "retrained" if "_trained" in file else "replaced"
        print(file, data["hue_string"])
        data["saved_macs"] = saved_macs
        dfs.append(data)  # append the data frame to the list

    df = pd.concat(
        dfs, ignore_index=True
    )  # concatenate all the data frames in the list.

    df = df.drop(columns="halut_layers")
    df["top_1_accuracy_100"] = df["top_1_accuracy"]  # * 100
    df.sort_values(
        by=["num_replaced_layers"], inplace=True, ignore_index=True, ascending=True
    )
    # df = df.reindex(sorted(df.columns), axis=1)
    return df


def plot_all_layers() -> None:
    data_path = "../data/accuracy/single_layer/training_data"
    dfs = []
    i = 0
    for l in all_layers:
        i = i + 1
        # if i > 12:
        #     break
        df = json_to_dataframe(data_path, l)
        dfs.append(df)

    df = pd.concat(dfs)
    # df.to_latex("test.tex")
    # with pd.ExcelWriter("output.xlsx") as writer:
    #     df.to_excel(writer, sheet_name=layer_name)
    # df2.to_excel(writer, sheet_name='Sheet_name_2')
    print(df)
    sns.set_context("paper")
    sns.set(font="serif")
    sns.set_style(
        "whitegrid",
        # {"font.family": "serif", "font.serif": ["Times", "Palatino", "serif"]},
    )
    colors = sns.color_palette("Greys", n_colors=5).as_hex()[-3:]
    colors2 = sns.color_palette("Blues", n_colors=5).as_hex()[2:4]
    colors = colors + colors2
    print(colors, colors2)
    # colors.reverse()
    customPalette = sns.color_palette(colors)
    grid = sns.FacetGrid(
        df,
        col="layer_name",
        hue="C",
        palette=customPalette,
        col_wrap=6,
        height=2.5,
        legend_out=True,
    )
    RESNET_ACC = 80.858
    # Draw a horizontal line to show the starting point
    grid.refline(y=RESNET_ACC, linestyle=":", color="red")
    # grid.refline(y=80.0, linestyle=":", color="grey")
    # grid.refline(y=79.0, linestyle=":", color="grey")
    MIN = 63.95
    # Draw a line plot to show the trajectory of each random walk
    grid.map(plt.plot, "rows", "top_1_accuracy_100", marker="o")
    for (row_i, col_j, hue_k), data_ijk in grid.facet_data():
        ax = grid.facet_axis(row_i, col_j)
        max_acc = np.max(data_ijk["top_1_accuracy_100"])
        ax.hlines(
            color=grid._facet_color(hue_k, None),
            y=max_acc,
            xmin=0.5,
            xmax=((40 * 256) + 1),
            linestyle="--",
        )
        if data_ijk.shape[0] and data_ijk.iloc[0]["C"] == 64:
            error = np.min(data_ijk["scaled_error"])
            C = data_ijk.iloc[0]["C"]
            K = data_ijk.iloc[0]["K"]
            D = data_ijk.iloc[0]["learned_d"]
            M = data_ijk.iloc[0]["learned_m"]
            representation_factor = C * np.log2(K) / D
            uncertainty_factor = int(np.log2(M / representation_factor))
            lut_size = data_ijk.iloc[0]["L_size"] / 1024 / 1024
            ax.text(
                y=MIN + 0.1,
                x=1,
                s=(
                    f"{max_acc:0.2f}% (-{RESNET_ACC - max_acc:0.2f}%) |"
                    rf" {error:.2e} | {lut_size:0.1f} MB | {uncertainty_factor}"
                ),
                fontsize=7,
            )

        if data_ijk.shape[0] and data_ijk.iloc[0]["C"] == 128:
            error = np.min(data_ijk["scaled_error"])
            lut_size = data_ijk.iloc[0]["L_size"] / 1024 / 1024
            ax.text(
                y=MIN + (RESNET_ACC - MIN) / 10,
                x=1,
                s=(
                    f"{max_acc:0.2f}% (-{RESNET_ACC - max_acc:0.2f}%) |"
                    rf" {error:.2e} | {lut_size:0.1f} MB"
                ),
                color=grid._facet_color(hue_k, None),
                fontsize=7,
            )

    grid.set_axis_labels("#Images for Halut", "Top1 Accuracy (%)")
    grid.set_titles(col_template="{col_name}")
    # Adjust the tick positions and labels
    grid.set(xscale="log")
    xticks = [2, 8, 32, 128, 2048]
    grid.set(
        xticks=xticks,
        xticklabels=xticks,
        yticks=[73, 78, 79, 80, 81],
        xlim=(0.9, (40 * 256) + 1),
        ylim=(MIN, 81),
    )
    grid.add_legend()
    grid._legend.set_title("C=")

    # grid.fig.tight_layout(w_pad=1)
    plt.savefig("../figures/all_layers.pdf", bbox_inches="tight", dpi=600)
    plt.savefig("../figures/all_layers.png", bbox_inches="tight", dpi=600)


def plot_comparision() -> None:
    data_path_1 = "../data/accuracy/single_layer/training_data"
    data_path_2 = "../data/accuracy/single_layer/updated_learning"
    dfs = []
    for l in layers_interesting:
        df = json_to_dataframe(data_path_1, l, 64, "old-")
        dfs.append(df)
        df = json_to_dataframe(data_path_2, l, 64, "new-")
        dfs.append(df)

    df = pd.concat(dfs)
    # df.to_latex("test.tex")
    # with pd.ExcelWriter("output.xlsx") as writer:
    #     df.to_excel(writer, sheet_name=layer_name)
    # df2.to_excel(writer, sheet_name='Sheet_name_2')
    print(df)
    sns.set_context("paper")
    sns.set(font="serif")
    sns.set_style(
        "whitegrid",
        # {"font.family": "serif", "font.serif": ["Times", "Palatino", "serif"]},
    )
    colors = sns.color_palette("Greys", n_colors=5).as_hex()[-3:]
    colors2 = sns.color_palette("Greens", n_colors=5).as_hex()[2:5]
    colors.reverse()
    colors2.reverse()
    colors = colors + colors2
    print(colors, colors2)

    customPalette = sns.color_palette(colors)
    grid = sns.FacetGrid(
        df,
        col="layer_name",
        hue="hue_string",
        palette=customPalette,
        col_wrap=4,
        height=2.5,
        legend_out=True,
    )
    RESNET_ACC = 80.858
    # Draw a horizontal line to show the starting point
    grid.refline(y=RESNET_ACC, linestyle=":", color="red")
    # grid.refline(y=80.0, linestyle=":", color="grey")
    # grid.refline(y=79.0, linestyle=":", color="grey")
    MIN = 63.95
    # Draw a line plot to show the trajectory of each random walk
    grid.map(plt.plot, "rows", "top_1_accuracy_100", marker="o")
    for (row_i, col_j, hue_k), data_ijk in grid.facet_data():
        ax = grid.facet_axis(row_i, col_j)
        max_acc = np.max(data_ijk["top_1_accuracy_100"])
        ax.hlines(
            color=grid._facet_color(hue_k, None),
            y=max_acc,
            xmin=0.5,
            xmax=((40 * 256) + 1),
            linestyle="--",
        )
        if data_ijk.shape[0] and data_ijk.iloc[0]["C"] == 64:
            error = np.min(data_ijk["scaled_error"])
            C = data_ijk.iloc[0]["C"]
            K = data_ijk.iloc[0]["K"]
            D = data_ijk.iloc[0]["learned_d"]
            M = data_ijk.iloc[0]["learned_m"]
            representation_factor = C * np.log2(K) / D
            uncertainty_factor = int(np.log2(M / representation_factor))
            lut_size = data_ijk.iloc[0]["L_size"] / 1024 / 1024
            ax.text(
                y=MIN
                + (
                    (RESNET_ACC - MIN) / 10
                    if "new" in data_ijk.iloc[0]["hue_string"]
                    else 0.1
                ),
                x=1,
                s=(
                    f"{max_acc:0.2f}% (-{RESNET_ACC - max_acc:0.2f}%) |"
                    rf" {error:.2e} | {lut_size:0.1f} MB | {uncertainty_factor}"
                ),
                color=grid._facet_color(hue_k, None),
                fontsize=7,
            )

        if data_ijk.shape[0] and data_ijk.iloc[0]["C"] == 128:
            error = np.min(data_ijk["scaled_error"])
            lut_size = data_ijk.iloc[0]["L_size"] / 1024 / 1024
            ax.text(
                y=MIN + (RESNET_ACC - MIN) / 10,
                x=1,
                s=(
                    f"{max_acc:0.2f}% (-{RESNET_ACC - max_acc:0.2f}%) |"
                    rf" {error:.2e} | {lut_size:0.1f} MB"
                ),
                color=grid._facet_color(hue_k, None),
                fontsize=7,
            )

    grid.set_axis_labels("#Images for Halut", "Top1 Accuracy (%)")
    grid.set_titles(col_template="{col_name}")
    # Adjust the tick positions and labels
    grid.set(xscale="log")
    xticks = [2, 8, 32, 128, 2048]
    grid.set(
        xticks=xticks,
        xticklabels=xticks,
        yticks=[73, 78, 79, 80, 81],
        xlim=(0.9, (40 * 256) + 1),
        ylim=(MIN, 81),
    )
    grid.add_legend()
    grid._legend.set_title("C=")

    # grid.fig.tight_layout(w_pad=1)
    plt.savefig("../figures/comp_new_old.pdf", bbox_inches="tight", dpi=600)
    plt.savefig("../figures/comp_new_old.png", bbox_inches="tight", dpi=600)


def plot_retraining() -> None:
    data_path_1 = "../data/resnet18-cifar10"
    df = json_to_multi_layer(data_path_1, 64)
    print(df)
    sns.set_context("paper")
    sns.set(font="serif", font_scale=2)
    sns.set_style(
        "whitegrid",
        # {"font.family": "serif", "font.serif": ["Times", "Palatino", "serif"]},
    )
    plt.figure(figsize=(14, 8))
    print(df["last_replaced"])
    plot = sns.barplot(
        data=df[df["hue_string"] == "retrained"],
        x="last_replaced",
        y="top_1_accuracy_100",
        color="green",
    )
    plot2 = sns.barplot(
        data=df[df["hue_string"] == "replaced"],
        x="last_replaced",
        y="top_1_accuracy_100",
        color="blue",
    )
    RESNET_ACC = 87.130  # 87.130, 68.250
    # Draw a horizontal line to show the starting point
    plt.hlines(y=RESNET_ACC, linestyle=":", color="red", xmin=0.0, xmax=19)
    plt.xticks(rotation=90)
    plot2.set_ylabel("Top1 Accuracy [%]")
    plot2.set_xlabel("Replaced Layers")
    plot2.set_title("ResNet18 CIFAR-10 Replace&Freeze Retraining C=64")
    plt.savefig("../figures/retrained_10.pdf", bbox_inches="tight", dpi=600)
    plt.savefig("../figures/retrained_10.png", bbox_inches="tight", dpi=600)


def plot_multi_layer() -> None:
    data_path_1 = "../data/resnet18-cifar100_real"
    data_path_2 = "../data/accuracy/single_layer/training_data"
    df = json_to_multi_layer(data_path_1, 32)

    print(df)
    df.to_csv("out.csv")
    print(df["saved_macs"])
    sns.set_context("paper")
    sns.set(font="serif")
    sns.set_style(
        "whitegrid",
        # {"font.family": "serif", "font.serif": ["Times", "Palatino", "serif"]},
    )
    TOTAL_MACS = 3.8e9
    df["macs_saved_percentage"] = df["saved_macs"] / TOTAL_MACS * 100
    grid = sns.lineplot(data=df, x="saved_macs", y="top_1_accuracy_100", marker="o")

    grid.set_xlabel("MACs saved")
    grid.set_ylabel("Accuracy")
    grid.set(xlim=(0, 1.7e9))
    RESNET_ACC = 68.250
    # Draw a horizontal line to show the starting point
    grid.hlines(y=RESNET_ACC, linestyle=":", color="red", xmin=0, xmax=2.0e9)

    def forward(x: float) -> float:
        return x / TOTAL_MACS * 100

    def inverse(x: float) -> float:
        return x * TOTAL_MACS / 100

    secax = grid.secondary_xaxis("top", functions=(forward, inverse))
    secax.set_xlabel("MACs saved in %")
    RESNET_ACC = 68.250
    # Draw a horizontal line to show the starting point
    # grid.refline(y=RESNET_ACC, linestyle=":", color="red")
    plt.savefig("../figures/multi_layer.pdf", bbox_inches="tight", dpi=600)
    plt.savefig("../figures/multi_layer.png", bbox_inches="tight", dpi=600)


def create_string(input: Any) -> float:
    # return f"{input[0]:.2f} ({input[1]:.2f})"
    return input[1]


def create_string_flops_rw(input: Any) -> str:
    return f"{input[0]:.1f} ({input[1]:.1f})"


def calc_flops(
    N: int, D: int, M: int, C: int, K: int, type_: Literal["gemm", 0, 1, 2]
) -> int:
    if type_ == "gemm":
        return 2 * N * D * M
    elif type_ == 0:  # Maddness
        return N * C * (ceil(sqrt(K)) + M)
    elif type_ == 1:  # DT
        return N * C * (ceil(sqrt(K)) + M)
    elif type_ == 2:  # PQ
        return N * C * M + N * C * K * ceil(D / C) * 3


def get_read_and_writes(
    N: int,
    D: int,
    M: int,
    C: int,
    K: int,
    type_: Literal["gemm", 0, 1, 2],
    dtype_byte_width: int = 4,
) -> int:
    read_write_accesses = 0
    if type_ == "gemm":
        read_write_accesses = M * D + N * D + M * N
    elif type_ == 0:  # Maddness
        # remove LUT reads = N * M * C
        read_write_accesses = N * D + N * M * (1)
    elif type_ == 1:  # DT
        # remove LUT reads = N * M * C
        read_write_accesses = N * D + N * M * (
            1
        )  # one lookup more at the end of encoding
    elif type_ == 2:  # PQ
        # remove LUT reads = N * M * C
        read_write_accesses = (
            N * C * (ceil(D / C) + K * ceil(D / C)) + N * C + N * M * (1)
        )
    return dtype_byte_width * read_write_accesses


def calc_flops_gemm_row(input: Any) -> int:
    return calc_flops(
        int(input["learned_n"]) // int(input["rows"]),
        int(input["learned_d"]),
        int(input["learned_m"]),
        min(int(input["learned_d"]), int(input["C"])),
        int(input["K"]),
        "gemm",
    )


def calc_flops_gemm_enc(input: Any) -> int:
    return calc_flops(
        int(input["learned_n"]) // int(input["rows"]),
        int(input["learned_d"]),
        int(input["learned_m"]),
        min(int(input["learned_d"]), int(input["C"])),
        int(input["K"]),
        int(input["encoding_algorithm"]),  # type: ignore[arg-type]
    )


def calc_rw_gemm_row(input: Any) -> int:
    return get_read_and_writes(
        int(input["learned_n"]) // int(input["rows"]),
        int(input["learned_d"]),
        int(input["learned_m"]),
        min(int(input["learned_d"]), int(input["C"])),
        int(input["K"]),
        "gemm",
    )


def calc_rw_enc(input: Any) -> int:
    return get_read_and_writes(
        int(input["learned_n"]) // int(input["rows"]),
        int(input["learned_d"]),
        int(input["learned_m"]),
        min(int(input["learned_d"]), int(input["C"])),
        int(input["K"]),
        int(input["encoding_algorithm"]),  # type: ignore[arg-type]
    )


def formatter_acc_reduction(input: Any) -> str:
    color = "green"
    input_num = float(input)
    if input_num < -1.0:
        color = "orange"
    if input_num < -5.0:
        color = "red"
    return r"\textcolor{" + color + "}{" + input + "}"


def formatter_remove_parts(input: str) -> str:
    input = input.replace("blocks.", "")
    input = input.replace("layer", "")
    input = input.replace("downsample", "down")
    return input


def create_tables() -> None:
    df_32 = pd.read_csv("export.csv")
    ref_acc = {"resnet-50": 80.858, "levit": 76.520, "ds-cnn": 92.94}
    for m in ["ds-cnn", "levit", "resnet-50"]:
        df_m = df_32[df_32["model"] == m].copy()
        df_m["N"] = df_m["learned_n"] // df_m["rows"]
        df_m["flops_gemm"] = df_m.apply(calc_flops_gemm_row, axis=1)
        df_m["flops_halut"] = df_m.apply(calc_flops_gemm_enc, axis=1)
        df_m["flops_reduction"] = df_m["flops_gemm"] / df_m["flops_halut"]
        df_m["read_write_bytes_gemm"] = df_m.apply(calc_rw_gemm_row, axis=1)
        df_m["read_write_bytes_halut"] = df_m.apply(calc_rw_enc, axis=1)
        df_m["read_write_bytes_reduction"] = (
            df_m["read_write_bytes_gemm"] / df_m["read_write_bytes_halut"]
        )
        df_m["accuracy_enc_0"] = 0.0
        df_m["accuracy_enc_1"] = 0.0
        df_m["accuracy_enc_2"] = 0.0
        df_m["flops_reduction_0"] = 0.0
        df_m["flops_reduction_1"] = 0.0
        df_m["flops_reduction_2"] = 0.0
        df_m["rw_reduction_0"] = 0.0
        df_m["rw_reduction_1"] = 0.0
        df_m["rw_reduction_2"] = 0.0

        df_m["scaled_error_0"] = 0.0
        df_m["scaled_error_1"] = 0.0
        df_m["scaled_error_2"] = 0.0

        print(df_m[["flops_halut", "flops_gemm"]])
        for name in pd.unique(df_m["layer_name_canonical"]):
            data = df_m.loc[df_m["layer_name_canonical"] == name]
            accuracies = [0.0, 0.0, 0.0]
            flops_reductions = [0.0, 0.0, 0.0]
            scaled_errors = [0.0, 0.0, 0.0]
            rw_reductions = [0.0, 0.0, 0.0]
            for enc in range(3):
                acc = data.loc[
                    data["encoding_algorithm"] == float(enc), "top_1_accuracy_100"
                ].values
                accuracies[enc] = acc[0]
                red = data.loc[
                    data["encoding_algorithm"] == float(enc), "flops_reduction"
                ].values
                flops_reductions[enc] = red[0]
                err = data.loc[
                    data["encoding_algorithm"] == float(enc), "scaled_error"
                ].values
                scaled_errors[enc] = err[0]
                rw = data.loc[
                    data["encoding_algorithm"] == float(enc),
                    "read_write_bytes_reduction",
                ].values
                rw_reductions[enc] = rw[0]
            df_m.loc[
                df_m["layer_name_canonical"] == name, "accuracy_enc_0"
            ] = accuracies[0]
            df_m.loc[
                df_m["layer_name_canonical"] == name, "accuracy_enc_1"
            ] = accuracies[1]
            df_m.loc[
                df_m["layer_name_canonical"] == name, "accuracy_enc_2"
            ] = accuracies[2]

            df_m.loc[
                df_m["layer_name_canonical"] == name, "flops_reduction_0"
            ] = flops_reductions[0]
            df_m.loc[
                df_m["layer_name_canonical"] == name, "flops_reduction_1"
            ] = flops_reductions[1]
            df_m.loc[
                df_m["layer_name_canonical"] == name, "flops_reduction_2"
            ] = flops_reductions[2]

            df_m.loc[
                df_m["layer_name_canonical"] == name, "scaled_error_0"
            ] = scaled_errors[0]
            df_m.loc[
                df_m["layer_name_canonical"] == name, "scaled_error_1"
            ] = scaled_errors[1]
            df_m.loc[
                df_m["layer_name_canonical"] == name, "scaled_error_2"
            ] = scaled_errors[2]

            df_m.loc[
                df_m["layer_name_canonical"] == name, "rw_reduction_0"
            ] = rw_reductions[0]
            df_m.loc[
                df_m["layer_name_canonical"] == name, "rw_reduction_1"
            ] = rw_reductions[1]
            df_m.loc[
                df_m["layer_name_canonical"] == name, "rw_reduction_2"
            ] = rw_reductions[2]

        df_m.drop_duplicates(
            subset=["layer_name_canonical"], keep="first", inplace=True
        )
        df_m["d_int"] = df_m["learned_d"].astype(int)
        df_m["acc_enc_0_diff"] = df_m["accuracy_enc_0"] - ref_acc[m]
        df_m["acc_enc_1_diff"] = df_m["accuracy_enc_1"] - ref_acc[m]
        df_m["acc_enc_2_diff"] = df_m["accuracy_enc_2"] - ref_acc[m]

        df_m["acc_enc_0_text"] = df_m[["accuracy_enc_0", "acc_enc_0_diff"]].apply(
            create_string, axis=1
        )
        df_m["acc_enc_1_text"] = df_m[["accuracy_enc_1", "acc_enc_1_diff"]].apply(
            create_string, axis=1
        )
        df_m["acc_enc_2_text"] = df_m[["accuracy_enc_2", "acc_enc_2_diff"]].apply(
            create_string, axis=1
        )

        df_m["flops_rw_text_0"] = df_m[["flops_reduction_0", "rw_reduction_0"]].apply(
            create_string_flops_rw, axis=1
        )
        df_m["flops_rw_text_1"] = df_m[["flops_reduction_1", "rw_reduction_1"]].apply(
            create_string_flops_rw, axis=1
        )
        df_m["flops_rw_text_2"] = df_m[["flops_reduction_2", "rw_reduction_2"]].apply(
            create_string_flops_rw, axis=1
        )

        df_m["layer_name_table"] = df_m["layer_name_canonical"].apply(
            formatter_remove_parts
        )

        df_m["lut_kb"] = df_m["L_size"] // 1024

        df_table = df_m[
            [
                "layer_name_table",
                "table_info",
                "d_int",
                "acc_enc_0_text",
                "acc_enc_1_text",
                "acc_enc_2_text",
                "flops_rw_text_0",
                "flops_rw_text_2",
                "lut_kb",
                "scaled_error_0",
                "scaled_error_1",
                "scaled_error_2",
            ]
        ]

        cidx = pd.MultiIndex.from_arrays(
            [
                [
                    "Name",
                    "[Out, In]",
                    "D",
                    "Accuracy drop [%]",
                    "Accuracy drop [%]",
                    "Accuracy drop [%]",
                    "FLOPs (RW) [x]",
                    "FLOPs (RW) [x]",
                    "LUT",
                    "Scaled error",
                    "Scaled error",
                    "Scaled error",
                ],
                [
                    "",
                    "",
                    "",
                    "Madd",
                    "DT",
                    "PQ",
                    "M&DT",
                    "PQ",
                    "[kB]",
                    "Madd",
                    "DT",
                    "PQ",
                ],
            ]
        )
        df_table.columns = cidx
        print(df_table.head(5))
        styler = df_table.style
        styler.format(subset="Accuracy drop [%]", precision=2).format(
            subset="LUT", precision=0
        ).format("{:.1E}", subset="Scaled error").format_index(
            escape="latex", axis=1
        ).format_index(
            escape="latex", axis=0
        ).hide(
            level=0, axis=0
        )
        styler.background_gradient(
            axis=None,
            cmap="RdYlGn",
            high=0.8,
            subset=["Accuracy drop [%]"],
        )
        styler.background_gradient(
            axis=None,
            cmap="RdYlGn_r",
            low=0.5,
            subset=["Scaled error"],
        )
        print(df_table.head(5))
        styler.to_latex(
            "../tables/" + m + ".tex",
            clines="skip-last;data",
            convert_css=True,
            position_float="centering",
            multicol_align="|c|",
            hrules=True,
            # float_format="%.2f",
        )

        df_m.to_csv(m + ".csv")


def data_to_sql() -> None:
    data_paths = {
        "resnet-50": "../data/accuracy/single_layer/c_k_sweep",
        "levit": "../data/accuracy/single_layer/levit",
        "ds-cnn": "../data/accuracy/single_layer/ds-cnn",
    }
    layers_dict = {
        "resnet-50": all_layers,
        "levit": layers_levit,
        "ds-cnn": ds_cnn_layers,
    }
    dfs = []
    i = 0
    for m in ["ds-cnn", "levit", "resnet-50"]:
        layers = layers_dict[m]
        for l in layers:
            i = i + 1
            # if i > 12:
            #     break
            df = json_to_dataframe(data_paths[m], l, model=m)
            df["model"] = m
            dfs.append(df)

    df = pd.concat(dfs)
    print(df, df.shape)

    # tables
    df_32 = df[(df["C"] == 32) & (df["K"] == 16)]

    df_32.sort_values(
        by=["top_1_accuracy"], inplace=True, ignore_index=True, ascending=False
    )
    df_32.drop_duplicates(
        subset=["model", "layer_name_canonical", "encoding_algorithm"],
        keep="first",
        inplace=True,
    )
    df_32.sort_values(
        by=["layer_name_canonical"], inplace=True, ignore_index=True, ascending=True
    )

    print(df_32)

    df_32.to_csv("export.csv")

    # create_tables()

    import sqlite3

    con = sqlite3.connect("halutdata.db")
    df.to_sql("parametersweep", con=con)

    cur = con.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    print(cur.fetchall())

    cur.close()
    con.close()


if __name__ == "__main__":
    # plot_all_layers()
    # plot_comparision()
    # plot_multi_layer()
    # data_to_sql()
    # create_tables()
    # plot_retraining()
    resnet18_layer_info_plot()
