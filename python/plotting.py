import re
import pandas as pd  # type: ignore[import]
import numpy as np
import seaborn as sns  # type: ignore[import]
import matplotlib.pyplot as plt  # type: ignore[import]
import glob


def json_to_dataframe(
    path: str, layer_name: str, max_C: int = 128, prefix: str = ""
) -> pd.DataFrame:
    files = glob.glob(path + "/*.json")
    regex = rf"{layer_name}_.+\.json"
    pattern = re.compile(regex)
    files_res = [x for x in files if pattern.search(x)]

    dfs = []  # an empty list to store the data frames
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
    df["top_1_accuracy_100"] = df["top_1_accuracy"] * 100
    df.columns = df.columns.str.replace(layer_name + ".", "")
    df.sort_values(
        by=["rows", "test_name"], inplace=True, ignore_index=True, ascending=False
    )
    # df = df.reindex(sorted(df.columns), axis=1)
    return df


layers = [
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


def plot_all_layers() -> None:
    data_path = "../data/accuracy/single_layer/training_data"
    dfs = []
    i = 0
    for l in layers:
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


if __name__ == "__main__":
    # plot_all_layers()
    plot_comparision()
