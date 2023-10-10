import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# read csv to pandas dataframe
df = pd.read_csv("data/accuracy_comparison_all.csv")

print(df)

# normalize accuracy in df
df["mse_normalized"] = df["mse_backprop"] / df["int6_mse"]
df["huber_normalized"] = df["huber_backprop"] / df["int4_huber"]
df["mae_normalized"] = df["mae_backprop"] / df["int4_mae"]

df["int6_mse_normalized"] = df["int6_mse"] / df["int6_mse"]
df["int7_mse_normalized"] = df["int7_mse"] / df["int6_mse"]
df["int8_mse_normalized"] = df["int8_mse"] / df["int6_mse"]
df["int10_mse_normalized"] = df["int10_mse"] / df["int6_mse"]

# huber normalized
# df["int6_huber_normalized"] = df["int6_huber"] / df["int4_huber"]
# df["int7_huber_normalized"] = df["int7_huber"] / df["int4_huber"]
# df["int8_huber_normalized"] = df["int8_huber"] / df["int4_huber"]
# df["int10_huber_normalized"] = df["int10_huber"] / df["int4_huber"]
#
# # mae normalized
# df["int6_mae_normalized"] = df["int6_mae"] / df["int4_mae"]
# df["int7_mae_normalized"] = df["int7_mae"] / df["int4_mae"]
# df["int8_mae_normalized"] = df["int8_mae"] / df["int4_mae"]
# df["int10_mae_normalized"] = df["int10_mae"] / df["int4_mae"]


def get_position(df_row):
    mse = df_row["mse_backprop"]
    check_list = ["4", "6", "7", "8", "10"]
    for i, checks in enumerate(check_list):
        print("check", checks, df_row[f"int{checks}_mse"], mse)
        if df_row[f"int{checks}_mse"] - mse < 0.0:
            return (i - 1) + (1 - (mse / df_row[f"int{check_list[i-1]}_mse"]))

    return 0


df["mse_position"] = df.apply(get_position, axis=1)


def energy_efficiency(df_row):
    codebook_width = df_row["D"] // df_row["C"]
    codebook_width_factor = codebook_width / 9
    C_factor = 16 / df_row["C"]
    return 17 * codebook_width_factor * C_factor


df["energy_efficiency"] = df.apply(energy_efficiency, axis=1)

# filter certain names
df = df[df["layer"].isin(["conv2.0_A.npy", "conv3.0_A.npy", "conv4.0_A.npy"])]

# scatter plot mse_position vs energy_efficiency color by layer
# style figure for paper
plt.style.use("seaborn-v0_8-paper")
print(df["layer"].astype("category").cat.codes)
fix, ax = plt.subplots()
plt.scatter(
    df["energy_efficiency"],
    df["mse_position"],
    marker="x",
    c=df["layer"].astype("category").cat.codes,
    cmap="viridis",
)
# plt.scatter(df["mse_position"], df["energy_efficiency"])
# add x ticks with string labels
plt.yticks([2, 3, 4], ["INT6", "INT7", "INT8"])
plt.ylabel("Accuracy")
plt.xlabel("Energy Efficiency [TOPS/W]")
plt.title("Accuracy vs Energy Efficiency")
# make x logarithmic
plt.xscale("log")

plt.tight_layout()
# add grid
plt.grid()
# add legend
plt.legend(
    handles=[
        plt.Line2D([], [], marker="o", linestyle="None", label="conv2"),
        plt.Line2D([], [], marker="o", linestyle="None", label="conv3"),
        plt.Line2D([], [], marker="o", linestyle="None", label="conv4"),
    ],
    labels=["conv2.0", "conv3.0", "conv4.0"],
    loc="upper right",
    title="Layer",
)
# reverse x axis
# plt.gca().invert_xaxis()
# add x ticks
plt.xticks(
    [1, 10, 100],
    [
        "1",
        "10",
        "100",
    ],
)
# add subgrid for x axis
plt.gca().xaxis.grid(which="minor", linestyle="--")
# save figure
plt.savefig("figures/accuracy_vs_energy_efficiency_paper.png")
print(df)

# delete plot env
plt.clf()
import seaborn as sns

f, ax = plt.subplots(figsize=(5, 4))
# sns.set_context("paper")
sns.set(font="serif")
sns.set_style(
    "whitegrid",
    # {"font.family": "serif", "font.serif": ["Times", "serif"]},
)
# colors.reverse()
g = sns.scatterplot(
    data=df,
    x="energy_efficiency",
    y="mse_position",
    hue="layer",
    ax=ax,
    markers="x",
)
# add x ticks with string labels
plt.yticks([1, 2, 3], ["INT6", "INT7", "INT8"])
plt.ylabel("Accuracy")
plt.xlabel("Energy Efficiency [TOPS/W]")
# plt.title("Accuracy vs Energy Efficiency")
# make x logarithmic
plt.xscale("log")
# add grid
plt.grid()
# subgrid
plt.gca().xaxis.grid(which="minor", linestyle="--")
plt.legend(
    handles=[
        plt.Line2D([], [], color="C0", marker="o", linestyle="None", label="conv2"),
        plt.Line2D([], [], color="C1", marker="o", linestyle="None", label="conv3"),
        plt.Line2D([], [], color="C2", marker="o", linestyle="None", label="conv4"),
    ],
    labels=["conv2.0", "conv3.0", "conv4.0"],
    loc="lower left",
    # title="Layer",
)
plt.xticks(
    [1, 10, 100],
    [
        "1",
        "10",
        "100",
    ],
)
# save
plt.savefig("figures/accuracy_vs_energy_efficiency_paper.png", dpi=600)
