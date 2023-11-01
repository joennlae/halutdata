from tbparse import SummaryReader

import matplotlib.pyplot as plt

log_dir = "/home/janniss/Documents/halutmatmul/src/python/runs"
reader = SummaryReader(
    log_dir,
    extra_columns={
        "dir_name",
    },
)

# dirs = [
#     "Sep10_12-53-59_vilan1.ee.ethz.chcheckpoints_128_224_0.001_cosineannealinglr_adam_resnet9-lpl-0.001-thresh-25-93.6",
#     "Sep10_13-44-12_vilan1.ee.ethz.chcheckpoints_128_248_0.001_cosineannealinglr_adam_resnet9-lpl-0.001-thresh-25-93.6",
#     "Sep10_14-56-42_vilan1.ee.ethz.chcheckpoints_128_272_0.001_cosineannealinglr_adam_resnet9-lpl-0.001-thresh-25-93.6",
#     "Sep10_16-32-10_vilan1.ee.ethz.chcheckpoints_128_296_0.001_cosineannealinglr_adam_resnet9-lpl-0.001-thresh-25-93.6",
#     "Sep10_18-40-36_vilan1.ee.ethz.chcheckpoints_128_320_0.001_cosineannealinglr_adam_resnet9-lpl-0.001-thresh-25-93.6",
#     "Sep10_21-05-36_vilan1.ee.ethz.chcheckpoints_128_344_0.001_cosineannealinglr_adam_resnet9-lpl-0.001-thresh-25-93.6",
#     "Sep10_23-38-25_vilan1.ee.ethz.chcheckpoints_128_368_0.001_cosineannealinglr_adam_resnet9-lpl-0.001-thresh-25-93.6",
#     "Sep06_15-53-37_vilan1.ee.ethz.chresnet9-basic-adam_128_200_0.001_cosineannealinglr_adam",
#     "Sep10_09-59-35_vilan2.ee.ethz.chcheckpoints_128_567_0.0005_cosineannealinglr_adam_resnet9-lpl-0.0005-cont-200-93.6",
# ]
#
# dirs_retraining = [
#     "Sep10_12-53-59_vilan1.ee.ethz.chcheckpoints_128_224_0.001_cosineannealinglr_adam_resnet9-lpl-0.001-thresh-25-93.6",
#     "Sep10_13-44-12_vilan1.ee.ethz.chcheckpoints_128_248_0.001_cosineannealinglr_adam_resnet9-lpl-0.001-thresh-25-93.6",
#     "Sep10_14-56-42_vilan1.ee.ethz.chcheckpoints_128_272_0.001_cosineannealinglr_adam_resnet9-lpl-0.001-thresh-25-93.6",
#     "Sep10_16-32-10_vilan1.ee.ethz.chcheckpoints_128_296_0.001_cosineannealinglr_adam_resnet9-lpl-0.001-thresh-25-93.6",
#     "Sep10_18-40-36_vilan1.ee.ethz.chcheckpoints_128_320_0.001_cosineannealinglr_adam_resnet9-lpl-0.001-thresh-25-93.6",
#     "Sep10_21-05-36_vilan1.ee.ethz.chcheckpoints_128_344_0.001_cosineannealinglr_adam_resnet9-lpl-0.001-thresh-25-93.6",
#     "Sep10_23-38-25_vilan1.ee.ethz.chcheckpoints_128_368_0.001_cosineannealinglr_adam_resnet9-lpl-0.001-thresh-25-93.6",
# ]
# dir_default_training = [
#     "Sep06_15-53-37_vilan1.ee.ethz.chresnet9-basic-adam_128_200_0.001_cosineannealinglr_adam",
# ]
# dir_end_training = [
#     "Sep10_09-59-35_vilan2.ee.ethz.chcheckpoints_128_567_0.0005_cosineannealinglr_adam_resnet9-lpl-0.0005-cont-200-93.6",
# ]

dirs = [
    "Sep28_23-19-04_vilan1.ee.ethz.chresnet9-lr-0.001-amp-lut8-base-2_128_200_0.001_cosineannealinglr_adam_",
    "Sep29_00-04-08_vilan1.ee.ethz.chcheckpoints_128_223_0.001_cosineannealinglr_adam_resnet9-lpl-int8-lut-ste-amp-25-93.59",
    "Sep29_01-57-17_vilan1.ee.ethz.chcheckpoints_128_247_0.001_cosineannealinglr_adam_resnet9-lpl-int8-lut-ste-amp-25-93.59",
    "Sep29_04-43-37_vilan1.ee.ethz.chcheckpoints_128_271_0.001_cosineannealinglr_adam_resnet9-lpl-int8-lut-ste-amp-25-93.59",
    "Sep29_08-21-45_vilan1.ee.ethz.chcheckpoints_128_295_0.001_cosineannealinglr_adam_resnet9-lpl-int8-lut-ste-amp-25-93.59",
    "Sep29_13-34-38_vilan1.ee.ethz.chcheckpoints_128_319_0.001_cosineannealinglr_adam_resnet9-lpl-int8-lut-ste-amp-25-93.59",
    "Sep29_19-40-29_vilan1.ee.ethz.chcheckpoints_128_343_0.001_cosineannealinglr_adam_resnet9-lpl-int8-lut-ste-amp-25-93.59",
    "Sep30_02-00-29_vilan1.ee.ethz.chcheckpoints_128_367_0.001_cosineannealinglr_adam_resnet9-lpl-int8-lut-ste-amp-25-93.59",
    "Oct24_07-15-59_4e835acf0571checkpoints_16_1366_0.0005_cosineannealinglr_adam_resnet9-ft-p100-1000-93.59-x8-16",
]

dirs_retraining = [
    "Sep29_00-04-08_vilan1.ee.ethz.chcheckpoints_128_223_0.001_cosineannealinglr_adam_resnet9-lpl-int8-lut-ste-amp-25-93.59",
    "Sep29_01-57-17_vilan1.ee.ethz.chcheckpoints_128_247_0.001_cosineannealinglr_adam_resnet9-lpl-int8-lut-ste-amp-25-93.59",
    "Sep29_04-43-37_vilan1.ee.ethz.chcheckpoints_128_271_0.001_cosineannealinglr_adam_resnet9-lpl-int8-lut-ste-amp-25-93.59",
    "Sep29_08-21-45_vilan1.ee.ethz.chcheckpoints_128_295_0.001_cosineannealinglr_adam_resnet9-lpl-int8-lut-ste-amp-25-93.59",
    "Sep29_13-34-38_vilan1.ee.ethz.chcheckpoints_128_319_0.001_cosineannealinglr_adam_resnet9-lpl-int8-lut-ste-amp-25-93.59",
    "Sep29_19-40-29_vilan1.ee.ethz.chcheckpoints_128_343_0.001_cosineannealinglr_adam_resnet9-lpl-int8-lut-ste-amp-25-93.59",
    "Sep30_02-00-29_vilan1.ee.ethz.chcheckpoints_128_367_0.001_cosineannealinglr_adam_resnet9-lpl-int8-lut-ste-amp-25-93.59",
]

dir_default_training = [
    "Sep28_23-19-04_vilan1.ee.ethz.chresnet9-lr-0.001-amp-lut8-base-2_128_200_0.001_cosineannealinglr_adam_",
]

dir_end_training = [
    # "Sep30_10-18-51_vilan1.ee.ethz.chcheckpoints_128_666_0.0005_cosineannealinglr_adam_resnet9-lpl-int8-lut-ste-amp-cont-300-93.59",
    "Oct24_07-15-59_4e835acf0571checkpoints_16_1366_0.0005_cosineannealinglr_adam_resnet9-ft-p100-1000-93.59-x8-16"
]
print(reader)
df = reader.scalars
filtered_df = df[df["dir_name"].isin(dirs)]
print(filtered_df)


def get_hue(x):
    if x in dir_default_training:
        return "default_training"
    elif x in dir_end_training:
        return "end_training"
    else:
        return "retraining"


filtered_df["hue"] = filtered_df["dir_name"].apply(get_hue)
print(filtered_df)

test_acc = filtered_df[filtered_df["tag"] == "test/acc"]
train_acc = filtered_df[filtered_df["tag"] == "train/acc"]
train_loss = filtered_df[filtered_df["tag"] == "train/loss"]
train_lr = filtered_df[filtered_df["tag"] == "train/lr"]


def get_hue_color(x):
    if x == "default_training":
        # use custom rgb color
        return "#023047"
        # return "red"
    elif x == "end_training":
        return "#FB8500"
    else:
        return "#219EBC"


# style figure for paper
plt.style.use("seaborn-v0_8-paper")
# set font size
plt.rcParams.update({"font.size": 9})
# set font size for title
plt.rcParams.update({"axes.titlesize": 10})
# set font size for axis labels
# plot a 2x2 grid of plots
fig, axes = plt.subplots(2, 2, figsize=(5.5, 4))
# plot the first plot in the top left corner
ax = axes[0, 0]
# plot test accuracy
# use same y axis for the top two plots
ax.set_ylim([70, 102])
# add grid
ax.grid(True, linestyle="--", linewidth=0.5, color="gray")
# set title
ax.set_title("Test Accuracy [%]")
for dir in dirs:
    ax.plot(
        test_acc[test_acc["dir_name"] == dir]["step"],
        test_acc[test_acc["dir_name"] == dir]["value"],
        color=get_hue_color(get_hue(dir)),
        label="",
    )

# plot the second plot in the top right corner
ax = axes[0, 1]
ax.grid(True, linestyle="--", linewidth=0.5, color="gray")
# set title
ax.set_title("Train Accuracy [%]")
for dir in dirs:
    ax.plot(
        train_acc[train_acc["dir_name"] == dir]["step"],
        train_acc[train_acc["dir_name"] == dir]["value"],
        color=get_hue_color(get_hue(dir)),
        label="",
    )
ax.set_ylim([70, 102])

# plot the third plot in the bottom left corner
ax = axes[1, 0]
ax.grid(True, linestyle="--", linewidth=0.5, color="gray")
# make y axis logarithmic
ax.set_yscale("log")
# set title
ax.set_title("Learning Rate")
# plot learning rate
for dir in dirs:
    ax.plot(
        train_lr[train_lr["dir_name"] == dir]["step"],
        train_lr[train_lr["dir_name"] == dir]["value"],
        color=get_hue_color(get_hue(dir)),
        label="",
    )

# plot the fourth plot in the bottom right corner
ax = axes[1, 1]
ax.grid(True, linestyle="--", linewidth=0.5, color="gray")
# make y axis logarithmic
ax.set_yscale("log")
ax.set_title("Train Loss")
# plot training loss
for dir in dirs:
    ax.plot(
        train_loss[train_loss["dir_name"] == dir]["step"],
        train_loss[train_loss["dir_name"] == dir]["value"],
        color=get_hue_color(get_hue(dir)),
        label="",
    )
# make combined x axis label
fig.text(
    0.5,
    0.0,
    "Epochs",
    ha="center",
)

fig.tight_layout()

# save the figure
fig.savefig("plots_train.png", dpi=600, bbox_inches="tight")
# save also as pdf
fig.savefig("plots_train.pdf", dpi=1200, bbox_inches="tight")
