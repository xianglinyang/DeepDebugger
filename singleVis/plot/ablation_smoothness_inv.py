
import os
import json

import numpy as np
import pandas as pd
import matplotlib as mpl
import seaborn as sns


def main():
    # hyperparameters
    datasets = ["mnist", "fmnist", "cifar10"]
    selected_epochs_dict = {"mnist":[4, 12, 20],"fmnist":[10,30,50], "cifar10":[40, 120,200]}
    selected_epochs_dict = {"mnist":[5],"fmnist":[2,6,11], "cifar10":[3,9,18,41]}

    col = np.array(["dataset", "method", "type", "hue","period", "eval"])
    df = pd.DataFrame({}, columns=col)

    for i in range(len(datasets)): # dataset
        dataset = datasets[i]
        data = np.array([])
        selected_epochs = selected_epochs_dict[dataset]
        
        # DeepDebugger smoothness
        eval_path = "/home/xianglin/projects/DVI_data/resnet18_{}/Model/test_evaluation_hybrid.json".format(dataset)
        with open(eval_path, "r") as f:
                eval = json.load(f)
        for epoch_id in range(len(selected_epochs)):
            epoch = selected_epochs[epoch_id]
            nn_train = round(eval["ppr_train"][str(epoch)], 3)
            nn_test = round(eval["ppr_test"][str(epoch)], 3)

            if len(data) == 0:
                data = np.array([[dataset, "DeepDebugger", "Train", "DeepDebugger-Train", "{}".format(str(epoch_id)), nn_train]])
            else:
                data = np.concatenate((data, np.array([[dataset, "DeepDebugger", "Train", "DeepDebugger-Train",  "{}".format(str(epoch_id)), nn_train]])), axis=0)
            data = np.concatenate((data, np.array([[dataset, "DeepDebugger", "Test", "DeepDebugger-Test", "{}".format(str(epoch_id)), nn_test]])), axis=0)
        
        # DeepDebugger without smoothness
        eval_path = "/home/xianglin/projects/DVI_data/resnet18_{}/Model/without_smoothness/test_evaluation_hybrid.json".format(dataset)
        with open(eval_path, "r") as f:
                eval = json.load(f)
        for epoch_id in range(len(selected_epochs)):
            epoch = selected_epochs[epoch_id]
            nn_train = round(eval["ppr_train"][str(epoch)], 3)
            nn_test = round(eval["ppr_test"][str(epoch)], 3)

            data = np.concatenate((data, np.array([[dataset, "no_Smoothness", "Train", "no_Smoothness-Train", "{}".format(str(epoch_id)), nn_train]])), axis=0)
            data = np.concatenate((data, np.array([[dataset, "no_Smoothness", "Test", "no_Smoothness-Test", "{}".format(str(epoch_id)), nn_test]])), axis=0)
        
        # DeepDebugger without tl
        eval_path = "/home/xianglin/projects/DVI_data/resnet18_{}/Model/without_tl/test_evaluation_hybrid.json".format(dataset)
        with open(eval_path, "r") as f:
                eval = json.load(f)
        for epoch_id in range(len(selected_epochs)):
            epoch = selected_epochs[epoch_id]
            nn_train = round(eval["ppr_train"][str(epoch)], 3)
            nn_test = round(eval["ppr_test"][str(epoch)], 3)

            data = np.concatenate((data, np.array([[dataset, "no_TL", "Train", "no_TL-Train","{}".format(str(epoch_id)), nn_train]])), axis=0)
            data = np.concatenate((data, np.array([[dataset, "no_TL", "Test", "no_TL-Test", "{}".format(str(epoch_id)), nn_test]])), axis=0)

        df_tmp = pd.DataFrame(data, columns=col)
        df = df.append(df_tmp, ignore_index=True)
        df[["period"]] = df[["period"]].astype(int)
        df[["eval"]] = df[["eval"]].astype(float)

    df.to_excel("./plot_results/ablation_smoothness_ppr.xlsx")

    pal20c = sns.color_palette('tab20c', 20)
    sns.set_theme(style="whitegrid", palette=pal20c)
    hue_dict = {
        "no_TL-Train": pal20c[0],
        "no_Smoothness-Train": pal20c[4],
        "DeepDebugger-Train": pal20c[8],

        "no_TL-Test": pal20c[3],
        "no_Smoothness-Test": pal20c[7],
        "DeepDebugger-Test": pal20c[11],
    }
    sns.palplot([hue_dict[i] for i in hue_dict.keys()])

    axes = {'labelsize': 10,
            'titlesize': 10,}
    mpl.rc('axes', **axes)
    mpl.rcParams['xtick.labelsize'] = 10


    hue_list = ["no_TL-Train", "no_TL-Test", "no_Smoothness-Train", "no_Smoothness-Test", "DeepDebugger-Train", "DeepDebugger-Test"]

    fg = sns.catplot(
        x="period",
        y="eval",
        hue="hue",
        hue_order=hue_list,
        # order = [1, 2, 3, 4, 5],
        # row="method",
        col="dataset",
        ci=0.001,
        height=2.5, #2.65,
        aspect=1.0,#3,
        data=df,
        kind="bar",
        palette=[hue_dict[i] for i in hue_list],
        legend=True
    )
    sns.move_legend(fg, "lower center", bbox_to_anchor=(.42, 0.92), ncol=2, title=None, frameon=False)
    mpl.pyplot.setp(fg._legend.get_texts(), fontsize='10')

    axs = fg.axes[0]
    # max_ = df["eval"].max()
    # min_ = df["eval"].min()
    # axs[0].set_ylim(0., max_*1.1)
    axs[0].set_title("MNIST")
    axs[1].set_title("FMNIST")
    axs[2].set_title("CIFAR-10")

    (fg.despine(bottom=False, right=False, left=False, top=False)
        # .set_xticklabels(['Begin', 'Mid', 'End'])
        .set_axis_labels("Period", "")
        )
    # fg.fig.suptitle("NN preserving property")

    fg.savefig(
        "./plot_results/ablation_smoothness_ppr.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.0,
        transparent=True,
    )


if __name__ == "__main__":
    main()

