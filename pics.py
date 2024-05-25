import argparse
import json
import os
import warnings

import matplotlib.pyplot as plt


def plots(folders, legends):
    root_path = "/Users/chaconne/PycharmProjects/saved-files"
    paths, data = [], []
    for folder in folders:
        paths.append(os.path.join(root_path, folder, "lists.json"))

    for path in paths:
        with open(path, 'r') as f:
            data.append(json.load(f))

    epc_lsts = [dt["epochs"] for dt in data]
    if not len(set(tuple(epc) for epc in epc_lsts)) == 1:
        warnings.warn('Training epochs not aligned!')
        width = 15
    else:
        width = 10

    # subplots
    plt.style.use('seaborn')
    fig, axs = plt.subplots(2, 2, figsize=(width, 10))

    # train_acc plot
    for idx in range(len(legends)):
        axs[0, 0].plot(epc_lsts[idx], data[idx]['test_acc'], label=legends[idx])  # json文件搞反了
    axs[0, 0].set_title('Train Accuracy')
    axs[0, 0].set_xlabel('epochs')
    axs[0, 0].set_ylabel('train_acc')
    axs[0, 0].set_yticks(ticks=[0.1 * x for x in range(0, 11)])
    axs[0, 0].legend()

    # test_acc plot
    for idx in range(len(legends)):
        axs[0, 1].plot(epc_lsts[idx], data[idx]['train_acc'], label=legends[idx])  # json文件搞反了
    axs[0, 1].set_title('Test Accuracy')
    axs[0, 1].set_xlabel('epochs')
    axs[0, 1].set_ylabel('test_acc')
    axs[0, 1].set_yticks(ticks=[0.1 * x for x in range(0, 11)])
    axs[0, 1].legend()

    # train_loss plot
    for idx in range(len(legends)):
        axs[1, 0].plot(epc_lsts[idx], data[idx]['test_loss'], label=legends[idx])  # json文件搞反了
    axs[1, 0].set_title('Train Loss')
    axs[1, 0].set_xlabel('epochs')
    axs[1, 0].set_ylabel('train_loss')
    axs[1, 0].legend()

    # test_loss plot
    for idx in range(len(legends)):
        axs[1, 1].plot(epc_lsts[idx], data[idx]['train_loss'], label=legends[idx])  # json文件搞反了
    axs[1, 1].set_title('Test Loss')
    axs[1, 1].set_xlabel('epochs')
    axs[1, 1].set_ylabel('test_loss')
    axs[1, 1].legend()

    fig.tight_layout()
    plt.savefig("/Users/chaconne/Downloads/tmp.png")
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Folders to draw pics.")
    parser.add_argument(
        "--folders",
        nargs="+",
        type=str,
        help="Folders to compare acc and loss"
    )
    parser.add_argument(
        "--legends",
        nargs="+",
        help="Legends in the pics"
    )

    args = parser.parse_args()
    assert len(args.folders) == len(args.legends)

    plots(args.folders, args.legends)
