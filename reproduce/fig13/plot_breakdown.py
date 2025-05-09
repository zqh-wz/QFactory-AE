import os
import json
import tqdm
import torch
import numpy
import matplotlib.pyplot as plt
from functools import partial

from qfactory.testing import QFactoryLatencyTest, BitBLASLatencyTest, MarlinLatencyTest


N_K = [(int(shape[0]), int(shape[1])) for shape in json.load(open('../../data/nkg_shapes.json', 'r'))["breakdown"]]

def bench(name, Test):
    return [Test(N=n, K=k).latency() for n, k in tqdm.tqdm(N_K, desc=f"Benchmarking {name}")]

def collect_data():
    data = {}
    data['BitBLAS:W4'] = bench("BitBLAS:W4", partial(BitBLASLatencyTest, M=1, B=4, GS=128, SYM=0))
    data['BitBLAS:W2'] = bench("BitBLAS:W2", partial(BitBLASLatencyTest, M=1, B=2, GS=128, SYM=0))
    data['Marlin:W4'] = bench("Marlin:W4", partial(MarlinLatencyTest, M=1, GS=128))
    for label in ['base', 'template', 'trans', 'full']:
        os.environ["QFACTORY_CONFIG"] = f"best_{label}.json"
        base_impl = (label == 'base')
        data[f'{label}:W4'] = bench(f"{label}:W4", partial(QFactoryLatencyTest, M=1, NBITS=4, GS=128, sym=False, base_impl=base_impl))
        data[f'{label}:W2'] = bench(f"{label}:W2", partial(QFactoryLatencyTest, M=1, NBITS=2, GS=128, sym=False, base_impl=base_impl))
    return data

def plot_figure(data):
    bar_width = 0.14
    x = numpy.arange(len(N_K))

    plt.rcParams.update({'font.size': 9})
    plt.rcParams.update({'hatch.linewidth': 0.5})
    fig, (ax1, ax2) = plt.subplots(figsize=(4.5, 2.4), ncols=2, sharey=True)

    bar_labels = [
        ['BitBLAS', 'Marlin', 'base', 'template', 'trans', 'full'],
        ['BitBLAS', 'base', 'template', 'trans', 'full'],
    ]

    for idx, nbits in enumerate([4, 2]):
        ax = [ax1, ax2][idx]
        ax.axhline(y=1, color="black", linestyle="--", linewidth=0.9)
        ref_bitblas = data[f'BitBLAS:W{nbits}']
        bar_label = bar_labels[idx]
        for i, label in enumerate(bar_label):
            d = data[f'{label}:W{nbits}']
            c = {
                "BitBLAS": "#78d3ac",
                "Marlin": "#9355b0",
                "base": "#ffd0a5",
                "template": "#ff9e45",
                "trans": "#e46d00",
                "full": "#c05b00",
            }[label]
            h = {
                "BitBLAS": None,
                "Marlin": None,
                "base": "--",
                "template": "++",
                "trans": "//",
                "full": "xxx",
            }[label]
            l = {
                "BitBLAS": "BitBLAS",
                "Marlin": "Marlin",
                "base": "QFactory-Base",
                "template": "+TemplateGen",
                "trans": "+Transformation",
                "full": "+Scheduling",
            }[label]
            ax.bar(x + i * bar_width, numpy.array(d) / numpy.array(ref_bitblas), bar_width, label=l, hatch=h, color=c, edgecolor="black", linewidth=0.5)
            ax.set_xticks(x + (len(bar_label) - 1) * bar_width / 2, ["M2", "M3", "M4"])
            if label == "BitBLAS":
                for j, di in enumerate(d):
                    ax.text(x[j] + i * bar_width, di / ref_bitblas[j] + 0.06, f"{di * 1e6:.1f} us", ha='center', va='bottom', fontsize=6.5, rotation=90)

    ax1.set_ylabel('Normalized Latency')
    ax1.set_title('Asym. W4A16 Kernel', fontsize=8.5)
    ax2.set_title('Asym. W2A16 Kernel', fontsize=8.5)

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc='upper center',
        bbox_to_anchor=(0.56, 0.98),
        ncol=3,
        fontsize=8
    )
    plt.tight_layout(rect=[-0.02, 0, 1, 0.83])
    output_file = f"breakdown-h100.pdf"
    plt.savefig(output_file)

if __name__ == "__main__":
    if os.path.exists('results.json'):
        with open('results.json', 'r') as f:
            data = json.load(f)
    else:
        data = collect_data()
        with open('results.json', 'w') as f:
            json.dump(data, f, indent=4)
    plot_figure(data)
