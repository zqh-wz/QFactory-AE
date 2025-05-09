import os
import copy
import json
import tqdm
import torch
import numpy
import matplotlib.pyplot as plt
from functools import partial

from qfactory.testing import QFactoryLatencyTest, CuBLASLatencyTest, BitBLASLatencyTest, MarlinLatencyTest


N_K = [(int(shape[0]), int(shape[1])) for shape in json.load(open('../../data/nkg_shapes.json', 'r'))["squares-pure"]]

def bench(name, Test):
    return [Test(N=n, K=k).latency() for n, k in tqdm.tqdm(N_K, desc=f"Benchmarking {name}")]

def collect_data():
    data = {}
    compile_flags={
        "enable_transform": True,
        "enable_schedule": False,
        "enable_lower": True,
    }
    data['BitBLAS:W8:128'] = bench("BitBLAS:W8:128", partial(BitBLASLatencyTest, M=1, B=8, GS=128, SYM=0))
    data['BitBLAS:W4:128'] = bench("BitBLAS:W4:128", partial(BitBLASLatencyTest, M=1, B=4, GS=128, SYM=0))
    data['BitBLAS:W2:128'] = bench("BitBLAS:W2:128", partial(BitBLASLatencyTest, M=1, B=2, GS=128, SYM=0))
    data['QFactory:W8:128'] = bench("QFactory:W8:128", partial(QFactoryLatencyTest, M=1, NBITS=8, GS=128, sym=False, compile_flags=compile_flags))
    data['QFactory:W4:128'] = bench("QFactory:W4:128", partial(QFactoryLatencyTest, M=1, NBITS=4, GS=128, sym=False, compile_flags=compile_flags))
    data['QFactory:W2:128'] = bench("QFactory:W2:128", partial(QFactoryLatencyTest, M=1, NBITS=2, GS=128, sym=False, compile_flags=compile_flags))
    return data

def get_memory_footprint(M, N, K, B, GS, kernel):
    if GS == -1:
        if kernel == "cuBLAS":
            return (M * K + K * N + M * N) * 2
        else:
            return (M * K + M * N) * 2 + (K * N) * (B / 8)
    else:
        return (M * K + M * N) * 2 + (K * N) * ((B / 8) + (2 + (B / 8)) / GS)

def plot(data):
    plt.rcParams.update({'font.size': 8})
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(4.2, 2.0), sharey=True)
    axes = [ax1, ax2]

    bar_labels1 = ['BitBLAS:W8:128', 'BitBLAS:W4:128', 'BitBLAS:W2:128']
    bar_labels2 = ['QFactory:W8:128', 'QFactory:W4:128', 'QFactory:W2:128']

    for bid, baseline in enumerate([bar_labels1, bar_labels2]):
        ax = axes[bid]
        for _kernel in baseline:
            kernel, nbits, gs = _kernel.split(":")
            nbits, gs = int(nbits[1:]), int(gs)
            x = [numpy.log(N * K * nbits / 8 / (1024 ** 2)) / numpy.log(2) for N, K in N_K]
            d = [get_memory_footprint(1, N, K, nbits, gs, kernel) / data[_kernel][i] / 10 ** 9 for i, (N, K) in enumerate(N_K)]
            color = {
                8: "#b1400d",
                4: "#d55e00",
                2: "#ff7c00",
            }[nbits]
            marker = {
                8: 's',
                4: '^',
                2: 'o',
            }[nbits]
            ax.plot(x[1:], d[1:], label=kernel, color=color, linestyle='--', marker=marker, markersize=5, linewidth=0.9)
    ax1.set_title('BitBLAS')
    ax2.set_title('QFactory')
    ax1.set_ylabel('Bandwidth (GB/s)')
    ax1.set_ylim(0, 2000)
    ax1.set_yticks(range(0, 2001, 400))
    for ax in axes:
        ax.set_xticks(list(range(1, 10, 2)))
        ax.set_xticklabels([f"{2**i}" for i in range(1, 10, 2)])
        ax.set_xlabel('Weight Matrix Size (MB)')
    handles, labels = ax1.get_legend_handles_labels()
    handles_copy = copy.deepcopy(handles)
    # for handle in handles_copy:
    #     handle.set_color('black')
    fig.legend(
        handles_copy,
        ["8-bit", "4-bit", "2-bit"],
        loc='upper center',
        bbox_to_anchor=(0.52, 0.99),
        ncol=4
    )
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig("fig10.pdf")

if __name__ == '__main__':
    if os.path.exists('results.json'):
        with open('results.json', 'r') as f:
            data = json.load(f)
    else:
        data = collect_data()
        with open('results.json', 'w') as f:
            json.dump(data, f, indent=4)
    plot(data)
