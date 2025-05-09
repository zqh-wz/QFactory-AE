import os
import json
import tqdm
import torch
import numpy
import matplotlib.pyplot as plt
from functools import partial

from qfactory.testing import QFactoryLatencyTest, CuBLASLatencyTest, BitBLASLatencyTest


N_K = [(int(shape[0]), int(shape[1])) for shape in json.load(open('../../data/nkg_shapes.json', 'r'))["operator"]]

def bench(name, Test):
    return [Test(N=n, K=k).latency() for n, k in tqdm.tqdm(N_K, desc=f"Benchmarking {name}")]

def collect_data():
    data = {}
    data['cublas'] = bench("cublas", partial(CuBLASLatencyTest, M=1))
    compile_flags={
        "enable_transform": True,
        "enable_schedule": False,
        "enable_lower": True,
    }
    data['BitBLAS:W8'] = bench("BitBLAS:W8", partial(BitBLASLatencyTest, M=1, B=8, GS=128, SYM=0))
    data['QFactory:W8'] = bench("QFactory:W8", partial(QFactoryLatencyTest, M=1, NBITS=8, GS=128, sym=False, compile_flags=compile_flags))
    data['BitBLAS:W4'] = bench("BitBLAS:W4", partial(BitBLASLatencyTest, M=1, B=4, GS=128, SYM=0))
    data['QFactory:W4'] = bench("QFactory:W4", partial(QFactoryLatencyTest, M=1, NBITS=4, GS=128, sym=False, compile_flags=compile_flags))
    data['BitBLAS:W2'] = bench("BitBLAS:W2", partial(BitBLASLatencyTest, M=1, B=2, GS=128, SYM=0))
    data['QFactory:W2'] = bench("QFactory:W2", partial(QFactoryLatencyTest, M=1, NBITS=2, GS=128, sym=False, compile_flags=compile_flags))
    return data

def plot_figure(data):
    plt.figure(figsize=(7.2, 1.25))
    plt.rcParams.update({'font.size': 8})
    plt.rcParams.update({'hatch.linewidth': 0.5})
    bar_width = 0.12
    x = numpy.arange(len(N_K) + 1)

    bar_labels = ['BitBLAS:W8', 'QFactory:W8', 'BitBLAS:W4', 'QFactory:W4', 'BitBLAS:W2', 'QFactory:W2']
    for i, label in enumerate(bar_labels):
        kernel, nbits = label.split(':')
        hatch = {
            "W8": "----",
            "W4": "||||",
            "W2": "xxxx",
        }[nbits]
        color = {
            "QFactory": "#f39e7c",
            "BitBLAS": "#97c6e2",
        }[kernel]
        speedup = numpy.array(data['cublas']) / numpy.array(data[label])
        geo_mean = numpy.prod(speedup) ** (1 / len(speedup))
        plt.bar(x + i * bar_width, numpy.append(speedup, geo_mean), bar_width, label=label, hatch=hatch, color=color, edgecolor="black", linewidth=0.9)
    plt.xticks(x + (len(bar_labels) - 1) * bar_width / 2, [f"M{i}" if i < len(x) - 1 else "GeoMean" for i in range(len(x))])
    plt.ylabel('Rel. Speedup')
    plt.yticks([0, 2, 4, 6, 8, 10], [0, 2, 4, 6, 8, 10]) # V100
    handles, labels = plt.gca().get_legend_handles_labels()
    fig = plt.gcf()
    fig.legend(handles, labels, loc='upper center', ncols=7, fontsize=6, bbox_to_anchor=(0.5, 0.98))
    plt.tight_layout(rect=[0, 0, 1, 0.90])
    plt.xlim(0 - bar_width * 1.5, len(x) - 1 + bar_width * (len(bar_labels) + 0.5))
    plt.savefig('fig8.pdf')


if __name__ == '__main__':
    if os.path.exists('results.json'):
        with open('results.json', 'r') as f:
            data = json.load(f)
    else:
        data = collect_data()
        with open('results.json', 'w') as f:
            json.dump(data, f, indent=4)
    plot_figure(data)
