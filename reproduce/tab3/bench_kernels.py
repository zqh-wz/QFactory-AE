import os
import json
import tqdm
import torch
import numpy
from prettytable import PrettyTable
from functools import partial

from qfactory.testing import QFactoryLatencyTest, CuBLASLatencyTest, BitBLASLatencyTest, MarlinLatencyTest


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
    data['BitBLAS:W4:1'] = bench("BitBLAS:W4:1", partial(BitBLASLatencyTest, M=1, B=4, GS=128, SYM=0))
    data['Marlin:W4:1'] = bench("Marlin:W4:1", partial(MarlinLatencyTest, M=1, GS=128))
    data['QFactory:W4:1'] = bench("QFactory:W4:1", partial(QFactoryLatencyTest, M=1, NBITS=4, GS=128, sym=False, compile_flags=compile_flags))
    data['BitBLAS:W4:2'] = bench("BitBLAS:W4:2", partial(BitBLASLatencyTest, M=2, B=4, GS=128, SYM=0))
    data['Marlin:W4:2'] = bench("Marlin:W4:2", partial(MarlinLatencyTest, M=2, GS=128))
    data['QFactory:W4:2'] = bench("QFactory:W4:2", partial(QFactoryLatencyTest, M=2, NBITS=4, GS=128, sym=False, compile_flags=compile_flags))
    data['BitBLAS:W4:4'] = bench("BitBLAS:W4:4", partial(BitBLASLatencyTest, M=4, B=4, GS=128, SYM=0))
    data['Marlin:W4:4'] = bench("Marlin:W4:4", partial(MarlinLatencyTest, M=4, GS=128))
    data['QFactory:W4:4'] = bench("QFactory:W4:4", partial(QFactoryLatencyTest, M=4, NBITS=4, GS=128, sym=False, compile_flags=compile_flags))
    return data

def plot_table(data):
    table = PrettyTable()
    table.field_names = ["Batch Size", "1", "2", "4"]
    labels = ['BitBLAS:W4', 'Marlin:W4', 'QFactory:W4']
    for label in labels:
        row = [label.split(':')[0]]
        for nbits in [1, 2, 4]:
            speedup = numpy.array(data['cublas']) / numpy.array(data[f'{label}:{nbits}'])
            geo_mean = numpy.prod(speedup) ** (1 / len(speedup))
            row.append(f"{geo_mean:.2f}x")
        table.add_row(row)
    print(table)


if __name__ == '__main__':
    if os.path.exists('results.json'):
        with open('results.json', 'r') as f:
            data = json.load(f)
    else:
        data = collect_data()
        with open('results.json', 'w') as f:
            json.dump(data, f, indent=4)
    plot_table(data)
