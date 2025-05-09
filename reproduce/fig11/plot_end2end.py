import re
import numpy
import matplotlib.pyplot as plt

models = ['L\n7B', 'L\n13B', 'L\n70B', 'Q\n7B', 'Q\n14B', 'Q\n32B', 'Q\n72B']

def parse(filename, regex):
    with open(filename, 'r') as f:
        lines = f.readlines()
    data = []
    for line in lines:
        match = re.search(regex, line)
        if match:
            data.append(float(match.group(1)))
    return data

def collect_data():
    data = {}
    def to_tps(data):
        return numpy.array([128 / x for x in data])
    data['bitblas'] = to_tps(parse('bitblas.log', r'Avg latency: (\d+\.\d+) seconds'))
    data['vllm'] = to_tps(parse('vllm.log', r'Avg latency: (\d+\.\d+) seconds'))
    data['qfactory'] = to_tps(parse('qfactory.log', r'Avg latency: (\d+\.\d+) seconds'))
    data['llama_cpp'] = numpy.array(parse('llama_cpp.log', r".*tg128.*\|\s*([\d.]+)(?:\s±\s[\d.]+)?\s*\|"))
    for baseline in ['bitblas', 'vllm', 'qfactory', 'llama_cpp']:
        data[f'{baseline}:4'] = data[baseline][:7]
        data[f'{baseline}:2'] = data[baseline][7:]
    return data

def plot(data):
    x = numpy.arange(len(models))

    speedup_llama_cpp_4bit = data['llama_cpp:4'] / data['qfactory:4']
    speedup_vllm_4bit = data['vllm:4'] / data['qfactory:4']
    speedup_bitblas_4bit = data['bitblas:4'] / data['qfactory:4']
    speedup_qfactory_4bit = numpy.ones(len(models))

    speedup_llama_cpp_2bit = data['llama_cpp:2'] / data['qfactory:2']
    speedup_bitblas_2bit = data['bitblas:2'] / data['qfactory:2']
    speedup_qfactory_2bit = numpy.ones(len(models) - 1)

    plt.rcParams.update({'font.size': 8})

    fig, (ax1, ax2) = plt.subplots(figsize=(4.5, 2.4), ncols=2, sharey=True)

    mask = list(range(0, len(models)))

    width = 0.2  # Bar width
    ax1.bar(x[mask] - 1.5 * width, speedup_llama_cpp_4bit[mask], width, label='llama.cpp', color='tab:red')
    ax1.bar(x[mask] - 0.5 * width, speedup_vllm_4bit[mask], width, label='Marlin', color='#9355B0')
    ax1.bar(x[mask] + 0.5 * width, speedup_bitblas_4bit[mask], width, label='BitBLAS', color='#78D3AC')
    ax1.bar(x[mask] + 1.5 * width, speedup_qfactory_4bit[mask], width, label='QFactory', color='#EF7F51')
    for i, v in zip(x[mask], speedup_qfactory_4bit[mask]):
        ax1.text(i, v + 0.05, f"({data['qfactory:4'][i]:.1f})", ha='center', fontsize=6, rotation=30)
    
    ax1.set_ylabel('Rel. Generation Speed')
    ax1.set_xticks(x)
    ax1.set_xticklabels(models)
    ax1.axhline(1, color='gray', linestyle='--')
    ax1.set_xlim(-0.5 - width, 6.5 + width)
    ax1.set_ylim(0.0, 1.26)
    ax1.set_yticks(numpy.arange(0, 1.1, 0.2))
    ax1.set_title('4-bit Quantization')

    ax2.bar(x[1:] - width, speedup_llama_cpp_2bit, width, label='llama.cpp', color='tab:red')
    ax2.bar(x[1:], speedup_bitblas_2bit, width, label='BitBLAS', color='#78D3AC')
    ax2.bar(x[1:] + width, speedup_qfactory_2bit, width, label='QFactory', color='#EF7F51')
    for i, v in zip(x[1:], speedup_qfactory_2bit):
        ax2.text(i, v + 0.05, f"({data['qfactory:2'][i - 1]:.1f})", ha='center', fontsize=6, rotation=30)
    ax2.text(0, 0.1, 'N/A', ha='center', fontsize=8, color='tab:red', weight='bold')

    ax2.set_xticks(x)
    ax2.set_xticklabels(models)
    ax2.axhline(1, color='gray', linestyle='--')
    ax2.set_xlim(-0.5 - width, 6.5 + width)
    ax2.set_title('2-bit Quantization')

    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc='upper center',
        bbox_to_anchor=(0.5, 0.98),
        ncol=4
    )
    plt.tight_layout(rect=[0, 0, 1, 0.90])

    plt.savefig(f'end2end_h100.pdf')


if __name__ == '__main__':
    data = collect_data()
    plot(data)
