import matplotlib
from matplotlib import pyplot as plt
import pandas as pd

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

df = pd.read_csv("benchmark_poly.csv", header=None, names=['name', 'size', 'time'])

agg = df.groupby(['name', 'size'])['time'].agg(['mean', 'std']).reset_index()

plt.figure(figsize=(5, 3))

color = None
for i, name in enumerate(['oversample', 'precompute_oversampling', 'recover', 'precompute_recovery', 'oversample_newton']):
    results = agg[agg['name'] == name]
    print(results)
    name = name.replace('_', ' ').title()
    line = plt.errorbar(results['size'], results['mean'], yerr=results['std'], label=name, marker='o' if i % 2 == 0 else 'x', color=color, linestyle='-' if i % 2 == 0 else '--')
    if i % 2 == 0:
        color = line[0].get_color()
    else:
        color = None

plt.yscale('log', base=2)
plt.xscale('log', base=2)
plt.grid()
plt.xlabel('symbols')
plt.ylabel('time (ns)')
plt.xticks([2**i for i in range(8, 18)])
plt.yticks([2**i for i in range(11, 35, 2)])
plt.legend(loc='upper right', fontsize='small')
plt.savefig('benchmarks_log_poly.pgf', bbox_inches='tight')
