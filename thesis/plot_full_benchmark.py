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

df = pd.read_csv("end_to_end_benchmark.csv", header=None, names=['name', 'size', 'duration', 'duration_cpu'])
print(df)
plt.figure(figsize=(5, 3))

encode = df[df['name'] == 'encode'][['size', 'duration']]
decode = df[df['name'] == 'decode'][['size', 'duration']]
encode_cpu = df[df['name'] == 'encode'][['size', 'duration_cpu']]
decode_cpu = df[df['name'] == 'decode'][['size', 'duration_cpu']]

e = plt.plot(encode['size'], encode['duration'], label='Encode', marker='o', linestyle='-')
plt.plot(encode_cpu['size'], encode_cpu['duration_cpu'], label='Encode CPU', marker='x', linestyle='--', color=e[0].get_color())
d = plt.plot(decode['size'], decode['duration'], label='Decode', marker='o', linestyle='-')
plt.plot(decode_cpu['size'], decode_cpu['duration_cpu'], label='Decode CPU', marker='x', linestyle='--', color=d[0].get_color())

plt.yscale('log', base=2)
plt.xscale('log', base=2)
plt.grid()

plt.xlabel('file size')
plt.ylabel('time (ns)')

plt.xticks([2**i for i in range(0, 8)])
plt.gca().set_xticklabels([f"{2**i} GB" for i in range(0, 8)])
plt.yticks([2**i for i in range(30, 44)])

plt.legend(loc='upper left', fontsize='small')
plt.savefig('end_to_end_benchmark.pgf', bbox_inches='tight')
