from matplotlib import pyplot as plt
import pandas as pd
from sys import argv

df = pd.read_csv(argv[1], header=None, names=['name', 'size', 'time'])

agg = df.groupby(['name', 'size'])['time'].agg(['mean', 'min']).reset_index()

for name in ['oversample', 'recover']:
    results = agg[agg['name'] == name]
    print(results)
    name = name.title()
    plt.plot(results['size'], results['mean'], label=f'{name} (mean)')
    plt.plot(results['size'], results['min'], label=f'{name} (min)', linestyle='--')

plt.xscale('log', base=2)
plt.yscale('log', base=2)
plt.xlabel('bytes')
plt.ylabel('time (ns)')
plt.legend()
plt.tight_layout()
plt.show()
