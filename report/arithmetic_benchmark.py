from os import system
from pathlib import Path
import csv

POW = 26
SYMBOL_NUM = 2 ** POW


FILES = [Path(x).with_suffix(f".{POW}.csv") for x in ["math_benchmarks_default", "math_benchmarks_no_clmul", "math_benchmarks_no_clmul_no_lzcnt"]]

if not FILES[0].exists():
    system(f"cargo run --release -F math_benchmarks math_benchmarks {SYMBOL_NUM} > {FILES[0]}")
if not FILES[1].exists():
    system(f"cargo run --release -F math_benchmarks -F no_clmul_check math_benchmarks {SYMBOL_NUM} > {FILES[1]}")
if not FILES[2].exists():
    system(f"cargo run --release -F math_benchmarks -F no_clmul_check -F no_lzcnt_check math_benchmarks {SYMBOL_NUM} > {FILES[2]}")

results = [[float("nan")] * 3 for _ in range(3)]
for i, file in enumerate(FILES):
    with open(file, "r", encoding="ascii") as f:
        reader = csv.reader(f)
        for j, name in enumerate(['multiplication', 'inversion', 'inversion_by_pow']):
            row = next(reader)
            assert row[0] == name
            results[i][j] = round(float(row[2]), ndigits=2)

table_results = f"""Multiplication     & {results[0][0]} ns & {results[1][0]} ns & {results[2][0]} ns \\\\
Inversion (Euclid) & {results[0][1]} ns & {results[1][1]} ns & {results[2][1]} ns \\\\
Inversion (Pow)    & {results[0][2]} ns & {results[1][2]} ns & {results[2][2]} ns \\\\
"""

table = """\\begin{table}[hbt]
\\centering
\\caption{Finite field arithmetic benchmark results, averaged over """ + f"$2^{{{POW}}}$"  + """ operations.}
\\begin{tabular}{lrrr}
\\toprule
\\textbf{Operation} & \\textbf{Default} & \\textbf{No CLMUL} & \\textbf{No CLMUL \\& LZCNT} \\\\
\\midrule
""" \
+ table_results + \
"""\\bottomrule
\\end{tabular}
\\end{table}
"""

Path("arithmetic_benchmark.tex").write_text(table, encoding="ascii")
