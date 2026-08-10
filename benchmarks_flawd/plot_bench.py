#!/usr/bin/env python3
"""
Gera gráficos a partir dos CSVs do benchmark glist vs gnx vs CPU.
Uso:  python3 plot_bench.py
Lê:  /tmp/bench_map.csv  e  /tmp/bench_reduce.csv
"""

import csv, os, math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

COLORS = {
    "glist":    "#2196F3",   # azul
    "gnx_tx":   "#FF9800",   # laranja
    "gnx_list": "#9C27B0",   # roxo
    "cpu":      "#4CAF50",   # verde
}
LABELS = {
    "glist":    "glist (GPU, lista→GPU→lista)",
    "gnx_tx":   "gnx (GPU, tensor pré-criado)",
    "gnx_list": "gnx (GPU, lista→tensor→GPU→tensor)",
    "cpu":      "CPU Elixir (Enum)",
}

def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({k: int(v) if k != "n" else int(v) for k, v in r.items()})
    return rows

def ratio_table(rows, baseline="gnx_tx"):
    print(f"\n  Razão  col / {baseline}:")
    print(f"  {'n':>10}  {'glist':>10}  {'gnx_tx':>10}  {'gnx_list':>10}  {'cpu':>10}")
    for r in rows:
        base = r[baseline]
        def fmt(k): return f"{r[k]/base:.2f}x"
        print(f"  {r['n']:>10}  {fmt('glist'):>10}  {fmt('gnx_tx'):>10}  {fmt('gnx_list'):>10}  {fmt('cpu'):>10}")

def plot_pair(map_rows, reduce_rows, out="benchmark_result.png"):
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("Benchmark end-to-end: glist vs gnx vs CPU\n"
                 "(mediana de 7 iterações — escala log/log)", fontsize=14, fontweight="bold")

    specs = [
        (axes[0, 0], map_rows,    "MAP — tempo (µs)",     "log"),
        (axes[0, 1], reduce_rows, "REDUCE — tempo (µs)",  "log"),
        (axes[1, 0], map_rows,    "MAP — razão vs gnx_tx","linear"),
        (axes[1, 1], reduce_rows, "REDUCE — razão vs gnx_tx","linear"),
    ]

    for ax, rows, title, yscale in specs:
        ns = [r["n"] for r in rows]
        is_ratio = "razão" in title

        for key in ["glist", "gnx_tx", "gnx_list", "cpu"]:
            if is_ratio:
                base = [r["gnx_tx"] for r in rows]
                vals = [r[key] / b for r, b in zip(rows, base)]
            else:
                vals = [r[key] for r in rows]

            ax.plot(ns, vals,
                    marker="o", linewidth=2, markersize=5,
                    color=COLORS[key], label=LABELS[key])

        if not is_ratio:
            ax.set_yscale("log")
            ax.set_ylabel("tempo (µs)")
        else:
            ax.axhline(1.0, color=COLORS["gnx_tx"], linestyle="--", alpha=0.4)
            ax.set_ylabel("razão (1.0 = igual ao gnx_tx)")

        ax.set_xscale("log")
        ax.set_xlabel("n (elementos)")
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, which="both", linestyle="--", alpha=0.4)
        ax.xaxis.set_major_formatter(ticker.FuncFormatter(
            lambda x, _: f"{int(x):,}".replace(",", ".")))

    plt.tight_layout()
    plt.savefig(out, dpi=150)
    print(f"\nGráfico salvo: {out}")
    plt.show()

if __name__ == "__main__":
    map_path    = "/tmp/bench_map.csv"
    reduce_path = "/tmp/bench_reduce.csv"

    if not os.path.exists(map_path):
        print(f"Arquivo não encontrado: {map_path}")
        print("Rode primeiro:  mix run benchmark_glist_vs_gnx_vs_cpu.exs")
        exit(1)

    map_rows    = load(map_path)
    reduce_rows = load(reduce_path)

    print("=== MAP ===")
    ratio_table(map_rows)
    print("\n=== REDUCE ===")
    ratio_table(reduce_rows)

    plot_pair(map_rows, reduce_rows, out="/tmp/benchmark_result.png")