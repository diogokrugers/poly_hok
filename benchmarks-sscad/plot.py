#!/usr/bin/env python3
"""
benchmarks-sscad/plot.py

Gera gráficos a partir dos CSVs em benchmarks-sscad/resultados/, produzidos
pelos benchmarks .exs (dot_product, black_scholes, nearest_neighbor, saxpy,
nbodies).

Rode isto na SUA máquina (não precisa Elixir/CUDA instalado, só Python +
matplotlib) depois de copiar a pasta benchmarks-sscad/resultados/ de volta
da máquina onde os benchmarks rodaram (ex.: scp da máquina do laboratório).

USO
    python3 plot.py <nome_do_benchmark>
    python3 plot.py <nome_do_benchmark> --formato png
    python3 plot.py --all
    python3 plot.py -help

BENCHMARKS RECONHECIDOS
    dot_product         benchmarks-sscad/resultados/dot_product.csv
    black_scholes        benchmarks-sscad/resultados/black_scholes.csv
    nearest_neighbor     benchmarks-sscad/resultados/nearest_neighbor.csv
    saxpy                benchmarks-sscad/resultados/saxpy.csv
    nbodies              benchmarks-sscad/resultados/nbodies.csv

Cada CSV precisa ter a coluna n e pelo menos flawd,gnx,cpu (tempos em
microssegundos, mesmo formato que BenchHelpers.save_csv/3 grava).
dot_product e nearest_neighbor têm ainda uma coluna flawd_fusion (o mesmo
pipeline map2+reduce rodando como um único kernel via Fusion.with_fusion);
quando essa coluna está presente, o script a inclui automaticamente nos
dois gráficos — não é preciso indicar nada na linha de comando.

SAÍDA
    Para cada benchmark, dois arquivos PNG em benchmarks-sscad/resultados/:
        <nome>_tempo.png     — tempo de execução vs. N (escala log-log)
        <nome>_speedup.png   — speedup de cada série (exceto CPU) em relação
                                à CPU vs. N (escala log-x, linha speedup=1
                                marcada)

EXEMPLOS
    python3 plot.py dot_product
    python3 plot.py nbodies
    python3 plot.py --all
"""

import argparse
import csv
import sys
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")  # não depende de display/backend interativo
    import matplotlib.pyplot as plt
except ImportError:
    print(
        "Erro: matplotlib não encontrado. Instale com:\n"
        "    pip3 install matplotlib\n"
        "(ou: python3 -m pip install matplotlib)",
        file=sys.stderr,
    )
    sys.exit(1)


HERE = Path(__file__).resolve().parent
RESULTS_DIR = HERE / "resultados"

# nome_do_benchmark -> (nome do arquivo CSV, título legível para os gráficos)
BENCHMARKS = {
    "dot_product": ("dot_product.csv", "Produto Escalar (Dot Product)"),
    "black_scholes": ("black_scholes.csv", "Black-Scholes"),
    "nearest_neighbor": ("nearest_neighbor.csv", "Nearest Neighbor"),
    "saxpy": ("saxpy.csv", "Saxpy (y = a*x + y)"),
    "nbodies": ("nbodies.csv", "N-Bodies"),
}

# Cores e rótulos fixos por série, consistentes em todos os gráficos do
# artigo. A ordem aqui é a ordem de desenho/legenda. "cpu" nunca aparece
# no gráfico de speedup (é a própria referência, linha speedup=1).
SERIES = [
    ("flawd", "flawd (listas Elixir)", "#2563eb", "o"),
    ("flawd_fusion", "flawd + fusão de kernel", "#16a34a", "D"),
    ("gnx", "gnx (PolyHok padrão)", "#f97316", "s"),
    ("cpu", "CPU (Enum)", "#6b7280", "^"),
]


def read_csv(path):
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        rows = []
        for row in reader:
            parsed = {"n": int(row["n"])}
            for key in fieldnames:
                if key == "n":
                    continue
                parsed[key] = float(row[key])
            rows.append(parsed)
    rows.sort(key=lambda r: r["n"])
    # colunas de série presentes neste CSV, na ordem definida em SERIES
    present_keys = [k for k, _, _, _ in SERIES if rows and k in rows[0]]
    return rows, present_keys


def us_to_ms(rows, key):
    return [r[key] / 1000.0 for r in rows]


def plot_tempo(bench_key, title, rows, present_keys, out_path, fmt):
    ns = [r["n"] for r in rows]

    fig, ax = plt.subplots(figsize=(7, 5))

    for key, label, color, marker in SERIES:
        if key not in present_keys:
            continue
        ax.plot(
            ns,
            us_to_ms(rows, key),
            label=label,
            color=color,
            marker=marker,
            linewidth=1.8,
            markersize=5,
        )

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Tamanho da entrada (N, escala log)")
    ax.set_ylabel("Tempo de execução (ms, escala log)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.6)

    fig.tight_layout()
    out_file = out_path / f"{bench_key}_tempo.{fmt}"
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    return out_file


def plot_speedup(bench_key, title, rows, present_keys, out_path, fmt):
    ns = [r["n"] for r in rows]

    fig, ax = plt.subplots(figsize=(7, 5))

    for key, label, color, marker in SERIES:
        if key == "cpu" or key not in present_keys:
            continue
        speedup = [r["cpu"] / r[key] for r in rows]
        ax.plot(
            ns,
            speedup,
            label=label,
            color=color,
            marker=marker,
            linewidth=1.8,
            markersize=5,
        )

    ax.axhline(
        1.0, color="#6b7280", linestyle=":", linewidth=1.2, label="CPU (speedup = 1)"
    )

    ax.set_xscale("log")
    ax.set_xlabel("Tamanho da entrada (N, escala log)")
    ax.set_ylabel("Speedup em relação à CPU (maior é melhor)")
    ax.set_title(f"{title} — Speedup vs. CPU")
    ax.legend()
    ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.6)

    fig.tight_layout()
    out_file = out_path / f"{bench_key}_speedup.{fmt}"
    fig.savefig(out_file, dpi=150)
    plt.close(fig)
    return out_file


def run_one(bench_key, fmt):
    csv_name, title = BENCHMARKS[bench_key]
    csv_path = RESULTS_DIR / csv_name

    if not csv_path.exists():
        print(
            f"Erro: não encontrei {csv_path}\n"
            f"Rode primeiro: mix run benchmarks-sscad/{bench_key}.exs",
            file=sys.stderr,
        )
        return False

    rows, present_keys = read_csv(csv_path)
    if not rows:
        print(f"Erro: {csv_path} está vazio.", file=sys.stderr)
        return False

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    tempo_file = plot_tempo(bench_key, title, rows, present_keys, RESULTS_DIR, fmt)
    speedup_file = plot_speedup(bench_key, title, rows, present_keys, RESULTS_DIR, fmt)

    print(f"[{bench_key}] gerado ({', '.join(present_keys)}):")
    print(f"  {tempo_file}")
    print(f"  {speedup_file}")
    return True


def main():
    parser = argparse.ArgumentParser(
        prog="plot.py",
        description="Plota os resultados dos benchmarks-sscad (CSVs em resultados/).",
        add_help=False,
    )
    parser.add_argument(
        "benchmark",
        nargs="?",
        help=f"Nome do benchmark a plotar. Um de: {', '.join(BENCHMARKS)}",
    )
    parser.add_argument(
        "--all", action="store_true", help="Plota todos os benchmarks reconhecidos."
    )
    parser.add_argument(
        "--formato",
        default="png",
        choices=["png", "pdf", "svg"],
        help="Formato de saída das imagens (default: png).",
    )
    parser.add_argument(
        "-help", "-h", "--help", action="store_true", dest="help_flag"
    )

    args = parser.parse_args()

    if args.help_flag or (args.benchmark is None and not args.all):
        parser.print_help()
        print("\nBenchmarks disponíveis:")
        for key, (csv_name, title) in BENCHMARKS.items():
            exists = "✓ CSV encontrado" if (RESULTS_DIR / csv_name).exists() else "✗ ainda não rodado"
            print(f"  {key:<20} {title:<30} [{exists}]")
        sys.exit(0)

    if args.all:
        ok = True
        for key in BENCHMARKS:
            ok = run_one(key, args.formato) and ok
        sys.exit(0 if ok else 1)

    if args.benchmark not in BENCHMARKS:
        print(
            f"Erro: benchmark '{args.benchmark}' não reconhecido.\n"
            f"Use um de: {', '.join(BENCHMARKS)}\n"
            f"Ou rode: python3 plot.py -help",
            file=sys.stderr,
        )
        sys.exit(1)

    ok = run_one(args.benchmark, args.formato)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()