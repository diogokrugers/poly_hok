#!/usr/bin/env python3
# plot_persistent.py — plota /tmp/bench_persistent.csv (Black-Scholes pipeline)
 
import csv
import matplotlib.pyplot as plt
 
rows = []
with open("/tmp/bench_persistent.csv") as f:
    for row in csv.DictReader(f):
        rows.append({k: int(v) for k, v in row.items()})
 
ns         = [r["n"]          for r in rows]
cpu        = [r["cpu"]        for r in rows]
gnx        = [r["gnx"]        for r in rows]
flawd      = [r["flawd"]      for r in rows]
persistent = [r["persistent"] for r in rows]
 
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(ns, cpu,        marker="o", label="CPU (Enum)")
ax.plot(ns, gnx,        marker="D", label="GNx (tensor)")
ax.plot(ns, flawd,      marker="s", label="flawd (roundtrip)")
ax.plot(ns, persistent, marker="^", label="flawd_ref (persistente)")
 
ax.set_xlabel("Tamanho da lista (n)")
ax.set_ylabel("Tempo mediano (µs)")
ax.set_title("Pipeline Black-Scholes |> normalização |> reduce(soma)\nCPU vs GNx vs flawd vs flawd_ref")
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_xscale("log")
 
plt.tight_layout()
plt.savefig("/tmp/bench_persistent.png", dpi=150)
print("Salvo em /tmp/bench_persistent.png")
plt.show()