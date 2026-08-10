import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("/tmp/bench_blackscholes.csv")

plt.figure(figsize=(10,6))

plt.plot(df["n"], df["flawd"], marker="o", label="flawd")
plt.plot(df["n"], df["gnx"], marker="o", label="gnx")
plt.plot(df["n"], df["cpu"], marker="o", label="cpu")

plt.xscale("log")
plt.yscale("log")

plt.xlabel("Input size (n)")
plt.ylabel("Time (microseconds)")
plt.title("Black-Scholes Benchmark")

plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig("blackscholes_benchmark.png", dpi=300)

print("saved: blackscholes_benchmark.png")

plt.show()