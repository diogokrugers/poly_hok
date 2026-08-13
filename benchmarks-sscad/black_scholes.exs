# =============================================================================
# benchmarks-sscad/black_scholes.exs
#
# Precificação de opções via aproximação de Black-Scholes (map elemento a
# elemento sobre um vetor de preços — log/exp/sqrt por elemento, então
# intensidade aritmética real, não um kernel trivial dominado por overhead
# de transferência).
#
# Compara três caminhos:
#   flawd — lista Elixir |> Ske.map(f), forma idiomática (new_flawd_list/
#           get_flawd_list acontecem por baixo, de forma transparente)
#   gnx   — caminho de baixo nível original do PolyHok (Nx.tensor -> GNx)
#   cpu   — Elixir puro, Enum.map
#
# Uso:
#   mix run benchmarks-sscad/black_scholes.exs
#
# Saída:
#   benchmarks-sscad/resultados/black_scholes.csv
# =============================================================================

Code.require_file("bench_helpers.exs", __DIR__)

require PolyHok
use Ske

defmodule BlackScholesBench do
  @sizes [1_000, 10_000, 100_000, 250_000, 500_000, 1_000_000, 2_000_000]
  @iterations 7
  @dtype {:f, 32}
  @warmup_n 10_000

  @out_csv Path.join([__DIR__, "resultados", "black_scholes.csv"])
  @bench_name "black_scholes"

  # ----------------------------------------------------------------
  # CPU
  # ----------------------------------------------------------------

  defp cpu_bs(x) do
    s = x + 50.0
    k = 100.0
    t = 1.0
    r = 0.02
    v = 0.30
    sqrt_t = :math.sqrt(t)

    d1 = (:math.log(s / k) + (r + v * v / 2.0) * t) / (v * sqrt_t)
    d2 = d1 - v * sqrt_t

    cnd_d1 = 1.0 / (1.0 + :math.exp(-1.702 * d1))
    cnd_d2 = 1.0 / (1.0 + :math.exp(-1.702 * d2))

    s * cnd_d1 - k * :math.exp(-r * t) * cnd_d2
  end

  # ----------------------------------------------------------------
  # GPU (mesmo corpo, escrito em PolyHok)
  # ----------------------------------------------------------------

  defp gpu_fun do
    PolyHok.phok fn x ->
      s = x + 50.0
      k = 100.0
      t = 1.0
      r = 0.02
      v = 0.30
      sqrt_t = sqrt(t)

      d1 = (log(s / k) + (r + v * v / 2.0) * t) / (v * sqrt_t)
      d2 = d1 - v * sqrt_t

      cnd_d1 = 1.0 / (1.0 + exp(-1.702 * d1))
      cnd_d2 = 1.0 / (1.0 + exp(-1.702 * d2))

      s * cnd_d1 - k * exp(-r * t) * cnd_d2
    end
  end

  # ----------------------------------------------------------------
  # Geração de dados
  # ----------------------------------------------------------------

  defp gen_prices(n), do: for(_ <- 1..n, do: :rand.uniform() * 100.0)

  # ----------------------------------------------------------------
  # Warmup
  # ----------------------------------------------------------------

  defp warmup do
    IO.puts("Warmup...")
    list = gen_prices(@warmup_n)
    nx = Nx.tensor([list], type: @dtype)
    f = gpu_fun()

    _ = list |> Ske.map(f)
    _ = PolyHok.new_gnx(nx) |> Ske.map(f) |> PolyHok.get_gnx()

    IO.puts("Pronto.\n")
  end

  # ----------------------------------------------------------------
  # Um ponto do benchmark
  # ----------------------------------------------------------------

  defp bench(n) do
    IO.write("black_scholes n=#{n}... ")

    list = gen_prices(n)
    nx = Nx.tensor([list], type: @dtype)
    f = gpu_fun()

    flawd = BenchHelpers.measure(fn -> list |> Ske.map(f) end, @iterations)

    gnx =
      BenchHelpers.measure(
        fn -> PolyHok.new_gnx(nx) |> Ske.map(f) |> PolyHok.get_gnx() end,
        @iterations
      )

    cpu = BenchHelpers.measure(fn -> Enum.map(list, &cpu_bs/1) end, @iterations)

    IO.puts(
      "flawd=#{BenchHelpers.fmt_time(flawd)} " <>
        "gnx=#{BenchHelpers.fmt_time(gnx)} " <>
        "cpu=#{BenchHelpers.fmt_time(cpu)} " <>
        "(speedup flawd/cpu=#{BenchHelpers.speedup(cpu, flawd)}x)"
    )

    %{n: n, flawd: flawd, gnx: gnx, cpu: cpu}
  end

  def run do
    warmup()
    rows = Enum.map(@sizes, &bench/1)
    BenchHelpers.save_csv(@out_csv, rows, [:n, :flawd, :gnx, :cpu])
    IO.puts("\nCSV salvo em: #{@out_csv}")
    BenchHelpers.print_plot_hint(@bench_name)
  end
end

BlackScholesBench.run()
