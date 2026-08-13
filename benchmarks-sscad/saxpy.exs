# =============================================================================
# benchmarks-sscad/saxpy.exs
#
# Saxpy (Single-precision A·X Plus Y): y[i] = a*x[i] + y[i]
#
# Diferente do dot product, aqui o escalar `a` é uma constante fixa dentro
# da própria função device (não um terceiro vetor) — mostra que o mesmo
# Ske.map2 funciona tanto para "vetor com vetor" quanto para "vetor com
# escalar fixo", sem nenhuma API adicional.
#
# Compara três caminhos:
#   flawd — listas Elixir |> Ske.map2/2, forma idiomática
#   gnx   — caminho de baixo nível original do PolyHok (Nx.tensor -> GNx)
#   cpu   — Elixir puro, Enum.zip_with
#
# Uso:
#   mix run benchmarks-sscad/saxpy.exs
#
# Saída:
#   benchmarks-sscad/resultados/saxpy.csv
# =============================================================================

Code.require_file("bench_helpers.exs", __DIR__)

require PolyHok
use Ske

defmodule SaxpyBench do
  @sizes [1_000, 10_000, 100_000, 250_000, 500_000, 1_000_000, 2_000_000]
  @iterations 7
  @dtype {:f, 32}
  @warmup_n 10_000
  @a 2.5

  @out_csv Path.join([__DIR__, "resultados", "saxpy.csv"])
  @bench_name "saxpy"

  # ----------------------------------------------------------------
  # GPU: `a` é escrito como literal direto no corpo do kernel — PolyHok
  # não suporta capturar variáveis Elixir livres dentro de funções device
  # (só literais e os próprios parâmetros da função/kernel), então o valor
  # de @a é embutido aqui como constante, e mantido em sincronia com
  # cpu_saxpy/2 abaixo, que usa o mesmo @a.
  # ----------------------------------------------------------------

  defp saxpy_fun do
    PolyHok.phok fn x, y -> 2.5 * x + y end
  end

  # ----------------------------------------------------------------
  # CPU
  # ----------------------------------------------------------------

  defp cpu_saxpy(xs, ys) do
    Enum.zip_with(xs, ys, fn x, y -> @a * x + y end)
  end

  # ----------------------------------------------------------------
  # Geração de dados
  # ----------------------------------------------------------------

  defp gen_vectors(n) do
    xs = for _ <- 1..n, do: :rand.uniform() * 10.0
    ys = for _ <- 1..n, do: :rand.uniform() * 10.0
    {xs, ys}
  end

  # ----------------------------------------------------------------
  # Warmup
  # ----------------------------------------------------------------

  defp warmup do
    IO.puts("Warmup...")
    {xs, ys} = gen_vectors(@warmup_n)
    nx_x = Nx.tensor([xs], type: @dtype)
    nx_y = Nx.tensor([ys], type: @dtype)
    f = saxpy_fun()

    _ = xs |> Ske.map2(ys, f)
    _ = PolyHok.new_gnx(nx_x) |> Ske.map2(PolyHok.new_gnx(nx_y), f) |> PolyHok.get_gnx()

    IO.puts("Pronto.\n")
  end

  # ----------------------------------------------------------------
  # Um ponto do benchmark
  # ----------------------------------------------------------------

  defp bench(n) do
    IO.write("saxpy n=#{n}... ")

    {xs, ys} = gen_vectors(n)
    nx_x = Nx.tensor([xs], type: @dtype)
    nx_y = Nx.tensor([ys], type: @dtype)
    f = saxpy_fun()

    flawd = BenchHelpers.measure(fn -> xs |> Ske.map2(ys, f) end, @iterations)

    gnx =
      BenchHelpers.measure(
        fn -> PolyHok.new_gnx(nx_x) |> Ske.map2(PolyHok.new_gnx(nx_y), f) |> PolyHok.get_gnx() end,
        @iterations
      )

    cpu = BenchHelpers.measure(fn -> cpu_saxpy(xs, ys) end, @iterations)

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

SaxpyBench.run()
