# =============================================================================
# benchmarks-sscad/dot_product.exs
#
# Produto escalar (dot product): d = sum(a[i] * b[i]), i em 1..n
#
# Clássico pipeline map2 + reduce — um exemplo motivador
# Compara quatro caminhos para o mesmo cálculo:
#
#   flawd        — listas Elixir |> Ske.map2/Ske.reduce, dois kernels
#                  (map2 e reduce lançados separadamente)
#   flawd_fusion — o mesmo pipeline, mas dentro de Fusion.with_fusion:
#                  map2+reduce viram um único kernel (map2Reduce) via
#                  fusão de kernel, sem nenhuma mudança na lógica escrita
#   gnx          — caminho "de baixo nível" original do PolyHok
#                  (Nx.tensor -> GNx), sem fusão
#   cpu          — Elixir puro, Enum.zip_reduce
#
# Uso:
#   mix run benchmarks-sscad/dot_product.exs
#
# Saída:
#   benchmarks-sscad/resultados/dot_product.csv
# =============================================================================

Code.require_file("bench_helpers.exs", __DIR__)

require PolyHok
use Ske
require Fusion

defmodule DotProductBench do
  @sizes [1_000, 10_000, 100_000, 250_000, 500_000, 1_000_000, 2_000_000]
  @iterations 7
  @dtype {:f, 32}
  @warmup_n 10_000

  @out_csv Path.join([__DIR__, "resultados", "dot_product.csv"])
  @bench_name "dot_product"

  # ----------------------------------------------------------------
  # Funções GPU — o mesmo par map2/reduce em qualquer um dos caminhos
  # ----------------------------------------------------------------

  defp mul_fun, do: PolyHok.phok(fn a, b -> a * b end)
  defp add_fun, do: PolyHok.phok(fn a, b -> a + b end)

  # ----------------------------------------------------------------
  # Referência CPU
  # ----------------------------------------------------------------

  defp cpu_dot(a, b) do
    Enum.zip_reduce(a, b, 0.0, fn x, y, acc -> acc + x * y end)
  end

  # ----------------------------------------------------------------
  # Geração de dados
  # ----------------------------------------------------------------

  defp gen_vectors(n) do
    a = for _ <- 1..n, do: :rand.uniform() * 10.0
    b = for _ <- 1..n, do: :rand.uniform() * 10.0
    {a, b}
  end

  # ----------------------------------------------------------------
  # Warmup
  # ----------------------------------------------------------------

  defp warmup do
    IO.puts("Warmup...")
    {a, b} = gen_vectors(@warmup_n)
    nx_a = Nx.tensor([a], type: @dtype)
    nx_b = Nx.tensor([b], type: @dtype)
    mf = mul_fun()
    rf = add_fun()

    # flawd: listas Elixir do início ao fim, forma idiomática, dois kernels
    _ = a |> Ske.map2(b, mf) |> Ske.reduce(0.0, rf)

    # flawd_fusion: mesmo pipeline, mas fundido em um único kernel.
    # Fusion.with_fusion analisa o AST em tempo de COMPILAÇÃO, então os
    # kernels precisam estar escritos inline aqui (PolyHok.phok(fn...end)
    # literal) — passar mf/rf como variável não funciona: nesse ponto a
    # macro só veria o nome da variável, não a função em si.
    _ =
      Fusion.with_fusion(
        Ske.map2(a, b, PolyHok.phok(fn x, y -> x * y end))
        |> Ske.reduce(0.0, PolyHok.phok(fn x, y -> x + y end))
      )

    # gnx: caminho de baixo nível, sem fusão
    _ =
      PolyHok.new_gnx(nx_a)
      |> Ske.map2(PolyHok.new_gnx(nx_b), mf)
      |> Ske.reduce(0.0, rf)
      |> PolyHok.get_gnx()

    IO.puts("Pronto.\n")
  end

  # ----------------------------------------------------------------
  # Um ponto do benchmark
  # ----------------------------------------------------------------

  defp bench(n) do
    IO.write("dot_product n=#{n}... ")

    {a, b} = gen_vectors(n)
    nx_a = Nx.tensor([a], type: @dtype)
    nx_b = Nx.tensor([b], type: @dtype)
    mf = mul_fun()
    rf = add_fun()

    flawd =
      BenchHelpers.measure(
        fn -> a |> Ske.map2(b, mf) |> Ske.reduce(0.0, rf) end,
        @iterations
      )

    flawd_fusion =
      BenchHelpers.measure(
        fn ->
          Fusion.with_fusion(
            Ske.map2(a, b, PolyHok.phok(fn x, y -> x * y end))
            |> Ske.reduce(0.0, PolyHok.phok(fn x, y -> x + y end))
          )
        end,
        @iterations
      )

    gnx =
      BenchHelpers.measure(
        fn ->
          PolyHok.new_gnx(nx_a)
          |> Ske.map2(PolyHok.new_gnx(nx_b), mf)
          |> Ske.reduce(0.0, rf)
          |> PolyHok.get_gnx()
        end,
        @iterations
      )

    cpu = BenchHelpers.measure(fn -> cpu_dot(a, b) end, @iterations)

    IO.puts(
      "flawd=#{BenchHelpers.fmt_time(flawd)} " <>
        "flawd_fusion=#{BenchHelpers.fmt_time(flawd_fusion)} " <>
        "gnx=#{BenchHelpers.fmt_time(gnx)} " <>
        "cpu=#{BenchHelpers.fmt_time(cpu)} " <>
        "(speedup flawd_fusion/flawd=#{BenchHelpers.speedup(flawd, flawd_fusion)}x, " <>
        "speedup flawd_fusion/cpu=#{BenchHelpers.speedup(cpu, flawd_fusion)}x)"
    )

    %{n: n, flawd: flawd, flawd_fusion: flawd_fusion, gnx: gnx, cpu: cpu}
  end

  def run do
    warmup()
    rows = Enum.map(@sizes, &bench/1)
    BenchHelpers.save_csv(@out_csv, rows, [:n, :flawd, :flawd_fusion, :gnx, :cpu])
    IO.puts("\nCSV salvo em: #{@out_csv}")
    BenchHelpers.print_plot_hint(@bench_name)
  end
end

DotProductBench.run()
