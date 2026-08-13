# =============================================================================
# benchmarks-sscad/nearest_neighbor.exs
#
# Nearest Neighbor: dado um conjunto de n pontos (lat, lng) e uma posição de
# referência fixa, encontra a menor distância euclidiana ao ponto de referência.
#
# Pipeline map2 (distância) + reduce (mínimo), a mesma composição do dot product
#
# Compara quatro caminhos:
#   flawd        — listas Elixir |> Ske.map2/Ske.reduce, dois kernels
#   flawd_fusion — o mesmo pipeline dentro de Fusion.with_fusion, um único
#                  kernel fundido (map2Reduce)
#   gnx          — caminho de baixo nível original do PolyHok (Nx.tensor
#                  -> GNx), sem fusão
#   cpu          — Elixir puro, Enum.zip_reduce
#
# Uso:
#   mix run benchmarks-sscad/nearest_neighbor.exs
#
# Saída:
#   benchmarks-sscad/resultados/nearest_neighbor.csv
# =============================================================================

Code.require_file("bench_helpers.exs", __DIR__)

require PolyHok
use Ske
require Fusion

defmodule NearestNeighborBench do
  @sizes [1_000, 10_000, 100_000, 250_000, 500_000, 1_000_000, 2_000_000]
  @iterations 7
  @dtype {:f, 32}
  @warmup_n 10_000

  @ref_lat 0.0
  @ref_lng 0.0

  @out_csv Path.join([__DIR__, "resultados", "nearest_neighbor.csv"])
  @bench_name "nearest_neighbor"

  # ----------------------------------------------------------------
  # Funções GPU — distância euclidiana ao ponto de referência, e o
  # operador de redução "menor de dois". @ref_lat/@ref_lng são 0.0 e o
  # kernel usa literais diretos (0.0 - lat, 0.0 - lng) em vez do atributo,
  # para não depender de nenhuma regra sutil sobre substituição de @attr
  # dentro de AST de macro — mesmo estilo dos literais fixos já usados em
  # black_scholes.exs (k = 100.0, t = 1.0, etc). Mantidos em sincronia
  # manual com @ref_lat/@ref_lng, usados apenas em cpu_nn/2 abaixo.
  # ----------------------------------------------------------------

  defp dist_fun do
    PolyHok.phok fn lat, lng ->
      dlat = lat - 0.0
      dlng = lng - 0.0
      sqrt(dlat * dlat + dlng * dlng)
    end
  end

  defp min_fun do
    # Reescrito para não usar if/else como EXPRESSÃO de retorno — o
    # inferidor de tipos do PolyHok (type_inference.ex, find_type_exp/2)
    # não tem clause para {:if, ...} quando ele é o próprio valor
    # retornado pela função (só sabe processar if/else como statement,
    # ver infer_if/2). Usar o if como statement de atribuição condicional
    # e retornar a variável simples em seguida é o padrão já usado e
    # validado em outros kernels do projeto (ex. Ske.reduce_kernel).
    PolyHok.phok fn x, y ->
      result = y

      if x < y do
        result = x
      end

      result
    end
  end

  # ----------------------------------------------------------------
  # CPU
  # ----------------------------------------------------------------

  defp cpu_nn(lats, lngs) do
    Enum.zip_reduce(lats, lngs, :infinity, fn lat, lng, acc ->
      dlat = lat - @ref_lat
      dlng = lng - @ref_lng
      d = :math.sqrt(dlat * dlat + dlng * dlng)
      min(d, acc)
    end)
  end

  # ----------------------------------------------------------------
  # Geração de dados: pontos espalhados num raio de ~100 unidades do
  # ponto de referência, como no gerador original (7..70 lat, 0..358 lng)
  # ----------------------------------------------------------------

  defp gen_points(n) do
    lats = for _ <- 1..n, do: 7.0 + :rand.uniform() * 63.0
    lngs = for _ <- 1..n, do: :rand.uniform() * 358.0
    {lats, lngs}
  end

  # ----------------------------------------------------------------
  # Warmup
  # ----------------------------------------------------------------

  defp warmup do
    IO.puts("Warmup...")
    {lats, lngs} = gen_points(@warmup_n)
    nx_lat = Nx.tensor([lats], type: @dtype)
    nx_lng = Nx.tensor([lngs], type: @dtype)
    df = dist_fun()
    mf = min_fun()

    _ = lats |> Ske.map2(lngs, df) |> Ske.reduce(1.0e9, mf)

    # flawd_fusion: mesmo pipeline, mas fundido em um único kernel. Os
    # kernels precisam estar inline aqui (ver nota equivalente em
    # dot_product.exs) — Fusion.with_fusion não resolve variáveis.
    _ =
      Fusion.with_fusion(
        Ske.map2(
          lats,
          lngs,
          PolyHok.phok(fn lat, lng ->
            dlat = lat - 0.0
            dlng = lng - 0.0
            sqrt(dlat * dlat + dlng * dlng)
          end)
        )
        |> Ske.reduce(
          1.0e9,
          PolyHok.phok(fn x, y ->
            result = y

            if x < y do
              result = x
            end

            result
          end)
        )
      )

    _ =
      PolyHok.new_gnx(nx_lat)
      |> Ske.map2(PolyHok.new_gnx(nx_lng), df)
      |> Ske.reduce(1.0e9, mf)
      |> PolyHok.get_gnx()

    IO.puts("Pronto.\n")
  end

  # ----------------------------------------------------------------
  # Um ponto do benchmark
  # ----------------------------------------------------------------

  defp bench(n) do
    IO.write("nearest_neighbor n=#{n}... ")

    {lats, lngs} = gen_points(n)
    nx_lat = Nx.tensor([lats], type: @dtype)
    nx_lng = Nx.tensor([lngs], type: @dtype)
    df = dist_fun()
    mf = min_fun()

    flawd =
      BenchHelpers.measure(
        fn -> lats |> Ske.map2(lngs, df) |> Ske.reduce(1.0e9, mf) end,
        @iterations
      )
    IO.write "\ncheck #{n} flawd!"
    flawd_fusion =
      BenchHelpers.measure(
        fn ->
          Fusion.with_fusion(
            Ske.map2(
              lats,
              lngs,
              PolyHok.phok(fn lat, lng ->
                dlat = lat - 0.0
                dlng = lng - 0.0
                sqrt(dlat * dlat + dlng * dlng)
              end)
            )
            |> Ske.reduce(
              1.0e9,
              PolyHok.phok(fn x, y ->
                result = y

                if x < y do
                  result = x
                end

                result
              end)
            )
          )
        end,
        @iterations
      )
    IO.write "\ncheck #{n} flawd fusion!"

    gnx =
      BenchHelpers.measure(
        fn ->
          PolyHok.new_gnx(nx_lat)
          |> Ske.map2(PolyHok.new_gnx(nx_lng), df)
          |> Ske.reduce(1.0e9, mf)
          |> PolyHok.get_gnx()
        end,
        @iterations
      )
    IO.write "\ncheck #{n} gnx!"

    cpu = BenchHelpers.measure(fn -> cpu_nn(lats, lngs) end, @iterations)
    IO.write "\ncheck #{n} cpu!\n"

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

NearestNeighborBench.run()
