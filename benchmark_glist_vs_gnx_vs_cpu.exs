# =============================================================================
# Benchmark: glist vs gnx vs CPU — map e reduce, end-to-end
#
# Mede o tempo TOTAL da operação do ponto de vista do usuário Elixir:
#   glist    : list → new_glist → Ske.map/reduce → get_glist  (GPU)
#   gnx_tx   : tensor pré-criado → new_gnx → Ske.map/reduce → get_gnx (GPU, melhor caso gnx)
#   gnx_list : list → Nx.tensor → new_gnx → Ske.map/reduce → get_gnx  (GPU, caso real gnx)
#   cpu      : Enum.map / Enum.reduce puro em Elixir
#
# Como rodar:
#   mix run benchmark_glist_vs_gnx_vs_cpu.exs
#   (coloque o arquivo na raiz do projeto poly_hok)
#
# Saída: imprime tabelas no terminal + salva CSVs em /tmp/bench_map.csv e /tmp/bench_reduce.csv
# =============================================================================

require PolyHok
use Ske

defmodule Bench do
  # ---- configuração --------------------------------------------------------
  @sizes      [1_000, 10_000, 100_000, 250_000, 500_000, 1_000_000]
  @iterations 7      # medições por ponto — vai ficar com a mediana
  @type_elixir {:s, 32}
  @warmup_n   10_000

  # ---- ponto de entrada ----------------------------------------------------
  def run do
    warmup()

    IO.puts("\n╔══════════════════════════════════════════════════════════════════╗")
    IO.puts("║              BENCHMARK  MAP  (end-to-end, µs)                   ║")
    IO.puts("╠════════════╦═══════════╦════════════╦═══════════╦═══════════════╣")
    IO.puts("║     n      ║  glist    ║  gnx_tx    ║  gnx_list ║  cpu_elixir   ║")
    IO.puts("╠════════════╬═══════════╬════════════╬═══════════╬═══════════════╣")

    map_rows = Enum.map(@sizes, fn n ->
      r = bench_map(n)
      IO.puts("║ #{pad(n, 10)} ║ #{pad(r.glist, 9)} ║ #{pad(r.gnx_tx, 10)} ║ #{pad(r.gnx_list, 9)} ║ #{pad(r.cpu, 13)} ║")
      r
    end)

    IO.puts("╚════════════╩═══════════╩════════════╩═══════════╩═══════════════╝")

    IO.puts("\n╔══════════════════════════════════════════════════════════════════╗")
    IO.puts("║             BENCHMARK  REDUCE  (end-to-end, µs)                 ║")
    IO.puts("╠════════════╦═══════════╦════════════╦═══════════╦═══════════════╣")
    IO.puts("║     n      ║  glist    ║  gnx_tx    ║  gnx_list ║  cpu_elixir   ║")
    IO.puts("╠════════════╬═══════════╬════════════╬═══════════╬═══════════════╣")

    reduce_rows = Enum.map(@sizes, fn n ->
      r = bench_reduce(n)
      IO.puts("║ #{pad(n, 10)} ║ #{pad(r.glist, 9)} ║ #{pad(r.gnx_tx, 10)} ║ #{pad(r.gnx_list, 9)} ║ #{pad(r.cpu, 13)} ║")
      r
    end)

    IO.puts("╚════════════╩═══════════╩════════════╩═══════════╩═══════════════╝")

    save_csv("/tmp/bench_map.csv",    map_rows)
    save_csv("/tmp/bench_reduce.csv", reduce_rows)

    IO.puts("\nCSVs salvos em /tmp/bench_map.csv e /tmp/bench_reduce.csv")
    IO.puts("Use o script Python abaixo para gerar os gráficos:")
    IO.puts("  python3 /tmp/plot_bench.py")
  end

  # ---- aquecimento ---------------------------------------------------------
  defp warmup do
    IO.puts("Aquecendo GPU (compilando kernels JIT)...")

    list = Enum.to_list(1..@warmup_n)
    nx   = Nx.tensor([list], type: @type_elixir)

    map_f    = PolyHok.phok fn x -> x * 2 end
    reduce_f = PolyHok.phok fn x, acc -> x + acc end

    # warm map
    list |> PolyHok.new_glist({1, @warmup_n}, @type_elixir) |> Ske.map(map_f)    |> PolyHok.get_glist()
    nx   |> PolyHok.new_gnx()                                |> Ske.map(map_f)    |> PolyHok.get_gnx()

    # warm reduce
    list |> PolyHok.new_glist({1, @warmup_n}, @type_elixir) |> Ske.reduce(0, reduce_f) |> PolyHok.get_glist()
    nx   |> PolyHok.new_gnx()                                |> Ske.reduce(0, reduce_f) |> PolyHok.get_gnx()

    IO.puts("Pronto.\n")
  end

  # ---- benchmark MAP -------------------------------------------------------
  defp bench_map(n) do
    IO.write("  map n=#{n}...")
    list = Enum.to_list(1..n)
    nx   = Nx.tensor([list], type: @type_elixir)   # pré-criado (melhor caso gnx)
    map_f = PolyHok.phok fn x -> x * 2 end

    # glist GPU — pipeline completo partindo da lista
    glist = measure(@iterations, fn ->
      list
      |> PolyHok.new_glist({1, n}, @type_elixir)
      |> Ske.map(map_f)
      |> PolyHok.get_glist()
    end)

    # gnx GPU — tensor pré-criado (não inclui custo de Nx.tensor/2)
    gnx_tx = measure(@iterations, fn ->
      nx
      |> PolyHok.new_gnx()
      |> Ske.map(map_f)
      |> PolyHok.get_gnx()
    end)

    # gnx GPU — partindo da lista (inclui Nx.tensor/2, comparação justa)
    gnx_list = measure(@iterations, fn ->
      list
      |> then(&Nx.tensor([&1], type: @type_elixir))
      |> PolyHok.new_gnx()
      |> Ske.map(map_f)
      |> PolyHok.get_gnx()
    end)

    # CPU puro — Enum.map
    cpu = measure(@iterations, fn ->
      Enum.map(list, fn x -> x * 2 end)
    end)

    IO.puts(" ok  glist=#{glist}µs  gnx_tx=#{gnx_tx}µs  gnx_list=#{gnx_list}µs  cpu=#{cpu}µs")
    %{n: n, glist: glist, gnx_tx: gnx_tx, gnx_list: gnx_list, cpu: cpu}
  end

  # ---- benchmark REDUCE ----------------------------------------------------
  defp bench_reduce(n) do
    IO.write("  reduce n=#{n}...")
    list = Enum.to_list(1..n)
    nx   = Nx.tensor([list], type: @type_elixir)
    reduce_f = PolyHok.phok fn x, acc -> x + acc end

    # glist GPU
    glist = measure(@iterations, fn ->
      list
      |> PolyHok.new_glist({1, n}, @type_elixir)
      |> Ske.reduce(0, reduce_f)
      |> PolyHok.get_glist()
    end)

    # gnx GPU — tensor pré-criado
    gnx_tx = measure(@iterations, fn ->
      nx
      |> PolyHok.new_gnx()
      |> Ske.reduce(0, reduce_f)
      |> PolyHok.get_gnx()
    end)

    # gnx GPU — partindo da lista
    gnx_list = measure(@iterations, fn ->
      list
      |> then(&Nx.tensor([&1], type: @type_elixir))
      |> PolyHok.new_gnx()
      |> Ske.reduce(0, reduce_f)
      |> PolyHok.get_gnx()
    end)

    # CPU puro — Enum.reduce
    cpu = measure(@iterations, fn ->
      Enum.reduce(list, 0, fn x, acc -> x + acc end)
    end)

    IO.puts(" ok  glist=#{glist}µs  gnx_tx=#{gnx_tx}µs  gnx_list=#{gnx_list}µs  cpu=#{cpu}µs")
    %{n: n, glist: glist, gnx_tx: gnx_tx, gnx_list: gnx_list, cpu: cpu}
  end

  # ---- utilidades ----------------------------------------------------------

  # Roda `f` `iters` vezes e retorna a mediana dos tempos em µs
  defp measure(iters, f) do
    times = for _ <- 1..iters do
      {t, _} = :timer.tc(f)
      t
    end
    median(times)
  end

  defp median(list) do
    sorted = Enum.sort(list)
    n = length(sorted)
    mid = div(n, 2)
    if rem(n, 2) == 0,
      do:   div(Enum.at(sorted, mid - 1) + Enum.at(sorted, mid), 2),
      else: Enum.at(sorted, mid)
  end

  defp pad(val, width) do
    s = to_string(val)
    String.pad_leading(s, width)
  end

  defp save_csv(path, rows) do
    header = "n,glist,gnx_tx,gnx_list,cpu\n"
    body = rows
      |> Enum.map(fn r ->
        "#{r.n},#{r.glist},#{r.gnx_tx},#{r.gnx_list},#{r.cpu}"
      end)
      |> Enum.join("\n")
    File.write!(path, header <> body <> "\n")
  end
end

Bench.run()
