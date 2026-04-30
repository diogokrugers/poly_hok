# benchmark_flawd_new.exs
# Benchmark end-to-end: flawd_list vs glist vs gnx vs CPU
# Inclui transferência de volta (get_*), usa floats, e pré-aquecimento adequado.

require PolyHok
use Ske

defmodule BenchFlawdNew do
  @sizes      [1_000, 10_000, 100_000, 250_000, 500_000, 1_000_000, 2_500_000]
  @iterations 7
  @type_float {:f, 32}
  @warmup_n   10_000

  def run do
    warmup()
    IO.puts("\n=== MAP benchmark (end-to-end, incluindo get) ===")
    map_rows = Enum.map(@sizes, &bench_map/1)
    IO.puts("\n=== REDUCE benchmark (end-to-end, incluindo get) ===")
    reduce_rows = Enum.map(@sizes, &bench_reduce/1)

    save_csv("/tmp/bench_map.csv", map_rows)
    save_csv("/tmp/bench_reduce.csv", reduce_rows)
    IO.puts("\nCSVs salvos em /tmp/bench_map.csv e /tmp/bench_reduce.csv")
    IO.puts("Rode: python3 plot_bench_new.py")
  end

  # -------------------- Warmup --------------------
  defp warmup do
    IO.puts("Aquecendo (JIT + pools)...")
    list = Enum.to_list(1..@warmup_n)
    nx   = Nx.tensor([list], type: @type_float)

    map_f = map_fun()
    red_f = reduce_fun()

    # Força compilação dos kernels e criação dos pools
    list |> PolyHok.new_flawd_list({1, @warmup_n}, @type_float) |> Ske.map(map_f) |> PolyHok.get_flawd_list()
    list |> PolyHok.new_glist({1, @warmup_n}, @type_float) |> Ske.map(map_f) |> PolyHok.get_glist()
    nx   |> PolyHok.new_gnx() |> Ske.map(map_f) |> PolyHok.get_gnx()

    list |> PolyHok.new_flawd_list({1, @warmup_n}, @type_float) |> Ske.reduce(0.0, red_f) |> PolyHok.get_flawd_list()
    list |> PolyHok.new_glist({1, @warmup_n}, @type_float) |> Ske.reduce(0.0, red_f) |> PolyHok.get_glist()
    nx   |> PolyHok.new_gnx() |> Ske.reduce(0.0, red_f) |> PolyHok.get_gnx()

    IO.puts("Pronto.\n")
  end

  # -------------------- Funções de teste --------------------
  defp map_fun do
    PolyHok.phok fn x ->
      y1 = x * 33.0 + 17.0
      y2 = y1 * 31.0 + 7.0
      y3 = y2 * 29.0 + 3.0
      y3 * 27.0 + 1.0
    end
  end

  defp reduce_fun do
    PolyHok.phok fn x, acc ->
      t1 = x * 3.0 + 1.0
      t2 = x * 5.0 + 7.0
      t3 = x * 7.0 + 11.0
      t4 = x * 11.0 + 13.0
      t5 = x * 13.0 + 17.0
      acc + t1 + t2 + t3 + t4 + t5
    end
  end

  # -------------------- Medições --------------------
  defp bench_map(n) do
    IO.write("map n=#{n}... ")
    list = Enum.to_list(1..n)
    nx   = Nx.tensor([list], type: @type_float)
    mf   = map_fun()

    flawd = measure(fn ->
      list
      |> PolyHok.new_flawd_list({1, n}, @type_float)
      |> Ske.map(mf)
      |> PolyHok.get_flawd_list()
    end)

    glist = measure(fn ->
      list
      |> PolyHok.new_glist({1, n}, @type_float)
      |> Ske.map(mf)
      |> PolyHok.get_glist()
    end)

    gnx = measure(fn ->
      nx
      |> PolyHok.new_gnx()
      |> Ske.map(mf)
      |> PolyHok.get_gnx()
    end)

    cpu = measure(fn ->
      Enum.map(list, fn x ->
        y1 = x * 33 + 17
        y2 = y1 * 31 + 7
        y3 = y2 * 29 + 3
        y3 * 27 + 1
      end)
    end)

    IO.puts("flawd=#{flawd}µs glist=#{glist}µs gnx=#{gnx}µs cpu=#{cpu}µs")
    %{n: n, flawd: flawd, glist: glist, gnx: gnx, cpu: cpu}
  end

  defp bench_reduce(n) do
    IO.write("reduce n=#{n}... ")
    list = Enum.to_list(1..n)
    nx   = Nx.tensor([list], type: @type_float)
    rf   = reduce_fun()
    initial = 0.0

    flawd = measure(fn ->
      list
      |> PolyHok.new_flawd_list({1, n}, @type_float)
      |> Ske.reduce(initial, rf)
      |> PolyHok.get_flawd_list()
    end)

    glist = measure(fn ->
      list
      |> PolyHok.new_glist({1, n}, @type_float)
      |> Ske.reduce(initial, rf)
      |> PolyHok.get_glist()
    end)

    gnx = measure(fn ->
      nx
      |> PolyHok.new_gnx()
      |> Ske.reduce(initial, rf)
      |> PolyHok.get_gnx()
    end)

    cpu = measure(fn ->
      Enum.reduce(list, initial, fn x, acc ->
        t1 = x * 3 + 1
        t2 = x * 5 + 7
        t3 = x * 7 + 11
        t4 = x * 11 + 13
        t5 = x * 13 + 17
        acc + t1 + t2 + t3 + t4 + t5
      end)
    end)

    IO.puts("flawd=#{flawd}µs glist=#{glist}µs gnx=#{gnx}µs cpu=#{cpu}µs")
    %{n: n, flawd: flawd, glist: glist, gnx: gnx, cpu: cpu}
  end

  # -------------------- Utilitários --------------------
  defp measure(f) do
    times = for _ <- 1..@iterations, do: elem(:timer.tc(f), 0)
    Enum.sort(times) |> Enum.at(div(@iterations, 2))
  end

  defp save_csv(path, rows) do
    header = "n,flawd,glist,gnx,cpu\n"
    body = Enum.map(rows, &"#{&1.n},#{&1.flawd},#{&1.glist},#{&1.gnx},#{&1.cpu}")
          |> Enum.join("\n")
    File.write!(path, header <> body <> "\n")
  end
end

BenchFlawdNew.run()
