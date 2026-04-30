# =============================================================================
# Benchmark: flawd_list vs glist vs gnx vs CPU — map e reduce, end-to-end
# =============================================================================

require PolyHok
use Ske

defmodule Bench2 do
  @sizes      [1_000, 10_000, 100_000, 250_000, 500_000, 1_000_000, 2_500_000]
  @iterations 7
  @type_elixir {:s, 32}
  @warmup_n   10_000

  def run do
    warmup()

    IO.puts("\nMAP benchmark")
    map_rows = Enum.map(@sizes, &bench_map/1)

    IO.puts("\nREDUCE benchmark")
    reduce_rows = Enum.map(@sizes, &bench_reduce/1)

    save_csv("/tmp/bench2_map.csv", map_rows)
    save_csv("/tmp/bench2_reduce.csv", reduce_rows)
  end

  # ======================
  # WARMUP
  # ======================

  defp warmup do
    IO.puts("Aquecendo...")
    list = Enum.to_list(1..@warmup_n)
    nx   = Nx.tensor([list], type: @type_elixir)

    mf = map_fun()
    rf = reduce_fun()

    list |> PolyHok.new_flawd_list({1, @warmup_n}, @type_elixir) |> Ske.map(mf)
    list |> PolyHok.new_glist({1, @warmup_n}, @type_elixir) |> Ske.map(mf)
    nx   |> PolyHok.new_gnx() |> Ske.map(mf)

    list |> PolyHok.new_flawd_list({1, @warmup_n}, @type_elixir) |> Ske.reduce(1, rf)
    list |> PolyHok.new_glist({1, @warmup_n}, @type_elixir) |> Ske.reduce(1, rf)
    nx   |> PolyHok.new_gnx() |> Ske.reduce(1, rf)

    IO.puts("Pronto.\n")
  end

  # ======================
  # MAP
  # ======================

  defp map_fun do
    PolyHok.phok fn x ->
      y1 = x * 33 + 17
      y2 = y1 * 31 + 7
      y3 = y2 * 29 + 3
      y3 * 27 + 1
    end
  end

  defp bench_map(n) do
    IO.write("map n=#{n}... ")

    list = Enum.to_list(1..n)
    nx   = Nx.tensor([list], type: @type_elixir)
    mf   = map_fun()

    flawd = measure(fn ->
      list |> PolyHok.new_flawd_list({1, n}, @type_elixir) |> Ske.map(mf)
    end)

    glist = measure(fn ->
      list |> PolyHok.new_glist({1, n}, @type_elixir) |> Ske.map(mf)
    end)

    gnx = measure(fn ->
      nx |> PolyHok.new_gnx() |> Ske.map(mf)
    end)

    cpu = measure(fn ->
      Enum.map(list, fn x ->
        y1 = x * 33 + 17
        y2 = y1 * 31 + 7
        y3 = y2 * 29 + 3
        y3 * 27 + 1
      end)
    end)

    IO.puts("cpu=#{cpu}µs")
    %{n: n, flawd: flawd, glist: glist, gnx: gnx, cpu: cpu}
  end

  # ======================
  # REDUCE
  # ======================

  defp reduce_fun do
    PolyHok.phok fn x, acc ->
      t1 = x * 3 + 1
      t2 = x * 5 + 7
      t3 = x * 7 + 11
      t4 = x * 11 + 13
      t5 = x * 13 + 17

      acc + t1 + t2 + t3 + t4 + t5
    end
  end

  defp bench_reduce(n) do
    IO.write("reduce n=#{n}... ")

    list = Enum.to_list(1..n)
    nx   = Nx.tensor([list], type: @type_elixir)
    rf   = reduce_fun()

    flawd = measure(fn ->
      list |> PolyHok.new_flawd_list({1, n}, @type_elixir) |> Ske.reduce(1, rf)
    end)

    glist = measure(fn ->
      list |> PolyHok.new_glist({1, n}, @type_elixir) |> Ske.reduce(1, rf)
    end)

    gnx = measure(fn ->
      nx |> PolyHok.new_gnx() |> Ske.reduce(1, rf)
    end)

    cpu = measure(fn ->
      Enum.reduce(list, 1, fn x, acc ->
        t1 = x * 3 + 1
        t2 = x * 5 + 7
        t3 = x * 7 + 11
        t4 = x * 11 + 13
        t5 = x * 13 + 17

        acc + t1 + t2 + t3 + t4 + t5
      end)
    end)

    IO.puts("cpu=#{cpu}µs")
    %{n: n, flawd: flawd, glist: glist, gnx: gnx, cpu: cpu}
  end

  # ======================

  defp measure(f) do
    times = for _ <- 1..@iterations, do: elem(:timer.tc(f), 0)
    Enum.sort(times) |> Enum.at(div(@iterations, 2))
  end

  defp save_csv(path, rows) do
    header = "n,flawd,glist,gnx,cpu\n"
    body =
      rows
      |> Enum.map(&"#{&1.n},#{&1.flawd},#{&1.glist},#{&1.gnx},#{&1.cpu}")
      |> Enum.join("\n")

    File.write!(path, header <> body <> "\n")
  end
end

Bench2.run()
