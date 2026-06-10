require PolyHok
use Ske

PolyHok.defmodule BlackScholesBench do

  @sizes [
    1_000,
    10_000,
    100_000,
    250_000,
    500_000,
    1_000_000,
    2_500_000,
    5_000_000
  ]

  @iterations 7
  @dtype {:f, 32}

  # ============================================================
  # CPU VERSION
  # ============================================================

  def cpu_bs(x) do

    s = x + 50.0
    k = 100.0
    t = 1.0
    r = 0.02
    v = 0.30

    sqrt_t = :math.sqrt(t)

    d1 =
      (
        :math.log(s / k) +
        (r + (v*v)/2.0) * t
      ) / (v * sqrt_t)

    d2 =
      d1 - v * sqrt_t

    cnd_d1 =
      1.0 / (1.0 + :math.exp(-1.702 * d1))

    cnd_d2 =
      1.0 / (1.0 + :math.exp(-1.702 * d2))

    s * cnd_d1 -
    k * :math.exp(-r*t) * cnd_d2

  end

  # ============================================================
  # GPU FUNCTION
  # ============================================================

  def gpu_fun do

    PolyHok.phok fn x ->

      s = x + 50.0
      k = 100.0
      t = 1.0
      r = 0.02
      v = 0.30

      sqrt_t = sqrt(t)

      d1 =
        (
          log(s / k) +
          (r + (v*v)/2.0) * t
        ) / (v * sqrt_t)

      d2 =
        d1 - v * sqrt_t

      cnd_d1 =
        1.0 / (1.0 + exp(-1.702 * d1))

      cnd_d2 =
        1.0 / (1.0 + exp(-1.702 * d2))

      s * cnd_d1 -
      k * exp(-r*t) * cnd_d2

    end

  end

  # ============================================================
  # WARMUP
  # ============================================================

  def warmup do

    IO.puts("Warmup...")

    n = 10_000

    list =
      Enum.map(1..n, fn _ ->
        :rand.uniform() * 100.0
      end)

    nx =
      Nx.tensor([list], type: @dtype)

    f = gpu_fun()

    list
    |> PolyHok.new_flawd_list({1,n}, @dtype)
    |> Ske.map(f)
    |> PolyHok.get_flawd_list()

    nx
    |> PolyHok.new_gnx()
    |> Ske.map(f)
    |> PolyHok.get_gnx()

    IO.puts("Done.\n")

  end

  # ============================================================
  # BENCHMARK
  # ============================================================

  def run do

    warmup()

    rows =
      Enum.map(@sizes, &bench/1)

    save_csv(
      "/tmp/bench_blackscholes.csv",
      rows
    )

    IO.puts("\nCSV salvo em:")
    IO.puts("/tmp/bench_blackscholes.csv")

    IO.puts("\nRode:")
    IO.puts("python3 plot_blackscholes.py")

  end

  def bench(n) do

    IO.write(
      "BlackScholes n=#{n}... "
    )

    list =
      Enum.map(1..n, fn _ ->
        :rand.uniform() * 100.0
      end)

    nx =
      Nx.tensor([list], type: @dtype)

    f = gpu_fun()

    flawd =

      measure(fn ->

        list
        |> PolyHok.new_flawd_list(
          {1,n},
          @dtype
        )
        |> Ske.map(f)
        |> PolyHok.get_flawd_list()

      end)

    gnx =

      measure(fn ->

        nx
        |> PolyHok.new_gnx()
        |> Ske.map(f)
        |> PolyHok.get_gnx()

      end)

    cpu =

      measure(fn ->

        Enum.map(list, fn x ->
          cpu_bs(x)
        end)

      end)

    IO.puts(
      "flawd=#{flawd}µs " <>
      "gnx=#{gnx}µs " <>
      "cpu=#{cpu}µs"
    )

    %{
      n: n,
      flawd: flawd,
      gnx: gnx,
      cpu: cpu
    }

  end

  # ============================================================
  # UTILS
  # ============================================================

  def measure(f) do

    times =

      for _ <- 1..@iterations do
        elem(:timer.tc(f), 0)
      end

    Enum.sort(times)
    |> Enum.at(div(@iterations,2))

  end

  def save_csv(path, rows) do

    header =
      "n,flawd,gnx,cpu\n"

    body =

      Enum.map(rows, fn r ->

        "#{r.n}," <>
        "#{r.flawd}," <>
        "#{r.gnx}," <>
        "#{r.cpu}"

      end)
      |> Enum.join("\n")

    File.write!(
      path,
      header <> body <> "\n"
    )

  end

end

BlackScholesBench.run()
