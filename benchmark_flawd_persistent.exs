# benchmark_flawd_persistent.exs
#
# Pipeline: map(black-scholes) |> map(normalização) |> reduce(soma)
#
# Cenários comparados:
#   :cpu         — Elixir puro (Enum, sem GPU)
#   :gnx         — GNx (tensor, estrutura não-funcional)
#   :flawd       — flawd existente (sobe/desce GPU a cada operação)
#   :persistent  — flawd_ref (lista fica na GPU durante toda a pipeline)
#
# Operações compute-pesadas para que o kernel domine em listas grandes
# e o overhead de transferência apareça claramente nas pequenas.

require PolyHok
use Ske

defmodule BenchPersistent do
  @sizes      [1_000, 10_000, 100_000, 250_000, 500_000, 1_000_000, 2_500_000, 5_000_000]
  @iterations 7
  @elem_type  {:f, 32}
  @warmup_n   10_000

  # ----------------------------------------------------------------
  # Funções GPU
  # ----------------------------------------------------------------

  # Map 1: Black-Scholes — compute pesado com log/exp/sqrt
  defp bs_fun do
    PolyHok.phok fn x ->
      s = x + 50.0
      k = 100.0
      t = 1.0
      r = 0.02
      v = 0.30
      sqrt_t = sqrt(t)
      d1 = (log(s / k) + (r + (v * v) / 2.0) * t) / (v * sqrt_t)
      d2 = d1 - v * sqrt_t
      cnd1 = 1.0 / (1.0 + exp(-1.702 * d1))
      cnd2 = 1.0 / (1.0 + exp(-1.702 * d2))
      s * cnd1 - k * exp(-r * t) * cnd2
    end
  end

  # Map 2: normalização simples encadeada após Black-Scholes
  defp norm_fun do
    PolyHok.phok fn x ->
      t1 = x * 0.01
      t2 = t1 * t1 + 1.0
      t1 / t2
    end
  end

  # Reduce: soma
  defp sum_fun do
    PolyHok.phok fn x, acc -> acc + x end
  end

  # ----------------------------------------------------------------
  # CPU equivalente
  # ----------------------------------------------------------------
  defp cpu_bs(x) do
    s = x + 50.0
    k = 100.0
    t = 1.0
    r = 0.02
    v = 0.30
    sqrt_t = :math.sqrt(t)
    d1 = (:math.log(s / k) + (r + (v * v) / 2.0) * t) / (v * sqrt_t)
    d2 = d1 - v * sqrt_t
    cnd1 = 1.0 / (1.0 + :math.exp(-1.702 * d1))
    cnd2 = 1.0 / (1.0 + :math.exp(-1.702 * d2))
    s * cnd1 - k * :math.exp(-r * t) * cnd2
  end

  defp cpu_norm(x) do
    t1 = x * 0.01
    t2 = t1 * t1 + 1.0
    t1 / t2
  end

  # ----------------------------------------------------------------
  # Warmup
  # ----------------------------------------------------------------
  defp warmup do
    IO.puts("Aquecendo (JIT + pools + flawd_ref)...")
    n    = @warmup_n
    list = Enum.map(1..n, fn _ -> :rand.uniform() * 100.0 end)
    nx   = Nx.tensor([list], type: @elem_type)
    bs   = bs_fun()
    nm   = norm_fun()
    sf   = sum_fun()

    # gnx
    nx |> PolyHok.new_gnx() |> Ske.map(bs) |> Ske.map(nm)
       |> Ske.reduce(0.0, sf) |> PolyHok.get_gnx()

    # flawd
    list |> PolyHok.new_flawd_list({1, n}, @elem_type)
         |> Ske.map(bs) |> Ske.map(nm)
         |> Ske.reduce(0.0, sf) |> PolyHok.get_flawd_list()

    # persistent
    list |> PolyHok.list_to_flawd(n, @elem_type)
         |> PolyHok.flawd_map(bs) |> PolyHok.flawd_map(nm)
         |> PolyHok.flawd_reduce(0.0, sf) |> PolyHok.flawd_to_list()

    IO.puts("Pronto.\n")
  end

  # ----------------------------------------------------------------
  # Pipelines
  # ----------------------------------------------------------------
  defp cpu_pipeline(list) do
    list
    |> Enum.map(&cpu_bs/1)
    |> Enum.map(&cpu_norm/1)
    |> Enum.reduce(0.0, fn x, acc -> acc + x end)
  end

  defp gnx_pipeline(list, nx) do
    bs = bs_fun()
    nm = norm_fun()
    sf = sum_fun()
    nx
    |> PolyHok.new_gnx()
    |> Ske.map(bs)
    |> Ske.map(nm)
    |> Ske.reduce(0.0, sf)
    |> PolyHok.get_gnx()
    |> then(fn t -> Nx.to_flat_list(t) |> hd() end)
  end

  defp flawd_pipeline(list, n) do
    bs = bs_fun()
    nm = norm_fun()
    sf = sum_fun()
    list
    |> PolyHok.new_flawd_list({1, n}, @elem_type)
    |> Ske.map(bs)
    |> Ske.map(nm)
    |> Ske.reduce(0.0, sf)
    |> PolyHok.get_flawd_list()
    |> hd()
  end

  defp persistent_pipeline(list, n) do
    bs = bs_fun()
    nm = norm_fun()
    sf = sum_fun()
    list
    |> PolyHok.list_to_flawd(n, @elem_type)
    |> PolyHok.flawd_map(bs)
    |> PolyHok.flawd_map(nm)
    |> PolyHok.flawd_reduce(0.0, sf)
    |> PolyHok.flawd_to_list()
    |> hd()
  end

  # ----------------------------------------------------------------
  # Corretude
  # ----------------------------------------------------------------
  defp verify do
    IO.puts("Verificando corretude...")
    n    = 1_000
    list = Enum.map(1..n, fn i -> i * 1.0 end)
    nx   = Nx.tensor([list], type: @elem_type)

    expected   = cpu_pipeline(list)
    got_gnx    = gnx_pipeline(list, nx)
    got_flawd  = flawd_pipeline(list, n)
    got_pers   = persistent_pipeline(list, n)

    tol = expected * 0.001  # 0.1% — float32 acumula erro

    for {label, got} <- [gnx: got_gnx, flawd: got_flawd, persistent: got_pers] do
      if abs(expected - got) > tol do
        raise "ERRO #{label}: esperado=#{expected} obtido=#{got} diff=#{abs(expected-got)}"
      end
    end

    IO.puts("OK (cpu=#{Float.round(expected, 4)}, todos dentro da tolerância float32).\n")
  end

  # ----------------------------------------------------------------
  # Medição
  # ----------------------------------------------------------------
  defp measure(f) do
    times = for _ <- 1..@iterations, do: elem(:timer.tc(f), 0)
    Enum.sort(times) |> Enum.at(div(@iterations, 2))
  end

  defp bench(n) do
    IO.write("n=#{n}... ")
    list = Enum.map(1..n, fn _ -> :rand.uniform() * 100.0 end)
    nx   = Nx.tensor([list], type: @elem_type)

    cpu        = measure(fn -> cpu_pipeline(list) end)
    gnx        = measure(fn -> gnx_pipeline(list, nx) end)
    flawd      = measure(fn -> flawd_pipeline(list, n) end)
    persistent = measure(fn -> persistent_pipeline(list, n) end)

    IO.puts("cpu=#{cpu}µs  gnx=#{gnx}µs  flawd=#{flawd}µs  persistent=#{persistent}µs")
    %{n: n, cpu: cpu, gnx: gnx, flawd: flawd, persistent: persistent}
  end

  defp save_csv(path, rows) do
    header = "n,cpu,gnx,flawd,persistent\n"
    body   = rows |> Enum.map(&"#{&1.n},#{&1.cpu},#{&1.gnx},#{&1.flawd},#{&1.persistent}")
                  |> Enum.join("\n")
    File.write!(path, header <> body <> "\n")
    IO.puts("CSV salvo em #{path}")
  end

  def run do
    verify()
    warmup()
    IO.puts("=== Pipeline: Black-Scholes |> normalização |> reduce(soma) ===")
    IO.puts("(mediana de #{@iterations} execuções, tempo em µs)\n")
    rows = Enum.map(@sizes, &bench/1)
    save_csv("/tmp/bench_persistent.csv", rows)
    IO.puts("\nPara plotar: python3 plot_persistent.py")
  end
end

BenchPersistent.run()
