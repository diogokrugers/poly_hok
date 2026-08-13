# =============================================================================
# benchmarks-sscad/nbodies.exs
#
# N-Bodies: simulação da interação gravitacional entre n corpos em 3D
# (posição + velocidade, 6 floats por corpo).
#
# ESCALA DIFERENTE DOS DEMAIS BENCHMARKS!
#
# Compara três caminhos:
#   flawd — listas Elixir |> PolyHok.new_flawd_list/get_flawd_list
#   gnx   — caminho de baixo nível original do PolyHok (Nx.tensor -> GNx)
#   cpu   — Elixir puro
#
# Uso:
#   mix run benchmarks-sscad/nbodies.exs
#
# Saída:
#   benchmarks-sscad/resultados/nbodies.csv
# =============================================================================

Code.require_file("bench_helpers.exs", __DIR__)

require PolyHok
use Ske

PolyHok.defmodule NBodiesBench do
  @sizes [128, 256, 512, 1024, 2048]
  @iterations 5
  @dtype {:f, 32}

  # ----------------------------------------------------------------
  # GPU kernels (mesma lógica de benchmark_nbodies.exs)
  # ----------------------------------------------------------------

  defd gpu_nBodies(p, c, n) do
    softening = 0.000000001
    dt = 0.01

    fx = 0.0
    fy = 0.0
    fz = 0.0

    for j in range(0, n) do
      dx = c[6 * j] - p[0]
      dy = c[6 * j + 1] - p[1]
      dz = c[6 * j + 2] - p[2]

      distSqr = dx * dx + dy * dy + dz * dz + softening
      invDist = 1.0 / sqrt(distSqr)
      invDist3 = invDist * invDist * invDist

      fx = fx + dx * invDist3
      fy = fy + dy * invDist3
      fz = fz + dz * invDist3
    end

    p[3] = p[3] + dt * fx
    p[4] = p[4] + dt * fy
    p[5] = p[5] + dt * fz
  end

  defd gpu_integrate(p, dt, n) do
    p[0] = p[0] + p[3] * dt
    p[1] = p[1] + p[4] * dt
    p[2] = p[2] + p[5] * dt
  end

  # ----------------------------------------------------------------
  # Launcher genérico (mesmo de benchmark_nbodies.exs)
  # ----------------------------------------------------------------

  defk map_step_2_para_no_resp_kernel(d_array, step, par1, par2, size, f) do
    globalId = blockDim.x * (gridDim.x * blockIdx.y + blockIdx.x) + threadIdx.x
    id = step * globalId

    if globalId < size do
      f(d_array + id, par1, par2)
    end
  end

  def map_2_para_no_resp(d_array, par1, par2, size, f) do
    block_size = 128
    step = 6
    nBlocks = floor((size + block_size - 1) / block_size)

    PolyHok.spawn(
      &NBodiesBench.map_step_2_para_no_resp_kernel/6,
      {nBlocks, 1, 1},
      {block_size, 1, 1},
      [d_array, step, par1, par2, size, f]
    )

    d_array
  end

  # ----------------------------------------------------------------
  # CPU
  # ----------------------------------------------------------------

  def cpu_step(bodies, n) do
    softening = 0.000000001
    dt = 0.01

    Enum.map(0..(n - 1), fn i ->
      base_i = i * 6

      px = Enum.at(bodies, base_i)
      py = Enum.at(bodies, base_i + 1)
      pz = Enum.at(bodies, base_i + 2)

      vx = Enum.at(bodies, base_i + 3)
      vy = Enum.at(bodies, base_i + 4)
      vz = Enum.at(bodies, base_i + 5)

      {fx, fy, fz} =
        Enum.reduce(0..(n - 1), {0.0, 0.0, 0.0}, fn j, {afx, afy, afz} ->
          base_j = j * 6

          dx = Enum.at(bodies, base_j) - px
          dy = Enum.at(bodies, base_j + 1) - py
          dz = Enum.at(bodies, base_j + 2) - pz

          distSqr = dx * dx + dy * dy + dz * dz + softening
          invDist = 1.0 / :math.sqrt(distSqr)
          invDist3 = invDist * invDist * invDist

          {afx + dx * invDist3, afy + dy * invDist3, afz + dz * invDist3}
        end)

      [px + vx * dt, py + vy * dt, pz + vz * dt, vx + fx * dt, vy + fy * dt, vz + fz * dt]
    end)
    |> List.flatten()
  end

  # ----------------------------------------------------------------
  # Geração de dados
  # ----------------------------------------------------------------

  defp gen_bodies(n), do: for(_ <- 1..(n * 6), do: :rand.uniform())

  # ----------------------------------------------------------------
  # Warmup
  # ----------------------------------------------------------------

  def warmup do
    IO.puts("Warmup...")
    n = 256
    list = gen_bodies(n)
    nx = Nx.tensor([list], type: @dtype)

    flawd_buf = PolyHok.new_flawd_list(list, {1, n * 6}, @dtype)

    flawd_buf
    |> map_2_para_no_resp(flawd_buf, n, n, &NBodiesBench.gpu_nBodies/3)
    |> map_2_para_no_resp(0.01, n, n, &NBodiesBench.gpu_integrate/3)
    |> PolyHok.get_flawd_list()

    gnx_buf = PolyHok.new_gnx(nx)

    gnx_buf
    |> map_2_para_no_resp(gnx_buf, n, n, &NBodiesBench.gpu_nBodies/3)
    |> map_2_para_no_resp(0.01, n, n, &NBodiesBench.gpu_integrate/3)
    |> PolyHok.get_gnx()

    IO.puts("Pronto.\n")
  end

  # ----------------------------------------------------------------
  # Um ponto do benchmark
  # ----------------------------------------------------------------

  def bench(n) do
    IO.write("nbodies n=#{n}... ")

    list = gen_bodies(n)
    nx = Nx.tensor([list], type: @dtype)

    flawd =
      BenchHelpers.measure(
        fn ->
          flawd_buf = PolyHok.new_flawd_list(list, {1, n * 6}, @dtype)

          flawd_buf
          |> map_2_para_no_resp(flawd_buf, n, n, &NBodiesBench.gpu_nBodies/3)
          |> map_2_para_no_resp(0.01, n, n, &NBodiesBench.gpu_integrate/3)
          |> PolyHok.get_flawd_list()
        end,
        @iterations
      )

    gnx =
      BenchHelpers.measure(
        fn ->
          gnx_buf = PolyHok.new_gnx(nx)

          gnx_buf
          |> map_2_para_no_resp(gnx_buf, n, n, &NBodiesBench.gpu_nBodies/3)
          |> map_2_para_no_resp(0.01, n, n, &NBodiesBench.gpu_integrate/3)
          |> PolyHok.get_gnx()
        end,
        @iterations
      )

    cpu = BenchHelpers.measure(fn -> cpu_step(list, n) end, @iterations)

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

    out_csv = Path.join([__DIR__, "resultados", "nbodies.csv"])
    BenchHelpers.save_csv(out_csv, rows, [:n, :flawd, :gnx, :cpu])
    IO.puts("\nCSV salvo em: #{out_csv}")
    BenchHelpers.print_plot_hint("nbodies")
  end
end

NBodiesBench.run()
