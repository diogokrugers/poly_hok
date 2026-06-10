require PolyHok
use Ske

PolyHok.defmodule NBodiesBench do
                             #2048, 4096
  @sizes [128, 256, 512, 1024]
  @iterations 5
  @dtype {:f, 32}

  # ============================================================
  # GPU kernels
  # ============================================================

  defd gpu_nBodies(p,c,n) do

    softening = 0.000000001
    dt = 0.01

    fx = 0.0
    fy = 0.0
    fz = 0.0

    for j in range(0,n) do

      dx = c[6*j] - p[0]
      dy = c[6*j+1] - p[1]
      dz = c[6*j+2] - p[2]

      distSqr =
        dx*dx + dy*dy + dz*dz + softening

      invDist =
        1.0 / sqrt(distSqr)

      invDist3 =
        invDist * invDist * invDist

      fx = fx + dx * invDist3
      fy = fy + dy * invDist3
      fz = fz + dz * invDist3

    end

    p[3] = p[3] + dt*fx
    p[4] = p[4] + dt*fy
    p[5] = p[5] + dt*fz

  end

  defd gpu_integrate(p, dt, n) do

    p[0] = p[0] + p[3] * dt
    p[1] = p[1] + p[4] * dt
    p[2] = p[2] + p[5] * dt

  end

  # ============================================================
  # Generic launcher
  # ============================================================

  defk map_step_2_para_no_resp_kernel(
    d_array,
    step,
    par1,
    par2,
    size,
    f
  ) do

    globalId =
      blockDim.x *
      (gridDim.x * blockIdx.y + blockIdx.x) +
      threadIdx.x

    id = step * globalId

    if globalId < size do
      f(d_array + id, par1, par2)
    end

  end

  def map_2_para_no_resp(
  d_array,
  par1,
  par2,
  size,
  f
) do

  block_size = 128

  step = 6

  nBlocks =
    floor((size + block_size - 1) / block_size)

  PolyHok.spawn(
    &NBodiesBench.map_step_2_para_no_resp_kernel/6,
    {nBlocks,1,1},
    {block_size,1,1},
    [d_array, step, par1, par2, size, f]
  )

  d_array

end

  # ============================================================
  # CPU reference
  # ============================================================

  def cpu_step(bodies, n) do

    softening = 0.000000001
    dt = 0.01

    Enum.map(0..(n-1), fn i ->

      base_i = i * 6

      px = Enum.at(bodies, base_i)
      py = Enum.at(bodies, base_i + 1)
      pz = Enum.at(bodies, base_i + 2)

      vx = Enum.at(bodies, base_i + 3)
      vy = Enum.at(bodies, base_i + 4)
      vz = Enum.at(bodies, base_i + 5)

      {fx, fy, fz} =

        Enum.reduce(
          0..(n-1),
          {0.0,0.0,0.0},
          fn j, {afx,afy,afz} ->

            base_j = j * 6

            dx = Enum.at(bodies, base_j) - px
            dy = Enum.at(bodies, base_j + 1) - py
            dz = Enum.at(bodies, base_j + 2) - pz

            distSqr =
              dx*dx + dy*dy + dz*dz + softening

            invDist =
              1.0 / :math.sqrt(distSqr)

            invDist3 =
              invDist * invDist * invDist

            {
              afx + dx * invDist3,
              afy + dy * invDist3,
              afz + dz * invDist3
            }

          end)

      [

        px + vx * dt,
        py + vy * dt,
        pz + vz * dt,

        vx + fx * dt,
        vy + fy * dt,
        vz + fz * dt

      ]

    end)
    |> List.flatten()

  end

  # ============================================================
  # Warmup
  # ============================================================

  def warmup do

    IO.puts("Warmup...")

    n = 256

    list =
      Enum.map(1..(n*6), fn _ ->
        :rand.uniform()
      end)

    nx =
      Nx.tensor([list], type: @dtype)

    flawd_buf =
      PolyHok.new_flawd_list(
        list,
        {1,n*6},
        @dtype
      )

    flawd_buf
    |> map_2_para_no_resp(
      flawd_buf,
      n,
      n,
      &NBodiesBench.gpu_nBodies/3
    )
    |> map_2_para_no_resp(
      0.01,
      n,
      n,
      &NBodiesBench.gpu_integrate/3
    )
    |> PolyHok.get_flawd_list()

    gnx_buf =
      PolyHok.new_gnx(nx)

    gnx_buf
    |> map_2_para_no_resp(
      gnx_buf,
      n,
      n,
      &NBodiesBench.gpu_nBodies/3
    )
    |> map_2_para_no_resp(
      0.01,
      n,
      n,
      &NBodiesBench.gpu_integrate/3
    )
    |> PolyHok.get_gnx()

    IO.puts("Done.\n")

  end

  # ============================================================
  # Benchmark
  # ============================================================

  def run do

    warmup()

    rows =
      Enum.map(@sizes, &bench/1)

    save_csv("/tmp/bench_nbodies.csv", rows)

    IO.puts("\nCSV salvo em:")
    IO.puts("/tmp/bench_nbodies.csv")

    IO.puts("\nRode:")
    IO.puts("python3 plot_nbodies.py")

  end

  def bench(n) do

    IO.write("nBodies=#{n}... ")

    list =
      Enum.map(1..(n*6), fn _ ->
        :rand.uniform()
      end)

    nx =
      Nx.tensor([list], type: @dtype)

    flawd =

      measure(fn ->

        flawd_buf =
          PolyHok.new_flawd_list(
            list,
            {1,n*6},
            @dtype
          )

        flawd_buf
        |> map_2_para_no_resp(
          flawd_buf,
          n,
          n,
          &NBodiesBench.gpu_nBodies/3
        )
        |> map_2_para_no_resp(
          0.01,
          n,
          n,
          &NBodiesBench.gpu_integrate/3
        )
        |> PolyHok.get_flawd_list()

      end)

    gnx =

      measure(fn ->

        gnx_buf =
          PolyHok.new_gnx(nx)

        gnx_buf
        |> map_2_para_no_resp(
          gnx_buf,
          n,
          n,
          &NBodiesBench.gpu_nBodies/3
        )
        |> map_2_para_no_resp(
          0.01,
          n,
          n,
          &NBodiesBench.gpu_integrate/3
        )
        |> PolyHok.get_gnx()

      end)

    cpu =

      measure(fn ->
        cpu_step(list, n)
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
  # Utils
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
        "#{r.n},#{r.flawd},#{r.gnx},#{r.cpu}"
      end)
      |> Enum.join("\n")

    File.write!(
      path,
      header <> body <> "\n"
    )

  end

end

NBodiesBench.run()
