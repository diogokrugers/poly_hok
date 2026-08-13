require PolyHok
use Ske

defmodule TesteSaxpy do
  def gen(n) do
    xs = for _ <- 1..n, do: :rand.uniform() * 10.0
    ys = for _ <- 1..n, do: :rand.uniform() * 10.0
    {xs, ys}
  end

  def bench(n) do
    IO.write("n=#{n}... ")
    {xs, ys} = gen(n)
    nx_x = Nx.tensor([xs], type: {:f, 32})
    nx_y = Nx.tensor([ys], type: {:f, 32})
    f = PolyHok.phok(fn x, y -> 2.5 * x + y end)

    IO.write("flawd...")
    for _ <- 1..7, do: xs |> Ske.map2(ys, f)
    IO.write("ok ")

    IO.write("gnx...")
    for _ <- 1..7 do
      PolyHok.new_gnx(nx_x) |> Ske.map2(PolyHok.new_gnx(nx_y), f) |> PolyHok.get_gnx()
    end
    IO.write("ok ")

    IO.puts("cpu... (pulando, não usa GPU)")
  end
end

TesteSaxpy.bench(1000)
TesteSaxpy.bench(10000)
TesteSaxpy.bench(100000)
