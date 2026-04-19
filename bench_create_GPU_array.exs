# bench.exs — Benchmark glist vs gnx (até 1_000_000)
# Uso: mix run bench.exs

IO.puts("=== Teste básico: new_glist / get_glist ===\n")

# ─────────────────────────────────────────
# 1. Sanidade
# ─────────────────────────────────────────

lista_1d = [1.0, 2.0, 3.0, 4.0]
g = PolyHok.new_glist(lista_1d, {1, 4}, {:f, 32})
IO.inspect(g, label: "glist criada")

resultado = PolyHok.get_glist(g)
IO.inspect(resultado, label: "get_glist")

if Enum.map(resultado, &Float.round(&1, 4)) == lista_1d do
  IO.puts("✓ Valores batem!\n")
else
  IO.puts("✗ Valores DIVERGEM: #{inspect(resultado)}\n")
end

# 2D
lista_2d = [[1.0, 2.0], [3.0, 4.0]]
g2 = PolyHok.new_glist(lista_2d, {2, 2}, {:f, 32})
resultado_2d = PolyHok.get_glist(g2)
IO.inspect(resultado_2d, label: "get_glist 2D (flat)")

# ─────────────────────────────────────────
# 2. Benchmark glist
# ─────────────────────────────────────────

IO.puts("\n=== Benchmark glist ===\n")

tamanhos = [
  1_000,
  10_000,
  100_000,
  250_000,
  500_000,
  1_000_000
]

for n <- tamanhos do
  lista = for i <- 1..n, do: i * 1.0

  {tempo_new, g} =
    :timer.tc(fn -> PolyHok.new_glist(lista, {1, n}, {:f, 32}) end)

  {tempo_get, _} =
    :timer.tc(fn -> PolyHok.get_glist(g) end)

  IO.puts("n=#{n} | new_glist: #{tempo_new}µs | get_glist: #{tempo_get}µs")
end

# ─────────────────────────────────────────
# 3. Comparação glist vs gnx
# ─────────────────────────────────────────

IO.puts("\n=== Comparação glist vs gnx ===\n")

IO.puts(
  String.pad_trailing("n", 12) <>
  String.pad_trailing("glist new (µs)", 20) <>
  String.pad_trailing("gnx new (µs)", 18) <>
  String.pad_trailing("glist get (µs)", 20) <>
  "gnx get (µs)"
)

IO.puts(String.duplicate("-", 90))

for n <- tamanhos do
  lista = for i <- 1..n, do: i * 1.0

  # ───── glist
  {t_glist_new, g_list} =
    :timer.tc(fn -> PolyHok.new_glist(lista, {1, n}, {:f, 32}) end)

  {t_glist_get, _} =
    :timer.tc(fn -> PolyHok.get_glist(g_list) end)

  # ───── gnx
  tensor = Nx.tensor(lista, type: {:f, 32})

  {t_gnx_new, g_nx} =
    :timer.tc(fn -> PolyHok.new_gnx(tensor) end)

  {t_gnx_get, _} =
    :timer.tc(fn -> PolyHok.get_gnx(g_nx) end)

  IO.puts(
    String.pad_trailing("#{n}", 12) <>
    String.pad_trailing("#{t_glist_new}", 20) <>
    String.pad_trailing("#{t_gnx_new}", 18) <>
    String.pad_trailing("#{t_glist_get}", 20) <>
    "#{t_gnx_get}"
  )
end

IO.puts("\nFim do benchmark.")
