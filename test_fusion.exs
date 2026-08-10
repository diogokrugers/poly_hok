require PolyHok
use Ske
require Fusion

# =============================================================================
# Teste 1: Fusion.with_fusion com lista Elixir nativa (flawd)
# =============================================================================

IO.puts("=== Teste 1: fusão com lista Elixir (flawd) ===")

lista = Enum.to_list(1..20)

resultado_lista =
  Fusion.with_fusion(
    Ske.map(lista, PolyHok.phok fn x -> x + 1 end)
    |> Ske.map(PolyHok.phok fn y -> y * 2 end)
  )

IO.inspect(resultado_lista, label: "resultado (lista)")
IO.inspect(is_list(resultado_lista), label: "é lista Elixir?")

# =============================================================================
# Teste 2: Fusion.with_fusion com gnx (Nx.tensor -> PolyHok.new_gnx)
# =============================================================================

IO.puts("\n=== Teste 2: fusão com gnx ===")

tensor = Nx.tensor([Enum.to_list(1..20)], type: {:s, 32})
gnx = PolyHok.new_gnx(tensor)

resultado_gnx_gpu =
  Fusion.with_fusion(
    Ske.map(gnx, PolyHok.phok fn x -> x + 1 end)
    |> Ske.map(PolyHok.phok fn y -> y * 2 end)
  )

resultado_gnx = resultado_gnx_gpu |> PolyHok.get_gnx() |> Nx.to_flat_list()

IO.inspect(resultado_gnx, label: "resultado (gnx -> Nx -> lista)")

# =============================================================================
# Checagem: os dois caminhos devem produzir o mesmo resultado numérico
# =============================================================================

IO.puts("\n=== Comparação ===")

if resultado_lista == resultado_gnx do
  IO.puts("OK: lista e gnx produziram o mesmo resultado.")
else
  IO.puts("DIVERGÊNCIA:")
  IO.inspect(resultado_lista, label: "lista")
  IO.inspect(resultado_gnx, label: "gnx")
end
