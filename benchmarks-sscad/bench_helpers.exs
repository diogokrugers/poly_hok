# =============================================================================
# benchmarks-sscad/bench_helpers.exs
#
# Utilitários compartilhados pelos benchmarks do artigo SSCAD-WIC.
# Cada benchmark faz `Code.require_file("bench_helpers.exs", __DIR__)` e usa
# BenchHelpers.measure/1 e BenchHelpers.save_csv/2.
#
# Metodologia de medição:
#   - @iterations execuções por ponto, tempo em microssegundos (:timer.tc)
#   - reporta a MEDIANA (não a média) — padrão comum em benchmarks de
#     sistemas para reduzir o efeito de outliers (GC, scheduler, ruído do SO)
#   - uma rodada de warmup antes de medir, para não contar o custo de
#     primeira compilação JIT do kernel (o cache de kernel do PolyHok só
#     "esquenta" depois da primeira invocação de cada kernel único)
# =============================================================================

defmodule BenchHelpers do
  @doc """
  Executa `f` `iterations` vezes e retorna a MEDIANA do tempo em
  microssegundos. `iterations` default 7 (ímpar — mediana bem definida).
  """
  def measure(f, iterations \\ 7) do
    times = for _ <- 1..iterations, do: elem(:timer.tc(f), 0)
    Enum.sort(times) |> Enum.at(div(iterations, 2))
  end

  @doc """
  Salva `rows` (lista de maps, todos com as mesmas chaves) em CSV.
  `columns` define a ordem das colunas explicitamente (a ordem de um Map
  Elixir não é garantida).
  """
  def save_csv(path, rows, columns) do
    path |> Path.dirname() |> File.mkdir_p!()

    header = Enum.join(columns, ",") <> "\n"

    body =
      rows
      |> Enum.map(fn row ->
        columns
        |> Enum.map(&Map.fetch!(row, &1))
        |> Enum.join(",")
      end)
      |> Enum.join("\n")

    File.write!(path, header <> body <> "\n")
  end

  @doc "Calcula o speedup (CPU / GPU) formatado para exibição no terminal."
  def speedup(cpu_us, gpu_us) when gpu_us > 0, do: Float.round(cpu_us / gpu_us, 2)
  def speedup(_cpu_us, _gpu_us), do: :infinity

  @doc "Formata um tempo em microssegundos para exibição legível (µs ou ms)."
  def fmt_time(us) when us >= 1000, do: "#{Float.round(us / 1000, 2)}ms"
  def fmt_time(us), do: "#{us}µs"

  @doc """
  Imprime, ao final de um benchmark, o comando python para gerar o gráfico
  correspondente. Mantém a mensagem consistente entre todos os benchmarks.
  """
  def print_plot_hint(bench_name) do
    IO.puts("\nCSV salvo. Para gerar o gráfico, rode na sua máquina:")
    IO.puts("    python3 benchmarks-sscad/plot.py #{bench_name}")
  end
end
