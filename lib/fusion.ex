defmodule Fusion do
  require PolyHok

  # NOTE on list (flawd) support:
  #
  # Fusion.with_fusion/1 parses a pipe chain of Ske.map/map2/map3/map4/reduce
  # calls at COMPILE TIME and rewrites it into a single call to one of
  # Ske.map, Ske.map2, Ske.map3, Ske.map4, Ske.mapReduce, Ske.map2Reduce or
  # Ske.map4Reduce (see emit_fused_chain/1 below). It never inspects the
  # runtime value of the data arguments — it only manipulates AST.
  #
  # Ske.map/2 (and map2/3/4, reduce, mapReduce, map2Reduce, map4Reduce) were
  # extended to accept plain Elixir lists as data arguments, in addition to
  # the usual gnx tensors: when a list is passed, it is transparently lifted
  # to the GPU via PolyHok.new_flawd_list/3, the fused kernel runs exactly
  # as it would for a gnx, and the result is lowered back to a native list
  # via PolyHok.get_flawd_list/1 before being returned.
  #
  # Because that detection happens inside Ske's runtime functions, Fusion
  # needs no changes to recognize lists: a pipeline written as
  #
  #     Fusion.with_fusion(
  #       Ske.map(lista, PolyHok.phok fn x -> x + 1 end)
  #       |> Ske.map(PolyHok.phok fn y -> y * 2 end)
  #     )
  #
  # is expanded to Ske.map(lista, fused_fn) exactly like the gnx case, and
  # Ske.map/2 then decides at runtime whether `lista` is a list or a gnx.
  # The whole fused kernel still runs as a single GPU launch — only the
  # list<->gnx boundary conversion (already implemented by flawd) wraps it.

  # NOTE on the implicit-first-input convenience form:
  #
  # The original fusion parser (parse_ske_call/1, below) requires every
  # stage of the pipe chain, including the first one, to be syntactically
  # an explicit Ske.map/map2/map3/map4/reduce call, e.g.
  #
  #     Ske.map(lista, f) |> Ske.map(g)
  #
  # A more "natural" pipe form, where the data is piped into the first
  # Ske call instead of passed explicitly:
  #
  #     lista |> Ske.map(f) |> Ske.map(g)
  #
  # is NOT recognized by that parser on its own, because after Elixir's
  # own pipe-macro expansion the first flattened node is just the bare
  # value `lista`, not an Ske.xxx call.
  #
  # with_fusion/1 now normalizes this convenience form into the explicit
  # one BEFORE any existing parsing logic runs (see
  # normalize_leading_pipe_value/1 below): if the leftmost node of the
  # whole pipe chain is not itself a recognizable Ske.xxx call, it is
  # spliced in as the first explicit argument of the first Ske.xxx call
  # found in the chain, producing exactly the same AST that writing the
  # explicit form by hand would have produced. Every already-working
  # explicit form is left completely untouched by this normalization
  # (see the guard in normalize_leading_pipe_value/1), so nothing that
  # previously worked changes behavior.

  defmodule AstCall do
    @moduledoc false

    @doc """
     :ske             -> :map, :map2, :map3, :map4, :reduce...
     :kernel_ast      -> raw AST
     :explicit_inputs -> tensor/scalar inputs written explicitly in the call
     :initial_ast     -> reduce initial value, when present
     :piped_input?    -> true when the stage expects the prior pipe value
     :stage_kind      -> :map or :reduce
     :terminal?       -> true for terminal stages such as reduce
     :output_kind     -> :tensor or :scalar
    """
    defstruct [
      :ske,
      :kernel_ast,
      explicit_inputs: [],
      initial_ast: nil,
      piped_input?: false,
      stage_kind: :map,
      terminal?: false,
      output_kind: :tensor
    ]
  end

  defp supported_skeleton_names, do: [:map, :map2, :map3, :map4, :reduce]

  defp skeleton_meta(:map) do
    %{
      first_inputs: 1,
      piped_inputs: 0,
      stage_kind: :map,
      terminal?: false,
      output_kind: :tensor
    }
  end

  defp skeleton_meta(:map2) do
    %{
      first_inputs: 2,
      piped_inputs: 1,
      stage_kind: :map,
      terminal?: false,
      output_kind: :tensor
    }
  end

  defp skeleton_meta(:map3) do
    %{
      first_inputs: 3,
      piped_inputs: 2,
      stage_kind: :map,
      terminal?: false,
      output_kind: :tensor
    }
  end

  defp skeleton_meta(:map4) do
    %{
      first_inputs: 4,
      piped_inputs: 3,
      stage_kind: :map,
      terminal?: false,
      output_kind: :tensor
    }
  end

  defp skeleton_meta(:reduce) do
    %{
      first_inputs: 1,
      piped_inputs: 0,
      stage_kind: :reduce,
      terminal?: true,
      output_kind: :scalar
    }
  end

  defp new_skecall(ske_name, explicit_inputs, kernel_ast, opts \\ []) do
    meta = skeleton_meta(ske_name)

    %AstCall{
      ske: ske_name,
      kernel_ast: kernel_ast,
      explicit_inputs: explicit_inputs,
      initial_ast: Keyword.get(opts, :initial_ast),
      piped_input?: Keyword.get(opts, :piped_input?, false),
      stage_kind: meta.stage_kind,
      terminal?: meta.terminal?,
      output_kind: meta.output_kind
    }
  end

  defp parse_ske_call({{:., _meta1, [{_alias, _meta2, [:Ske]}, ske_name]}, _meta3, args})
       when ske_name in [:map, :map2, :map3, :map4, :reduce] do
    do_parse_ske_call(ske_name, args)
  end

  defp parse_ske_call({{:., _meta1, [{_alias, _meta2, [:Ske]}, ske_name]}, _meta3, _args}) do
    raise ArgumentError,
          "Fusion.with_fusion/1 does not support Ske.#{ske_name}/...; supported skeletons are #{supported_skeletons_message()}"
  end

  defp parse_ske_call(other) do
    raise ArgumentError,
          "Fusion.with_fusion/1 expects explicit #{supported_skeletons_message()} calls in a pipe chain, got: #{Macro.to_string(other)}"
  end

  defp do_parse_ske_call(ske_name, args) do
    case {ske_name, args} do
      {:map, [kernel_ast]} ->
        new_skecall(:map, [], kernel_ast, piped_input?: true)

      {:map, [data_ast, kernel_ast]} ->
        new_skecall(:map, [data_ast], kernel_ast)

      {:map2, [data_ast, kernel_ast]} ->
        new_skecall(:map2, [data_ast], kernel_ast, piped_input?: true)

      {:map2, [data1, data2, kernel_ast]} ->
        new_skecall(:map2, [data1, data2], kernel_ast)

      {:map3, [data1, data2, kernel_ast]} ->
        new_skecall(:map3, [data1, data2], kernel_ast, piped_input?: true)

      {:map3, [data1, data2, data3, kernel_ast]} ->
        new_skecall(:map3, [data1, data2, data3], kernel_ast)

      {:map4, [data1, data2, data3, kernel_ast]} ->
        new_skecall(:map4, [data1, data2, data3], kernel_ast, piped_input?: true)

      {:map4, [data1, data2, data3, data4, kernel_ast]} ->
        new_skecall(:map4, [data1, data2, data3, data4], kernel_ast)

      {:reduce, [initial, kernel_ast]} ->
        new_skecall(:reduce, [], kernel_ast, initial_ast: initial, piped_input?: true)

      {:reduce, [data_ast, initial, kernel_ast]} ->
        new_skecall(:reduce, [data_ast], kernel_ast, initial_ast: initial)

      _ ->
        raise ArgumentError,
              "malformed Ske.#{ske_name} call in Fusion.with_fusion/1: expected #{skeleton_call_shapes(ske_name)}, got #{length(args)} argument(s)"
    end
  end

  defp supported_skeletons_message do
    supported_skeleton_names()
    |> Enum.map_join("/", &"Ske.#{&1}")
  end

  defp skeleton_call_shapes(:map), do: "Ske.map(input, kernel) or piped Ske.map(kernel)"

  defp skeleton_call_shapes(:map2),
    do: "Ske.map2(input1, input2, kernel) or piped Ske.map2(input2, kernel)"

  defp skeleton_call_shapes(:map3),
    do: "Ske.map3(input1, input2, input3, kernel) or piped Ske.map3(input2, input3, kernel)"

  defp skeleton_call_shapes(:map4),
    do:
      "Ske.map4(input1, input2, input3, input4, kernel) or piped Ske.map4(input2, input3, input4, kernel)"

  defp skeleton_call_shapes(:reduce),
    do: "Ske.reduce(input, initial, kernel) or piped Ske.reduce(initial, kernel)"

  defp flatten_pipe_ast({:|>, _meta, [lhs, rhs]}) do
    flatten_pipe_ast(lhs) ++ flatten_pipe_ast(rhs)
  end

  defp flatten_pipe_ast(ast), do: [ast]

  # Recognizes a bare Ske.xxx(...) call node, the same shape parse_ske_call/1
  # matches on. Used only to decide whether normalization is needed — it does
  # not replace or duplicate any of parse_ske_call/1's own validation.
  defp ske_call_node?({{:., _meta1, [{_alias, _meta2, [:Ske]}, ske_name]}, _meta3, _args})
       when ske_name in [:map, :map2, :map3, :map4, :reduce],
       do: true

  defp ske_call_node?(_other), do: false

  # Splices `value_ast` in as the first argument of an Ske.xxx(...) call
  # node. Only called from normalize_leading_pipe_value/1 below, on the
  # right-hand side of the leftmost pipe segment (i.e. before the chain is
  # flattened). The guard below does the actual recognition; see the
  # fallback clause just after for what happens when rhs isn't one.
  defp splice_leading_value({{:., meta1, [alias_ast, ske_name]}, meta3, args}, value_ast)
       when ske_name in [:map, :map2, :map3, :map4, :reduce] do
    {{:., meta1, [alias_ast, ske_name]}, meta3, [value_ast | args]}
  end

  # rhs is something other than a recognizable Ske.xxx call (e.g. a stray
  # function call between the bare value and the first Ske stage). Rather
  # than raising an opaque FunctionClauseError, fall through to the same
  # descriptive ArgumentError parse_ske_call/1 already raises for any
  # other unrecognized pipe node, so error quality for malformed chains is
  # unchanged by this normalization.
  defp splice_leading_value(other, _value_ast), do: parse_ske_call(other)

  # Normalizes `value |> Ske.xxx(...) |> ...` into
  # `Ske.xxx(value, ...) |> ...` by splicing `value` into the first Ske.xxx
  # call of the chain, but ONLY when the leading node is not already a
  # recognizable Ske.xxx call itself. When it already is one (the existing,
  # already-supported explicit form), the AST is returned completely
  # unchanged, so every previously-working pipeline keeps parsing exactly
  # as before.
  defp normalize_leading_pipe_value({:|>, meta, [lhs, rhs]} = ast) do
    if match?({:|>, _, _}, lhs) do
      {:|>, meta, [normalize_leading_pipe_value(lhs), rhs]}
    else
      if ske_call_node?(lhs) do
        ast
      else
        splice_leading_value(rhs, lhs)
      end
    end
  end

  defp normalize_leading_pipe_value(ast), do: ast

  defp validate_call_positions(calls) do
    calls
    |> Enum.with_index()
    |> Enum.each(fn {call, idx} -> validate_call_position!(call, idx) end)

    calls
  end

  defp validate_call_position!(%AstCall{} = call, idx) do
    meta = skeleton_meta(call.ske)
    actual = length(call.explicit_inputs)

    {expected_piped?, expected_inputs, stage_label} =
      if idx == 0 do
        {false, meta.first_inputs, "first stage #{call.ske}"}
      else
        {true, meta.piped_inputs, "stage #{idx + 1} #{call.ske}"}
      end

    cond do
      call.piped_input? != expected_piped? ->
        expected_shape =
          if expected_piped? do
            "piped #{call.ske} shape with #{expected_inputs} explicit input(s)"
          else
            "standalone #{call.ske} shape with #{expected_inputs} explicit input(s)"
          end

        raise ArgumentError,
              "#{stage_label} expects #{expected_shape}, got #{call_shape_description(call)}"

      actual != expected_inputs ->
        raise ArgumentError,
              "#{stage_label} expects #{expected_inputs} explicit input(s), got #{actual}"

      true ->
        :ok
    end
  end

  defp call_shape_description(%AstCall{} = call) do
    shape =
      if call.piped_input? do
        "piped"
      else
        "standalone"
      end

    "#{shape} #{call.ske} shape with #{length(call.explicit_inputs)} explicit input(s)"
  end

  defp foldable_scalar_ast?(ast) when is_integer(ast) or is_float(ast), do: true
  defp foldable_scalar_ast?({:-, _, [v]}) when is_integer(v) or is_float(v), do: true
  defp foldable_scalar_ast?(_), do: false

  defp resolve_external_input(data_ast, state) do
    if foldable_scalar_ast?(data_ast) do
      {data_ast, state}
    else
      key = Macro.to_string(data_ast)

      case state.input_vars[key] do
        nil ->
          var_atom = String.to_atom("arg#{state.next_input_idx}")
          var_ast = {var_atom, [], nil}

          new_state = %{
            state
            | next_input_idx: state.next_input_idx + 1,
              input_vars: Map.put(state.input_vars, key, var_ast),
              input_order: state.input_order ++ [{data_ast, var_ast}]
          }

          {var_ast, new_state}

        var_ast ->
          {var_ast, state}
      end
    end
  end

  defp resolve_external_inputs([], state), do: {[], state}

  defp resolve_external_inputs([data_ast | rest], state) do
    {arg_ast, state1} = resolve_external_input(data_ast, state)
    {other_asts, state2} = resolve_external_inputs(rest, state1)
    {[arg_ast | other_asts], state2}
  end

  defp collect_pattern_vars(ast, acc) do
    Macro.prewalk(ast, acc, fn
      {var, _meta, ctx} = node, vars
      when is_atom(var) and (is_atom(ctx) or is_nil(ctx)) ->
        {node, MapSet.put(vars, var)}

      node, vars ->
        {node, vars}
    end)
    |> elem(1)
  end

  defp collect_local_vars(body, param_vars) do
    Enum.reduce(List.wrap(body), MapSet.new(), fn node, acc ->
      case node do
        {:=, _, [lhs, _rhs]} ->
          MapSet.union(acc, collect_pattern_vars(lhs, MapSet.new()))

        {:type, _, [decl]} ->
          case decl do
            {var, _meta, _args} when is_atom(var) ->
              MapSet.put(acc, var)

            _ ->
              acc
          end

        _ ->
          acc
      end
    end)
    |> MapSet.difference(param_vars)
    |> MapSet.difference(MapSet.new([:return, :type]))
  end

  defp rename_local_vars(body, stage_idx, local_vars) do
    local_map =
      local_vars
      |> Enum.map(fn name -> {name, String.to_atom("s#{stage_idx}_#{name}")} end)
      |> Map.new()

    Macro.prewalk(body, fn
      {var, meta, ctx} when is_atom(var) and (is_atom(ctx) or is_nil(ctx)) ->
        case local_map[var] do
          nil -> {var, meta, ctx}
          renamed -> {renamed, meta, ctx}
        end

      node ->
        node
    end)
  end

  defp declared_var_from_type({var, _meta, _args}) when is_atom(var), do: var
  defp declared_var_from_type(_), do: nil

  defp substitute_expr({:=, meta, [lhs, rhs]}, param_map) do
    {:=, meta, [substitute_pattern(lhs, param_map), substitute_expr(rhs, param_map)]}
  end

  defp substitute_expr({:type, meta, [decl]}, param_map) do
    {:type, meta, [substitute_pattern(decl, param_map)]}
  end

  defp substitute_expr({var, _meta, ctx} = node, param_map)
       when is_atom(var) and (is_atom(ctx) or is_nil(ctx)) do
    case param_map[var] do
      nil -> node
      ast -> ast
    end
  end

  defp substitute_expr({form, meta, args}, param_map) when is_list(args) do
    {form, meta, Enum.map(args, &substitute_expr(&1, param_map))}
  end

  defp substitute_expr(list, param_map) when is_list(list) do
    Enum.map(list, &substitute_expr(&1, param_map))
  end

  defp substitute_expr(other, _param_map), do: other

  defp substitute_pattern({var, _meta, ctx} = node, _param_map)
       when is_atom(var) and (is_atom(ctx) or is_nil(ctx)) do
    case var do
      :_ -> node
      _ -> node
    end
  end

  defp substitute_pattern({form, meta, args}, param_map) when is_list(args) do
    {form, meta, Enum.map(args, &substitute_pattern(&1, param_map))}
  end

  defp substitute_pattern(list, param_map) when is_list(list) do
    Enum.map(list, &substitute_pattern(&1, param_map))
  end

  defp substitute_pattern(other, _param_map), do: other

  defp substitute_params(body, param_map) do
    {nodes, _final_map} =
      body
      |> List.wrap()
      |> Enum.map_reduce(param_map, fn node, env ->
        case node do
          {:=, meta, [lhs, rhs]} ->
            lhs_sub = substitute_pattern(lhs, env)
            rhs_sub = substitute_expr(rhs, env)

            next_env =
              lhs
              |> collect_pattern_vars(MapSet.new())
              |> Enum.reduce(env, fn var, acc -> Map.delete(acc, var) end)

            {{:=, meta, [lhs_sub, rhs_sub]}, next_env}

          {:type, meta, [decl]} ->
            decl_sub = substitute_pattern(decl, env)

            next_env =
              case declared_var_from_type(decl) do
                nil -> env
                var -> Map.delete(env, var)
              end

            {{:type, meta, [decl_sub]}, next_env}

          other ->
            {substitute_expr(other, env), env}
        end
      end)

    nodes
  end

  defp inline_kernel_with_args(kernel_ast, actual_args, stage_idx) do
    {formal_args, body} = decompose_kernel(kernel_ast)

    if length(formal_args) != length(actual_args) do
      raise ArgumentError,
            "fusion arity mismatch in stage #{stage_idx}: kernel expects #{length(formal_args)} args, got #{length(actual_args)}"
    end

    param_vars =
      formal_args
      |> Enum.map(fn {var, _meta, _ctx} -> var end)
      |> MapSet.new()

    local_vars = collect_local_vars(body, param_vars)
    renamed = rename_local_vars(body, stage_idx, local_vars)

    param_map =
      Enum.zip(formal_args, actual_args)
      |> Enum.map(fn {{name, _meta, _ctx}, arg_ast} -> {name, arg_ast} end)
      |> Map.new()

    substituted = substitute_params(renamed, param_map)
    split_body_and_return(substituted)
  end

  defp fuse_map_stage(call, stage_idx, state) do
    {actual_args, state1, stage_prelude} =
      if call.piped_input? do
        {extra_args, state_after_inputs} = resolve_external_inputs(call.explicit_inputs, state)
        input_var = {String.to_atom("__fuse_in_#{stage_idx}"), [], nil}
        prelude = [{:=, [], [input_var, state.value_ast]}]
        {[input_var | extra_args], state_after_inputs, prelude}
      else
        {resolved, s} = resolve_external_inputs(call.explicit_inputs, state)
        {resolved, s, []}
      end

    {stage_prefix, stage_value} = inline_kernel_with_args(call.kernel_ast, actual_args, stage_idx)

    %{
      state1
      | body: state1.body ++ stage_prelude ++ stage_prefix,
        value_ast: stage_value
    }
  end

  defp emit_map_from_inputs_and_fun([t1], fun_ast) do
    quote do
      Ske.map(unquote(t1), unquote(fun_ast))
    end
  end

  defp emit_map_from_inputs_and_fun([t1, t2], fun_ast) do
    quote do
      Ske.map2(unquote(t1), unquote(t2), unquote(fun_ast))
    end
  end

  defp emit_map_from_inputs_and_fun([t1, t2, t3], fun_ast) do
    quote do
      Ske.map3(unquote(t1), unquote(t2), unquote(t3), unquote(fun_ast))
    end
  end

  defp emit_map_from_inputs_and_fun([t1, t2, t3, t4], fun_ast) do
    quote do
      Ske.map4(unquote(t1), unquote(t2), unquote(t3), unquote(t4), unquote(fun_ast))
    end
  end

  defp emit_map_from_inputs_and_fun(inputs, _fun_ast) do
    raise ArgumentError,
          "full-chain fusion requires <= 4 tensor inputs with current Ske API, got #{length(inputs)}"
  end

  defp fusion_data_key(data_ast, state) do
    if foldable_scalar_ast?(data_ast) do
      {{:scalar, strip_ast_metadata(data_ast)}, state}
    else
      key = Macro.to_string(strip_ast_metadata(data_ast))

      case state.input_vars[key] do
        nil ->
          input_key = {:input, state.next_input_idx}

          new_state = %{
            state
            | next_input_idx: state.next_input_idx + 1,
              input_vars: Map.put(state.input_vars, key, input_key)
          }

          {input_key, new_state}

        input_key ->
          {input_key, state}
      end
    end
  end

  defp fusion_data_keys([], state), do: {[], state}

  defp fusion_data_keys([data_ast | rest], state) do
    {data_key, state1} = fusion_data_key(data_ast, state)
    {data_keys, state2} = fusion_data_keys(rest, state1)
    {[data_key | data_keys], state2}
  end

  defp normalize_fusion_key(calls) do
    state0 = %{next_input_idx: 0, input_vars: %{}}

    {stages, _state} =
      calls
      |> Enum.map_reduce(state0, fn call, state ->
        {data_keys, state1} = fusion_data_keys(call.explicit_inputs, state)
        kernel_key = normalize_kernel_key(call.kernel_ast)

        {{call.ske, call.piped_input?, data_keys, kernel_key}, state1}
      end)

    {:fusion_v2, stages}
  end

  defp normalize_kernel_key(kernel_ast) do
    {args, body} = decompose_kernel(kernel_ast)
    {:kernel, strip_ast_metadata(args), strip_ast_metadata(body)}
  end

  defp strip_ast_metadata(ast) do
    Macro.prewalk(ast, fn
      {name, _meta, args} when is_atom(name) ->
        {name, [], args}

      node ->
        node
    end)
  end

  defp collect_fusion_inputs(calls) do
    state0 = %{next_input_idx: 0, input_vars: %{}, input_order: []}

    state =
      calls
      |> Enum.reduce(state0, fn call, state ->
        {_arg_asts, next_state} = resolve_external_inputs(call.explicit_inputs, state)
        next_state
      end)

    Enum.map(state.input_order, fn {data_ast, _var_ast} -> data_ast end)
  end

  defp fusion_cache_get(key) do
    case Process.whereis(:module_server) do
      nil ->
        nil

      _pid ->
        send(:module_server, {:get_fusion, key, self()})

        receive do
          {:fusion, cached} -> cached
        after
          5_000 -> nil
        end
    end
  end

  defp fusion_cache_put(key, value) do
    case Process.whereis(:module_server) do
      nil -> :ok
      _pid -> send(:module_server, {:put_fusion, key, value})
    end

    value
  end

  defp stable_fusion_name(key) do
    hash =
      key
      |> :erlang.term_to_binary()
      |> then(&:crypto.hash(:sha256, &1))
      |> Base.encode16(case: :lower)
      |> binary_part(0, 16)

    "fusion_" <> hash
  end

  defp build_fn_ast(args, body) do
    body_ast =
      case List.wrap(body) do
        [single] -> single
        nodes -> {:__block__, [], nodes}
      end

    {:fn, [], [{:->, [], [args, PolyHok.CudaBackend.add_return(body_ast)]}]}
  end

  defp build_anon_fun_ast(name, function_ast) do
    quote do
      {:anon, unquote(name), unquote(Macro.escape(function_ast))}
    end
  end

  defp build_cached_anon_fun(key, args, body) do
    function_ast = build_fn_ast(args, body)
    build_anon_fun_ast(stable_fusion_name(key), function_ast)
  end

  defp fuse_map_chain_calls(calls) do
    if Enum.empty?(calls) do
      raise ArgumentError, "empty fusion chain"
    end

    invalid = Enum.find(calls, fn call -> call.ske not in [:map, :map2, :map3, :map4] end)

    if invalid do
      raise ArgumentError,
            "full-chain map fusion supports only map/map2/map3/map4 calls, got #{inspect(invalid.ske)}"
    end

    key = normalize_fusion_key(calls)

    case fusion_cache_get(key) do
      nil ->
        state0 = %{body: [], value_ast: nil, next_input_idx: 0, input_vars: %{}, input_order: []}

        state =
          calls
          |> Enum.with_index()
          |> Enum.reduce(state0, fn {call, idx}, acc ->
            fuse_map_stage(call, idx, acc)
          end)

        input_data_asts = Enum.map(state.input_order, fn {data_ast, _var_ast} -> data_ast end)
        input_vars = Enum.map(state.input_order, fn {_data_ast, var_ast} -> var_ast end)
        fused_fun = build_cached_anon_fun(key, input_vars, state.body ++ [state.value_ast])

        fusion_cache_put(key, fused_fun)
        %{inputs: input_data_asts, fun: fused_fun}

      fused_fun ->
        %{inputs: collect_fusion_inputs(calls), fun: fused_fun}
    end
  end

  defp emit_chain_with_reduce(calls) do
    reduce_idx =
      calls
      |> Enum.with_index()
      |> Enum.find_value(fn
        {%AstCall{ske: :reduce}, idx} -> idx
        _ -> nil
      end)

    map_calls = Enum.take(calls, reduce_idx)
    reduce_call = Enum.at(calls, reduce_idx)

    %{inputs: inputs, fun: map_fun} = fuse_map_chain_calls(map_calls)
    red_fun = normalize_kernel_ast(reduce_call.kernel_ast)
    initial = reduce_call.initial_ast

    case inputs do
      [t1] ->
        quote do
          Ske.mapReduce(unquote(t1), unquote(initial), unquote(map_fun), unquote(red_fun))
        end

      [t1, t2] ->
        quote do
          Ske.map2Reduce(
            unquote(t1),
            unquote(t2),
            unquote(initial),
            unquote(map_fun),
            unquote(red_fun)
          )
        end

      [t1, t2, t3, t4] ->
        quote do
          Ske.map4Reduce(
            unquote(t1),
            unquote(t2),
            unquote(t3),
            unquote(t4),
            unquote(initial),
            unquote(map_fun),
            unquote(red_fun)
          )
        end

      _ ->
        raise ArgumentError,
              "map-chain |> reduce fusion currently supports fused map arity 1, 2, or 4, got #{length(inputs)}"
    end
  end

  defp emit_fused_chain(calls) do
    reduce_positions =
      calls
      |> Enum.with_index()
      |> Enum.filter(fn {%AstCall{ske: ske}, _idx} -> ske == :reduce end)
      |> Enum.map(fn {_call, idx} -> idx end)

    case reduce_positions do
      [] ->
        %{inputs: inputs, fun: fun_ast} = fuse_map_chain_calls(calls)
        emit_map_from_inputs_and_fun(inputs, fun_ast)

      [idx] when idx == length(calls) - 1 ->
        emit_chain_with_reduce(calls)

      [idx] ->
        raise ArgumentError,
              "reduce must be the final stage in fusion chains, found at stage #{idx + 1}"

      _ ->
        raise ArgumentError, "fusion chain supports at most one reduce stage"
    end
  end

  defp split_body_and_return(body) do
    body = List.wrap(body)

    case body do
      [] ->
        raise ArgumentError, "Empty function body in split_body_and_return/1"

      _ ->
        {prefix, [last]} = Enum.split(body, length(body) - 1)

        case last do
          {:return, _meta, [expr]} ->
            {prefix, expr}

          other ->
            {prefix, other}
        end
    end
  end

  defp build_phok_fun(args, body) do
    quote do
      PolyHok.phok(fn unquote_splicing(args) ->
        (unquote_splicing(body))
      end)
    end
  end

  defp decompose_kernel(kernel_ast) do
    case kernel_ast do
      # PolyHok.phok(...)
      {{:., _, [{:__aliases__, _, [:PolyHok]}, :phok]}, _, _} ->
        comp_ast_phok(kernel_ast)

      # &Mod.fun/arity
      {:&, _, _} ->
        comp_ast_device(kernel_ast)

      other ->
        raise ArgumentError,
              "unsupported kernel AST in Fusion: #{Macro.to_string(other)}"
    end
  end

  defp comp_ast_phok(
         {{:., _, [{:__aliases__, _, [:PolyHok]}, :phok]}, _,
          [{:fn, _, [{:->, _, [args, body_ast]}]}]}
       ) do
    {args, normalize_body_ast(body_ast)}
  end

  defp comp_ast_device(
         {:&, _, [{:/, _, [{{:., _, [{:__aliases__, _, _}, f_name]}, _, []}, _f_arity]}]}
       ) do
    pid = self()
    send(:module_server, {:get_ast, f_name, pid})

    {{:defd, _m, fn_body}, _} =
      receive do
        {:ast, body} -> body
      end

    [args | block] = fn_body
    args = extract_args(args)
    block = extract_block(block)
    {args, block}
  end

  defp extract_args({_name, _m, args}) do
    args
  end

  defp extract_block([[do: body_ast]]), do: normalize_body_ast(body_ast)
  defp extract_block(do: body_ast), do: normalize_body_ast(body_ast)

  defp normalize_body_ast(body_ast) do
    case body_ast do
      {:__block__, _m, block_body} when is_list(block_body) ->
        case block_body do
          [{:return, _rm, [expr]}] -> [expr]
          other -> other
        end

      {:return, _m, [expr]} ->
        [expr]

      expr ->
        [expr]
    end
  end

  defp normalize_kernel_ast(kernel_ast) do
    {args, body} = decompose_kernel(kernel_ast)
    build_phok_fun(args, body)
  end

  defp find_in_node_list([head | tail], pred) do
    case find_ast_node(head, pred) do
      {:ok, node} -> {:ok, node}
      :not_found -> find_ast_node(tail, pred)
    end
  end

  defp find_in_node_list([], _pred) do
    :not_found
  end

  def find_ast_node(ast_node, predicate) do
    cond do
      predicate.(ast_node) ->
        {:ok, ast_node}

      is_tuple(ast_node) ->
        ast_node
        |> Tuple.to_list()
        |> find_in_node_list(predicate)

      is_list(ast_node) ->
        find_in_node_list(ast_node, predicate)

      true ->
        :not_found
    end
  end

  def extract_defk_body_to_list({{:defk, _, [_, [do: {:__block__, _, body_list}]]}, _}) do
    body_list
  end

  defmacro with_fusion({:|>, _meta, _args} = ast) do
    calls =
      ast
      |> normalize_leading_pipe_value()
      |> flatten_pipe_ast()
      |> Enum.map(&parse_ske_call/1)
      |> validate_call_positions()

    emit_fused_chain(calls)
  end

  defmacro with_fusion(ast), do: ast
end
