require PolyHok
#PolyHok.defmodule SkeKernels do
#
# defk map_step_2_para_no_resp_kernel(d_array,  step, par1, par2,size,f) do
#    globalId  = blockDim.x * ( gridDim.x * blockIdx.y + blockIdx.x ) + threadIdx.x
#    id  = step * globalId
#    #f(id,id)
#    if (globalId < size) do
#      f(d_array+id,par1,par2)
#    end
#  end
#
#end

PolyHok.defmodule Ske do
  #defmacro __using__(_opts) do
  #     IO.puts "You are USIng!"
  #    end

  PolyHok.include CAS_Poly

  #######################################
  ##  Suporte a listas Elixir nativas (flawd)
  ##
  ##  Ske.map, Ske.map2, Ske.map3, Ske.map4, Ske.reduce, Ske.mapReduce,
  ##  Ske.map2Reduce e Ske.map4Reduce aceitam listas Elixir nos argumentos
  ##  de dados, além dos gnx tradicionais.
  ##
  ##  Regra: se QUALQUER argumento de dados de uma chamada for uma lista
  ##  Elixir, TODOS os argumentos de dados dessa chamada devem ser listas
  ##  (não dá para misturar lista com gnx na mesma chamada), a conversão
  ##  para flawd é feita internamente (PolyHok.new_flawd_list) para cada
  ##  uma, o skeleton roda exatamente como já rodava sobre gnx, e o
  ##  resultado desce de volta para uma lista Elixir nativa
  ##  (PolyHok.get_flawd_list) antes de retornar. Nenhuma otimização do
  ##  flawd (prefetch de cons cells, pinned memory, streams, etc) é
  ##  contornada — a chamada só encapsula new_flawd_list/get_flawd_list.
  ##
  ##  Isso permite escrever:
  ##
  ##      lista |> Ske.map(f)
  ##
  ##  em vez de:
  ##
  ##      lista |> PolyHok.new_flawd_list(shape, type) |> Ske.map(f) |> PolyHok.get_flawd_list()
  ##
  ##  O custo de conversão é o mesmo — o ganho é exclusivamente de API:
  ##  o programador nunca precisa tocar em nada além de listas.
  #######################################

  # não usada no momento, mantida comentada
  # defp list_data_arg?(list) when is_list(list), do: true
  # defp list_data_arg?(_), do: false

  # Infere {rows, cols, type} de uma lista Elixir 1D ou 2D (lista de listas).
  def infer_list_shape_and_type(list) do
    case list do
      [h | _] when is_list(h) ->
        rows = length(list)
        cols = length(h)
        type = infer_elem_type(List.flatten(list))
        {rows, cols, type}

      _ ->
        cols = length(list)
        type = infer_elem_type(list)
        {1, cols, type}
    end
  end

  # Tipo default: {:s, 32} para inteiros, {:f, 32} assim que um float é encontrado
  def infer_elem_type([]), do: {:s, 32}
  def infer_elem_type([x | rest]) do
    cond do
      is_float(x) -> {:f, 32}
      is_integer(x) -> {:s, 32}
      true -> infer_elem_type(rest)
    end
  end

  # Sobe uma lista Elixir para a GPU via flawd, inferindo shape/type.
  def list_to_gnx(list) do
    {rows, cols, type} = infer_list_shape_and_type(list)
    PolyHok.new_flawd_list(list, {rows, cols}, type)
  end

  # Desce um gnx para lista Elixir via flawd, usando o type/shape do
  # próprio gnx (já conhecidos nesse ponto, pois foi criado por list_to_gnx
  # ou por um skeleton anterior operando sobre o mesmo tipo).
  def gnx_to_list(gnx) do
    PolyHok.get_flawd_list(gnx)
  end

  #######################################
  ##  reduce / mapReduce / map2Reduce / map4Reduce
  #######################################

  def reduce(ref, initial, f) when is_list(ref) do
    ref
    |> list_to_gnx()
    |> reduce(initial, f)
    |> gnx_to_list()
  end
  def reduce(ref, initial, f) do

     shape = PolyHok.get_shape_gnx(ref)
     type = PolyHok.get_type_gnx(ref)
     size = Tuple.product(shape)
      result_gpu  = PolyHok.new_gnx(Nx.tensor([[initial]] , type: type))

      threadsPerBlock = 256
      blocksPerGrid = div(size + threadsPerBlock - 1, threadsPerBlock)
      numberOfBlocks = blocksPerGrid

      case type do
        {:f,32} -> cas = PolyHok.phok (fn (x,y,z) -> cas_float(x,y,z) end)
            PolyHok.spawn(&Ske.reduce_kernel/6,{numberOfBlocks,1,1},{threadsPerBlock,1,1},[ref,result_gpu, initial, size, cas, f])

        {:f,64} -> cas = PolyHok.phok (fn (x,y,z) -> cas_double(x,y,z) end)
            PolyHok.spawn(&Ske.reduce_kernel/6,{numberOfBlocks,1,1},{threadsPerBlock,1,1},[ref,result_gpu, initial, size, cas, f])

        {:s,32} -> cas = PolyHok.phok (fn (x,y,z) -> cas_int(x,y,z) end)
            PolyHok.spawn(&Ske.reduce_kernel/6,{numberOfBlocks,1,1},{threadsPerBlock,1,1},[ref,result_gpu, initial, size, cas, f])

        x -> raise "new_gnx: type #{x} not suported"
     end



      result_gpu
  end
  defk reduce_kernel(a, ref4, initial, n, cas, f) do

    __shared__ cache[256]

    tid = threadIdx.x + blockIdx.x * blockDim.x;
    cacheIndex = threadIdx.x

    temp = initial

    while (tid < n) do
      temp = f(a[tid], temp)
      tid = blockDim.x * gridDim.x + tid
    end

    cache[cacheIndex] = temp
      __syncthreads()

    i = blockDim.x/2

    while (i != 0 ) do  ###&& tid < n) do
      #tid = blockDim.x * gridDim.x + tid
      if (cacheIndex < i) do
        cache[cacheIndex] = f(cache[cacheIndex + i] , cache[cacheIndex])
      end

    __syncthreads()
    i = i/2
    end

  if (cacheIndex == 0) do
    current_value = ref4[0]
    while(!(current_value == cas(ref4,current_value,f(cache[0],current_value)))) do
      current_value = ref4[0]
    end
  end

  end

  def mapReduce(ref, initial, map_f, red_f) when is_list(ref) do
    ref
    |> list_to_gnx()
    |> mapReduce(initial, map_f, red_f)
    |> gnx_to_list()
  end
  def mapReduce(ref, initial, map_f, red_f) do
    shape = PolyHok.get_shape_gnx(ref)
    type = PolyHok.get_type_gnx(ref)
    size = Tuple.product(shape)

    result_gpu = PolyHok.new_gnx(Nx.tensor([[initial]], type: type))

    threadsPerBlock = 256
    blocksPerGrid = div(size + threadsPerBlock - 1, threadsPerBlock)
    numberOfBlocks = blocksPerGrid

    case type do
      {:f, 32} ->
        cas = PolyHok.phok(fn x, y, z -> cas_float(x, y, z) end)

        PolyHok.spawn(&Ske.mapReduce_kernel/7, {numberOfBlocks, 1, 1}, {threadsPerBlock, 1, 1}, [
          ref,
          result_gpu,
          initial,
          size,
          cas,
          map_f,
          red_f
        ])

      {:f, 64} ->
        cas = PolyHok.phok(fn x, y, z -> cas_double(x, y, z) end)

        PolyHok.spawn(&Ske.mapReduce_kernel/7, {numberOfBlocks, 1, 1}, {threadsPerBlock, 1, 1}, [
          ref,
          result_gpu,
          initial,
          size,
          cas,
          map_f,
          red_f
        ])

      {:s, 32} ->
        cas = PolyHok.phok(fn x, y, z -> cas_int(x, y, z) end)

        PolyHok.spawn(&Ske.mapReduce_kernel/7, {numberOfBlocks, 1, 1}, {threadsPerBlock, 1, 1}, [
          ref,
          result_gpu,
          initial,
          size,
          cas,
          map_f,
          red_f
        ])

      x ->
        raise "mapReduce: type #{inspect(x)} not supported"
    end

    result_gpu
  end

  defk mapReduce_kernel(a, ref4, initial, n, cas, map_f, red_f) do
    __shared__ cache[256]

    tid = threadIdx.x + blockIdx.x * blockDim.x
    cacheIndex = threadIdx.x
    stride = blockDim.x * gridDim.x

    temp = initial

    while tid < n do
      mapped = map_f(a[tid])
      temp = red_f(mapped, temp)
      tid = tid + stride
    end

    cache[cacheIndex] = temp
    __syncthreads()

    i = blockDim.x / 2

    while i != 0 do
      if cacheIndex < i do
        cache[cacheIndex] = red_f(cache[cacheIndex + i], cache[cacheIndex])
      end

      __syncthreads()
      i = i / 2
    end

    if cacheIndex == 0 do
      current_value = ref4[0]

      while(!(current_value == cas(ref4, current_value, red_f(cache[0], current_value)))) do
        current_value = ref4[0]
      end
    end
  end

  def map2Reduce(t1, t2, initial, map_f, red_f) when is_list(t1) or is_list(t2) do
    if not (is_list(t1) and is_list(t2)) do
      raise "map2Reduce: os dois argumentos de dados devem ser ambos listas ou ambos gnx, não é possível misturar"
    end

    g1 = list_to_gnx(t1)
    g2 = list_to_gnx(t2)

    map2Reduce(g1, g2, initial, map_f, red_f)
    |> gnx_to_list()
  end
  def map2Reduce(t1, t2, initial, map_f, red_f) do
    shape1 = PolyHok.get_shape_gnx(t1)
    shape2 = PolyHok.get_shape_gnx(t2)

    if shape1 != shape2 do
      raise "map2Reduce: input shapes must match, got #{inspect(shape1)} and #{inspect(shape2)}"
    end

    type1 = PolyHok.get_type_gnx(t1)
    type2 = PolyHok.get_type_gnx(t2)

    if type1 != type2 do
      raise "map2Reduce: input types must match, got #{inspect(type1)} and #{inspect(type2)}"
    end

    shape = shape1
    type = type1
    size = Tuple.product(shape)

    # resultado final escalar (mesma estratégia de reduce/mapReduce)
    result_gpu = PolyHok.new_gnx(Nx.tensor([[initial]], type: type))

    threadsPerBlock = 256
    # blocksPerGrid = div(size + threadsPerBlock - 1, threadsPerBlock)
    blocksPerGrid = 4096
    numberOfBlocks = blocksPerGrid

    case type do
      {:f, 32} ->
        cas = PolyHok.phok(fn x, y, z -> cas_float(x, y, z) end)

        PolyHok.spawn(
          &Ske.map2Reduce_kernel/8,
          {numberOfBlocks, 1, 1},
          {threadsPerBlock, 1, 1},
          [t1, t2, result_gpu, initial, size, cas, map_f, red_f]
        )

      {:f, 64} ->
        cas = PolyHok.phok(fn x, y, z -> cas_double(x, y, z) end)

        PolyHok.spawn(
          &Ske.map2Reduce_kernel/8,
          {numberOfBlocks, 1, 1},
          {threadsPerBlock, 1, 1},
          [t1, t2, result_gpu, initial, size, cas, map_f, red_f]
        )

      {:s, 32} ->
        cas = PolyHok.phok(fn x, y, z -> cas_int(x, y, z) end)

        PolyHok.spawn(
          &Ske.map2Reduce_kernel/8,
          {numberOfBlocks, 1, 1},
          {threadsPerBlock, 1, 1},
          [t1, t2, result_gpu, initial, size, cas, map_f, red_f]
        )

      x ->
        raise "map2Reduce: type #{inspect(x)} not supported"
    end

    result_gpu
  end

  defk map2Reduce_kernel(t1, t2, ref_out, initial, n, cas, map_f, red_f) do
    __shared__ cache[256]

    tid = threadIdx.x + blockIdx.x * blockDim.x
    cacheIndex = threadIdx.x
    stride = blockDim.x * gridDim.x

    temp = initial

    while tid < n do
      mapped = map_f(t1[tid], t2[tid])
      temp = red_f(mapped, temp)
      tid = tid + stride
    end

    cache[cacheIndex] = temp
    __syncthreads()

    i = blockDim.x / 2

    while i != 0 do
      if cacheIndex < i do
        cache[cacheIndex] = red_f(cache[cacheIndex + i], cache[cacheIndex])
      end

      __syncthreads()
      i = i / 2
    end

    if cacheIndex == 0 do
      current_value = ref_out[0]

      while(!(current_value == cas(ref_out, current_value, red_f(cache[0], current_value)))) do
        current_value = ref_out[0]
      end
    end
  end

  def map4Reduce(t1, t2, t3, t4, initial, map_f, red_f)
      when is_list(t1) or is_list(t2) or is_list(t3) or is_list(t4) do
    if not (is_list(t1) and is_list(t2) and is_list(t3) and is_list(t4)) do
      raise "map4Reduce: todos os argumentos de dados devem ser listas ou todos gnx, não é possível misturar"
    end

    g1 = list_to_gnx(t1)
    g2 = list_to_gnx(t2)
    g3 = list_to_gnx(t3)
    g4 = list_to_gnx(t4)

    map4Reduce(g1, g2, g3, g4, initial, map_f, red_f)
    |> gnx_to_list()
  end
  def map4Reduce(t1, t2, t3, t4, initial, map_f, red_f) do
    shape1 = PolyHok.get_shape_gnx(t1)
    shape2 = PolyHok.get_shape_gnx(t2)
    shape3 = PolyHok.get_shape_gnx(t3)
    shape4 = PolyHok.get_shape_gnx(t4)

    if shape1 != shape2 or shape1 != shape3 or shape1 != shape4 do
      raise "map4Reduce: input shapes must match, got #{inspect(shape1)}, #{inspect(shape2)}, #{inspect(shape3)}, #{inspect(shape4)}"
    end

    type1 = PolyHok.get_type_gnx(t1)
    type2 = PolyHok.get_type_gnx(t2)
    type3 = PolyHok.get_type_gnx(t3)
    type4 = PolyHok.get_type_gnx(t4)

    if type1 != type2 or type1 != type3 or type1 != type4 do
      raise "map4Reduce: input types must match, got #{inspect(type1)}, #{inspect(type2)}, #{inspect(type3)}, #{inspect(type4)}"
    end

    shape = shape1
    type = type1
    size = Tuple.product(shape)

    result_gpu = PolyHok.new_gnx(Nx.tensor([[initial]], type: type))

    threadsPerBlock = 256
    blocksPerGrid = 4096
    numberOfBlocks = blocksPerGrid

    case type do
      {:f, 32} ->
        cas = PolyHok.phok(fn x, y, z -> cas_float(x, y, z) end)

        PolyHok.spawn(
          &Ske.map4Reduce_kernel/10,
          {numberOfBlocks, 1, 1},
          {threadsPerBlock, 1, 1},
          [t1, t2, t3, t4, result_gpu, initial, size, cas, map_f, red_f]
        )

      {:f, 64} ->
        cas = PolyHok.phok(fn x, y, z -> cas_double(x, y, z) end)

        PolyHok.spawn(
          &Ske.map4Reduce_kernel/10,
          {numberOfBlocks, 1, 1},
          {threadsPerBlock, 1, 1},
          [t1, t2, t3, t4, result_gpu, initial, size, cas, map_f, red_f]
        )

      {:s, 32} ->
        cas = PolyHok.phok(fn x, y, z -> cas_int(x, y, z) end)

        PolyHok.spawn(
          &Ske.map4Reduce_kernel/10,
          {numberOfBlocks, 1, 1},
          {threadsPerBlock, 1, 1},
          [t1, t2, t3, t4, result_gpu, initial, size, cas, map_f, red_f]
        )

      x ->
        raise "map4Reduce: type #{inspect(x)} not supported"
    end

    result_gpu
  end

  defk map4Reduce_kernel(t1, t2, t3, t4, ref_out, initial, n, cas, map_f, red_f) do
    __shared__ cache[256]

    tid = threadIdx.x + blockIdx.x * blockDim.x
    cacheIndex = threadIdx.x
    stride = blockDim.x * gridDim.x

    temp = initial

    while tid < n do
      mapped = map_f(t1[tid], t2[tid], t3[tid], t4[tid])
      temp = red_f(mapped, temp)
      tid = tid + stride
    end

    cache[cacheIndex] = temp
    __syncthreads()

    i = blockDim.x / 2

    while i != 0 do
      if cacheIndex < i do
        cache[cacheIndex] = red_f(cache[cacheIndex + i], cache[cacheIndex])
      end

      __syncthreads()
      i = i / 2
    end

    if cacheIndex == 0 do
      current_value = ref_out[0]

      while(!(current_value == cas(ref_out, current_value, red_f(cache[0], current_value)))) do
        current_value = ref_out[0]
      end
    end
  end

  #######################################
  ##  map / map2 / map3 / map4
  #######################################

  @defaults %{coord: false, return: true, dim: :one}
  def map(a,b,c,options \\[])
  def map({:nx, type, shape, name , ref}, func, [par1,par2], options )do
    %{coord: coord, return: return, dim: dim} = Enum.into(options, @defaults)
  case dim do
    :one ->   if (not coord && not return )do
                   map_2_para_no_resp({:nx, type, shape, name , ref},  par1, par2, func)
              end

              if (not coord && return) do
                # map_2_para
              end
     :two ->  if (coord && not return) do
                    map_coord_2D_2_para_no_resp({:nx, type, shape, name , ref}, par1, par2, func)
              end


  end
end
  def map({:nx, type, shape, name , ref}, func, [par1], options )do
    %{coord: coord, return: return, dim: dim} = Enum.into(options, @defaults)
  case dim do
    :one ->   if (not coord && not return )do
                   # map_1_para_no_resp({:nx, type, shape, name , ref},  par1, func)
              end

              if (not coord && return) do
                # map_2_para
              end
     :two ->  if (coord && not return) do
                    map_coord_2D_1_para_no_resp({:nx, type, shape, name , ref}, par1,  func)
              end


  end
  end
  def map({:nx, type, shape, name, ref}, {:nx, type2, shape2, name2, ref2}, func, options) do
    %{coord: coord, return: return, dim: dim} = Enum.into(options, @defaults)

    if (coord || not return || dim == :two) do
      raise "The only options for a map2 are: coord: false, return: true, dim: :one"
    else
      map2({:nx, type, shape, name , ref}, {:nx, type2, shape2, name2, ref2}, func)
    end
  end
  def map(input, f) when is_list(input) do
    input
    |> list_to_gnx()
    |> map(f)
    |> gnx_to_list()
  end
  def map(input, f) do
    shape = PolyHok.get_shape(input)
    type = PolyHok.get_type(input)
    result_gpu = PolyHok.new_gnx(shape,type)
    size = Tuple.product(shape)
    threadsPerBlock = 128;
    numberOfBlocks = div(size + threadsPerBlock - 1, threadsPerBlock)

    PolyHok.spawn(&Ske.map_ker/4,
              {numberOfBlocks,1,1},
              {threadsPerBlock,1,1},
              [input,result_gpu,size, f])
    result_gpu
  end
  defk map2_kernel(a1,a2,a3,size,f) do
    id = blockIdx.x * blockDim.x + threadIdx.x
    if(id < size) do
      a3[id] = f(a1[id],a2[id])
    end
  end
  def map2(t1, t2, func) when is_list(t1) or is_list(t2) do
    if not (is_list(t1) and is_list(t2)) do
      raise "map2: os dois argumentos de dados devem ser ambos listas ou ambos gnx, não é possível misturar"
    end

    g1 = list_to_gnx(t1)
    g2 = list_to_gnx(t2)

    map2(g1, g2, func)
    |> gnx_to_list()
  end
  def map2(t1,t2,func) do

    shape = PolyHok.get_shape_gnx(t1)
    type = PolyHok.get_type_gnx(t2)
    size = Tuple.product shape
    result_gpu = PolyHok.new_gnx(shape, type)

      threadsPerBlock = 256;
      numberOfBlocks = div(size + threadsPerBlock - 1, threadsPerBlock)

      PolyHok.spawn(&Ske.map2_kernel/5,{numberOfBlocks,1,1},{threadsPerBlock,1,1},[t1,t2,result_gpu,size,func])


      result_gpu
  end

  defk map3_kernel(a1, a2, a3, out, size, f) do
    id = blockIdx.x * blockDim.x + threadIdx.x

    if id < size do
      out[id] = f(a1[id], a2[id], a3[id])
    end
  end

  def map3(t1, t2, t3, func) when is_list(t1) or is_list(t2) or is_list(t3) do
    if not (is_list(t1) and is_list(t2) and is_list(t3)) do
      raise "map3: todos os argumentos de dados devem ser listas ou todos gnx, não é possível misturar"
    end

    g1 = list_to_gnx(t1)
    g2 = list_to_gnx(t2)
    g3 = list_to_gnx(t3)

    map3(g1, g2, g3, func)
    |> gnx_to_list()
  end
  def map3(t1, t2, t3, func) do
    shape1 = PolyHok.get_shape_gnx(t1)
    shape2 = PolyHok.get_shape_gnx(t2)
    shape3 = PolyHok.get_shape_gnx(t3)

    if shape1 != shape2 or shape1 != shape3 do
      raise "map3: input shapes must match, got #{inspect(shape1)}, #{inspect(shape2)}, #{inspect(shape3)}"
    end

    type1 = PolyHok.get_type_gnx(t1)
    type2 = PolyHok.get_type_gnx(t2)
    type3 = PolyHok.get_type_gnx(t3)

    if type1 != type2 or type1 != type3 do
      raise "map3: input types must match, got #{inspect(type1)}, #{inspect(type2)}, #{inspect(type3)}"
    end

    size = Tuple.product(shape1)
    result_gpu = PolyHok.new_gnx(shape1, type1)

    threadsPerBlock = 256
    numberOfBlocks = div(size + threadsPerBlock - 1, threadsPerBlock)

    PolyHok.spawn(&Ske.map3_kernel/6, {numberOfBlocks, 1, 1}, {threadsPerBlock, 1, 1}, [
      t1,
      t2,
      t3,
      result_gpu,
      size,
      func
    ])

    result_gpu
  end

  defk map4_kernel(a1, a2, a3, a4, out, size, f) do
    id = blockIdx.x * blockDim.x + threadIdx.x

    if id < size do
      out[id] = f(a1[id], a2[id], a3[id], a4[id])
    end
  end

  def map4(t1, t2, t3, t4, func)
      when is_list(t1) or is_list(t2) or is_list(t3) or is_list(t4) do
    if not (is_list(t1) and is_list(t2) and is_list(t3) and is_list(t4)) do
      raise "map4: todos os argumentos de dados devem ser listas ou todos gnx, não é possível misturar"
    end

    g1 = list_to_gnx(t1)
    g2 = list_to_gnx(t2)
    g3 = list_to_gnx(t3)
    g4 = list_to_gnx(t4)

    map4(g1, g2, g3, g4, func)
    |> gnx_to_list()
  end
  def map4(t1, t2, t3, t4, func) do
    shape1 = PolyHok.get_shape_gnx(t1)
    shape2 = PolyHok.get_shape_gnx(t2)
    shape3 = PolyHok.get_shape_gnx(t3)
    shape4 = PolyHok.get_shape_gnx(t4)

    if shape1 != shape2 or shape1 != shape3 or shape1 != shape4 do
      raise "map4: input shapes must match, got #{inspect(shape1)}, #{inspect(shape2)}, #{inspect(shape3)}, #{inspect(shape4)}"
    end

    type1 = PolyHok.get_type_gnx(t1)
    type2 = PolyHok.get_type_gnx(t2)
    type3 = PolyHok.get_type_gnx(t3)
    type4 = PolyHok.get_type_gnx(t4)

    if type1 != type2 or type1 != type3 or type1 != type4 do
      raise "map4: input types must match, got #{inspect(type1)}, #{inspect(type2)}, #{inspect(type3)}, #{inspect(type4)}"
    end

    size = Tuple.product(shape1)
    result_gpu = PolyHok.new_gnx(shape1, type1)

    threadsPerBlock = 256
    numberOfBlocks = div(size + threadsPerBlock - 1, threadsPerBlock)

    PolyHok.spawn(&Ske.map4_kernel/7, {numberOfBlocks, 1, 1}, {threadsPerBlock, 1, 1}, [
      t1,
      t2,
      t3,
      t4,
      result_gpu,
      size,
      func
    ])

    result_gpu
  end

  defk map_coord_2D_1_para_no_resp_kernel(d_array,  step, par1,sizex,sizey,f) do

    x = threadIdx.x + blockIdx.x * blockDim.x
    y = threadIdx.y + blockIdx.y * blockDim.y
    offset = x + y * blockDim.x * gridDim.x

     id  = step * offset
    #f(id,id)
    if (offset < (sizex*sizey)) do
      f(d_array+id,par1,x,y)
    end
  end
  def map_coord_2D_1_para_no_resp(d_array, par1, f) do

   {sizex,sizey,step} =  case PolyHok.get_shape_gnx(d_array) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end

    #IO.inspect {sizex,sizey,step}
    block_size = 16
    grid_rows = trunc ((sizex + block_size - 1) / block_size)
    grid_cols = trunc ((sizey + block_size - 1) / block_size)


    PolyHok.spawn(&Ske.map_coord_2D_1_para_no_resp_kernel/6,{grid_cols,grid_rows,1},{block_size,block_size,1},[d_array,step,par1,sizex,sizey,f])
      d_array
  end
  defk map_coord_2D_2_para_no_resp_kernel(d_array,  step, par1, par2,sizex,sizey,f) do

    x = threadIdx.x + blockIdx.x * blockDim.x
    y = threadIdx.y + blockIdx.y * blockDim.y
    offset = x + y * blockDim.x * gridDim.x

     id  = step * offset
    #f(id,id)
    if (offset < (sizex*sizey)) do
      f(d_array+id,par1,par2,x,y)
    end
  end
  def map_coord_2D_2_para_no_resp(d_array, par1, par2, f) do

   {sizex,sizey,step} =  case PolyHok.get_shape_gnx(d_array) do
                            {l,c} -> {l,c,1}
                            {l,c,step} -> {l,c,step}
                            x -> raise "Invalid shape for a 2D map: #{inspect x}!"
                          end

    #IO.inspect {sizex,sizey,step}
    block_size = 16
    grid_rows = trunc ((sizex + block_size - 1) / block_size)
    grid_cols = trunc ((sizey + block_size - 1) / block_size)


    PolyHok.spawn(&Ske.map_coord_2D_2_para_no_resp_kernel/7,{grid_cols,grid_rows,1},{block_size,block_size,1},[d_array,step,par1,par2,sizex,sizey,f])
      d_array
  end
  def map_2_para_no_resp(d_array,  par1, par2, f) do
      block_size =  128;
      {l,step} = PolyHok.get_shape_gnx(d_array)
      size = l*step
      nBlocks = floor ((size + block_size - 1) / block_size)

      PolyHok.spawn(&Ske.map_step_2_para_no_resp_kernel/6,{nBlocks,1,1},{block_size,1,1},[d_array,step,par1,par2,l,f])
      d_array
  end
  defk map_step_2_para_no_resp_kernel(d_array,  step, par1, par2,size,f) do
        globalId  = blockDim.x * ( gridDim.x * blockIdx.y + blockIdx.x ) + threadIdx.x
        id  = step * globalId
        #f(id,id)
        if (globalId < size) do
          f(d_array+id,par1,par2)
        end
  end
  defk map_ker(a1,a2,size,f) do
      index = blockIdx.x * blockDim.x + threadIdx.x
      stride = blockDim.x * gridDim.x

      for i in range(index,size,stride) do
            a2[i] = f(a1[i])
      end
    end
  end




#### kernels
