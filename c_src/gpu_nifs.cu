#include "erl_nif.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <dlfcn.h>
#include <assert.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>
#include <pthread.h>
#include <string.h>

CUcontext  context = NULL;

/* ================================================================
 * KERNEL CACHE — evita recompilação NVRTC a cada chamada
 * Mapeia source_code → (CUmodule + CUfunction)
 * Thread-safe via pthread_mutex
 * ================================================================ */

#define KERNEL_CACHE_SLOTS 512

typedef struct {
    uint64_t   hash;      /* FNV-1a do source; 0 = slot vazio    */
    char      *source;    /* strdup do source (NULL = vazio)      */
    CUmodule   module;
    CUfunction function;
} KernelCacheEntry;

static KernelCacheEntry g_kcache[KERNEL_CACHE_SLOTS];
static pthread_mutex_t  g_kcache_mutex = PTHREAD_MUTEX_INITIALIZER;

/* FNV-1a 64-bit */
static uint64_t fnv1a(const char *s) {
    uint64_t h = 14695981039346656037ULL;
    while (*s) { h ^= (uint8_t)(*s++); h *= 1099511628211ULL; }
    return h;
}

/* Procura no cache. Deve ser chamado com o mutex. */
static CUfunction kcache_lookup(const char *src, uint64_t hash) {
    uint32_t slot = (uint32_t)(hash & (KERNEL_CACHE_SLOTS - 1));
    for (int i = 0; i < KERNEL_CACHE_SLOTS; i++) {
        uint32_t idx = (slot + i) & (KERNEL_CACHE_SLOTS - 1);
        KernelCacheEntry *e = &g_kcache[idx];
        if (e->source == NULL)   return NULL;  /* slot vazio */
        if (e->hash == hash && strcmp(e->source, src) == 0)
            return e->function;                /* cache hit  */
    }
    return NULL;  /* tabela cheia (impossível com 512 slots e <20 kernels) */
}

/* Armazena no cache. Deve ser chamado COM o mutex. */
static void kcache_store(const char *src, uint64_t hash,
                         CUmodule mod, CUfunction fn) {
    uint32_t slot = (uint32_t)(hash & (KERNEL_CACHE_SLOTS - 1));
    for (int i = 0; i < KERNEL_CACHE_SLOTS; i++) {
        uint32_t idx = (slot + i) & (KERNEL_CACHE_SLOTS - 1);
        KernelCacheEntry *e = &g_kcache[idx];
        if (e->source == NULL) {
            e->hash     = hash;
            e->source   = strdup(src);   /* cópia permanente no heap */
            e->module   = mod;
            e->function = fn;
            return;
        }
    }
    /* tabela cheia — ignora silenciosamente (próxima chamada recompila) */
}

void init_cuda(ErlNifEnv *env)
{
  if (context == NULL)
    {
       CUresult err;
       //CUdevice   device;
       int device = 0;
      // printf("aqui!\n");
       cuInit(0);

       //err = cuDeviceGet(&device, 0);
       //if(err != CUDA_SUCCESS)  
      //{ char message[200];
      //  const char *error;
      //  cuGetErrorString(err, &error);
      //  strcpy(message,"Error get device (init_cuda): ");
      //  strcat(message, error);
      //  enif_raise_exception(env,enif_make_string(env, message, ERL_NIF_LATIN1));
      //}

       err = cuCtxCreate(&context, 0, device);
      // printf("Context created: %p",context);
       if(err != CUDA_SUCCESS)  
      { char message[200];
        const char *error;
        cuGetErrorString(err, &error);
        strcpy(message,"Error INIT CUDA: ");
        strcat(message, error);
        enif_raise_exception(env,enif_make_string(env, message, ERL_NIF_LATIN1));
      }
    
      
    } else {cuCtxSetCurrent(context);}
}

#define MX_ROWS(matrix) (((uint32_t*)matrix)[0])
#define MX_COLS(matrix) (((uint32_t*)matrix)[1])
#define MX_SET_ROWS(matrix, rows) ((uint32_t*)matrix)[0] = rows
#define MX_SET_COLS(matrix, cols) ((uint32_t*)matrix)[1] = cols
#define MX_LENGTH(matrix) ((((uint32_t*)matrix)[0])*(((uint32_t*)matrix)[1]) + 2)


ErlNifResourceType *KERNEL_TYPE;
ErlNifResourceType *ARRAY_TYPE;
ErlNifResourceType *PINNED_ARRAY;
ErlNifResourceType *FLAWD_ARRAY_TYPE;

/* Forward declaration: definida junto com o resto da infraestrutura de
 * stream do flawd, mais abaixo no arquivo. Precisa estar visível aqui
 * para que jit_compile_and_launch_nif possa lançar o kernel na mesma
 * stream usada pelas transferências H2D/D2H do flawd (ver comentário
 * "STREAM DO KERNEL" dentro de jit_compile_and_launch_nif). */
static cudaStream_t flawd_get_stream(void);

void
dev_array_destructor(ErlNifEnv *env, void *res) {
  CUdeviceptr *dev_array = (CUdeviceptr*) res;
  cuMemFree(*dev_array);
}

void
dev_pinned_array_destructor(ErlNifEnv *env, void *res) {
  float **dev_array = (float**) res;
  cudaFreeHost(*dev_array);
}

static void flawd_dev_array_destructor(ErlNifEnv *env, void *res) {
    (void)env; (void)res;
}

static int
load(ErlNifEnv *env, void **priv_data, ERL_NIF_TERM load_info) {

    KERNEL_TYPE =
        enif_open_resource_type(env, NULL, "kernel",
                                NULL, ERL_NIF_RT_CREATE, NULL);

    ARRAY_TYPE =
        enif_open_resource_type(env, NULL, "gpu_ref",
                                dev_array_destructor, ERL_NIF_RT_CREATE, NULL);

    PINNED_ARRAY =
        enif_open_resource_type(env, NULL, "pinned_array",
                                dev_pinned_array_destructor, ERL_NIF_RT_CREATE, NULL);

    FLAWD_ARRAY_TYPE =
        enif_open_resource_type(env, NULL, "flawd_gpu_ref",
                                flawd_dev_array_destructor, ERL_NIF_RT_CREATE, NULL);

    return 0;
}

/////////////////////////////
////////
///////////////////////
//////////////////////////////
//////////////// BEGIN CUDA DRIVER API CODE
/////////////////////////////
/////////////////////////////
/////////////////////////////

///////////////////////
/////////
//////// BEGIN JIT COMPILATION KERNEL
////////
///////////////////

void fail_cuda(ErlNifEnv *env, CUresult result, const char *obs){
        
       char message[1000];
        const char *error;
        strcpy(message,"Error  CUDA ");
        strcat(message,obs);
        strcat(message,": ");
        cuGetErrorString(result, &error);
        strcat(message, error);
        enif_raise_exception(env,enif_make_string(env, message, ERL_NIF_LATIN1));

}


void fail_nvrtc(ErlNifEnv *env,nvrtcResult result, const char *obs){
        
        char message[1000];
        
        strcpy(message,"Error  NVRTC ");
        strcat(message,obs);
        strcat(message,": ");
        strcat(message, nvrtcGetErrorString(result));
        enif_raise_exception(env,enif_make_string(env, message, ERL_NIF_LATIN1));

}


char* compile_to_ptx(ErlNifEnv *env, char* program_source) {
    nvrtcResult rv;

   
    // create nvrtc program
    nvrtcProgram prog;
    rv = nvrtcCreateProgram(
        &prog,
        program_source,
        nullptr,
        0,
        nullptr,
        nullptr
    );
    if(rv != NVRTC_SUCCESS) fail_nvrtc(env,rv,"nvrtcCreateProgram");
   
    
    int size_options = 10;
     const char* options[10] = {
        "--include-path=/lib/erlang/usr/include/",
        "--include-path=/usr/include/",
        "--include-path=/usr/lib/",
        "--include-path=/usr/include/x86_64-linux-gnu/",
        "--include-path=/usr/include/c++/11",
        "--include-path=/usr/include/x86_64-linux-gnu/c++/11",
        "--include-path=/usr/include/c++/11/backward",
        "--include-path=/usr/lib/gcc/x86_64-linux-gnu/11/include",
        "--include-path=/usr/include/i386-linux-gnu/",
        "--include-path=/usr/local/include"
 };
 
    rv = nvrtcCompileProgram(prog, size_options, options);
    if(rv != NVRTC_SUCCESS) {
        nvrtcResult erro_g = rv;
        size_t log_size;
        rv = nvrtcGetProgramLogSize(prog, &log_size);
        if(rv != NVRTC_SUCCESS) fail_nvrtc(env,rv,"nvrtcGetProgramLogSize");
        //auto log = std::make_unique<char[]>(log_size);
        char log[log_size];
        rv = nvrtcGetProgramLog(prog, log);
        if(rv != NVRTC_SUCCESS) fail_nvrtc(env,rv,"nvrtcGetProgramLog");
        assert(log[log_size - 1] == '\0');

        printf("Compilation error; log: %s\n", log);

        fail_nvrtc(env,erro_g,"nvrtcCompileProgram");
        //return enif_make_int(env, 0);

    }
    // get ptx code
    size_t ptx_size;
    rv = nvrtcGetPTXSize(prog, &ptx_size);
    if(rv != NVRTC_SUCCESS) fail_nvrtc(env,rv,"nvrtcGetPTXSize");
    char* ptx_source = new char[ptx_size];
    nvrtcGetPTX(prog, ptx_source);
  
   
    if(rv != NVRTC_SUCCESS) fail_nvrtc(env,rv,"nvrtcGetPTX");
    assert(ptx_source[ptx_size - 1] == '\0');

    nvrtcDestroyProgram(&prog);

    return ptx_source;
}



static ERL_NIF_TERM jit_compile_and_launch_nif(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
  
    ERL_NIF_TERM list_types;
    ERL_NIF_TERM head_types;
    ERL_NIF_TERM tail_types;

    ERL_NIF_TERM list_args;
    ERL_NIF_TERM head_args;
    ERL_NIF_TERM tail_args;

    const ERL_NIF_TERM *tuple_blocks;
    const ERL_NIF_TERM *tuple_threads;
    int arity;

    CUmodule   module;
    CUfunction function;
    CUresult err;

    /// START COLLECTING TIME

    //float time;
    //cudaEvent_t start, stop;   
    // cudaEventCreate(&start) ;
    //cudaEventCreate(&stop) ;
    //cudaEventRecord(start, 0) ;
  
  /////////// get name kernel

    ERL_NIF_TERM e_name = argv[0];
    unsigned int size_name;
    if (!enif_get_list_length(env,e_name,&size_name)) {
      return enif_make_badarg(env);
    }

   char kernel_name[size_name+1];
   
   enif_get_string(env,e_name,kernel_name,size_name+1,ERL_NIF_LATIN1);
  

  
  ///////////// get code

    ERL_NIF_TERM e_code = argv[1];
    unsigned int size_code;
    if (!enif_get_list_length(env,e_code,&size_code)) {
      return enif_make_badarg(env);
    }

   char code[size_code+1];
   
   enif_get_string(env,e_code,code,size_code+1,ERL_NIF_LATIN1);
  
   if (!enif_get_tuple(env, argv[2], &arity, &tuple_blocks)) {
      printf ("spawn: blocks argument is not a tuple");
    }

    if (!enif_get_tuple(env, argv[3], &arity, &tuple_threads)) {
      printf ("spawn:threads argument is not a tuple");
    }
    int b1,b2,b3,t1,t2,t3;

    enif_get_int(env,tuple_blocks[0],&b1);
    enif_get_int(env,tuple_blocks[1],&b2);
    enif_get_int(env,tuple_blocks[2],&b3);
    enif_get_int(env,tuple_threads[0],&t1);
    enif_get_int(env,tuple_threads[1],&t2);
    enif_get_int(env,tuple_threads[2],&t3);

  int size_args;

  if (!enif_get_int(env, argv[4], &size_args)) {
      return enif_make_badarg(env);
  }

  CUdeviceptr arrays[size_args];
  float floats[size_args];
  int ints[size_args];
  double doubles[size_args];
  int arrays_ptr=0;
  int floats_ptr=0;
  int doubles_ptr=0;
  int ints_ptr=0;
  /* Marcada quando algum argumento tint/tfloat/tdouble vier de um
   * resource FLAWD_ARRAY_TYPE (ver "STREAM DO KERNEL" mais abaixo). */
  int uses_flawd_array = 0;
  // printf("%s\n",code);
   //printf("Args: %d %d %d %d %d %d\n",b1,b2,b3,t1,t2,t3);
 
  /* ----------------------------------------------------------
   * KERNEL CACHE: compila apenas uma vez por source único.
   * Cache hit: pula NVRTC (~100ms economizados por chamada).
   * ---------------------------------------------------------- */
  {
    uint64_t src_hash = fnv1a(code);
    pthread_mutex_lock(&g_kcache_mutex);
    function = kcache_lookup(code, src_hash);
    if (function == NULL) {
      /* Cache miss — compilar agora */
      char* ptx = compile_to_ptx(env, code);
      init_cuda(env);
      err = cuModuleLoadDataEx(&module, ptx, 0, 0, 0);
      delete[] ptx;                              /* libera PTX imediatamente */
      if (err != CUDA_SUCCESS) {
        pthread_mutex_unlock(&g_kcache_mutex);
        fail_cuda(env, err, "cuModuleLoadData jit compile");
        return enif_make_int(env, 0);
      }
      err = cuModuleGetFunction(&function, module, kernel_name);
      if (err != CUDA_SUCCESS) {
        pthread_mutex_unlock(&g_kcache_mutex);
        fail_cuda(env, err, "cuModuleGetFunction jit compile");
        return enif_make_int(env, 0);
      }
      kcache_store(code, src_hash, module, function);
      /* Descomente para depurar: */
      /* printf("[kcache] compiled & cached: %s\n", kernel_name); */
    } else {
      /* Descomente para depurar: */
      /* printf("[kcache] hit: %s\n", kernel_name); */
    }
    pthread_mutex_unlock(&g_kcache_mutex);
  }


  void *args[size_args];

  list_types = argv[5];
  list_args = argv[6];

  for(int i=0; i<size_args; i++)
  {
    char type_name[1024];
    unsigned int size_type;
    if (!enif_get_list_cell(env,list_types,&head_types,&tail_types))
    { printf("erro get list cell\n");
      return enif_make_badarg(env);
    }
    if (!enif_get_list_length(env,head_types,&size_type))
    { printf("erro get list length\n");
      return enif_make_badarg(env);
    }
    
    enif_get_string(env,head_types,type_name,size_type+1,ERL_NIF_LATIN1);

     if (!enif_get_list_cell(env,list_args,&head_args,&tail_args))
       { printf("erro get list cell\n");
          return enif_make_badarg(env);
       }

    
    if (strcmp(type_name, "int") == 0) 
    {
       int iarg;
      if(! enif_get_int(env, head_args, &iarg))
       { printf("error getting int arg\n");
          return enif_make_badarg(env);
       }
       ints[ints_ptr] = iarg;
       args[i] = (void*)  &ints[ints_ptr];
       ints_ptr++;


    } else if (strcmp(type_name, "float") == 0)
    {

      
      double darg;
      if(! enif_get_double(env, head_args, &darg))
      { printf("error getting float arg\n");
          return enif_make_badarg(env);
       }

      floats[floats_ptr] = (float) darg;
       args[i] = (void*)  &floats[floats_ptr];
       floats_ptr++;  

    } else if (strcmp(type_name, "double") == 0)
    {

      
      double darg;
      if(! enif_get_double(env, head_args, &darg))
      { printf("error getting double arg\n");
          return enif_make_badarg(env);
       }

      doubles[doubles_ptr] =  darg;
       args[i] = (void*)  &doubles[doubles_ptr];
       doubles_ptr++;  

    }
    else if (strcmp(type_name, "tint") == 0)
    {
  
      CUdeviceptr *array_res;
      if (!enif_get_resource(env, head_args, ARRAY_TYPE, (void **)&array_res)) {
        enif_get_resource(env, head_args, FLAWD_ARRAY_TYPE, (void **)&array_res);
        uses_flawd_array = 1;
      }
      arrays[arrays_ptr] = *array_res;
      args[i] = (void*)  &arrays[arrays_ptr];
      arrays_ptr++;
    } else if (strcmp(type_name, "tfloat") == 0)
    {
      CUdeviceptr *array_res;
      if (!enif_get_resource(env, head_args, ARRAY_TYPE, (void **)&array_res)) {
        enif_get_resource(env, head_args, FLAWD_ARRAY_TYPE, (void **)&array_res);
        uses_flawd_array = 1;
      }
      arrays[arrays_ptr] = *array_res;
      args[i] = (void*)  &arrays[arrays_ptr];
      arrays_ptr++;

     } else if (strcmp(type_name, "tdouble") == 0)
    {

      CUdeviceptr *array_res;
      if (!enif_get_resource(env, head_args, ARRAY_TYPE, (void **)&array_res)) {
        enif_get_resource(env, head_args, FLAWD_ARRAY_TYPE, (void **)&array_res);
        uses_flawd_array = 1;
      }
      arrays[arrays_ptr] = *array_res;
      //printf("pointer %p\n",arrays[arrays_ptr]);
      args[i] = (void*)  &arrays[arrays_ptr];
      arrays_ptr++;
    
    } else{
        printf("Type %s not suported\n", type_name);
          return enif_make_badarg(env);
    }
    
    
    list_types = tail_types;
    list_args = tail_args;
  }

//printf("after arguments and types\n");

  // LAUNCH KERNEL

  /// END COLLECTING TIME

 // cudaEventRecord(stop, 0) ;
  //  cudaEventSynchronize(stop) ;
  //  cudaEventElapsedTime(&time, start, stop) ;

   // printf("cuda%s\t%3.1f\n", kernel_name,time);

    init_cuda(env);

  /* ----------------------------------------------------------------
   * STREAM DO KERNEL
   *
   * Quando algum argumento tint/tfloat/tdouble veio de um array flawd
   * (uses_flawd_array == 1), lança o kernel na MESMA stream usada por
   * new_flawd_list_nif/get_flawd_list_nif (g_flawd_stream) em vez da
   * stream default (0). Dentro de uma única stream CUDA, operações
   * são sempre executadas na ordem em que foram enfileiradas — então
   * H2D -> kernel -> D2H ficam ordenados pela própria stream, sem
   * depender de um cudaEvent/cudaStreamWaitEvent para costurar a
   * ordem entre streams diferentes.
   *
   * Para chamadas que não envolvem flawd, uses_flawd_array permanece
   * 0 e o kernel continua sendo lançado na stream default (0), exatamente
   * como antes desta mudança — nenhum outro caminho (gnx puro, glist,
   * etc.) é afetado.
   * ---------------------------------------------------------------- */
  CUstream launch_stream = uses_flawd_array
                              ? (CUstream) flawd_get_stream()
                              : (CUstream) 0;

  err = cuLaunchKernel(function, b1, b2, b3,  // Nx1x1 blocks
                                    t1, t2, t3,            // 1x1x1 threads
                                    0, launch_stream, args, 0) ;
 // printf("after kernel launch\n");
  if (err != CUDA_SUCCESS) fail_cuda(env,err,"cuLaunchKernel jit compile");
   
  cuCtxSynchronize();

  // int ptr_matrix[1000];
   //CUdeviceptr *dev_array = (CUdeviceptr*) args[1];
   //err=  cuMemcpyDtoH(ptr_matrix, dev_array, 3*sizeof(int)) ;
   // printf("pointer %p\n",*dev_array);
  //  printf("blah %p\n",args[0]);
    if(err != CUDA_SUCCESS)  
      { char message[200];//printf("its ok\n");
        const char *error;
        cuGetErrorString(err, &error);
        strcpy(message,"Error at kernel launch: ");
        strcat(message, error);
        enif_raise_exception(env,enif_make_string(env, message, ERL_NIF_LATIN1));
    }

   return enif_make_int(env, 0);
}  


///////////////////////
/////////
////////  JIT COMPILATION KERNEL
////////
///////////////////




static ERL_NIF_TERM get_gpu_array_nif(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
  int nrow;
  int ncol;
  char type_name[1024];

  ERL_NIF_TERM  result;
  CUdeviceptr dev_array;
  CUresult err;

 init_cuda(env);
// printf("entrou get array\n");
  CUdeviceptr *array_res;

    if (!enif_get_resource(env, argv[0], ARRAY_TYPE, (void **) &array_res)) {
       return enif_make_badarg(env);
    }

  dev_array = *array_res;

   
  if (!enif_get_int(env, argv[1], &nrow)) {
      return enif_make_badarg(env);
  }
  
  if (!enif_get_int(env, argv[2], &ncol)) {
      return enif_make_badarg(env);
  }

  ERL_NIF_TERM e_type_name = argv[3];
  unsigned int size_type_name;
  if (!enif_get_list_length(env,e_type_name,&size_type_name)) {
      return enif_make_badarg(env);
  }

  
  enif_get_string(env,e_type_name,type_name,size_type_name+1,ERL_NIF_LATIN1);

  
  if (strcmp(type_name, "float") == 0) 
  {
    

    int result_size = sizeof(float) * (nrow*ncol);
    int data_size = sizeof(float) * (nrow*ncol);
    float *result_data = (float *) enif_make_new_binary(env, result_size, &result);

    float *ptr_matrix ;
    ptr_matrix = result_data;
    
    
    //// MAKE CUDA CALL
    err = cuMemcpyDtoH(ptr_matrix, dev_array, data_size) ;
  
    if(err != CUDA_SUCCESS)  
      { char message[200]; 
        const char *error;
        cuGetErrorString(err, &error);
        strcpy(message,"Error (get_gpu_array_nif): error compying data from device to host: ");
        strcat(message, error);
        enif_raise_exception(env,enif_make_string(env, message, ERL_NIF_LATIN1));
    }
     //////// END CUDA CALL

  } else if (strcmp(type_name, "int") == 0)
  {
    
    int result_size = sizeof(int) * (nrow*ncol);
    int data_size = sizeof(int) * (nrow*ncol);
    int *result_data = (int *) enif_make_new_binary(env, result_size, &result);

    int *ptr_matrix ;
    ptr_matrix = result_data;
    
    //// MAKE CUDA CALL
     // printf("cuda get\n");
     // printf("pointer %p\n",dev_array);
    err=  cuMemcpyDtoH(ptr_matrix, dev_array, data_size) ;
    
    if(err != CUDA_SUCCESS)  
      { char message[200]; //printf("cuda get\n");
        const char *error;
        cuGetErrorString(err, &error);
        strcpy(message,"Error (get_gpu_array_nif) copying data from device to host: ");
        strcat(message, error);
        enif_raise_exception(env,enif_make_string(env, message, ERL_NIF_LATIN1));
    }
     //////// END CUDA CALL

    
  } else if (strcmp(type_name, "double") == 0)
  {
   
    int result_size = sizeof(double) * (nrow*ncol);
    int data_size = sizeof(double) * (nrow*ncol);
    double *result_data = (double *) enif_make_new_binary(env, result_size, &result);

    double *ptr_matrix ;
    ptr_matrix = result_data;
    
    //// MAKE CUDA CALL
    err=  cuMemcpyDtoH(ptr_matrix, dev_array, data_size) ;
    
    if(err != CUDA_SUCCESS)  
      { char message[200];
        const char *error;
        cuGetErrorString(err, &error);
        strcpy(message,"Error (get_gpu_array_nif) copying data from device to host: ");
        strcat(message, error);
        enif_raise_exception(env,enif_make_string(env, message, ERL_NIF_LATIN1));
    }
     //////// END CUDA CALL

    
  } else /* default: */
 {
    char message[200];
        strcpy(message,"Error (get_gpu_array_nif) copying data from device to host: ");
        strcat(message, type_name);
        enif_raise_exception(env,enif_make_string(env, message, ERL_NIF_LATIN1));
 }
  return result;
}


static ERL_NIF_TERM create_gpu_array_nx_nif(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
  ErlNifBinary  array_el;
  
  int nrow,ncol;

  CUresult err;
  CUdeviceptr     dev_array;
  

 init_cuda(env);

 

  if (!enif_inspect_binary(env, argv[0], &array_el)) return enif_make_badarg(env);

  if (!enif_get_int(env, argv[1], &nrow)) {
      return enif_make_badarg(env);
  }
  
  if (!enif_get_int(env, argv[2], &ncol)) {
      return enif_make_badarg(env);
  }

  ERL_NIF_TERM e_type_name = argv[3];
  unsigned int size_type_name;
  if (!enif_get_list_length(env,e_type_name,&size_type_name)) {
      return enif_make_badarg(env);
  }

 char type_name[1024];
  
  enif_get_string(env,e_type_name,type_name,size_type_name+1,ERL_NIF_LATIN1);

  // FINAL TERM TO BE RETURNED:
  ERL_NIF_TERM term;

  if (strcmp(type_name, "float") == 0) 
  { 
    
    float         *array;
    
    array = (float *) array_el.data;

   size_t data_size = sizeof(float)* ncol*nrow;
  // printf("size float: %d data size: %d\n", sizeof(float),data_size);
    ///// MAKE CUDA CALL
    err = cuMemAlloc(&dev_array, data_size) ;
  
    if(err != CUDA_SUCCESS)  
      { char message[200];
        const char *error;
        cuGetErrorString(err, &error);
       // printf("nrow %d ncol %d size %d size bytes %d\n", nrow, ncol,nrow*ncol, data_size);
        strcpy(message,"Error (create_gpu_array_nx_nif:) cuMemAlloc size: %d");
        strcat(message, error);
        enif_raise_exception(env,enif_make_string(env, message, ERL_NIF_LATIN1));
    }
  ///// MAKE CUDA CALL
    err= cuMemcpyHtoD(dev_array, array, data_size) ;
     if(err != CUDA_SUCCESS)  
      { char message[200];
        const char *error;
        cuGetErrorString(err, &error);
        strcpy(message,"Error create_gpu_array_nx_nif: ");
        strcat(message, error);
        enif_raise_exception(env,enif_make_string(env, message, ERL_NIF_LATIN1));
    }
  
  /////////// END CUDA CALL

    

  } 
  else if (strcmp(type_name, "int") == 0)
  {
    int         *array;
    array = (int *) array_el.data;
    
    size_t data_size = sizeof(int)* ncol*nrow;

     ///// MAKE CUDA CALL
    err = cuMemAlloc(&dev_array, data_size) ;
  
    if(err != CUDA_SUCCESS)  
      { char message[200];
        const char *error;
        cuGetErrorString(err, &error);
        strcpy(message,"Error create_gpu_array_nx_nif1: ");
        strcat(message, error);
        enif_raise_exception(env,enif_make_string(env, message, ERL_NIF_LATIN1));
    }
  ///// MAKE CUDA CALL
    err= cuMemcpyHtoD(dev_array, array, data_size) ;
     if(err != CUDA_SUCCESS)  
      { char message[200];
        const char *error;
        cuGetErrorString(err, &error);
        strcpy(message,"Error create_gpu_array_nx_nif2: ");
        strcat(message, error);
        enif_raise_exception(env,enif_make_string(env, message, ERL_NIF_LATIN1));
    }
  
  /////////// END CUDA CALL



  }
  else if (strcmp(type_name, "double") == 0)
  {
    double        *array;
 
    array = (double *) array_el.data;

    size_t data_size = sizeof(double)* ncol*nrow;
    
    err = cuMemAlloc(&dev_array, data_size) ;
    if(err != CUDA_SUCCESS)  
      { char message[200];
        const char *error;
        cuGetErrorString(err, &error);
        strcpy(message,"Error create_gpu_array_nx_nif1: ");
        strcat(message, error);
        enif_raise_exception(env,enif_make_string(env, message, ERL_NIF_LATIN1));
    }
  ///// MAKE CUDA CALL
    err= cuMemcpyHtoD(dev_array, array, data_size) ;
     if(err != CUDA_SUCCESS)  
      { char message[200];
        const char *error;
        cuGetErrorString(err, &error);
        strcpy(message,"Error create_gpu_array_nx_nif2: ");
        strcat(message, error);
        enif_raise_exception(env,enif_make_string(env, message, ERL_NIF_LATIN1));
    }
  
  /////////// END CUDA CALL

     
  }
 /* more else if clauses */
 else /* default: */
 {

  
    char message[200];
        strcpy(message,"Error create_gpu_array_nx_nif: unknown type: ");
        strcat(message, type_name);
        enif_raise_exception(env,enif_make_string(env, message, ERL_NIF_LATIN1));
 }

  CUdeviceptr *gpu_res = (CUdeviceptr*)enif_alloc_resource(ARRAY_TYPE, sizeof(CUdeviceptr));
  *gpu_res = dev_array;
  term = enif_make_resource(env, gpu_res);
  // ...and release the resource so that it will be freed when Erlang garbage collects
  enif_release_resource(gpu_res);
  return term;
}






static ERL_NIF_TERM new_gpu_array_nif(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
  size_t data_size;
  int nrow,ncol;
  ERL_NIF_TERM term;
  
  CUresult err;
  CUdeviceptr dev_array;

   init_cuda(env);
  
  if (!enif_get_int(env, argv[0], &nrow)) {
      return enif_make_badarg(env);
  }
  
  if (!enif_get_int(env, argv[1], &ncol)) {
      return enif_make_badarg(env);
  }

  ERL_NIF_TERM e_type_name = argv[2];
  unsigned int size_type_name;
  if (!enif_get_list_length(env,e_type_name,&size_type_name)) {
      return enif_make_badarg(env);
  }
  char type_name[1024];
  enif_get_string(env,e_type_name,type_name,size_type_name+1,ERL_NIF_LATIN1);

  if (strcmp(type_name, "float") == 0) 
  {

    
    data_size = nrow * ncol * sizeof(float);

    //// MAKE CUDA CALL
    err = cuMemAlloc(&dev_array, data_size) ;
    
    if(err != CUDA_SUCCESS)  
      { char message[200];
        const char *error;
        cuGetErrorString(err, &error);
        strcpy(message,"Error new_gpu_array_nif: ");
        strcat(message, error);
        enif_raise_exception(env,enif_make_string(env, message, ERL_NIF_LATIN1));
    }

     //END CUDA CALL


   
  } else if (strcmp(type_name, "int") == 0)
  {
    data_size = nrow * ncol * sizeof(int);

    //// MAKE CUDA CALL
    err = cuMemAlloc(&dev_array, data_size) ;
    
    if(err != CUDA_SUCCESS)  
      { char message[200];
        const char *error;
        cuGetErrorString(err, &error);
        strcpy(message,"Error new_gpu_array_nif: ");
        strcat(message, error);
        enif_raise_exception(env,enif_make_string(env, message, ERL_NIF_LATIN1));
    }

     //END CUDA CALL

 // printf("pointer %p\n",dev_array);
    
  } else if (strcmp(type_name, "double") == 0)
  {
   data_size = nrow * ncol * sizeof(double);

    //// MAKE CUDA CALL
    err = cuMemAlloc(&dev_array, data_size) ;
    
    if(err != CUDA_SUCCESS)  
      { char message[200];
        const char *error;
        cuGetErrorString(err, &error);
        strcpy(message,"Error new_gpu_array_nif:");
        strcat(message, error);
        enif_raise_exception(env,enif_make_string(env, message, ERL_NIF_LATIN1));
    }

     //END CUDA CALL


     
  } else /* default: */
 {
    char message[200];
        strcpy(message,"Error new_gpu_array_nif: unknown type: ");
        strcat(message, type_name);
        enif_raise_exception(env,enif_make_string(env, message, ERL_NIF_LATIN1));
 }

     CUdeviceptr *gpu_res = (CUdeviceptr*)enif_alloc_resource(ARRAY_TYPE, sizeof(CUdeviceptr));
     *gpu_res = dev_array;
      term = enif_make_resource(env, gpu_res);
  // ...and release the resource so that it will be freed when Erlang garbage collects
      enif_release_resource(gpu_res);
  return term;
}

/////////////////////////////
////////
///////////////////////
//////////////////////////////
//////////////// END CUDA DRIVER API CODE
/////////////////////////////
/////////////////////////////
/////////////////////////////

static ERL_NIF_TERM new_pinned_nif(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {

  float *host_matrix;
  cudaError_t error_gpu;
  int length;
  ERL_NIF_TERM list;
  ERL_NIF_TERM head;
  ERL_NIF_TERM tail;
  if (!enif_get_list_cell(env,argv[0],&head,&tail)) return enif_make_badarg(env);
  if (!enif_get_int(env, argv[1], &length))  return enif_make_badarg(env);




  int data_size = sizeof(float)*(length+2);
  

  ///// MAKE CUDA CALL
 // teste host_matrix = (float*) malloc(data_size);
  cudaMallocHost( (void**)&host_matrix, data_size);
  error_gpu = cudaGetLastError();
  if(error_gpu != cudaSuccess)  
      { char message[200];
        strcpy(message,"Error new_pinned_nif: ");
        strcat(message, cudaGetErrorString(error_gpu));
        enif_raise_exception(env,enif_make_string(env, message, ERL_NIF_LATIN1));
      }
  MX_SET_ROWS(host_matrix, 1);
  MX_SET_COLS(host_matrix, length);

  list = argv[0];

  for(int i=2;i<(length+2);i++)
  {
    enif_get_list_cell(env,list,&head,&tail);
    double dvalue;
    enif_get_double(env, head, &dvalue);
    host_matrix[i] = (float) dvalue;
    list = tail;

  }


  float **pinned_res = (float**)enif_alloc_resource(PINNED_ARRAY, sizeof(float *));
  *pinned_res = host_matrix;
  ERL_NIF_TERM term = enif_make_resource(env, pinned_res);
  // ...and release the resource so that it will be freed when Erlang garbage collects
  enif_release_resource(pinned_res);

  return term;

}

static ERL_NIF_TERM new_gmatrex_pinned_nif(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
  float         *matrix;
  float         *dev_matrix;
  cudaError_t error_gpu;
  float **array_res;

  if (!enif_get_resource(env, argv[0], PINNED_ARRAY, (void **) &array_res)) {
    return enif_make_badarg(env);
  }

  matrix = (float *) *array_res;
  uint64_t data_size = sizeof(float)*(MX_LENGTH(matrix)-2);
  
  matrix +=2; 

  ///// MAKE CUDA CALL
  cudaMalloc( (void**)&dev_matrix, data_size);
  error_gpu = cudaGetLastError();
  if(error_gpu != cudaSuccess)  
      { char message[200];
        strcpy(message,"Error create_ref_nif: ");
        strcat(message, cudaGetErrorString(error_gpu));
        enif_raise_exception(env,enif_make_string(env, message, ERL_NIF_LATIN1));
      }
  ///// MAKE CUDA CALL
  cudaMemcpy( dev_matrix, matrix, data_size, cudaMemcpyHostToDevice );
  error_gpu = cudaGetLastError();
  if(error_gpu != cudaSuccess)  
      { char message[200];
        strcpy(message,"Error create_ref_nif: ");
        strcat(message, cudaGetErrorString(error_gpu));
        enif_raise_exception(env,enif_make_string(env, message, ERL_NIF_LATIN1));
      }
  
  /////////// END CUDA CALL

  float **gpu_res = (float**)enif_alloc_resource(ARRAY_TYPE, sizeof(float *));
  *gpu_res = dev_matrix;
  ERL_NIF_TERM term = enif_make_resource(env, gpu_res);
  // ...and release the resource so that it will be freed when Erlang garbage collects
  enif_release_resource(gpu_res);

  return term;
}

static ERL_NIF_TERM create_nx_ref_nif(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
  ErlNifBinary  matrix_el;
  cudaError_t error_gpu;
  int nrow,ncol;
  
  if (!enif_inspect_binary(env, argv[0], &matrix_el)) return enif_make_badarg(env);

  if (!enif_get_int(env, argv[1], &nrow)) {
      return enif_make_badarg(env);
  }
  
  if (!enif_get_int(env, argv[2], &ncol)) {
      return enif_make_badarg(env);
  }

  ERL_NIF_TERM e_type_name = argv[3];
  unsigned int size_type_name;
  if (!enif_get_list_length(env,e_type_name,&size_type_name)) {
      return enif_make_badarg(env);
  }

  

  char type_name[1024];
  
  enif_get_string(env,e_type_name,type_name,size_type_name+1,ERL_NIF_LATIN1);

  // FINAL TERM TO BE RETURNED:
  ERL_NIF_TERM term;

  if (strcmp(type_name, "float") == 0) 
  { 
    
    float         *matrix;
    float         *dev_matrix;
    matrix = (float *) matrix_el.data;

    uint64_t data_size = sizeof(float)* ncol*nrow;

    ///// MAKE CUDA CALL
    cudaMalloc( (void**)&dev_matrix, data_size);
    error_gpu = cudaGetLastError();
    if(error_gpu != cudaSuccess)  
      { char message[200];
        strcpy(message,"Error create_nx_ref_nif: ");
        strcat(message, cudaGetErrorString(error_gpu));
        enif_raise_exception(env,enif_make_string(env, message, ERL_NIF_LATIN1));
      }
  ///// MAKE CUDA CALL
    cudaMemcpy( dev_matrix, matrix, data_size, cudaMemcpyHostToDevice );
    error_gpu = cudaGetLastError();
    if(error_gpu != cudaSuccess)  
       { char message[200];
         strcpy(message,"Error create_nx_ref_nif: ");
         strcat(message, cudaGetErrorString(error_gpu));
         enif_raise_exception(env,enif_make_string(env, message, ERL_NIF_LATIN1));
       }
  
  /////////// END CUDA CALL

    float **gpu_res = (float**)enif_alloc_resource(ARRAY_TYPE, sizeof(float *));
    *gpu_res = dev_matrix;
    term = enif_make_resource(env, gpu_res);
  // ...and release the resource so that it will be freed when Erlang garbage collects
     enif_release_resource(gpu_res);

  } 
  else if (strcmp(type_name, "int") == 0)
  {
    int         *matrix;
    int        *dev_matrix;
    matrix = (int *) matrix_el.data;
    
    uint64_t data_size = sizeof(int)* ncol*nrow;

    ///// MAKE CUDA CALL
    cudaMalloc( (void**)&dev_matrix, data_size);
    error_gpu = cudaGetLastError();
    if(error_gpu != cudaSuccess)  
      { char message[200];
        strcpy(message,"Error create_nx_ref_nif: ");
        strcat(message, cudaGetErrorString(error_gpu));
        enif_raise_exception(env,enif_make_string(env, message, ERL_NIF_LATIN1));
      }
  ///// MAKE CUDA CALL
    cudaMemcpy( dev_matrix, matrix, data_size, cudaMemcpyHostToDevice );
    error_gpu = cudaGetLastError();
    if(error_gpu != cudaSuccess)  
       { char message[200];
         strcpy(message,"Error create_nx_ref_nif: ");
         strcat(message, cudaGetErrorString(error_gpu));
         enif_raise_exception(env,enif_make_string(env, message, ERL_NIF_LATIN1));
       }
  
  /////////// END CUDA CALL

    int **gpu_res = (int**)enif_alloc_resource(ARRAY_TYPE, sizeof(int *));
    *gpu_res = dev_matrix;
    term = enif_make_resource(env, gpu_res);
  // ...and release the resource so that it will be freed when Erlang garbage collects
     enif_release_resource(gpu_res);

  }
  else if (strcmp(type_name, "double") == 0)
  {
    double        *matrix;
    double      *dev_matrix;
    matrix = (double *) matrix_el.data;

    uint64_t data_size = sizeof(double)* ncol*nrow;
    
    ///// MAKE CUDA CALL
    cudaMalloc( (void**)&dev_matrix, data_size);
    error_gpu = cudaGetLastError();
    if(error_gpu != cudaSuccess)  
      { char message[200];
        strcpy(message,"Error create_nx_ref_nif: ");
        strcat(message, cudaGetErrorString(error_gpu));
        enif_raise_exception(env,enif_make_string(env, message, ERL_NIF_LATIN1));
      }
  ///// MAKE CUDA CALL
    cudaMemcpy( dev_matrix, matrix, data_size, cudaMemcpyHostToDevice );
    error_gpu = cudaGetLastError();
    if(error_gpu != cudaSuccess)  
       { char message[200];
         strcpy(message,"Error create_nx_ref_nif: ");
         strcat(message, cudaGetErrorString(error_gpu));
         enif_raise_exception(env,enif_make_string(env, message, ERL_NIF_LATIN1));
       }
  
  /////////// END CUDA CALL

    double **gpu_res = (double**)enif_alloc_resource(ARRAY_TYPE, sizeof(double *));
    *gpu_res = dev_matrix;
    term = enif_make_resource(env, gpu_res);
  // ...and release the resource so that it will be freed when Erlang garbage collects
     enif_release_resource(gpu_res);
   
  }
 /* more else if clauses */
 else /* default: */
 {
    char message[200];
        strcpy(message,"Error create_nx_ref_nif: unknown type: ");
        strcat(message, type_name);
        enif_raise_exception(env,enif_make_string(env, message, ERL_NIF_LATIN1));
 }

  return term;
}




static ERL_NIF_TERM create_ref_nif(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
  ErlNifBinary  matrix_el;
  float         *matrix;
  float         *dev_matrix;
  cudaError_t error_gpu;
  
  if (!enif_inspect_binary(env, argv[0], &matrix_el)) return enif_make_badarg(env);

  matrix = (float *) matrix_el.data;
  uint64_t data_size = sizeof(float)*(MX_LENGTH(matrix)-2);
  
  matrix +=2; 

  ///// MAKE CUDA CALL
  cudaMalloc( (void**)&dev_matrix, data_size);
  error_gpu = cudaGetLastError();
  if(error_gpu != cudaSuccess)  
      { char message[200];
        strcpy(message,"Error create_ref_nif: ");
        strcat(message, cudaGetErrorString(error_gpu));
        enif_raise_exception(env,enif_make_string(env, message, ERL_NIF_LATIN1));
      }
  ///// MAKE CUDA CALL
  cudaMemcpy( dev_matrix, matrix, data_size, cudaMemcpyHostToDevice );
  error_gpu = cudaGetLastError();
  if(error_gpu != cudaSuccess)  
      { char message[200];
        strcpy(message,"Error create_ref_nif: ");
        strcat(message, cudaGetErrorString(error_gpu));
        enif_raise_exception(env,enif_make_string(env, message, ERL_NIF_LATIN1));
      }
  
  /////////// END CUDA CALL

  float **gpu_res = (float**)enif_alloc_resource(ARRAY_TYPE, sizeof(float *));
  *gpu_res = dev_matrix;
  ERL_NIF_TERM term = enif_make_resource(env, gpu_res);
  // ...and release the resource so that it will be freed when Erlang garbage collects
  enif_release_resource(gpu_res);

  return term;
}

static ERL_NIF_TERM new_gpu_nx_nif(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
  int data_size;
  cudaError_t error_gpu;
  int nrow,ncol;
  ERL_NIF_TERM term;

  if (!enif_get_int(env, argv[0], &nrow)) {
      return enif_make_badarg(env);
  }
  
  if (!enif_get_int(env, argv[1], &ncol)) {
      return enif_make_badarg(env);
  }

  ERL_NIF_TERM e_type_name = argv[2];
  unsigned int size_type_name;
  if (!enif_get_list_length(env,e_type_name,&size_type_name)) {
      return enif_make_badarg(env);
  }
  char type_name[1024];
  enif_get_string(env,e_type_name,type_name,size_type_name+1,ERL_NIF_LATIN1);

  if (strcmp(type_name, "float") == 0) 
  {

    float         *dev_matrix;
    data_size = nrow * ncol * sizeof(float);

    //// MAKE CUDA CALL
    cudaMalloc( (void**)&dev_matrix, data_size);
    error_gpu = cudaGetLastError();
    if(error_gpu != cudaSuccess)  
       { char message[200];
         strcpy(message,"Error new_gpu_nx_nif: ");
         strcat(message, cudaGetErrorString(error_gpu));
         enif_raise_exception(env,enif_make_string(env, message, ERL_NIF_LATIN1));
       }

     //END CUDA CALL


     float **gpu_res = (float**)enif_alloc_resource(ARRAY_TYPE, sizeof(float *));
     *gpu_res = dev_matrix;
      term = enif_make_resource(env, gpu_res);
  // ...and release the resource so that it will be freed when Erlang garbage collects
      enif_release_resource(gpu_res);
  } else if (strcmp(type_name, "int") == 0)
  {
    int         *dev_matrix;
    data_size = nrow * ncol * sizeof(int);

    //// MAKE CUDA CALL
    cudaMalloc( (void**)&dev_matrix, data_size);
    error_gpu = cudaGetLastError();
    if(error_gpu != cudaSuccess)  
       { char message[200];
         strcpy(message,"Error new_gpu_nx_nif: ");
         strcat(message, cudaGetErrorString(error_gpu));
         enif_raise_exception(env,enif_make_string(env, message, ERL_NIF_LATIN1));
       }

     //END CUDA CALL


     int **gpu_res = (int**)enif_alloc_resource(ARRAY_TYPE, sizeof(int *));
     *gpu_res = dev_matrix;
      term = enif_make_resource(env, gpu_res);
  // ...and release the resource so that it will be freed when Erlang garbage collects
      enif_release_resource(gpu_res);
  } else if (strcmp(type_name, "double") == 0)
  {
    double         *dev_matrix;
    data_size = nrow * ncol * sizeof(double);

    //// MAKE CUDA CALL
    cudaMalloc( (void**)&dev_matrix, data_size);
    error_gpu = cudaGetLastError();
    if(error_gpu != cudaSuccess)  
       { char message[200];
         strcpy(message,"Error new_gpu_nx_nif: ");
         strcat(message, cudaGetErrorString(error_gpu));
         enif_raise_exception(env,enif_make_string(env, message, ERL_NIF_LATIN1));
       }

     //END CUDA CALL


     double **gpu_res = (double**)enif_alloc_resource(ARRAY_TYPE, sizeof(double *));
     *gpu_res = dev_matrix;
      term = enif_make_resource(env, gpu_res);
  // ...and release the resource so that it will be freed when Erlang garbage collects
      enif_release_resource(gpu_res);
  } else /* default: */
 {
    char message[200];
        strcpy(message,"Error gpu_nx_nif: unknown type: ");
        strcat(message, type_name);
        enif_raise_exception(env,enif_make_string(env, message, ERL_NIF_LATIN1));
 }
  return term;
}


static ERL_NIF_TERM new_ref_nif(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
  float         *dev_matrix;
  int data_size;
  cudaError_t error_gpu;
  
  if (!enif_get_int(env, argv[0], &data_size)) {
      return enif_make_badarg(env);
  }
 
  data_size = data_size * sizeof(float);

  //// MAKE CUDA CALL
  cudaMalloc( (void**)&dev_matrix, data_size);
  error_gpu = cudaGetLastError();
  if(error_gpu != cudaSuccess)  
      { char message[200];
        strcpy(message,"Error new_ref_nif: ");
        strcat(message, cudaGetErrorString(error_gpu));
        enif_raise_exception(env,enif_make_string(env, message, ERL_NIF_LATIN1));
      }

  //END CUDA CALL


  float **gpu_res = (float**)enif_alloc_resource(ARRAY_TYPE, sizeof(float *));
  *gpu_res = dev_matrix;
  ERL_NIF_TERM term = enif_make_resource(env, gpu_res);
  // ...and release the resource so that it will be freed when Erlang garbage collects
  enif_release_resource(gpu_res);

  return term;
}

static ERL_NIF_TERM get_nx_nif(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
  int nrow;
  int ncol;
  ERL_NIF_TERM  result;
  cudaError_t error_gpu;

  ERL_NIF_TERM e_type_name = argv[3];
  unsigned int size_type_name;
  if (!enif_get_list_length(env,e_type_name,&size_type_name)) {
      return enif_make_badarg(env);
  }

  char type_name[1024];
  enif_get_string(env,e_type_name,type_name,size_type_name+1,ERL_NIF_LATIN1);

  
  if (!enif_get_int(env, argv[1], &nrow)) {
      return enif_make_badarg(env);
  }
  
  if (!enif_get_int(env, argv[2], &ncol)) {
      return enif_make_badarg(env);
  }
  
  if (strcmp(type_name, "float") == 0) 
  {
    float **array_res;

    if (!enif_get_resource(env, argv[0], ARRAY_TYPE, (void **) &array_res)) {
       return enif_make_badarg(env);
    }

    float *dev_array = *array_res;

    int result_size = sizeof(float) * (nrow*ncol);
    int data_size = sizeof(float) * (nrow*ncol);
    float *result_data = (float *) enif_make_new_binary(env, result_size, &result);

    float *ptr_matrix ;
    ptr_matrix = result_data;
    
    //// MAKE CUDA CALL
    cudaMemcpy(ptr_matrix, dev_array, data_size, cudaMemcpyDeviceToHost );
    error_gpu = cudaGetLastError();
    if(error_gpu != cudaSuccess)  
       { char message[200];
         strcpy(message,"Error get_nx_nif: ");
         strcat(message, cudaGetErrorString(error_gpu));
         enif_raise_exception(env,enif_make_string(env, message, ERL_NIF_LATIN1));
       }
     //////// END CUDA CALL

  } else if (strcmp(type_name, "int") == 0)
  {
    int **array_res;

    if (!enif_get_resource(env, argv[0], ARRAY_TYPE, (void **) &array_res)) {
       return enif_make_badarg(env);
    }

    int *dev_array = *array_res;

    int result_size = sizeof(int) * (nrow*ncol);
    int data_size = sizeof(int) * (nrow*ncol);
    int *result_data = (int *) enif_make_new_binary(env, result_size, &result);

    int *ptr_matrix ;
    ptr_matrix = result_data;
    
    //// MAKE CUDA CALL
    cudaMemcpy(ptr_matrix, dev_array, data_size, cudaMemcpyDeviceToHost );
    error_gpu = cudaGetLastError();
    if(error_gpu != cudaSuccess)  
       { char message[200];
         strcpy(message,"Error get_nx_nif: ");
         strcat(message, cudaGetErrorString(error_gpu));
         enif_raise_exception(env,enif_make_string(env, message, ERL_NIF_LATIN1));
       }
     //////// END CUDA CALL

    
  } else if (strcmp(type_name, "double") == 0)
  {
    double **array_res;

    if (!enif_get_resource(env, argv[0], ARRAY_TYPE, (void **) &array_res)) {
       return enif_make_badarg(env);
    }

    double *dev_array = *array_res;

    double result_size = sizeof(double) * (nrow*ncol);
    double data_size = sizeof(double) * (nrow*ncol);
    double *result_data = (double *) enif_make_new_binary(env, result_size, &result);

    double *ptr_matrix ;
    ptr_matrix = result_data;
    
    //// MAKE CUDA CALL
    cudaMemcpy(ptr_matrix, dev_array, data_size, cudaMemcpyDeviceToHost );
    error_gpu = cudaGetLastError();
    if(error_gpu != cudaSuccess)  
       { char message[200];
         strcpy(message,"Error get_nx_nif: ");
         strcat(message, cudaGetErrorString(error_gpu));
         enif_raise_exception(env,enif_make_string(env, message, ERL_NIF_LATIN1));
       }
     //////// END CUDA CALL

    
  } else /* default: */
 {
    char message[200];
        strcpy(message,"Error get_nx_nif: unknown type: ");
        strcat(message, type_name);
        enif_raise_exception(env,enif_make_string(env, message, ERL_NIF_LATIN1));
 }
  return result;
}

static ERL_NIF_TERM get_matrex_nif(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
  int nrow;
  int ncol;
  ERL_NIF_TERM  result;
  float **array_res;
  cudaError_t error_gpu;
  
  if (!enif_get_resource(env, argv[0], ARRAY_TYPE, (void **) &array_res)) {
    return enif_make_badarg(env);
  }
  
  if (!enif_get_int(env, argv[1], &nrow)) {
      return enif_make_badarg(env);
  }
  
  if (!enif_get_int(env, argv[2], &ncol)) {
      return enif_make_badarg(env);
  }
  
  float *dev_array = *array_res;

  int result_size = sizeof(float) * (nrow*ncol+2);
  int data_size = sizeof(float) * (nrow*ncol);
  float *result_data = (float *) enif_make_new_binary(env, result_size, &result);

  float *ptr_matrix ;
  ptr_matrix = result_data;
  ptr_matrix +=2;


  //// MAKE CUDA CALL
  cudaMemcpy(ptr_matrix, dev_array, data_size, cudaMemcpyDeviceToHost );
  error_gpu = cudaGetLastError();
  if(error_gpu != cudaSuccess)  
      { char message[200];
        strcpy(message,"Error get_matrex_nif: ");
        strcat(message, cudaGetErrorString(error_gpu));
        enif_raise_exception(env,enif_make_string(env, message, ERL_NIF_LATIN1));
      }
  //////// END CUDA CALL

  MX_SET_ROWS(result_data, nrow);
  MX_SET_COLS(result_data, ncol);
  
  return result;
}

static ERL_NIF_TERM synchronize_nif(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
  cudaError_t error_gpu;

  ////// MAKE CUDA CALL
  cudaDeviceSynchronize();
  error_gpu = cudaGetLastError();
  if(error_gpu != cudaSuccess)  
      { char message[200];
        strcpy(message,"Error synchronize_nif: ");
        strcat(message, cudaGetErrorString(error_gpu));
        enif_raise_exception(env,enif_make_string(env, message, ERL_NIF_LATIN1));
      }
  //// END CUDA CALL
  return enif_make_int(env, 0);
}

static ERL_NIF_TERM load_kernel_nif(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
   
  ERL_NIF_TERM e_name_module = argv[0];
  ERL_NIF_TERM e_name_fun = argv[1];
  
  unsigned int size_name_module;
  unsigned int size_name_fun;
  

  enif_get_list_length(env,e_name_fun,&size_name_fun);
  enif_get_list_length(env,e_name_module,&size_name_module);

  char kernel_name[1024];
  char func_name[1024];
  char lib_name[1024];
  char module_name[1024];

  enif_get_string(env,e_name_fun,kernel_name,size_name_fun+1,ERL_NIF_LATIN1);
  enif_get_string(env,e_name_module,module_name,size_name_module+1,ERL_NIF_LATIN1);

  strcpy(func_name,kernel_name);
  strcat(func_name,"_call");
  strcpy(lib_name,"priv/");
  strcat(lib_name,module_name);
  strcat(lib_name,".so");

  
  //printf("libname %s\n",lib_name);
 // printf("func name a %s\n",func_name);
  //printf("module name %s\n", module_name);
  
  void * m_handle = dlopen(lib_name, RTLD_NOW);
  if(m_handle== NULL)  
      { fprintf(stderr, "dlopen failure: %s\n", dlerror()); 
        char message[200];
        strcpy(message,"Error opening shared library for the programa. It was not found.\n");
        enif_raise_exception(env,enif_make_string(env, message, ERL_NIF_LATIN1));
        return enif_make_int(env, 0);
      }

 // printf("Pointer %p\n",m_handle);

  void (*fn)();
  fn= (void (*)())dlsym( m_handle, func_name);

   if(fn == NULL)  
        { 
          fprintf(stderr, "dlopen failure: %s\n", dlerror()); 
          char message[200];
        strcpy(message,"Error opening .so file: ");
        strcat(message, func_name);
        strcat(message, " was not found!");
        enif_raise_exception(env,enif_make_string(env, message, ERL_NIF_LATIN1));
        return enif_make_int(env, 0);
      }
  
  void (**kernel_res)() = (void (**)()) enif_alloc_resource(KERNEL_TYPE, sizeof(void *));

  // Let's create conn and let the resource point to it
  
  *kernel_res = fn;
  
  // We can now make the Erlang term that holds the resource...
  ERL_NIF_TERM term = enif_make_resource(env, kernel_res);
  // ...and release the resource so that it will be freed when Erlang garbage collects
  enif_release_resource(kernel_res);
 

  return term;
}

static ERL_NIF_TERM load_fun_nif(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
   
  ERL_NIF_TERM e_name_module = argv[0];
  ERL_NIF_TERM e_name_fun = argv[1];
  
  unsigned int size_name_module;
  unsigned int size_name_fun;
  

  enif_get_list_length(env,e_name_fun,&size_name_fun);
  enif_get_list_length(env,e_name_module,&size_name_module);

  char kernel_name[1024];
  char func_name[1024];
  char lib_name[1024];
  char module_name[1024];

  enif_get_string(env,e_name_fun,kernel_name,size_name_fun+1,ERL_NIF_LATIN1);
  enif_get_string(env,e_name_module,module_name,size_name_module+1,ERL_NIF_LATIN1);

  strcpy(func_name,"get_");
  strcat(func_name,kernel_name);
  strcat(func_name,"_ptr");
  strcpy(lib_name,"priv/");
  strcat(lib_name,module_name);
  strcat(lib_name,".so");

 
 // printf("libname %s\n",lib_name);
  
  void * m_handle = dlopen(lib_name, RTLD_NOW);
  if(m_handle== NULL)  
      { char message[200];
        strcpy(message,"Error opening shared library for the programa. It was not found.");
        enif_raise_exception(env,enif_make_string(env, message, ERL_NIF_LATIN1));
        return enif_make_int(env, 0);
      }

//printf("Pointer %p\n",m_handle);

  //printf("function name %s \nlib name %s pointer %p\n", func_name, lib_name, m_handle);

  void* (*fn)();
  fn= (void* (*)())dlsym( m_handle, func_name);
  
 // printf("pointer function a %p %li\n",fn, (long int) fn);

  if(fn == NULL)  
        { 
          
          char message[200];
        strcpy(message,"Error opening .so file: ");
        strcat(message, kernel_name);
        strcat(message, " was not found!");
        enif_raise_exception(env,enif_make_string(env, message, ERL_NIF_LATIN1));
        return enif_make_int(env, 0);
      }


  void* ptr = fn();

  ///printf("function pointer %p\n",ptr);

  void** kernel_res = (void**) enif_alloc_resource(KERNEL_TYPE, sizeof(void *));

  // Let's create conn and let the resource point to it
  
  *kernel_res = ptr;

 // printf("kernel resource %p\n", *kernel_res);
 // printf("erlang resource %p\n", kernel_res);
  
  // We can now make the Erlang term that holds the resource...
  ERL_NIF_TERM term = enif_make_resource(env, kernel_res);
  // ...and release the resource so that it will be freed when Erlang garbage collects
  enif_release_resource(kernel_res);
 // printf("saiu\n");
  //printf("term %p\n", term);
  return term;
}



static ERL_NIF_TERM spawn_nif(ErlNifEnv *env, int argc, const ERL_NIF_TERM argv[]) {
  void (**kernel_res)(ErlNifEnv *env, const ERL_NIF_TERM argv[], ErlNifResourceType* type,ErlNifResourceType* ftype);
  //void (**kernel_res)();
  //float **array_res;
  //printf("spawn begin\n");
  //fflush(stdout);
  if (!enif_get_resource(env, argv[0], KERNEL_TYPE, (void **) &kernel_res)) {
    return enif_make_badarg(env);
  }
  
  void (*fn)(ErlNifEnv *env, const ERL_NIF_TERM argv[], ErlNifResourceType* type,ErlNifResourceType* ftype) = *kernel_res;
  //void (*fn)() = *kernel_res;
  //float *array = *array_res;
  //printf("ok nif");
  (*fn)(env,argv,ARRAY_TYPE,KERNEL_TYPE);
  //(*fn)();



  return enif_make_int(env, 0);
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////


/* ================================================================
 * FLAWD_LIST — Functional Lists All The Way Down
 *
 * otimizações em relação ao glist:
 *   1. Linearização direta em C — zero alocações no BEAM
 *   2. Software prefetch (__builtin_prefetch) com lookahead de 8 nós
 *      sobre os ponteiros taggeados das cons cells do BEAM
 *   3. Pinned (page-locked) memory via cudaMallocHost — DMA direto,
 *      sem cópia de staging buffer pelo driver
 *   4. cudaMemcpyAsync + stream dedicada — H2D e D2H não bloqueiam
 *      o scheduler do BEAM desnecessariamente  (WORK IN PROGRESS)
 *   5. Reconstrução da lista resultado inteiramente em C —
 *      zero overhead de decode binário em Elixir
 * ================================================================ */

#define FLAWD_PREFETCH_DIST 8

/* ---------------------------------------------------------------
 * Stream CUDA global reutilizada entre chamadas (WORK IN PROGRESS)
 * --------------------------------------------------------------- */
static cudaStream_t g_flawd_stream          = NULL;
static pthread_mutex_t g_flawd_stream_mutex = PTHREAD_MUTEX_INITIALIZER;

static cudaStream_t flawd_get_stream(void) {
    pthread_mutex_lock(&g_flawd_stream_mutex);
    if (g_flawd_stream == NULL) {
        cudaStreamCreate(&g_flawd_stream);
    }
    pthread_mutex_unlock(&g_flawd_stream_mutex);
    return g_flawd_stream;
}

/* ---------------------------------------------------------------
 * Pinned Buffer Pool
 *
 * Problema resolvido: cudaMallocHost/cudaFreeHost custam ~2-3ms
 * cada. Chamadas por operação (new + get) somavam ~5-6ms de piso
 * fixo, dominando o benchmark para todos os tamanhos pequenos.
 *
 * Solução: um pool de buffers pinned persistentes por direção (H2D
 * e D2H), cada buffer alocado sob demanda e reaproveitado via
 * acquire/release, "grow-only" por slot (um slot que já tem
 * capacidade suficiente é reaproveitado sem nova alocação).
 *
 * Múltiplos slots (em vez de um único buffer travado por mutex de
 * exclusão) são necessários porque Ske.map2/map3/map4/map2Reduce/
 * map4Reduce sobem MAIS DE UM array flawd simultaneamente antes de
 * qualquer um deles ser liberado (o release só acontece dentro de
 * get_flawd_list_nif, no fim do pipeline) — um único slot causava
 * deadlock: a segunda chamada de new_flawd_list_nif ficava travada
 * esperando um mutex que só seria liberado depois que o pipeline
 * inteiro (que nunca chegava a rodar) terminasse.
 * --------------------------------------------------------------- */
#define FLAWD_POOL_SLOTS 8

typedef struct {
    void  *buf;
    size_t capacity;
    int    in_use;
} PinnedSlot;

typedef struct {
    PinnedSlot      slots[FLAWD_POOL_SLOTS];
    pthread_mutex_t lock;
} PinnedPool;

/* Pool H2D: usado em new_flawd_list_nif */
static PinnedPool g_flawd_h2d = { .lock = PTHREAD_MUTEX_INITIALIZER };
/* Pool D2H: usado em get_flawd_list_nif */
static PinnedPool g_flawd_d2h = { .lock = PTHREAD_MUTEX_INITIALIZER };

/* Adquire um slot livre do pool com capacidade >= needed, crescendo-o
 * (ou alocando-o pela primeira vez) se necessário. Retorna NULL se
 * todos os FLAWD_POOL_SLOTS estiverem em uso simultaneamente (ver
 * FLAWD_POOL_SLOTS acima) ou se a alocação falhar.
 * IMPORTANTE: o chamador DEVE invocar pinned_pool_release() com o
 * MESMO ponteiro retornado aqui, após o uso. */
static void* pinned_pool_acquire(PinnedPool *pool, size_t needed) {
    pthread_mutex_lock(&pool->lock);

    int free_idx = -1;
    for (int i = 0; i < FLAWD_POOL_SLOTS; i++) {
        if (!pool->slots[i].in_use) {
            /* prefere um slot livre que já tenha capacidade suficiente */
            if (pool->slots[i].capacity >= needed) {
                pool->slots[i].in_use = 1;
                pthread_mutex_unlock(&pool->lock);
                return pool->slots[i].buf;
            }
            if (free_idx == -1) free_idx = i;
        }
    }

    if (free_idx == -1) {
        /* todos os slots estão em uso ao mesmo tempo */
        pthread_mutex_unlock(&pool->lock);
        return NULL;
    }

    PinnedSlot *slot = &pool->slots[free_idx];
    if (slot->buf != NULL) {
        cudaFreeHost(slot->buf);
        slot->buf      = NULL;
        slot->capacity = 0;
    }
    cudaError_t e = cudaMallocHost(&slot->buf, needed);
    if (e != cudaSuccess) {
        pthread_mutex_unlock(&pool->lock);
        return NULL;   /* caller verifica e reporta erro */
    }
    slot->capacity = needed;
    slot->in_use   = 1;

    void *result = slot->buf;
    pthread_mutex_unlock(&pool->lock);
    return result;
}

/* Libera o slot correspondente a `ptr` para reutilização futura. */
static void pinned_pool_release(PinnedPool *pool, void *ptr) {
    pthread_mutex_lock(&pool->lock);
    for (int i = 0; i < FLAWD_POOL_SLOTS; i++) {
        if (pool->slots[i].buf == ptr) {
            pool->slots[i].in_use = 0;
            break;
        }
    }
    pthread_mutex_unlock(&pool->lock);
}

typedef struct {
    CUdeviceptr ptr;
    size_t      capacity;
    int         in_use;
} DeviceSlot;

typedef struct {
    DeviceSlot      slots[FLAWD_POOL_SLOTS];
    pthread_mutex_t lock;
} DevicePool;

static DevicePool g_flawd_dev = { .lock = PTHREAD_MUTEX_INITIALIZER };

/* Mesma lógica de pinned_pool_acquire, para memória device. */
static CUdeviceptr device_pool_acquire(DevicePool *pool, size_t needed) {
    pthread_mutex_lock(&pool->lock);

    int free_idx = -1;
    for (int i = 0; i < FLAWD_POOL_SLOTS; i++) {
        if (!pool->slots[i].in_use) {
            if (pool->slots[i].capacity >= needed) {
                pool->slots[i].in_use = 1;
                pthread_mutex_unlock(&pool->lock);
                return pool->slots[i].ptr;
            }
            if (free_idx == -1) free_idx = i;
        }
    }

    if (free_idx == -1) {
        pthread_mutex_unlock(&pool->lock);
        return 0;
    }

    DeviceSlot *slot = &pool->slots[free_idx];
    if (slot->ptr != 0) {
        cuMemFree(slot->ptr);
        slot->ptr      = 0;
        slot->capacity = 0;
    }
    CUresult e = cuMemAlloc(&slot->ptr, needed);
    if (e != CUDA_SUCCESS) {
        pthread_mutex_unlock(&pool->lock);
        return 0;
    }
    slot->capacity = needed;
    slot->in_use   = 1;

    CUdeviceptr result = slot->ptr;
    pthread_mutex_unlock(&pool->lock);
    return result;
}

/* Libera o slot correspondente a `ptr` para reutilização futura. */
static void device_pool_release(DevicePool *pool, CUdeviceptr ptr) {
    pthread_mutex_lock(&pool->lock);
    for (int i = 0; i < FLAWD_POOL_SLOTS; i++) {
        if (pool->slots[i].ptr == ptr) {
            pool->slots[i].in_use = 0;
            break;
        }
    }
    pthread_mutex_unlock(&pool->lock);
}

/* userData do callback carrega tanto o pool quanto o ponteiro do slot
 * a liberar (antes bastava o pool, pois havia um único slot possível). */
typedef struct {
    PinnedPool *pool;
    void       *ptr;
} ReleaseH2DCallbackData;

static void CUDART_CB release_h2d_pool_callback(cudaStream_t stream,
                                                  cudaError_t  status,
                                                  void        *userData) {
    (void)stream; (void)status;
    ReleaseH2DCallbackData *data = (ReleaseH2DCallbackData *)userData;
    pinned_pool_release(data->pool, data->ptr);
    free(data);
}

/* ---------------------------------------------------------------
 * new_flawd_list_nif(list, rows, cols, type_charlist)
 *
 * Percorre a lista Elixir em C com prefetch de ponteiros de cons
 * cells, serializa direto em pinned memory e transfere para GPU
 * via cudaMemcpyAsync.
 * Retorna resource idêntico ao ARRAY_TYPE dos outros NIFs —
 * compatível com Ske.map/reduce sem nenhuma mudança.
 * --------------------------------------------------------------- */
static ERL_NIF_TERM new_flawd_list_nif(ErlNifEnv *env, int argc,
                                        const ERL_NIF_TERM argv[]) {
    /* ---- Argumentos: list, rows, cols, type_charlist ---- */
    int rows, cols;
    if (!enif_get_int(env, argv[1], &rows)) return enif_make_badarg(env);
    if (!enif_get_int(env, argv[2], &cols)) return enif_make_badarg(env);

    unsigned int type_slen;
    if (!enif_get_list_length(env, argv[3], &type_slen)) return enif_make_badarg(env);
    char type_name[64];
    enif_get_string(env, argv[3], type_name, sizeof(type_name), ERL_NIF_LATIN1);

    int n = rows * cols;
    if (n <= 0) return enif_make_badarg(env);

    size_t elem_size;
    if      (strcmp(type_name, "int")    == 0) elem_size = sizeof(int32_t);
    else if (strcmp(type_name, "float")  == 0) elem_size = sizeof(float);
    else if (strcmp(type_name, "double") == 0) elem_size = sizeof(double);
    else return enif_make_badarg(env);

    size_t data_size = (size_t)n * elem_size;

    void *h_buf = pinned_pool_acquire(&g_flawd_h2d, data_size);
    if (h_buf == NULL) {
        return enif_raise_exception(env,
            enif_make_string(env,
                "new_flawd_list_nif: pinned_pool_acquire falhou (cudaMallocHost, pool cheio ou sem memoria)",
                ERL_NIF_LATIN1));
    }

    ERL_NIF_TERM cur       = argv[0];
    ERL_NIF_TERM head, tail;
    ERL_NIF_TERM lookahead = argv[0];
    ERL_NIF_TERM la_head, la_tail;

    /* Avança lookahead FLAWD_PREFETCH_DIST passos */
    for (int k = 0; k < FLAWD_PREFETCH_DIST; k++) {
        if (!enif_get_list_cell(env, lookahead, &la_head, &la_tail)) break;
        __builtin_prefetch((void *)((uintptr_t)la_tail & ~0x3UL), 0, 1);
        lookahead = la_tail;
    }

    if (strcmp(type_name, "int") == 0) {
        int32_t *buf = (int32_t *)h_buf;
        for (int i = 0; i < n; i++) {
            if (!enif_get_list_cell(env, cur, &head, &tail)) break;
            /* prefetch próxima cons cell */
            if (enif_get_list_cell(env, lookahead, &la_head, &la_tail)) {
                __builtin_prefetch((void *)((uintptr_t)la_tail & ~0x3UL), 0, 1);
                lookahead = la_tail;
            }
            if (__builtin_expect((head & 0xF) == 0xF, 1)) {
                /* fast path: small integer imediato */
                buf[i] = (int32_t)((intptr_t)head >> 4);
            } else {
                /* slow path: big integer ou tipo inesperado */
                long val = 0;
                enif_get_int64(env, head, &val);
                buf[i] = (int32_t)val;
            }
            cur = tail;
        }
    } else if (strcmp(type_name, "float") == 0) {
        float *buf = (float *)h_buf;
        for (int i = 0; i < n; i++) {
            if (!enif_get_list_cell(env, cur, &head, &tail)) break;
            if (enif_get_list_cell(env, lookahead, &la_head, &la_tail)) {
                __builtin_prefetch((void *)((uintptr_t)la_tail & ~0x3UL), 0, 1);
                lookahead = la_tail;
            }
            double val = 0.0;
            enif_get_double(env, head, &val);
            buf[i] = (float)val;
            cur = tail;
        }
    } else { /* double */
        double *buf = (double *)h_buf;
        for (int i = 0; i < n; i++) {
            if (!enif_get_list_cell(env, cur, &head, &tail)) break;
            if (enif_get_list_cell(env, lookahead, &la_head, &la_tail)) {
                __builtin_prefetch((void *)((uintptr_t)la_tail & ~0x3UL), 0, 1);
                lookahead = la_tail;
            }
            double val = 0.0;
            enif_get_double(env, head, &val);
            buf[i] = val;
            cur = tail;
        }
    }

    init_cuda(env);
    CUdeviceptr dev_array = device_pool_acquire(&g_flawd_dev, data_size);
    if (dev_array == 0) {
        pinned_pool_release(&g_flawd_h2d, h_buf);   /* H2D pool: libera manualmente aqui */
        return enif_raise_exception(env,
            enif_make_string(env,
                "new_flawd_list_nif: device_pool_acquire falhou (cuMemAlloc, pool cheio ou sem memoria)",
                ERL_NIF_LATIN1));
    }

   cudaStream_t fstream = flawd_get_stream();

    cudaError_t cerr = cudaMemcpyAsync((void *)dev_array, h_buf, data_size,
                                        cudaMemcpyHostToDevice, fstream);
    if (cerr != cudaSuccess) {
        device_pool_release(&g_flawd_dev, dev_array);
        pinned_pool_release(&g_flawd_h2d, h_buf);
        char msg[256];
        snprintf(msg, sizeof(msg),
                 "new_flawd_list_nif: cudaMemcpyAsync H2D: %s",
                 cudaGetErrorString(cerr));
        return enif_raise_exception(env,
                    enif_make_string(env, msg, ERL_NIF_LATIN1));
    }

    /* userData precisa sobreviver até o callback rodar (assíncrono, após
     * esta função já ter retornado) — alocado no heap, liberado pelo
     * próprio callback (ver release_h2d_pool_callback). */
    ReleaseH2DCallbackData *cb_data =
        (ReleaseH2DCallbackData *)malloc(sizeof(ReleaseH2DCallbackData));
    cb_data->pool = &g_flawd_h2d;
    cb_data->ptr  = h_buf;
    cudaStreamAddCallback(fstream, release_h2d_pool_callback, cb_data, 0);

    cudaEvent_t h2d_done;
    cudaEventCreateWithFlags(&h2d_done, cudaEventDisableTiming);
    cudaEventRecord(h2d_done, fstream);

    cudaStreamWaitEvent(0 /*default stream*/, h2d_done, 0);
    cudaEventDestroy(h2d_done);   /* seguro destruir após gravar */

    CUdeviceptr *gpu_res = (CUdeviceptr *)enif_alloc_resource(
                                FLAWD_ARRAY_TYPE, sizeof(CUdeviceptr));
    *gpu_res = dev_array;
    ERL_NIF_TERM term = enif_make_resource(env, gpu_res);
    enif_release_resource(gpu_res);
    return term;
}

/* ---------------------------------------------------------------
 * release_flawd_list_nif: libera o slot do DevicePool de um array
 * FLAWD_ARRAY_TYPE sem descer os dados para a lista Elixir.
 *
 * Necessária porque Ske.map/map2/map3/map4/reduce (ver ske.ex) sobem o(s)
 * array(s) de ENTRADA via new_flawd_list, mas o resultado do kernel é um
 * array de SAÍDA separado (criado via PolyHok.new_gnx, não vem do device
 * pool). get_flawd_list só é chamado sobre o array de saída — o slot do
 * array de entrada nunca seria liberado sem esta função, vazando um slot
 * do pool a cada chamada até os FLAWD_POOL_SLOTS se esgotarem (erro
 * "device_pool_acquire falhou... pool cheio").
 *
 * Chamar sobre um recurso ARRAY_TYPE (gnx normal, não veio do pool) é um
 * no-op seguro — não há slot para liberar.
 * --------------------------------------------------------------- */
static ERL_NIF_TERM release_flawd_list_nif(ErlNifEnv *env, int argc,
                                            const ERL_NIF_TERM argv[]) {
    CUdeviceptr *array_res;
    if (enif_get_resource(env, argv[0], FLAWD_ARRAY_TYPE, (void **)&array_res)) {
        device_pool_release(&g_flawd_dev, *array_res);
    }
    /* ARRAY_TYPE ou tipo desconhecido: nada a fazer, não é erro */
    return enif_make_atom(env, "ok");
}

/* ---------------------------------------------------------------
 * get_flawd_list_nif(ref, n, type_charlist)
 *
 * D2H via pinned memory assíncrona.
 * Reconstrói a lista (de trás pra frente)
 * --------------------------------------------------------------- */
static ERL_NIF_TERM get_flawd_list_nif(ErlNifEnv *env, int argc,
                                        const ERL_NIF_TERM argv[]) {
    /* Aceita tanto ARRAY_TYPE quanto FLAWD_ARRAY_TYPE. Só arrays
     * FLAWD_ARRAY_TYPE vieram do device pool (g_flawd_dev) — só esses
     * devem ter seu slot liberado ao final. */
    CUdeviceptr *array_res;
    int from_flawd_pool = 1;
    if (!enif_get_resource(env, argv[0], FLAWD_ARRAY_TYPE, (void **)&array_res)) {
        from_flawd_pool = 0;
        if (!enif_get_resource(env, argv[0], ARRAY_TYPE, (void **)&array_res))
            return enif_make_badarg(env);
    }

    int n;
    if (!enif_get_int(env, argv[1], &n)) return enif_make_badarg(env);
    if (n <= 0) return enif_make_badarg(env);

    unsigned int type_slen;
    if (!enif_get_list_length(env, argv[2], &type_slen)) return enif_make_badarg(env);
    char type_name[64];
    enif_get_string(env, argv[2], type_name, sizeof(type_name), ERL_NIF_LATIN1);

    size_t elem_size;
    if      (strcmp(type_name, "int")    == 0) elem_size = sizeof(int32_t);
    else if (strcmp(type_name, "float")  == 0) elem_size = sizeof(float);
    else if (strcmp(type_name, "double") == 0) elem_size = sizeof(double);
    else return enif_make_badarg(env);

    size_t data_size = (size_t)n * elem_size;

    /* ---- Adquire buffer pinned D2H do pool ---- */
    void *h_buf = pinned_pool_acquire(&g_flawd_d2h, data_size);
    if (h_buf == NULL) {
        if (from_flawd_pool) device_pool_release(&g_flawd_dev, *array_res);
        return enif_raise_exception(env,
            enif_make_string(env,
                "get_flawd_list_nif: pinned_pool_acquire falhou (cudaMallocHost, pool cheio ou sem memoria)",
                ERL_NIF_LATIN1));
    }

    cudaStream_t fstream = flawd_get_stream();
    cudaError_t cerr = cudaMemcpyAsync(h_buf, (const void *)*array_res, data_size,
                                        cudaMemcpyDeviceToHost, fstream);
    if (cerr != cudaSuccess) {
        pinned_pool_release(&g_flawd_d2h, h_buf);
        if (from_flawd_pool) device_pool_release(&g_flawd_dev, *array_res);
        char msg[256];
        snprintf(msg, sizeof(msg),
                 "get_flawd_list_nif: cudaMemcpyAsync D2H: %s",
                 cudaGetErrorString(cerr));
        return enif_raise_exception(env,
                    enif_make_string(env, msg, ERL_NIF_LATIN1));
    }
    /* Sync necessário: precisamos dos dados antes de construir a lista */
    cudaStreamSynchronize(fstream);

    if (from_flawd_pool) device_pool_release(&g_flawd_dev, *array_res);

    /* Threshold para stack vs heap: 4096 termos * 8 bytes = 32KB — seguro */
#define FLAWD_STACK_TERMS 4096
    ERL_NIF_TERM  stack_terms[FLAWD_STACK_TERMS];
    ERL_NIF_TERM *terms;
    int           terms_heap = 0;

    if (n <= FLAWD_STACK_TERMS) {
        terms = stack_terms;
    } else {
        terms = (ERL_NIF_TERM *)enif_alloc((size_t)n * sizeof(ERL_NIF_TERM));
        if (terms == NULL) {
            /* device pool já foi liberado logo acima — só falta o pinned D2H */
            pinned_pool_release(&g_flawd_d2h, h_buf);
            return enif_raise_exception(env,
                enif_make_string(env,
                    "get_flawd_list_nif: enif_alloc falhou para array de termos",
                    ERL_NIF_LATIN1));
        }
        terms_heap = 1;
    }

    /* Constrói o array de termos em ordem direta (0..n-1) */
    if (strcmp(type_name, "int") == 0) {
        int32_t *buf = (int32_t *)h_buf;
        for (int i = 0; i < n; i++)
            terms[i] = enif_make_int(env, (int)buf[i]);
    } else if (strcmp(type_name, "float") == 0) {
        float *buf = (float *)h_buf;
        for (int i = 0; i < n; i++)
            terms[i] = enif_make_double(env, (double)buf[i]);
    } else { /* double */
        double *buf = (double *)h_buf;
        for (int i = 0; i < n; i++)
            terms[i] = enif_make_double(env, buf[i]);
    }

    /* Uma única alocação bulk na heap do BEAM — constrói a lista completa */
    ERL_NIF_TERM result = enif_make_list_from_array(env, terms, (unsigned)n);

    if (terms_heap) enif_free(terms);

    pinned_pool_release(&g_flawd_d2h, h_buf);
    return result;
}

static ErlNifFunc nif_funcs[] = {
    {"jit_compile_and_launch_nif", 7, jit_compile_and_launch_nif},
    {"new_gpu_array_nif",          3, new_gpu_array_nif},
    {"get_gpu_array_nif",          4, get_gpu_array_nif},
    {"create_gpu_array_nx_nif",    4, create_gpu_array_nx_nif},
    {"load_kernel_nif",            2, load_kernel_nif},
    {"load_fun_nif",               2, load_fun_nif},
    {"new_pinned_nif",             2, new_pinned_nif},
    {"new_gmatrex_pinned_nif",     1, new_gmatrex_pinned_nif},
    {"spawn_nif",                  4, spawn_nif},
    {"create_nx_ref_nif",          4, create_nx_ref_nif},
    {"get_nx_nif",                 4, get_nx_nif},
    {"new_gpu_nx_nif",             3, new_gpu_nx_nif},
    {"create_ref_nif",             1, create_ref_nif},
    {"new_ref_nif",                1, new_ref_nif},
    {"get_matrex_nif",             3, get_matrex_nif},
    {"synchronize_nif",            0, synchronize_nif},
    /* dirty NIFs não bloqueiam schedulers limpos */
    {"new_flawd_list_nif",         4, new_flawd_list_nif, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"get_flawd_list_nif",         3, get_flawd_list_nif, ERL_NIF_DIRTY_JOB_CPU_BOUND},
    {"release_flawd_list_nif",     1, release_flawd_list_nif, 0},
};

ERL_NIF_INIT(Elixir.PolyHok, nif_funcs, &load, NULL, NULL, NULL)

