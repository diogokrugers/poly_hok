all: priv/gpu_nifs.so 

priv/gpu_nifs.so: c_src/gpu_nifs.cu
	nvcc --shared -g -lcuda -lnvrtc -lpthread --compiler-options '-fPIC' -o priv/gpu_nifs.so c_src/gpu_nifs.cu -ccbin gcc-10


bmp: c_src/bmp_nifs.cu 
	nvcc --shared -g --compiler-options '-fPIC' -o priv/bmp_nifs.so c_src/bmp_nifs.cu -ccbin gcc-10

clean:
	rm priv/gpu_nifs.so