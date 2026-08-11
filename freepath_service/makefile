FC 	?= gfortran
CUDA_HOME ?= /usr/local/cuda-13.0
NVCC 	?= $(CUDA_HOME)/bin/nvcc
FCFLAGS	= -O3 -DREALSIZE=8
NVFLAGS	= -O3 -std=c++17
HERE_c	= ../Cuba-4.2.2
CUDA_LIB ?= $(CUDA_HOME)/lib64
LIBS_c 	= -lm
LIB_c 	= $(HERE_c)/libcuba.a
HEADERS = $(HERE_c)/cuba.h
LDFLAGS	= -llapack -lopenblas
#---------------------------------------------------

test=SNOP_op_Ross3

test:$(test).F lib functions.mod par_register.mod cubapar.mod gpu_bridge.mod gpu_integrand.o
	$(FC) $(FCFLAGS) -o SNOP_op_Ross3 $(test).F functions.F par_register.F cubapar.f90 gpu_bridge.f90 gpu_integrand.o $(LIB_c) $(LIBS_c) -lstdc++ $(LDFLAGS) -L$(CUDA_LIB) -lcudart
	
lib: $(LIB_c) $(HEADERS)

functions.mod:functions.F
	$(FC) $(FCFLAGS) -c functions.F 
	
par_register.mod:par_register.F
	$(FC) $(FCFLAGS) -c par_register.F
	
cubapar.mod: cubapar.f90
	$(FC) $(FCFLAGS) -c cubapar.f90

gpu_bridge.mod: gpu_bridge.f90
	$(FC) $(FCFLAGS) -c gpu_bridge.f90

gpu_integrand.o: gpu_integrand.cu
	$(NVCC) $(NVFLAGS) -c gpu_integrand.cu -o gpu_integrand.o
	
#---------------------------------------------------

clean:
	-rm -f *.o *.mod SNOP_op_Ross3 T_* res* nsp_out*.txt Enout*.txt error_info* abs* emi* data.txt
	
# DO NOT DELETE
