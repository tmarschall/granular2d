# This is a sample makefile
# By changing the compiler flags and directory variables
#   it should successfully compile on most systems
# Run `make depend` to automatically generate dependencies
#

# set project root directory
PROJECT_ROOT=/home/tmarscha/granular2d
MAKE_ROOT=$(PROJECT_ROOT)/Custom

# define compilers
NVCC=nvcc
CXX=g++

# set where the cuda toolkit is installed on the system
CUDAPATH=/usr/local/cuda/5.0/

# set additional directories to search for header files
CXXINCLUDES= -I$(CUDAPATH)/include -I$(PROJECT_ROOT)/include
NVCCINCLUDES= -I$(CUDAPATH)/include -I$(PROJECT_ROOT)/include

# set compiler flags for object compilation
CXXFLAGS= -Wall -O3
NVCCFLAGS= -O3 --ptxas-options=-v -gencode arch=compute_20,code=sm_20 

# set additional directories to search for libraries
LIBFLAGS= -L$(CUDAPATH)/lib64
# set linker flags
LDFLAGS= -Wall -lcudart

# list of c++ and cuda source files
CXXSRCS= box.cpp cell_list.cpp device_config.cpp disk.cpp error_check.cpp neighbor_list.cpp spherocyl2d.cpp
TESTSRCS= main.cpp
NVCCSRCS= find_neighbors.cu

# set paths to search for prerequisites
VPATH= ../src/cpp:../src/cu:../tests:../include

# list of objects
OBJDIR=obj
CXXOBJS= $(patsubst %.cpp,$(OBJDIR)/cpp/%.o, $(CXXSRCS))
TESTOBJS= $(patsubst %.cpp,$(OBJDIR)/tests/%.o,$(TESTSRCS))
NVCCOBJS= $(patsubst %.cu,$(OBJDIR)/cu/%.o, $(NVCCSRCS))


# executables
MAINEXE= granular2d_test

all: $(MAINEXE)
main: $(MAINEXE)

# add object directory as order-only prerequisites (objects will not get remade because timestamp changes) 
$(CXXOBJS): | $(OBJDIR)
$(TESTOBJS): | $(OBJDIR)
$(NVCCOBJS): | $(OBJDIR)

$(MAINEXE): $(CXXOBJS) $(NVCCOBJS) $(OBJDIR)/tests/main.o
	$(CXX) -o $(MAINEXE) $(OBJDIR)/tests/main.o $(CXXOBJS) $(NVCCOBJS) $(LIBFLAGS) $(LDFLAGS)

# make all object directories if required
$(OBJDIR):
	mkdir $(OBJDIR) && mkdir $(OBJDIR)/cpp $(OBJDIR)/cu $(OBJDIR)/tests


$(CXXOBJS): $(OBJDIR)/cpp/%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) $(CXXINCLUDES) $< -o $@

$(TESTOBJS): $(OBJDIR)/tests/%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) $(CXXINCLUDES) $< -o $@

$(NVCCOBJS): $(OBJDIR)/cu/%.o: %.cu
	$(NVCC) -c $(NVCCFLAGS) $(NVCCINCLUDES) $< -o $@


cpp_depend:
	touch $(OBJDIR)/cpp/cpp.deps; cd ../src/cpp; makedepend $(CXXINCLUDES) -p $(OBJDIR)/cpp/ -f $(MAKE_ROOT)/$(OBJDIR)/cpp/cpp.deps $(CXXSRCS);

cu_depend:
	touch $(OBJDIR)/cu/cu.deps; cd ../src/cu; makedepend $(NVCCINCLUDES) -p $(OBJDIR)/cu/ -f $(MAKE_ROOT)/$(OBJDIR)/cu/cu.deps $(NVCCSRCS);

depend: cpp_depend cu_depend
	touch $(OBJDIR)/tests/tests.deps; cd ../tests; makedepend $(CXXINCLUDES) -p $(OBJDIR)/tests/ -f $(MAKE_ROOT)/$(OBJDIR)/tests/tests.deps $(TESTSRCS)


clean:
	rm -f *.o *.ptx $(OBJDIR)/*.o $(OBJDIR)/*.ptx $(OBJDIR)/cpp/*.o $(OBJDIR)/cu/*.o $(OBJDIR)/cu/*.ptx $(OBJDIR)/tests/*.o

-include $(OBJDIR)/cpp/cpp.deps
-include $(OBJDIR)/cu/cu.deps
-include $(OBJDIR)/tests/tests.deps
