# This is a sample makefile
# By changing the directories variables to reflect the local filesystem
#  it should successfully compile on most systems


# set project root directory
PROJECT_ROOT=/Users/marschat/cuda/workspace/granular2D

# define compilers
NVCC=nvcc
CXX=g++

# set where the cuda toolkit is installed on the system
CUDAPATH=/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/NVIDIA/CUDA-6.0/

# set additional directories to search for header files
CXXINCLUDES= -I$(CUDAPATH)/include -I$(PROJECT_ROOT)/include
NVCCINCLUDES= -I$(CUDAPATH)/include -I$(PROJECT_ROOT)/include

# set compiler flags for object compilation
CXXFLAGS= -Wall -O3
NVCCFLAGS= -O3 --ptxas-options=-v -arch=sm_30 

# set additional directories to search for libraries
LIBFLAGS= -L$(CUDAPATH)/lib
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

# add object directories as order-only prerequisites (objects do not get remade because timestamp changes) 
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


depend:
	makedepend $(CXXSRCS) $(TESTSRCS) $(NVCCSRCS) $(CXXINCLUDES)

clean:
	rm -f *.o *.ptx $(OBJDIR)/*.o $(OBJDIR)/*.ptx $(OBJDIR)/cpp/*.o $(OBJDIR)/cu/*.o $(OBJDIR)/cu/*.ptx $(OBJDIR)/tests/*.o

		