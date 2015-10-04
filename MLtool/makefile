.PHONY: clean
CXX = mpic++
TARGET = Kmeans
LIB_FLAGS = /home/septicmk/lib/lib/libboost_serialization.so /home/septicmk/lib/lib/libboost_mpi.so
all: 
	$(CXX) Kmeans.cpp $(LIB_FLAGS) -o $(TARGET)
clean:
	-rm -f *.o $(TARGET)

