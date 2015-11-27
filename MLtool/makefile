CC      = mpic++
CFLAGS  = -Wall -O
INCLUDEFLAGS = -I ./Core -I ./Collection -I ./Kmeans -I ./MLalgorithm -I ./DT 
#INCLUDEFLAGS = -I ./Core -I ./Collection -I ./Cluster -I ./MLalgorithm 
#LDFLAGS = -L/home/ylf/system_tools/boost_1_59/lib -lboost_serialization -lboost_mpi
LDFLAGS = -lboost_mpi #-lboost_serialization 
#LDFLAGS = /home/ylf/system_tools/boost_1_59/lib/libboost_serialization.so /home/ylf/system_tools/boost_1_59/lib/libboost_mpi.so
#LDFLAGS = /home/septicmk/lib/lib/libboost_serialization.so /home/septicmk/lib/lib/libboost_mpi.so
#OBJS    = ./Core/Core.o ./Collection/Flexible_vector.o ./Collection/Collection.o ./MLalgorithm/MLalgorithm.o ./Cluster/Kmeans_Lloyd.o 
OBJS    = ./Core/Core.o ./Collection/Flexible_vector.o ./Collection/Collection.o ./MLalgorithm/MLalgorithm.o ./Kmeans/Kmeans.o ./DT/DT.o
TARGETS = main

.PHONY:all 
all:$(TARGETS)
main: main_dt.o $(OBJS)
	$(CC) -o  $@  $^ $(LDFLAGS)
%.o:%.cpp
	$(CC) -o $@ -c $< $(CFLAGS) $(INCLUDEFLAGS)
%.d:%.cpp
	@set -e; rm -f $@; $(CC) -MM $< $(INCLUDEFLAGS) > $@.$$$$; \
		sed 's,\($*\)\.o[ :]*,\1.o $@ : ,g' < $@.$$$$ > $@; \
		rm -f $@.$$$$
-include $(OBJS:.o=.d)
.PHONY:clean 
clean:
	@rm -f $(TARGETS)
	@find ./ -name *.o -o -name *.d -o -name *.d.* | xargs rm -f
	@find ./ -name *.o | xargs rm -f
