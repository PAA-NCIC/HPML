/*
ID: septicmk
LANG: C++
TASK: Core.h
*/
#ifndef __CORE
#define __CORE

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/vector.hpp>
//#include <boost/serialization/queue.hpp>
#include <boost/mpi.hpp>
#include <algorithm>
#include <vector>
#include <queue>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <fstream>  
#include <string>
#include "../Collection/Flexible_vector.h"
inline int Random(int mod){
    return static_cast<int> (static_cast<double>(rand())/ RAND_MAX * mod);
}

inline int Round(double r){  
    return static_cast<int> ((r > 0.0) ? floor(r + 0.5) : ceil(r - 0.5));
} 

class Core{
    public:
        //Flexible_vector f_vector;
        Collection *f_vector;
    

        std::string Trim (std::string &str);
        std::vector<std::string> ReadCSV(std::string pwd);
        double praser(std::vector<std::string> data, int feat_dim, size_t i, size_t j);
        std::vector<double> partition(std::vector<std::string> origin,
                size_t number, size_t loc, size_t feat_dim);
        //inline MLalgorithm* merge(int rank, boost::mpi::communicator world, MLalgorithm* local_update);
        //MLalgorithm* mainLoop(int argc, char* argv[], 
        //        MLalgorithm *ptr,
        //        double eps,
        //        size_t feat_dim);
};
#endif
