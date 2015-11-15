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
#include <boost/serialization/queue.hpp>
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
#include "../MLalgorithm/MLalgorithm.h"

namespace core{

    inline int Random(int mod){
        return static_cast<int> (static_cast<double>(rand())/ RAND_MAX * mod);
    }

    inline int Round(double r){  
        return static_cast<int> ((r > 0.0) ? floor(r + 0.5) : ceil(r - 0.5));
    } 

    template<class T>
        class Core{
            public:
                std::string Trim (std::string &str);
                std::vector<std::string> ReadCSV(std::string pwd);
                double praser(std::vector<std::string> data, int feat_dim, size_t i, size_t j);
                std::vector<double> partition(std::vector<std::string> origin,
                        size_t number, size_t loc, size_t feat_dim);
                T* merge(int rank, boost::mpi::communicator world, T* local_update);
                T* mainLoop(int argc, char* argv[], 
                        T* ptr,
                        double eps,
                        size_t feat_dim);
        };

}
#include "Core_impl.h"
#endif
