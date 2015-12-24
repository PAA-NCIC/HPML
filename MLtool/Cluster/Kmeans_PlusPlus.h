/*
ID:neo-white 
LANG: C++
TASK: Kmeans_PlusPlus.h
*/

#ifndef __KMEANS_PLUSPLUS
#define __KMEANS_PLUSPLUS

#include "../MLalgorithm/MLalgorithm.h"
#include "../Collection/Flexible_vector.h"
#include <boost/serialization/vector.hpp>
#include <vector>

namespace kmeans_plusplus{

class Kmeans_PlusPlus{
    public:
        Flexible_vector fvkl;
        Flexible_vector result_kpp;
        int k;
        //Flexible_vector *kmeans_plusplus(Flexible_vector *problem,int k);
        std::vector<int> kmeans_plusplus(Flexible_vector *problem,int k);
        void build_model(Flexible_vector *model);

        //std::vector<int> init_random(size_t length, size_t random_seed);
        double euclidean_distance(std::vector<node> px,std::vector<node> py);
};
}
#endif



