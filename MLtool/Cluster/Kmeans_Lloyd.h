/*
ID:neo-white 
LANG: C++
TASK: Kmeans_lloyd.h
*/

#ifndef __KMEANS_LLOYD
#define __KMEANS_LLOYD

#include "../MLalgorithm/MLalgorithm.h"
#include "../Collection/Flexible_vector.h"
#include <boost/serialization/vector.hpp>
#include <vector>

namespace kmeans_lloyd{

class Kmeans_Lloyd:public MLalgorithm<Kmeans_Lloyd>{
    private:
        friend class boost::serialization::access;
        template <typename Archive>
            void serialize (Archive &ar, const unsigned int version){
                ar & boost::serialization::base_object<MLalgorithm<Kmeans_Lloyd> >(*this);
                ar & fvkl;
            }
    public:
        Flexible_vector fvkl;
        //size_t n_cluster;
        Kmeans_Lloyd* operator +(Kmeans_Lloyd &rhs);
        void beginDataScan(Flexible_vector *records, size_t n_cluster);
        void endDataScan();
        Kmeans_Lloyd* processRecord(Flexible_vector *records, size_t feat_dim);
        bool isConverged(Kmeans_Lloyd* rhs, size_t feat_dim, double eps);
        void finish(Flexible_vector *records, size_t feat_dim);

        std::vector<int> init_random(size_t length, size_t random_seed);
        double euclidean_distance(std::vector<node> px,std::vector<node> py);
};
}
#endif



