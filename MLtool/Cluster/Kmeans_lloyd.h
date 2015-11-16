/*
ID:neo-white 
LANG: C++
TASK: Kmeans_lloyd.h
*/

#ifndef __KMEANS_LLOYD
#define __KMEANS_LLOYD

#include "../MLalgorithm/MLalgorithm.h"
#include <boost/serialization/vector.hpp>
#include <vector>

class Kmeans_lloyd:public MLalgorithm{
    private:
        friend class boost::serialization::access;
        template <typename Archive>
            void serialize (Archive &ar, const unsigned int version){
                ar & boost::serialization::base_object<MLalgorithm>(*this);
            }
    public:
        size_t n_cluster;
        MLalgorithm* operator +(MLalgorithm &rhs);
        void beginDataScan(Flexible_vector *records, size_t feat_dim);
        void endDataScan();
        MLalgorithm* processRecord(Flexible_vector *records, size_t feat_dim);
        bool isConverged(MLalgorithm* rhs, size_t feat_dim, double eps);
        void finish(Flexible_vector *records, size_t feat_dim);

        std::vector<int> init_random(size_t length, size_t random_seed);
};

#endif



