/*
ID:neo-white 
LANG: C++
TASK: Kmeans_Parallel.h
*/

#ifndef __KMEANS_PARALLEL
#define __KMEANS_PARALLEL

#include "../MLalgorithm/MLalgorithm.h"
#include "../Collection/Flexible_vector.h"
#include <boost/serialization/vector.hpp>
#include <vector>

namespace kmeans_parallel{

struct Model{
private:
    friend class boost::serialization::access;
    template <typename Archive>
        void serialize (Archive &ar, const unsigned int version){
            ar &centers;
            ar &dis;
            ar &costX;
            ar &oversample;
            ar &members;
        }
public:
    std::vector<int> centers;
    std::vector<double> dis;  //the distances from xi to centers
    double costX;   //sum of distances
    size_t oversample;   //oversample parameter
    size_t members;  //data partition
};

class Kmeans_Parallel:public MLalgorithm<Kmeans_Parallel>{
    private:
        friend class boost::serialization::access;
        template <typename Archive>
            void serialize (Archive &ar, const unsigned int version){
                ar & boost::serialization::base_object<MLalgorithm<Kmeans_Parallel> >(*this);
                ar & mbasic_;
            }
    public:
        Flexible_vector fvkl_;
        Model mbasic_;
        std::vector<int> global_weights;
        std::vector<int> final_centers;
        //size_t n_cluster;
        Kmeans_Parallel* operator +(Kmeans_Parallel &rhs);
        void beginDataScan(Flexible_vector *records, size_t n_cluster);
        void endDataScan();
        Kmeans_Parallel* processRecord(Model *prior, size_t rank);
        bool isConverged(Kmeans_Parallel* rhs, size_t feat_dim, double eps);
        void finish(Flexible_vector *records, size_t feat_dim);

        //std::vector<int> init_random(size_t length, size_t random_seed);
        Kmeans_Parallel *ProcessDistanceSum(Flexible_vector *records, Model *prior, size_t loc);
        double CountEuclideanDistance(std::vector<node> px,std::vector<node> py);
        void SampleC(Model *model, std::vector<int> centers, size_t loc);
        std::vector<int> SetWeight(Flexible_vector *records, Model *prior, size_t loc);
        //test
        void OutputModel();
};
} //namespace kmeans_parallel
#endif



