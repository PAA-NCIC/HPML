/*
ID: septicmk
LANG: C++
TASK: Kmeans.h
*/

#ifndef __KMEANS
#define __KMEANS

#include "../MLalgorithm/MLalgorithm.h"
#include <boost/serialization/vector.hpp>
#include <vector>

namespace kmeans{

    class Matrix{
        private:
            friend class boost::serialization::access;
            template <typename Archive>
                void serialize (Archive &ar, const unsigned int version){
                    ar &data;
                    ar &ncol;
                    ar &nrow;
                }
        public:
            size_t ncol;
            size_t nrow;
            std::vector<double> data;

            void init(size_t nrow, size_t ncol);
            void add(std::vector<double> record, size_t i);
            std::vector<double> get_value(size_t i);
    };


    class Kmeans:public MLalgorithm<Kmeans>{
        private:
            friend class boost::serialization::access;
            template <typename Archive>
                void serialize (Archive &ar, const unsigned int version){
                    ar & boost::serialization::base_object<MLalgorithm<Kmeans> >(*this);
                    ar & mat;
                }
        public:
            Matrix mat;
            Kmeans* operator +(Kmeans &rhs);
            void beginDataScan(std::vector<double> records, size_t feat_dim);
            void endDataScan();
            Kmeans* processRecord(std::vector<double> records, size_t feat_dim);
            bool isConverged(Kmeans* rhs, size_t feat_dim, double eps);
            void finish(std::vector<double> records, size_t feat_dim);
    };


}
#endif



