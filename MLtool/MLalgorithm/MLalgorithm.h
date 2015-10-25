/*
ID: septicmk
LANG: C++
TASK: MLalgorithm.h
*/
#ifndef __MLALGORITHM
#define __MLALGORITHM

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <vector>

class MLalgorithm{
    private:
        friend class boost::serialization::access;
        template <typename Archive>
            void serialize (Archive &ar, const unsigned int version){
                ar &nrow;
                ar &ncol;
                ar &data;
            }
    public:
        size_t nrow, ncol;
        std::vector<double> data;
        void Init(size_t nrow, size_t ncol);
        void add (std::vector<double> record, size_t i);
        std::vector<double> get_value(size_t i);

        //
        virtual MLalgorithm* operator + (MLalgorithm &rhs)=0;
        virtual void beginDataScan(std::vector<double> records, size_t feat_dim) = 0;
        virtual MLalgorithm* processRecord (std::vector<double> records, size_t feat_dim)=0;
        virtual void endDataScan()=0;
        virtual bool isConverged(MLalgorithm* rhs, size_t feat_dim, double eps)=0;
        virtual void finish(std::vector<double> records, size_t feat_dim)=0;
};
#endif
