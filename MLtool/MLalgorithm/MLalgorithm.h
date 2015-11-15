/*
ID: septicmk
LANG: C++
TASK: MLalgorithm.h
*/
#ifndef __MLALGORITHM
#define __MLALGORITHM

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>


template<class Derived>
class MLalgorithm{
    private:
        friend class boost::serialization::access;
        template <typename Archive>
            void serialize (Archive &ar, const unsigned int version){
            }
        Derived* cast(){
            return static_cast<Derived*>(this);
        }
    public:

        Derived* operator + (Derived &rhs){
            return *cast() + *(rhs.cast());
        }
        void beginDataScan(std::vector<double> records, size_t feat_dim){
            cast()->beginDataScan(records, feat_dim);
        }
        Derived* processRecord (std::vector<double> records, size_t feat_dim){
            return cast()->processRecord (records, feat_dim);
        }
        void endDataScan(){
            cast()->endDataScan();
        }
        bool isConverged(Derived* rhs, size_t feat_dim, double eps){
            return cast()->isConverged(rhs->cast(), feat_dim, eps);
        }
        void finish(std::vector<double> records, size_t feat_dim){
            cast()->finish(records, feat_dim);
        }
};
#endif
