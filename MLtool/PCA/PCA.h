/*
ID: septicmk
LANG: C++
TASK: 
*/

#ifndef __PCA
#define __PCA
#include "../MLalgorithm/MLalgorithm.h"
#include <boost/serialization/vector.hpp>
#include <vector>

namespace pca{

    extern "C" void dgesdd_( char* jobz, int* m, int* n, double* a,
                int* lda, double* s, double* u, int* ldu, double* vt, int* ldvt,
                double* work, int* lwork, int* iwork, int* info );
/* Auxiliary routines prototypes */
    extern "C" void print_matrix( char* desc, int m, int n, double* a, int lda );

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
            std::vector<double>data;
            void init(size_t nrow, size_t ncol ,std::vector<double> data);
    };

    class PCA:public MLalgorithm<PCA>{
        private:
            friend class boost::serialization::access;
            template <typename Archive>
                void serialize (Archive &ar, const unsigned int version){
                    ar & boost::serialization::base_object<MLalgorithm<PCA> >(*this);
                    ar & mat;

                }
        public:
            Matrix mat;
            PCA* operator +(PCA &rhs);
            void beginDataScan(std::vector<double> records, size_t feat_dim);
            void endDataScan();
            PCA* processRecord(std::vector<double> records, size_t feat_dim);
            bool isConverged(PCA *rhs, size_t feat_dim, double eps);
            void finish(std::vector<double> records, size_t feat_dim);
    };
}



#endif
