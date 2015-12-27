/*
ID: septicmk
LANG: C++
TASK: 
*/

#include "PCA.h"
#include "../Core/Core.h"
#include <boost/serialization/vector.hpp>
#include <vector>
BOOST_CLASS_EXPORT(pca::PCA)
namespace  pca{
    void Matrix::init (size_t nrow, size_t ncol, std::vector<double> data){
        this->nrow = nrow;
        this->ncol = ncol;
        this-data = data;
    }

    PCA* PCA::operator +(PCA &rhs){
        PCA *pret = new PCA();
        return pret;
    }

    void PCA::beginDataScan(std::vector<double> records, size_t feat_dim){
        return; 
    }

    void PCA::endDataScan(){
        return;
    }

    PCA* PCA::processRecord(std::vector<double> records, size_t feat_dim){
        PCA* pret = new PCA();
        return pret;
    }

    bool PCA::isConverged(PCA* rhs, size_t feat_dim, double eps){
        return true;
    }

    void PCA::finish(std::vector<double> records, size_t feat_dim){
        size_t n = feat_dim;
        size_t m = records.size()/feat_dim;
        size_t LDA = m;
        size_t LDU = m;
        size_t LDVT = n;
        int info, lwork;
        double wkopt;
        double* work;
        int *iwork = (int *)malloc(8*n*sizeof(int));
        double *s = (double *)malloc(n*sizeof(double));
        double *u = (double *)malloc(LDU*n*sizeof(double));
        double *vt = (double *)malloc(LDVT*n*sizeof(double));
        std::vector<double> A;
        A.clear();
        for(size_t i = 0 ; i < n ; i++){
            for(size_t j = 0; j < m ; j ++){
                A.push_back(records[j*feat_dim+i]);
            }
        }
        printf("PCA Results\n");
        lwork = -1;
        dgesdd_("Singular vectors", &m, &n, &*A.begin(), &LDA, s, u, &LDU, vt, &LDVT, &iwork,
                &lwork, iwork, &info);
        lwork = (int)wkopt;
        work = (double *)malloc( lwork*sizeof(double) );
        dgesdd_("Singular vectors", &m, &n, &*A.begin(), &LDA, s, u, &LDU, vt, &LDVT, work,
                &lwork, iwork, &info);
        if( info > 0){
            printf("The algorithm computing SVD failed to cinverge.\n");
            return;
        }

        print_matrix( "Right sigual vectors(rowwise)", n, n, vt, ldvt);
        free( (void*)work);
    }
}

