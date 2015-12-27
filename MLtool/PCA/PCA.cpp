/*
ID: septicmk
LANG: C++
TASK: 
*/

#include "PCA.h"
#include "../Core/Core.h"
#include <boost/serialization/vector.hpp>
#include <boost/serialization/export.hpp>
#include <vector>

BOOST_CLASS_EXPORT(pca::PCA)
namespace pca{
    void print_matrix( char* desc, int m, int n, double* a, int lda ) {
        int i, j;
        printf( "\n %s\n", desc );
        for( i = 0; i < m; i++ ) {
                for( j = 0; j < n; j++ ) printf( " %6.2f", a[i+j*lda] );
                printf( "\n" );
        }
    }
    void Matrix::init (size_t nrow, size_t ncol, std::vector<double> data){
        this->nrow = nrow;
        this->ncol = ncol;
        this->data = data;
    }

    PCA* PCA::operator +(PCA &rhs){
        PCA *pret = new PCA();
        pret->mat.init(mat.nrow,mat.ncol,mat.data);
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
        pret->mat.init(mat.nrow,mat.ncol,mat.data);
        return pret;
    }

    bool PCA::isConverged(PCA* rhs, size_t feat_dim, double eps){
        return true;
    }

    void PCA::finish(std::vector<double> records, size_t feat_dim){
        printf("PCA finish\n");
        int n = feat_dim;
        int m = records.size()/feat_dim;
        int LDA = m;
        int LDU = m;
        int LDVT = n;
        int info, lwork;
        double wkopt;
        double* work;
        int *iwork = (int *)malloc(8*n*sizeof(int));
        double *s = (double *)malloc(n*sizeof(double));
        double *u = (double *)malloc(LDU*n*sizeof(double));
        double *vt = (double *)malloc(LDVT*n*sizeof(double));
        std::vector<double> A;
        A.clear();
        for(int i = 0 ; i < n ; i++){
            for(int j = 0; j < m ; j ++){
                A.push_back(records[j*feat_dim+i]);
            }
        }
        printf("PCA Results\n");
        lwork = -1;
        dgesdd_("Singular vectors", &m, &n, &*A.begin(), &LDA, s, u, &LDU, vt, &LDVT, &wkopt,
                &lwork, iwork, &info);
        lwork = (int)wkopt;
        work = (double *)malloc( lwork*sizeof(double) );
        dgesdd_("Singular vectors", &m, &n, &*A.begin(), &LDA, s, u, &LDU, vt, &LDVT, work,
                &lwork, iwork, &info);
        if( info > 0){
            printf("The algorithm computing SVD failed to cinverge.\n");
            return;
        }

        print_matrix ( "Right sigual vectors(rowwise)", n, n, vt, LDVT);
        free( (void*)work);
    }
}

