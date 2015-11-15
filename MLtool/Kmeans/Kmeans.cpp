/*
ID: septicmk
LANG: C++
TASK: Kmeans.cpp
*/

#include "Kmeans.h"
#include "../Core/Core.h"
#include <boost/serialization/vector.hpp>
#include <boost/serialization/queue.hpp>
#include <boost/serialization/export.hpp> 
#include <vector>
#include <queue>
#include <fstream>  


BOOST_CLASS_EXPORT(kmeans::Kmeans)
namespace kmeans{

    void Matrix::init (size_t nrow, size_t ncol){
        this->nrow = nrow;
        this->ncol = ncol;
        data.resize(nrow*ncol);
        std::fill(data.begin(), data.end(), 0.0f);
    }

    void Matrix::add (std::vector<double> record, size_t i){
        for(size_t j = 0; j < record.size(); ++j){
            data[ i*ncol + j ] += record[j];
        }
    }

    std::vector<double> Matrix::get_value(size_t i){
        std::vector<double> ret;
        for (size_t j = 0; j < ncol; ++j){
            ret.push_back(data[ i*ncol + j ]);
        }
        return ret;
    }



    Kmeans* Kmeans::operator +(Kmeans &rhs){
        Kmeans *pret = new Kmeans();
        pret->mat.init(mat.nrow, mat.ncol);
        for(size_t i = 0; i< mat.data.size() ; ++i)
            pret->mat.data[i] = mat.data[i] + rhs.mat.data[i];
        return pret;
    }

    void Kmeans::beginDataScan(std::vector<double> records, size_t feat_dim){
        std::priority_queue<int> q;
        std::vector<int> weight;
        size_t n_records = records.size();
        size_t n_cluster = mat.nrow;
        for (size_t i = 0; i < n_records; ++ i){
            int x = core::Random(10007);
            weight.push_back(x);
            if (q.size() < n_cluster){
                q.push(i);
            }else if (x < weight[q.top()]){
                q.pop();
                q.push(i);
            }
        }
        while(!q.empty()){
            std::vector<double> record;
            record.clear();
            size_t i = q.top();
            for(size_t j = 0; j < feat_dim ; ++j){
                record.push_back(records[i*feat_dim+j]);
            }
            mat.add(record, n_cluster - q.size());
            q.pop();
        }
    }


    void Kmeans::endDataScan(){
        for(size_t i = 0; i < mat.nrow; ++i){
            for(size_t j = 0; j < mat.ncol; ++j){
                mat.data[i*mat.ncol+j] /= mat.data[i*mat.ncol + mat.ncol-1];
            }
            mat.data[i*mat.ncol + mat.ncol-1] = 1;
        } 

        //for(size_t i = 0; i < nrow; ++i){
        //for(size_t j = 0; j < ncol; ++j){
        //std::cout << data[i*ncol+j] << " ";
        //}
        //std::puts("");
        //}
    }

    Kmeans* Kmeans::processRecord(std::vector<double> records, size_t feat_dim){
        Kmeans* pret = new Kmeans();
        //std::cout << this->test << std::endl;
        pret->mat.init(this->mat.nrow, this->mat.ncol);
        size_t n_records = records.size()/feat_dim;
        for (size_t i = 0 ; i < n_records ; i++){
            std::vector<double> record;
            record.clear();
            for(size_t j = 0; j < feat_dim ; j ++){
                record.push_back(records[i*feat_dim+j]);
            }
            double dis = (1L<<16)-1;
            int who = -1;
            //std::cout<<"bp"<<std::endl;
            for(size_t k = 0; k < mat.nrow; k++){
                std::vector<double> centroid = mat.get_value(k);
                double tmp = 0;
                for (size_t u = 0; u < record.size(); ++u){
                    tmp += (centroid[u] - record[u])*(centroid[u] - record[u]);
                }

                if (tmp < dis){
                    who = k;
                    dis = tmp;
                }
            }
            record.push_back(1);
            pret->mat.add(record, who);
        }
        return pret;
    }

    bool Kmeans::isConverged(Kmeans* rhs, size_t feat_dim, double eps){
        double diff = 0;
        for(size_t i = 0; i < mat.nrow; i ++){
            double tmp = 0;
            for(size_t j = 0;  j < feat_dim; j++){
                tmp += (mat.data[i*mat.ncol+j] - rhs->mat.data[i*mat.ncol+j]) * (mat.data[i*mat.ncol+j] - rhs->mat.data[i*mat.ncol+j]);
            }
            tmp = sqrt(tmp);
            diff += tmp;
        }
        if (diff <= eps) return true;
        else return false;

    }

    void Kmeans::finish(std::vector<double> records, size_t feat_dim){
        std::ofstream file;
        file.open("./ans.txt");	
        std::vector<float> record;
        size_t n_records = records.size()/feat_dim;
        for(size_t i = 0; i < n_records; ++i){
            std::vector<double> record;
            record.clear();
            for(size_t j = 0; j < feat_dim; ++ j){
                record.push_back(records[i*feat_dim+j]);
            }
            double dis = (1<<16)-1;
            int who = -1;
            for(size_t k = 0; k < this->mat.nrow; k++){
                std::vector<double> centroid = this->mat.get_value(k);
                double tmp = 0;
                for (size_t u = 0; u < record.size(); ++u){
                    tmp += (centroid[u] - record[u])*(centroid[u] - record[u]);
                }

                if (tmp < dis){
                    who = k;
                    dis = tmp;
                }
            }
            file << who << std::endl;

        }

    }

}
