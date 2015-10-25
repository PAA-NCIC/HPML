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

BOOST_CLASS_EXPORT(Kmeans)
MLalgorithm* Kmeans::operator +(MLalgorithm &rhs){
    MLalgorithm *pret = new Kmeans();
    pret->Init(this->nrow, this->ncol);
    for(size_t i = 0; i< data.size() ; ++i)
        pret->data[i] = data[i] + rhs.data[i];
    return pret;
}

void Kmeans::beginDataScan(std::vector<double> records, size_t feat_dim){
    std::priority_queue<int> q;
    std::vector<int> weight;
    size_t n_records = records.size();
    size_t n_cluster = this->nrow;
    for (size_t i = 0; i < n_records; ++ i){
        int x = Random(10007);
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
        add(record, n_cluster - q.size());
        q.pop();
    }
}


void Kmeans::endDataScan(){
    for(size_t i = 0; i < nrow; ++i){
        for(size_t j = 0; j < ncol; ++j){
            data[i*ncol+j] /= data[i*ncol + ncol-1];
        }
        data[i*ncol + ncol-1] = 1;
    } 

    //for(size_t i = 0; i < nrow; ++i){
    //for(size_t j = 0; j < ncol; ++j){
    //std::cout << data[i*ncol+j] << " ";
    //}
    //std::puts("");
    //}
}

MLalgorithm* Kmeans::processRecord(std::vector<double> records, size_t feat_dim){
    MLalgorithm* pret = new Kmeans();
    pret->Init(this->nrow, this->ncol);
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
        for(size_t k = 0; k < this->nrow; k++){
            std::vector<double> centroid = this->get_value(k);
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
        pret->add(record, who);
    }
    return pret;
}

bool Kmeans::isConverged(MLalgorithm* rhs, size_t feat_dim, double eps){
    double diff = 0;
    for(size_t i = 0; i < nrow; i ++){
        double tmp = 0;
        for(size_t j = 0;  j < feat_dim; j++){
            tmp += (data[i*ncol+j] - rhs->data[i*ncol+j]) * (data[i*ncol+j] - rhs->data[i*ncol+j]);
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
        for(size_t k = 0; k < this->nrow; k++){
            std::vector<double> centroid = this->get_value(k);
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

