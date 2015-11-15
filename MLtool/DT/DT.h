/*
ID: septicmk
LANG: C++
TASK: DT.h
*/

#ifndef __DT
#define __DT

#include "../MLalgorithm/MLalgorithm.h"
#include <boost/serialization/vector.hpp>
#include "../Core/Core.h"
#include <vector>
#include <cmath>
#include <iostream>

namespace dt{

    class Matrix{
        private:
            friend class boost::serialization::access;
            template <typename Archive>
                void serialize (Archive &ar, const unsigned int version){
                    ar& mat;
                    ar& nrow;
                    ar& ncol;
                }
        public:
            std::vector<int> mat;
            size_t nrow, ncol;
            int get(size_t i, size_t j){
                return mat[i*ncol+j];
            }

            void print(){
                for(size_t i = 0 ; i < nrow; i ++){
                    for(size_t j = 0 ; j < ncol ; j++){
                        std::cout << get(i,j) << " ";
                    }
                    std::cout << std::endl;
                }
            }

            void init(size_t nrow, size_t ncol){
                this->nrow = nrow;
                this->ncol = ncol;
                mat.resize(nrow*ncol);
                std::fill(mat.begin(), mat.end(), 0);
            }

            void add(size_t i, size_t j, int addon){
                mat[i*ncol+j] += addon;
            }

            Matrix merge(Matrix rhs){
                Matrix ret;
                ret.init(this->nrow, this->ncol);
                for(size_t i = 0; i < nrow; i++){
                    for(size_t j = 0; j < ncol; j++){
                        ret.mat[i*ncol+j] = mat[i*ncol + j] + rhs.mat[i*ncol+j];
                    }
                }
                return ret;
            }

            double infoGainRate(){
                double before = 0;
                double after = 0;
                double sum = 0;
                //std::cout << "=====================" << std::endl;
                //print();
                //std::cout << "=====================" << std::endl;
                std::vector<double> attr;
                attr.resize(nrow);
                std::fill(attr.begin(), attr.end(), 0.0f);
                std::vector<double> cls;
                cls.resize(ncol);
                std::fill(cls.begin(), cls.end(), 0.0f);

                for(size_t i = 0; i < nrow; i++)
                    for(size_t j = 0; j < ncol; j++){
                        attr[i] += get(i,j);
                        cls[j] += get(i,j);
                        sum += get(i,j);
                    }

                double spaninfo = 0;
                double check = 0;
                for(size_t j = 0; j < ncol; j++){
                    check += cls[j];
                    if(abs(cls[j]) > 1e-5) before += -(cls[j]/sum)*log(cls[j]/sum)/log(2);
                }
                //std::cout << check << " " <<  sum << std::endl;

                for(size_t i = 0; i < nrow; i++){
                    double tmp = 0;
                    for(size_t j = 0; j < ncol; j++){
                        if(abs(get(i,j)) > 1e-5) tmp += -(get(i,j)/cls[j])*log(get(i,j)/cls[j])/log(2);
                    }
                    if(abs(attr[i]) > 1e-5) {
                        spaninfo += -((attr[i])/sum) * log((attr[i])/sum) / log(2);
                        after += (attr[i])*tmp/sum;
                    }
                }
                //std::cout << "@@" << before << " " << after << " " << spaninfo << std::endl;
                //std::cout << "$$"<< (before-after)/spaninfo<< std::endl;
                return (before - after)/spaninfo;
            }
    };

    class Node{
        private:
            friend class boost::serialization::access;
            template <typename Archive>
                void serialize (Archive &ar, const unsigned int version){
                    ar & vec;
                    ar & index;
                    ar & hashcode;
                    ar & D;
                    ar & who;
                    ar & tag;
                }
        public:
            long long hashcode;
            bool tag;
            std::vector<int> index;
            std::vector<Matrix> vec;
            std::vector<int> D;
            int who;

            void print(){
                std::cout<< "##" << hashcode << " " << who << " " << tag << std::endl;
                for(size_t i = 0 ; i < index.size() ; i ++){
                    std::cout << index[i] << " ";
                }
                std::cout << std::endl;

            }

            void check(){
                for(size_t i = 0 ; i < vec.size() ; i ++)
                    vec[i].print();
            }

            void init(std::vector<size_t>feat_depth, size_t cls_number){
                tag = false;
                who = -1;
                vec.clear();
                for(size_t i = 0; i < index.size(); i++){
                    Matrix hist;
                    hist.init(feat_depth[i], cls_number);
                    vec.push_back(hist);
                }
                D.clear();
                D.resize(cls_number);
                std::fill(D.begin(), D.end(),0);
                hashcode = get_hash();
            }

            void update(std::vector<int> record){
                int cls = record[record.size()-1];
                D[cls] ++;
                for(size_t j = 0; j < index.size(); j++){
                    if(index[j] == -1){
                        vec[j].add(record[j],cls,1);
                    }
                }
            }

            long long get_hash(){
                long long ret = 0;
                for(size_t i = 0; i < index.size(); i++){
                    ret += (1LL<<i)*index[i];
                    ret %= 1000000007LL;
                }
                return ret;
            }

            bool operator == (Node &rhs) const{
                if(hashcode != rhs.hashcode)
                    return false;
                bool flag = true;
                for(size_t i = 0; i < index.size(); i++)
                    if(index[i] != rhs.index[i])
                        flag = false;
                return flag; 
            }

            Node merge(Node &rhs){
                Node ret = (*this);
                if(ret.tag) return ret;
                for(size_t i = 0; i < vec.size(); i++){
                    ret.vec[i] = vec[i].merge(rhs.vec[i]);
                }
                for(size_t i = 0; i < D.size(); i++){
                    ret.D[i] = D[i] + rhs.D[i];
                }
                return ret;
            }

            int split(){
                
                //std::cout << "wwwwwwwwwwwwwwwwwww" << std::endl;
                //print();
                //check();
                //std::cout << "wwwwwwwwwwwwwwwwwwww" << std::endl;

                double mmax = -999999999;
                int win = -1;
                //if(vec.size()!=0) vec.clear();
                for(size_t i = 0 ; i < index.size() ; i++){
                    if(index[i] == -1){
                        double tmp = vec[i].infoGainRate();
                        //std::cout << tmp << std::endl;
                        if(tmp > mmax){
                            mmax = tmp;
                            win = i;
                        }
                    }
                }
                return win;
            }
    };

    class DT:public MLalgorithm<DT>{
        private:
            friend class boost::serialization::access;
            template <typename Archive>
                void serialize (Archive &ar, const unsigned int version){
                    ar & boost::serialization::base_object<MLalgorithm<DT> >(*this);
                    ar & nodes;
                    ar & feat_depth;
                    ar & feat_dim;
                    ar & cls_number;
                }
        public:
            std::vector<Node> nodes;
            std::vector<size_t> feat_depth;
            size_t feat_dim;
            size_t cls_number;

            void check(){
                std::cout << "################" << std::endl;
                for (size_t i = 0 ; i < nodes.size();i++) nodes[i].check();
                std::cout << "################" << std::endl;
            }

            void init(std::vector<size_t>feat_depth, size_t cls_number);
            DT* operator +(DT &rhs);
            void beginDataScan(std::vector<double> records, size_t feat_dim);
            void endDataScan();
            DT* processRecord(std::vector<double> records, size_t feat_dim);
            bool isConverged(DT* rhs, size_t feat_dim, double eps);
            void finish(std::vector<double> records, size_t feat_dim);
    };

}
#endif



