/*
ID: septicmk
LANG: C++
TASK: DT.cpp
*/

#include "DT.h"
#include "../Core/Core.h"
#include <boost/serialization/vector.hpp>
#include <boost/serialization/queue.hpp>
#include <boost/serialization/export.hpp> 
#include <vector>
#include <queue>
#include <fstream>  
#include <iostream>


BOOST_CLASS_EXPORT(dt::DT)
namespace dt{

    void DT::init(std::vector<size_t>feat_depth, size_t cls_number){
        this->feat_depth = feat_depth;
        this->feat_dim = feat_depth.size();
        this->cls_number = cls_number;
    }

    DT* DT::operator +(DT &rhs){
        DT* ret = new DT;
        ret->nodes = this->nodes;
        ret->feat_depth = this->feat_depth;
        ret->feat_dim = this->feat_dim;
        ret->cls_number = this->cls_number;
        for(size_t i = 0; i < nodes.size(); i++){
            if(nodes[i] == rhs.nodes[i]){
                ret->nodes[i] = nodes[i].merge(rhs.nodes[i]);
            }
        }
        return ret;
    }

    void DT::beginDataScan(std::vector<double> records, size_t feat_dim){
        Node head;
        head.index.resize(feat_dim-1);
        std::fill(head.index.begin(), head.index.end(), -1);
        head.init(feat_depth, cls_number);
        nodes.clear();
        nodes.push_back(head);
    }

    void DT::endDataScan(){
        std::vector<Node> next;
        next.clear();
        for(size_t i = 0; i < nodes.size(); i++){
            Node cur = nodes[i];
            if(cur.tag) continue;
            int pure = 0;
            int major = -1;
            int num = -1;
            for(size_t j = 0; j < cls_number; j++){
                if(cur.D[j] != 0) pure ++;
                if(cur.D[j] > num){
                    num = cur.D[j];
                    major = j;
                }
            }

            if(pure == 1){
                cur.vec.clear();
                cur.tag = true;
                cur.who = major;
                cur.D.clear();
                next.push_back(cur);
                continue;
            }
            int end = 0;
            for(size_t j = 0; j < cur.index.size(); j++){
                if(cur.index[j] == -1) end++;
            }
            int win = cur.split();
            //std::cout << pure << " "<<major  << std::endl;
            //std::cout << end << " "<<win  << std::endl;
            for(size_t k = 0; k < feat_depth[win]; k++){
                Node tmp;
                tmp.index = cur.index;
                tmp.index[win] = k;
                tmp.init(feat_depth, cls_number);
                //std::cout << "##"<<tmp.D.size() << std::endl;

                int tot = 0;
                for(size_t w = 0; w < cls_number; w++){
                    tot += cur.vec[win].get(k,w);
                }
                //std::cout << "**" << tot << " " << end << std::endl;
                if(tot == 0 || end == 1){
                    tmp.tag = true;
                    tmp.D.clear();
                    tmp.vec.clear();
                    tmp.who = major;
                }
                //std::cout << "$$"<<tmp.D.size() << std::endl;
                next.push_back(tmp);
            }
        }
        this->nodes = next;
    }

    DT* DT::processRecord(std::vector<double> records, size_t feat_dim){
        DT* ret = new DT;
        ret->nodes = this->nodes;
        ret->feat_depth = this->feat_depth;
        ret->feat_dim = this->feat_dim;
        ret->cls_number = this->cls_number;
        size_t n_records = records.size()/feat_dim;

        for(size_t k = 0; k < n_records; k++){
            std::vector<int> record; 
            record.clear();
            for(size_t w = 0; w < feat_dim ; w++){
                record.push_back(records[w + k*feat_dim]);
            }

            for(size_t i = 0; i < ret->nodes.size(); i++){
                Node cur = ret->nodes[i];

                bool tag = true;
                for(size_t j = 0; j < cur.index.size(); j++){
                    if(cur.index[j] != -1 && cur.index[j] != record[j]){
                        tag = false;
                        break;
                    }
                } 

                if(tag){
                    if(cur.tag) break;
                    cur.update(record);
                    ret->nodes[i] = cur;
                    break;
                }
            }
        }
        return ret;
    }

    bool DT::isConverged(DT* rhs, size_t feat_dim, double eps){
        for(size_t i = 0; i < nodes.size(); i ++){
            if(nodes[i].tag == false) return false;
        }
        return true;
    }

    void DT::finish(std::vector<double> records, size_t feat_dim){
        std::ofstream file;
        file.open("./DT.txt");	

        for(size_t i = 0; i < nodes.size(); i++){
            for(size_t j = 0; j < nodes[i].index.size(); j ++){
                file <<  nodes[i].index[j] << ' ';
            }
            file << nodes[i].who << std::endl;
        }
    }


}


