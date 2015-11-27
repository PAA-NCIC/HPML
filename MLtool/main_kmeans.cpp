/*
ID: septicmk
LANG: C++
TASK:main.cpp
*/
#include "MLalgorithm/MLalgorithm.h"
#include "Core/Core.h"
#include "Kmeans/Kmeans.h"
//#include "DT/DT.h"

using namespace core;
//using namespace dt;
using namespace kmeans;

int main(int argc,char *argv[]){
    //Core<DT> *p = new Core<DT>();
    Core<Kmeans> *p = new Core<Kmeans>();
    p->f_vector = new Flexible_vector();

    //std::string file_in = "../data/car_nor.csv";
    std::string file_in = "pmlp_iris.csv";
    std::string file_out = "test";

    //Load data
    p->f_vector->Load(file_in);
    //DT *ptr = new DT();
    //std::vector<size_t> feat_depth;
    //feat_depth.push_back(4);
    //feat_depth.push_back(4);
    //feat_depth.push_back(4);
    //feat_depth.push_back(3);
    //feat_depth.push_back(3);
    //feat_depth.push_back(3);
    //ptr->init(feat_depth, 4);

    Kmeans *ptr = new Kmeans();
    ptr->mat.init(3,5);
    p->mainLoop(argc, argv, ptr, 1e-7, 4);
    //p->pause();
    return 0;
}

