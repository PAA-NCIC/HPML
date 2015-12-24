/*
ID: neo-white
LANG: C++
TASK:main.cpp
*/
#include "Collection/Flexible_vector.h"
#include "Core/Core.h"
#include "MLalgorithm/MLalgorithm.h"
//#include "Kmeans/Kmeans.h"
//#include "Cluster/Kmeans_Lloyd.h"
//#include "Cluster/Kmeans_PlusPlus.h"
#include "Cluster/Kmeans_Parallel.h"
//#include "DT/DT.h"

using namespace core;
//using namespace dt;
using namespace kmeans_parallel;
//using namespace kmeans_plusplus;

int main(int argc,char *argv[]){
    Core<Kmeans_Parallel> *p = new Core<Kmeans_Parallel>();
    p->f_vector = new Flexible_vector();

    std::string file_in = "pmlp_iris.csv";
    //std::string file_out = "test";

    //Load data
    p->f_vector->Load(file_in);
    std::cerr<<"In main: prob.l= "<<p->f_vector->prob.l<<" prob.max_feature= "<<p->f_vector->prob.max_feature<<std::endl;
    //p->f_vector->output_problem(file_out,p->f_vector->prob);
    //Flexible_vector *f = (Flexible_vector *)p->f_vector;
    //f->output_problem(file_out,f->prob);

    //set model
    Kmeans_Parallel *ptr = new Kmeans_Parallel();
    //double max_feature = p->f_vector->prob.max_feature;
    //ptr->fvkl.init(3,4);
    //ptr->fvkl.init(3,max_feature);
    //p->mainLoop(argc, argv, ptr, 1e-7, 4);
    int k = 3;
    size_t oversample = k;
    size_t round =5;
    p->mainLoop(argc, argv, ptr, oversample, round, k);
    std::cout<<"Kmeans || over!"<<std::endl;
    return 0;
}
