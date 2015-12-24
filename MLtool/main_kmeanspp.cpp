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
#include "Cluster/Kmeans_PlusPlus.h"
//#include "DT/DT.h"

//using namespace core;
//using namespace dt;
using namespace kmeans_plusplus;

int main(int argc,char *argv[]){
    //Core<Kmeans_Lloyd> *p = new Core<Kmeans_Lloyd>();
    Flexible_vector *f_vector = new Flexible_vector();

    std::string file_in = "pmlp_iris.csv";
    std::string file_out = "test";

    //Load data
    f_vector->Load(file_in);
    std::cerr<<"In main: prob.l= "<<f_vector->prob.l<<" prob.max_feature= "<<f_vector->prob.max_feature<<std::endl;
    //p->f_vector->output_problem(file_out,p->f_vector->prob);
    //Flexible_vector *f = (Flexible_vector *)p->f_vector;
    //f->output_problem(file_out,f->prob);

    //set model
    Kmeans_PlusPlus *ptr = new Kmeans_PlusPlus();
    //double max_feature = p->f_vector->prob.max_feature;
    //ptr->fvkl.init(3,4);
    //ptr->fvkl.init(3,max_feature);
    //p->mainLoop(argc, argv, ptr, 1e-7, 4);
    int k = 6;
    std::vector<int> result;
    result = ptr->kmeans_plusplus(f_vector,k);
    std::cout<<"iterator Kmeans ++ center: "<<std::endl;
    int counter=0;
    for(std::vector<int>::iterator i=result.begin();i != result.end();i++)
    {
        if(*i==1)
            std::cout<<counter<<std::endl;
        counter++;
    }
    std::cout<<"size Kmeans ++ center: "<<std::endl;
    for(int ii=0;ii < result.size();ii++)
    {
        if(result[ii]==1)
            std::cout<<ii<<std::endl;
    }
    std::cout<<"Kmeans ++ over!"<<std::endl;
    //p->mainLoop(argc, argv, ptr, 1e-7, k);
    return 0;
}
