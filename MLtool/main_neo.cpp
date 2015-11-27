/*
ID: septicmk
LANG: C++
TASK:main.cpp
*/
#include "MLalgorithm/MLalgorithm.h"
#include "Core/Core.h"
#include "Kmeans/Kmeans.h"

using namespace core;
using namespace kmeans;
int main(int argc,char *argv[]){
    Core<Kmeans> *p = new Core<Kmeans>();
    Kmeans *ptr = new Kmeans();
    ptr->Init(3,5);

    p->mainLoop(argc, argv, ptr, 1e-7, 4);
    return 0;
}

