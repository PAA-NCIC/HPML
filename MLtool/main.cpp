/*
ID: neo-white
LANG: C++
TASK:main.cpp
*/
#include "Collection/Flexible_vector.h"
#include "Core/Core.h"


int main(int argc,char *argv[]){
    Core *p = new Core();
    p->f_vector = new Flexible_vector();

    std::string file_in = "pmlp_iris.csv";
    std::string file_out = "test";

    p->f_vector->Load(file_in);
    //p->f_vector->output_problem(file_out,p->f_vector->prob);
    Flexible_vector *f = (Flexible_vector *)p->f_vector;
    f->output_problem(file_out,f->prob);

    return 0;
}

