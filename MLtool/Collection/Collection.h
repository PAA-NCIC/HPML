/*
LANG: C++
TASK: Collection.h 
*/

#ifndef __COLLECTION
#define __COLLECTION

#include <string>
#include <vector>

class Collection{
    public:
        virtual void Load(std::string filename)=0;
        //virtual Collection* partition(size_t number, size_t rank)=0;
        //重载[]操作符，作为右值
        virtual std::vector<double> operator [](size_t loc) const =0;
        virtual size_t size()=0;
    
        //problem prob;
        //virtual void output_problem(std::string file_out,problem prob)=0;
};

#endif
