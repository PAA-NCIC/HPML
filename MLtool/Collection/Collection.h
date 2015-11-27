/*
LANG: C++
TASK: Collection.h 
*/

#ifndef __COLLECTION
#define __COLLECTION

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <string>
#include <vector>

class Collection{
private:
    friend class boost::serialization::access;
    template <typename Archive>
        void serialize (Archive &ar, const unsigned int version){
        }
    public:
        virtual void Load(std::string filename)=0;
        //virtual void Store(std::string filename)=0;
        //virtual Collection* partition(size_t number, size_t rank)=0;
        //重载[]操作符，作为右值
        virtual std::vector<double> operator [](size_t loc) const =0;
        virtual size_t size()=0;
    
        //problem prob;
        //virtual void output_problem(std::string file_out,problem prob)=0;
};

#endif
