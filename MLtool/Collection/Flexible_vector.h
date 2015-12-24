/*
ID: neo-white
LANG: C++
TASK: Flexible_vector.h
*/
#ifndef __FLEXIBLE_VECTOR
#define __FLEXIBLE_VECTOR

#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/queue.hpp>
#include <boost/mpi.hpp>
#include <algorithm>
#include <vector>
#include <queue>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <sstream>
#include <fstream>  
#include <string>
#include <iomanip>
#include "Collection.h"

struct node
{
    friend class boost::serialization::access;
    template <typename Archive>
        void serialize (Archive &ar, const unsigned int version){
            ar & index;
            ar & value;
            }

     int index;
	 double value;
};
struct problem
{
    friend class boost::serialization::access;
    template <typename Archive>
        void serialize (Archive &ar, const unsigned int version){
            ar & l;
            ar & max_feature;
            ar & y;
            ar & x_ptr;
            ar & x_interval;
            ar & x;
            }
    
    int l;
	int max_feature;
    std::vector<double> y;
    std::vector<int> x_ptr;
    std::vector<int> x_interval;
    std::vector<node> x;
};

template <class Type>
Type stringToNum(const std::string& str)
{
    std::istringstream iss(str);
    Type num;
    iss >> num;
    return num;
}

class Flexible_vector:public Collection{
    private:
        friend class boost::serialization::access;
        template <typename Archive>
            void serialize (Archive &ar, const unsigned int version){
                ar & boost::serialization::base_object<Collection>(*this);
                ar & prob;
            }
    public:
        problem prob;
	    //~Flexible_vector();

        problem Read_file_without_index(std::string filename);

        //From Collection
        void Load(std::string filename);
        //void Store(std::string filename);
        //Collection* partition(size_t number, size_t rank);
        Flexible_vector* partition(Flexible_vector *origin, size_t number, size_t loc, size_t feat_dim);
        size_t partition(int problem_l, size_t worker_number, size_t loc);
        std::vector<double> operator [](size_t loc) const ;
        size_t size();

        void init();
        void init(size_t l, size_t max_feature);
        void GenerateFVfromVector(std::vector<int> src);    //use int vector to generate Flexible_vector
        void insert_end(Flexible_vector src, size_t loc);   //insert the loc-th sample of src to the end of prob
        std::vector<node> get_value_without_label(size_t loc);
        void add(std::vector<node> src, size_t loc);   //add src to the loc-th sample of prob in fvkl 
        std::vector<double> FvtoVector() const ;
        std::vector<std::string> Vector_DtoS(std::vector<double> data) const ;
        //test
        void print_fv();
        void output_problem(std::string filename,problem prb);
        void main_vector(int argc,char* argv[]);
    
/*
    private:
        int get_data(const int index, Qfloat **data, int len);
    	void swap_index(int i, int j);
	    int l;
	    long int size;
	    struct head_t
    	{
	    	head_t *prev, *next;	// a circular list
		    Qfloat *data;
	    	int len;		// data[0,len) is cached in this entry
	    };

    	head_t *head;
    	head_t lru_head;
    	void lru_delete(head_t *h);
    	void lru_insert(head_t *h);

*/
};

#endif
