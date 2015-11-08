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
//#include <boost/serialization/queue.hpp>
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

struct node
{
     int index;
	 double value;
};
struct problem
{
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

class Flexible_vector{
    public:
        problem prob;
	    ~Flexible_vector();

        problem Read_file_without_index(std::string filename);

        //test
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
