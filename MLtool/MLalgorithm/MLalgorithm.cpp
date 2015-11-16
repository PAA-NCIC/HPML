/*
ID: septicmk
LANG: C++
TASK: 
*/
/*
 * modified by Yuan Lufeng in 20151113
 * function:data format is changed to Collection.
 */

#include "MLalgorithm.h"
#include <vector>
#include <boost/serialization/vector.hpp>


void MLalgorithm::Init(size_t nrow, size_t ncol){
    this->nrow = nrow;
    this->ncol = ncol;
    data.resize(nrow * ncol);
    std::fill(data.begin(), data.end(), 0.0f);
}

void MLalgorithm::add (std::vector<double> record, size_t i){
    for (size_t j = 0; j < record.size(); ++j){
        data[ i*ncol + j ] += record[j];
    }
}


std::vector<double> MLalgorithm::get_value(size_t i){
    std::vector<double> ret;
    for (size_t j = 0; j < ncol; ++j){
        ret.push_back(data[i*ncol + j]);
    }
    return ret;
}


