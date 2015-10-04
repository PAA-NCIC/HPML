/*
ID: septicmk
LANG: C++
TASK: Kmeans.cpp
*/
#include <algorithm>
#include <vector>
#include <cmath>
#include <cstdio>

namespace Kmeans{
inline int Random(int mod){
    return static_cast<int> (static_cast<double>(rand())/ RAND_MAX * mod);
}


// use Matrix to store centroids
struct Matrix {
    // data
    size_t nrow, ncol;
    std::vector<float> data;
    std::vector<int> num;
     
    inline void Init(size_t nrow, size_t ncol){
        this->nrow = nrow;
        this->ncol = ncol;
        data.resize(nrow * ncol);
        num.resize(nrow);
        std::fill(data.begin(), data.end(), 0.0f);
        std::fill(data.begin(), data.end(), 0);
    }

    inline Matrix operator + (Matrix &rhs){
        Matrix ret;
        ret.Init(this->nrow, this->ncol);
        for(size_t i = 0; i < data.size(); ++ i)
            ret.data[i] = data[i] + rhs.data[i];
        for(size_t i = 0; i < num.size(); ++ i)
            ret.num[i] = num[i] + rhs.num[i];
        return ret;
    }

    inline void Print() {
        for (size_t i = 0; i < data.size(); ++ i){
            char ch = ((i+1) % ncol == 0) ? '\n' : ' ';
            printf("%.2f%c", data[i], ch);
        }
    }

};



}

int main() {
    return 0;
}

