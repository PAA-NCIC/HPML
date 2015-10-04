/*
ID: septicmk
LANG: C++
TASK: Kmeans.cpp
*/
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

namespace mpi = boost::mpi;


inline int Random(int mod){
    return static_cast<int> (static_cast<double>(rand())/ RAND_MAX * mod);
}

inline int Round(double r){  
    return static_cast<int> ((r > 0.0) ? floor(r + 0.5) : ceil(r - 0.5));
}  

// use Matrix to store centroids
struct Matrix {
    private:
        //data
        size_t nrow, ncol;
        std::vector<float> data;
        std::vector<int> num;
        
        //serialization
        friend class boost::serialization::access;
        template <typename Archive>
            friend void serialize (Archive &ar, Matrix &m, const unsigned int version){
                ar &m.nrow;
                ar &m.ncol;
                ar &m.data;
                ar &m.num;
            }

    public:
        // init the centroids
        inline void Init(size_t nrow, size_t ncol){
            this->nrow = nrow;
            this->ncol = ncol;
            data.resize(nrow * ncol);
            num.resize(nrow);
            std::fill(data.begin(), data.end(), 0.0f);
            std::fill(data.begin(), data.end(), 0);
        }

        inline void set (std::vector<float> record, size_t locata){
            num[locata] += 1;
            for (size_t i = 0; i < record.size(); ++i){
                data[ locata * ncol + i ] += record[i];
            }
        }
        
        // update the centroids
        inline Matrix operator + (Matrix &rhs){
            Matrix ret;
            ret.Init(this->nrow, this->ncol);
            for (size_t i = 0; i < data.size(); ++ i)
                ret.data[i] = data[i] + rhs.data[i];
            for (size_t i = 0; i < num.size(); ++ i)
                ret.num[i] = num[i] + rhs.num[i];
            return ret;
        }

        inline double operator - (Matrix &rhs){
            double eps = 0;
            for (size_t i = 0; i < nrow; ++i){
                double tmp = 0;
                for (size_t j = 0; j < ncol; ++j){
                    float t = data[ i*ncol + j] / num[i];
                    float w = rhs.data[ i*ncol + j] / rhs.num[i];
                    tmp += (t - w)*(t - w);
                }
                tmp = sqrt(tmp);
                eps += tmp;
            }
            return eps; 
        }

        // assign cluster
        inline int closest(std::vector<float> record){
            float dis = (1<<16)-1;
            int who = -1;
            for (size_t i = 0; i < nrow; ++i){
                float tmp = 0;
                for (size_t j = 0; j < ncol; ++j){
                    float t = data[ i*ncol + j] / num[i];
                    tmp += (t - record[j])*(t - record[j]);
                }
                if (tmp < dis){
                    who = i;
                    dis = tmp;
                }
            }
            return who;
        }

        // logging(tmp)
        inline void Print() {
            for (size_t i = 0; i < data.size(); ++ i){
                char ch = ((i+1) % ncol == 0) ? '\n' : ' ';
                printf("%.2f%c", data[i], ch);
            }
            for (size_t i = 0; i < num.size(); ++i){
                char ch = (i == num.size()-1) ? '\n' : ' ';
                printf("%d%c", num[i], ch);
            }
        }

};


std::string Trim (std::string &str){
    str.erase(0,str.find_first_not_of(" \t\r\n"));
    str.erase(str.find_last_not_of(" \t\r\n") + 1);
    return str;
}

inline std::vector<std::string> ReadCSV(std::string pwd){
    std::ifstream fin(pwd.c_str());
    std::string e,line;
    std::vector<std::string> data;
    while (std::getline(fin, line)){
        std::stringstream  lineStream(line);
        while(std::getline(lineStream, e,',')){
            data.push_back(e);
        }
    }
    return data;
}

inline float praser(std::vector<std::string> data, int feat_dim, size_t i, size_t j){
    return atof(data[i*feat_dim+j].c_str());
}

bool beginDataScan(Matrix &m, 
        std::vector<std::string> data,
        size_t number, size_t partion,
        size_t iter_num, mpi::communicator world,
        size_t n_cluster, int feat_dim,
        float eps){
    size_t rank = world.rank();
    size_t n_records = data.size()/feat_dim;
    Matrix centroids = m;
    boost::mpi::broadcast(world, centroids, 0);

    size_t lim = std::min(rank*number+number, n_records);
    std::vector<float> record;
    Matrix addon;
    addon.Init(n_cluster, feat_dim);


    for (size_t i = rank*number; i < lim; ++ i){
        record.clear();
        for (size_t j = 0; j < feat_dim; ++ j)
            record.push_back(praser(data, feat_dim, i, j));
        int who = centroids.closest(record);
        addon.set(record, who);
    }
    //std::cout << "#################" << std::endl;
    //std::cout << rank << std::endl;
    //addon.Print();
    //std::cout << "#################" << std::endl;
    bool flag = false;
    Matrix cur_centroids;
    if(rank == 0){
        std::cout << "[iteration] " << iter_num;
        cur_centroids = centroids + addon;
        for(size_t i = 1; i < world.size(); ++i){
            world.recv(boost::mpi::any_source, i, addon);
            //std::cout << i << std::endl;
            //addon.Print();
            cur_centroids = cur_centroids + addon;
        }
        float diff = cur_centroids - centroids;
        std::cout<< " diff : " << diff << std::endl;
        flag = diff > eps ? false : true;
    }else{
        world.send(0, rank, addon);
    }
    boost::mpi::broadcast(world, flag, 0);
    m = cur_centroids;
    return flag;
}


int main(int argc, char* argv[]){
    size_t partion = 5;
    std::vector<std::string> data = ReadCSV("./data/iris_n.csv");
    size_t n_cluster = 3;
    size_t feat_dim = 4;
    size_t n_records = data.size() / feat_dim;
    size_t number = n_records / partion;
    float eps = 1e-5;
    Matrix centroids;
    centroids.Init(n_cluster, feat_dim);

    //todo
    std::priority_queue<int> q;
    std::vector<int> weight;
    for (size_t i = 0; i < n_records; ++ i){
        int x = Random(10007);
        weight.push_back(x);
        if (q.size() < n_cluster){
            q.push(i);
        }else if (x < weight[q.top()]){
            q.pop();
            q.push(i);
        }
    }
    while(!q.empty()){
        std::vector<float> record;
        record.clear();
        size_t i = q.top();
        for(size_t j = 0; j < feat_dim ; ++j){
            record.push_back(praser(data, feat_dim, i, j));
        }
        centroids.set(record, n_cluster - q.size());
        q.pop();
    }
    //

    mpi::environment env(argc, argv);
    mpi::communicator world;
    if (world.rank() == 0)
        std::cout << "[processor] "<<world.size() << std::endl;

    size_t iter_num = 0;
    bool flag = false;
    do{
        flag = beginDataScan(centroids, data, number, partion, iter_num, world, n_cluster, feat_dim, eps); 
        iter_num ++;
    }while(!flag);

    if(world.rank() == 0){

        std::ofstream file;
        file.open("./ans.txt");	
        std::vector<float> record;
        for (size_t i = 0; i < n_records; ++ i){
            record.clear();
            for (size_t j = 0; j < feat_dim; ++ j){
                record.push_back(praser(data, feat_dim, i, j));
            }
            int who = centroids.closest(record);
            file << who << std::endl;
        }
        file.close();

    }

}
