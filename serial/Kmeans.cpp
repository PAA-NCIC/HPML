/*
ID: septicmk
LANG: C++
TASK: Kmeans.cpp
*/
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <queue>
#include <sstream>
#include <fstream>
#include <string>

inline int Random(int mod){ return static_cast<int> (static_cast<double>(rand())/ RAND_MAX * mod);
}

inline int Round(double r){  
    return static_cast<int> ((r > 0.0) ? floor(r + 0.5) : ceil(r - 0.5));
}  

struct Matrix {
    private:
        //data
        size_t nrow, ncol;
        std::vector<float> data;
        std::vector<int> num;
        
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
            ret.Init(nrow, ncol);
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

        inline void reset(){
            for (size_t i = 0; i < nrow; ++i){
                for (size_t j = 0; j < ncol; ++j){
                    data[i*ncol + j] /= num[i];
                }
                num[i] = 1;
            }
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
        size_t iter_num,
        size_t n_cluster, int feat_dim,
        float eps){
    size_t n_records = data.size()/feat_dim;
    Matrix centroids = m;

    std::vector<float> record;
    Matrix addon;
    addon.Init(n_cluster, feat_dim);

    for (size_t i = 0; i < n_records; ++ i){
        record.clear();
        for (size_t j = 0; j < feat_dim; ++ j)
            record.push_back(praser(data, feat_dim, i, j));
        int who = centroids.closest(record);
        addon.set(record, who);
    }
    bool flag = false;
    Matrix cur_centroids;
    std::cout << "[iteration] " << iter_num;
    cur_centroids = centroids + addon;
    cur_centroids = cur_centroids + addon;
    float diff = cur_centroids - centroids;
    std::cout<< " diff : " << diff << std::endl;
    flag = diff > eps ? false : true;
    m = cur_centroids;
    //m.Print();
    return flag;
}


int main(int argc, char* argv[]){
    std::vector<std::string> data = ReadCSV("../data/iris_n.csv");
    size_t n_cluster = 3;
    size_t feat_dim = 4;
    size_t n_records = data.size() / feat_dim;
    std::cout<<n_records << std::endl;
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

    size_t iter_num = 0;
    bool flag = false;
    do{
        flag = beginDataScan(centroids, data, iter_num, n_cluster, feat_dim, eps); 
        centroids.reset();
        iter_num ++;
    }while(!flag);


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
    return 0;
}
