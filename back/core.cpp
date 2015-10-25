/*
ID: septicmk
LANG: C++
TASK: Kmeans.cpp
*/
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/queue.hpp>
#include <boost/serialization/export.hpp> 
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

//namespace mpi = boost::mpi;

inline int Random(int mod){
    return static_cast<int> (static_cast<double>(rand())/ RAND_MAX * mod);
}

inline int Round(double r){  
    return static_cast<int> ((r > 0.0) ? floor(r + 0.5) : ceil(r - 0.5));
}  


class MLalgorithm{
    private:
        friend class boost::serialization::access;
        template <typename Archive>
        void serialize (Archive &ar, const unsigned int version){
                ar &nrow;
                ar &ncol;
                ar &data;
            }

    public:
        size_t nrow, ncol;
        std::vector<double> data;

        void Init(size_t nrow, size_t ncol){
            this->nrow = nrow;
            this->ncol = ncol;
            data.resize(nrow * ncol);
            std::fill(data.begin(), data.end(), 0.0f);
        }

        void add (std::vector<double> record, size_t i){
            for (size_t j = 0; j < record.size(); ++j){
                data[ i*ncol + j ] += record[j];
            }
        }

        std::vector<double> get_value(size_t i){
            std::vector<double> ret;
            for (size_t j = 0; j < ncol; ++j){
                ret.push_back(data[i*ncol + j]);
            }
            return ret;
        }
        
        virtual MLalgorithm* operator + (MLalgorithm &rhs)=0;
        virtual void beginDataScan(std::vector<double> records, size_t feat_dim) = 0;
        virtual MLalgorithm* processRecord (std::vector<double> records, size_t feat_dim)=0;
        virtual void endDataScan()=0;
        virtual bool isConverged(MLalgorithm* rhs, size_t feat_dim, double eps)=0;
        virtual void finish(std::vector<double> records, size_t feat_dim)=0;
};

class Kmeans:public MLalgorithm{
    private:
        friend class boost::serialization::access;
        template <typename Archive>
        void serialize (Archive &ar, const unsigned int version){
            ar & boost::serialization::base_object<MLalgorithm>(*this);
        }

    public:
        MLalgorithm* operator +(MLalgorithm &rhs){
            MLalgorithm *pret = new Kmeans();
            pret->Init(this->nrow, this->ncol);
            for(size_t i = 0; i< data.size() ; ++i)
                pret->data[i] = data[i] + rhs.data[i];
            return pret;
        }

        void beginDataScan(std::vector<double> records, size_t feat_dim){
            std::priority_queue<int> q;
            std::vector<int> weight;
            size_t n_records = records.size();
            size_t n_cluster = this->nrow;
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
                std::vector<double> record;
                record.clear();
                size_t i = q.top();
                for(size_t j = 0; j < feat_dim ; ++j){
                    record.push_back(records[i*feat_dim+j]);
                }
                add(record, n_cluster - q.size());
                q.pop();
            }
        }

        void endDataScan(){
            for(size_t i = 0; i < nrow; ++i){
                for(size_t j = 0; j < ncol; ++j){
                    data[i*ncol+j] /= data[i*ncol + ncol-1];
                }
                data[i*ncol + ncol-1] = 1;
            } 

            //for(size_t i = 0; i < nrow; ++i){
                //for(size_t j = 0; j < ncol; ++j){
                    //std::cout << data[i*ncol+j] << " ";
                //}
                //std::puts("");
            //}
        }

        MLalgorithm* processRecord(std::vector<double> records, size_t feat_dim){
            MLalgorithm* pret = new Kmeans();
            pret->Init(this->nrow, this->ncol);
            size_t n_records = records.size()/feat_dim;
            for (size_t i = 0 ; i < n_records ; i++){
                std::vector<double> record;
                record.clear();
                for(size_t j = 0; j < feat_dim ; j ++){
                    record.push_back(records[i*feat_dim+j]);
                }
                double dis = (1L<<16)-1;
                int who = -1;
                //std::cout<<"bp"<<std::endl;
                for(size_t k = 0; k < this->nrow; k++){
                    std::vector<double> centroid = this->get_value(k);
                    double tmp = 0;
                    for (size_t u = 0; u < record.size(); ++u){
                        tmp += (centroid[u] - record[u])*(centroid[u] - record[u]);
                    }

                    if (tmp < dis){
                        who = k;
                        dis = tmp;
                    }
                }
                record.push_back(1);
                pret->add(record, who);
            }
            return pret;
        }

        bool isConverged(MLalgorithm* rhs, size_t feat_dim, double eps){
            double diff = 0;
            for(size_t i = 0; i < nrow; i ++){
                double tmp = 0;
                for(size_t j = 0;  j < feat_dim; j++){
                    tmp += (data[i*ncol+j] - rhs->data[i*ncol+j]) * (data[i*ncol+j] - rhs->data[i*ncol+j]);
                }
                tmp = sqrt(tmp);
                diff += tmp;
            }
            if (diff <= eps) return true;
            else return false;

        }

        void finish(std::vector<double> records, size_t feat_dim){
            std::ofstream file;
            file.open("./ans.txt");	
            std::vector<float> record;
            size_t n_records = records.size()/feat_dim;
            for(size_t i = 0; i < n_records; ++i){
                std::vector<double> record;
                record.clear();
                for(size_t j = 0; j < feat_dim; ++ j){
                    record.push_back(records[i*feat_dim+j]);
                }
                double dis = (1<<16)-1;
                int who = -1;
                for(size_t k = 0; k < this->nrow; k++){
                    std::vector<double> centroid = this->get_value(k);
                    double tmp = 0;
                    for (size_t u = 0; u < record.size(); ++u){
                        tmp += (centroid[u] - record[u])*(centroid[u] - record[u]);
                    }

                    if (tmp < dis){
                        who = k;
                        dis = tmp;
                    }
                }
                file << who << std::endl;

            }

        }
};
BOOST_CLASS_EXPORT(Kmeans)


class Core{
    public:
     
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

        inline double praser(std::vector<std::string> data, int feat_dim, size_t i, size_t j){
            return atof(data[i*feat_dim+j].c_str());
        }

        inline std::vector<double> partition(std::vector<std::string> origin, size_t number, size_t loc, size_t feat_dim){
            std::vector<double> ret;
            ret.clear();
            size_t n_records = origin.size()/feat_dim;
            int members = n_records/number;
            size_t lim = std::min(loc*members+members, n_records);
            for (size_t i = loc*members; i < lim ; i++){
                for(size_t j = 0 ; j < feat_dim; j ++){
                    ret.push_back(praser(origin, feat_dim, i, j));
                }
            }
            return ret;
        }

        MLalgorithm* mainLoop(int argc, char* argv[], 
                MLalgorithm *ptr,
                double eps,
                size_t feat_dim){
            std::vector<std::string> data = ReadCSV("../data/iris_n.csv");
            ptr->beginDataScan(partition(data,1,0,feat_dim), feat_dim);
            boost::mpi::environment env(argc, argv);
            boost::mpi::communicator world;
            int rank = world.rank();
            size_t number = world.size();
            size_t n_records = data.size()/ feat_dim;
            size_t iter_num = 0;
            std::vector<double> records = partition(data, number, rank, feat_dim);
            if (rank == 0) std::cout << "[processor] "<< world.size() << std::endl;

            bool done = 0;
            while(!done){
                boost::mpi::broadcast(world, ptr, 0);
                MLalgorithm* local_update = ptr->processRecord(records, feat_dim);
                
                world.barrier();
                //if(rank ==0 ) std::cout<<"flag"<<std::endl;
                if(rank == 0){
                    std::cout << "[iteration] " << iter_num++ << std::endl;
                    MLalgorithm* ptmp;
                    MLalgorithm* global_updata = local_update;
                    for(size_t i = 1; i < world.size(); ++i){
                        world.recv(boost::mpi::any_source, i, ptmp);
                        global_updata = *global_updata + *ptmp;
                    }
                    MLalgorithm *pafter = *ptr + *global_updata;
                    pafter->endDataScan();
                    done = pafter->isConverged(ptr, feat_dim, eps);
                    ptr = pafter;
                }else{
                    world.send(0, rank, local_update);
                }
                boost::mpi::broadcast(world, done, 0);
            }
            if(rank == 0){
                std::cout << "done" << std::endl;
                std::vector<double> records;
                for(size_t i = 0; i < data.size(); ++i){
                    records.push_back(atof(data[i].c_str()));
                }
                ptr->finish(records, feat_dim);
            }
        }
};

int main(int argc,char *argv[]){
    Core *p = new Core();
    MLalgorithm *ptr = new Kmeans();
    ptr->Init(3,5);

    p->mainLoop(argc, argv, ptr, 1e-7, 4);
    return 0;
}
