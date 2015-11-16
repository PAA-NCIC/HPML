/*
ID: septicmk
LANG: C++
TASK: Core.cpp
*/

//#include "../MLalgorithm/MLalgorithm.h"
#include "Core.h"


std::string Core::Trim (std::string &str){
    str.erase(0,str.find_first_not_of(" \t\r\n"));
    str.erase(str.find_last_not_of(" \t\r\n") + 1);
    return str;
}

inline std::vector<std::string> Core::ReadCSV(std::string pwd){
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

inline double Core::praser(std::vector<std::string> data, int feat_dim, size_t i, size_t j){
    return atof(data[i*feat_dim+j].c_str());
}

inline std::vector<double> Core::partition(std::vector<std::string> origin, size_t number, size_t loc, size_t feat_dim){
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

/*
inline MLalgorithm* Core::merge(int rank, boost::mpi::communicator world, MLalgorithm* local_update){
        MLalgorithm* ptmp;
        MLalgorithm* update = local_update;
        if(rank*2+1 < world.size()){
            world.recv(rank*2+1, rank*2+1, ptmp);
            update = *update + *ptmp;
        }
        if(rank*2+2 < world.size()){
            world.recv(rank*2+2, rank*2+2, ptmp);
            update = *update + *ptmp;
        }
    if(rank == 0){
        return update;
    }else{
        world.send((rank-1)/2, rank, update);
        return NULL;
    }
}

MLalgorithm* Core::mainLoop(int argc, char* argv[], 
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
        }
        MLalgorithm* global_updata = merge(rank, world, local_update);
        if(rank == 0){
            MLalgorithm *pafter = *ptr + *global_updata;
            pafter->endDataScan();
            done = pafter->isConverged(ptr, feat_dim, eps);
            ptr = pafter;
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
*/
