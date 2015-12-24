/*
ID: septicmk
LANG: C++
TASK: Core_impl.h
*/

#include "../MLalgorithm/MLalgorithm.h"
#include "../Cluster/Kmeans_PlusPlus.h"
//#include "Core.h"

namespace core{

    template<class T>
        std::string Core<T>::Trim (std::string &str){
            str.erase(0,str.find_first_not_of(" \t\r\n"));
            str.erase(str.find_last_not_of(" \t\r\n") + 1);
            return str;
        }

    template<class T>
        inline std::vector<std::string> Core<T>::ReadCSV(std::string pwd){
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

    template<class T>
        inline double Core<T>::praser(std::vector<std::string> data, int feat_dim, size_t i, size_t j){
            return atof(data[i*feat_dim+j].c_str());
        }

    /*
    template<class T>
        inline std::vector<double> Core<T>::partition(std::vector<std::string> origin, size_t number, size_t loc, size_t feat_dim){
            std::vector<double> ret;
            ret.clear();
            size_t n_records = origin.size()/feat_dim;
            int members = n_records/number;
            if (number*members < n_records) members += 1;
            size_t lim = std::min(loc*members+members, n_records);
            for (size_t i = loc*members; i < lim ; i++){
                for(size_t j = 0 ; j < feat_dim; j ++){
                    ret.push_back(praser(origin, feat_dim, i, j));
                }
            }
            return ret;
        }
    */
    template<class T>
        T* Core<T>::merge(int rank, boost::mpi::communicator world, T* local_update){
            T* ptmp;
            T* update = local_update;
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

    template<class T>
        std::vector<int> Core<T>::AddWeight(std::vector<int> x,std::vector<int> y){
            std::vector<int> sum;
            sum.resize(x.size());
            for(size_t i = 0; i < x.size(); ++i)
                sum[i] = x[i] + y[i];
            return sum;
        }

    template<class T>
        std::vector<int> Core<T>::MergeWeight(int rank, boost::mpi::communicator world, std::vector<int> local_weights){
            std::vector<int> ptmp;
            std::vector<int> update = local_weights;
            if(rank*2+1 < world.size()){
                world.recv(rank*2+1, rank*2+1, ptmp);
                update = AddWeight(update,ptmp);
            }
            if(rank*2+2 < world.size()){
                world.recv(rank*2+2, rank*2+2, ptmp);
                update = AddWeight(update,ptmp);
            }
            if(rank == 0){
                return update;
            }else{
                world.send((rank-1)/2, rank, update);
                return update;
            }
        }

    template<class T>
        T* Core<T>::mainLoop(int argc, char* argv[], 
                T *ptr,
                double eps,
                size_t feat_dim){
            //std::vector<std::string> data = ReadCSV("../data/car_nor.csv");
            int k = feat_dim;
            //std::cout<<"ptr.l= "<<ptr->fvkl.prob.l<<" ptr.max_feature= "<<ptr->fvkl.prob.max_feature<<std::endl;
            //std::cout<<"this.l= "<<this->f_vector->prob.l<<" this.max_feature= "<<this->f_vector->prob.max_feature<<std::endl;
            //std::cout<<"l= "<<f_vector->prob.l<<" max_feature= "<<f_vector->prob.max_feature<<std::endl;
            ptr->beginDataScan(this->f_vector, k);
            std::cout<<"Initial centroid set:"<<std::endl;
            ptr->fvkl.print_fv();
            boost::mpi::environment env(argc, argv);
            boost::mpi::communicator world;
            int rank = world.rank();
            size_t number = world.size();
            //size_t n_records = data.size()/ feat_dim;
            size_t iter_num = 0;
            //std::cerr<<"number= "<<number<<" rank= "<<rank<<std::endl;
            Flexible_vector *records = this->f_vector->partition(this->f_vector, number, rank, feat_dim);
            //std::cout<<"records.l= "<<records->prob.l<<" records.max_feature= "
             //       <<records->prob.max_feature<<std::endl;
            //records->output_problem("test_out",records->prob);
            if (rank == 0) std::cout << "[processor] "<< world.size() << std::endl;

            bool done = 0;
            while(!done){
                boost::mpi::broadcast(world, ptr, 0);
                //if(rank ==0 ) std::cout<<"flag"<<std::endl;
                if(rank == 0){
                    std::cout << "[iteration] " << iter_num++ << std::endl;
                }
                //world.barrier();
                T* local_update = ptr->processRecord(records, feat_dim);
                std::cerr<<"Rank"<<rank<<"local_update = "<<std::endl;
                local_update->fvkl.print_fv();
                //world.barrier();
                T* global_update = merge(rank, world, local_update);
                std::cerr<<"Rank"<<rank<<"global_update= "<<std::endl;
                global_update->fvkl.print_fv();
                //std::cerr<<"After merge"<<std::endl;
                if(rank == 0){
                    //T *pafter = *ptr + *global_update;  //mk
                    T *pafter =  global_update;  //yuan
                    pafter->endDataScan();

                    //std::cerr<<"After endDataScan"<<std::endl;
                    std::cerr<<"Rank0ptr = "<<std::endl;
                    ptr->fvkl.print_fv();
                    std::cerr<<"Rank0pafter = "<<std::endl;
                    pafter->fvkl.print_fv();
                    done = pafter->isConverged(ptr, feat_dim, eps);
                    ptr = pafter;
                }
                boost::mpi::broadcast(world, done, 0);
            }
            if(rank == 0){
                std::cout << "done" << std::endl;
                ptr->finish(this->f_vector, feat_dim);
            }
            return ptr;
        }
    //new mainLoop for Kmeans||
    template<class T>
        T* Core<T>::mainLoop(int argc, char* argv[], 
                T *ptr,
                size_t oversample,
                size_t round,
                int k){
            //std::vector<std::string> data = ReadCSV("../data/car_nor.csv");
            //std::cout<<"ptr.l= "<<ptr->fvkl.prob.l<<" ptr.max_feature= "<<ptr->fvkl.prob.max_feature<<std::endl;
            //std::cout<<"this.l= "<<this->f_vector->prob.l<<" this.max_feature= "<<this->f_vector->prob.max_feature<<std::endl;
            //std::cout<<"l= "<<f_vector->prob.l<<" max_feature= "<<f_vector->prob.max_feature<<std::endl;
            ptr->beginDataScan(this->f_vector, k);
            std::cout<<"Initial centroid set:"<<ptr->mbasic_.centers[0]<<std::endl;
            boost::mpi::environment env(argc, argv);
            boost::mpi::communicator world;
            int rank = world.rank();
            size_t number = world.size();
            //size_t n_records = data.size()/ feat_dim;
            size_t iter_num = 0;
            //std::cerr<<"number= "<<number<<" rank= "<<rank<<std::endl;
            size_t members = this->f_vector->partition(this->f_vector->prob.l, number, rank);
            ptr->mbasic_.oversample = oversample;
            ptr->mbasic_.members = members;
            //std::cout<<"records.l= "<<records->prob.l<<" records.max_feature= "
             //       <<records->prob.max_feature<<std::endl;
            //records->output_problem("test_out",records->prob);
            if (rank == 0) std::cout << "[processor] "<< world.size() << " members="<<members<< std::endl;
            boost::mpi::broadcast(world, ptr, 0);
            std::cout<<"process"<<rank<<":Initial ptr centers: "<<std::endl;
            ptr->OutputModel();
            T *local_model = ptr->ProcessDistanceSum(this->f_vector, &ptr->mbasic_, rank);
            std::cout<<"process"<<rank<<": local distance "<<local_model->mbasic_.costX<<std::endl;
            double re;
            boost::mpi::all_reduce(world,local_model->mbasic_.costX,re,std::plus<double>());
            //boost::mpi::all_reduce(world,local_model->mbasic_.costX,std::plus<double>());
            //std::cout<<"process"<<rank<<": sum distance "<<re<<std::endl;
            local_model->mbasic_.costX = re;
            std::cout<<"process"<<rank<<": sum distance "<<local_model->mbasic_.costX<<std::endl;
            //std::cout<<"process"<<rank<<":local_model centers: "<<std::endl;
            //local_model->OutputModel();
            bool done = 0;
            while(!done){
                //if(rank ==0 ) std::cout<<"flag"<<std::endl;
                if(rank == 0){
                    std::cout << "[iteration] " << iter_num++ << std::endl;
                }
                //world.barrier();
                local_model->SampleC(&local_model->mbasic_, ptr->mbasic_.centers,rank);
                std::cout<<"process"<<rank<<":local_model centers: "<<std::endl;
                local_model->OutputModel();
                //std::cerr<<"Rank"<<rank<<"local_update = "<<std::endl;
                //local_update->fvkl.print_fv();
                //world.barrier();
                T* global_update = merge(rank, world, local_model);
                if(rank == 0){
                    std::cout<<"process"<<rank<<":global update centers: "<<std::endl;
                    global_update->OutputModel();
                }
                //std::cerr<<"Rank"<<rank<<"global_update= "<<std::endl;
                //global_update->fvkl.print_fv();
                //std::cerr<<"After merge"<<std::endl;
                if(rank == 0){
                    T *pafter = *ptr + *global_update;
                    ptr = pafter;
                    std::cout<<"process"<<rank<<":global model centers: "<<std::endl;
                    ptr->OutputModel();
                }
                local_model->endDataScan();
                --round;
                done = (round == 0)?1:0; 
                if(done)
                    break;
                boost::mpi::broadcast(world, ptr, 0);
                local_model = ptr->ProcessDistanceSum(this->f_vector, &ptr->mbasic_, rank);
                //std::cerr<<"After endDataScan"<<std::endl;
            }
            //Set weight of each ci in C
            boost::mpi::broadcast(world, ptr, 0);
            std::vector<int> local_weights = ptr->SetWeight(this->f_vector, &ptr->mbasic_, rank); 
            std::vector<int> global_weights = MergeWeight(rank, world, local_weights);
            if(rank == 0){
                std::cout << "done" << std::endl;
                std::cout << "centers size :"<< ptr->mbasic_.centers.size() <<std::endl;
                std::cout << "centers :" << std::endl;
                for(size_t i = 0; i < ptr->mbasic_.centers.size(); ++i)
                    std::cout<<ptr->mbasic_.centers[i]<<" ";
                std::cout<<std::endl;
                ptr->global_weights = global_weights;
                std::cout << "weights size :"<< ptr->global_weights.size() <<std::endl;
                std::cout << "weights :" << std::endl;
                int sum_weights = 0;
                for(size_t i = 0; i < ptr->global_weights.size(); ++i){
                    sum_weights += ptr->global_weights[i];
                    std::cout<<ptr->global_weights[i]<<" ";
                }
                std::cout<<std::endl;
                std::cout << "sum weights :" << sum_weights << std::endl;
                //ptr->finish(this->f_vector, round);

                //Kmeans ++ recluster
                ptr->fvkl_.GenerateFVfromVector(ptr->global_weights);   //generate fv of the weighted points in C
                kmeans_plusplus::Kmeans_PlusPlus *recluster_kmeanspp = new kmeans_plusplus::Kmeans_PlusPlus();
                std::vector<int> final_centers = recluster_kmeanspp->kmeans_plusplus(&ptr->fvkl_,k);
                ptr->final_centers = final_centers;
                std::cout << "final centers size :"<< k <<std::endl;
                std::cout << "final centers :" << std::endl;
                for(size_t i = 0; i < ptr->final_centers.size(); ++i){
                    if(ptr->final_centers[i] == 1)
                        std::cout<<ptr->mbasic_.centers[i]<<" ";
                }
                std::cout<<std::endl;
                ptr->finish(this->f_vector, round);
            }
            //if(rank == 0){
            //    std::cout << "done" << std::endl;
            //    ptr->finish(this->f_vector, round);
            //}

            return ptr;
        }
}
