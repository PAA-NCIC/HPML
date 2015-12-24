/*
ID: neo-white 
LANG: C++
TASK: Kmeans_Parallel.cpp
*/

#include "Kmeans_Parallel.h"
#include "../Core/Core.h"
#include <boost/serialization/vector.hpp>
#include <boost/serialization/queue.hpp>
#include <boost/serialization/export.hpp> 
#include <vector>
#include <queue>
#include <fstream>  
#include <time.h>
#include <sys/time.h>

BOOST_CLASS_EXPORT(kmeans_parallel::Kmeans_Parallel)
namespace kmeans_parallel{

    Kmeans_Parallel* Kmeans_Parallel::operator +(Kmeans_Parallel &rhs)
    {
        Kmeans_Parallel *pret = new Kmeans_Parallel();
        int l = mbasic_.centers.size() + rhs.mbasic_.centers.size();
        //pret->mbasic_.centers.resize(l);
        for(size_t i = 0; i < mbasic_.centers.size();++i)
            pret->mbasic_.centers.push_back(mbasic_.centers[i]);
        for(size_t i = 0; i < rhs.mbasic_.centers.size();++i)
            pret->mbasic_.centers.push_back(rhs.mbasic_.centers[i]);
        pret->mbasic_.oversample = mbasic_.oversample;
        pret->mbasic_.members = mbasic_.members;
        return pret;
    }
   
    /*
    std::vector<int> Kmeans_Lloyd::init_random(size_t length, size_t random_seed)
    {
        int tmp;
        std::vector<int> queue;
        for(size_t i=0;i<length;i++)
            queue.push_back(0);
           
        srand((unsigned)time(NULL));
        for(size_t j=0;j<random_seed;j++)
        {
            while(1)
            {
                tmp = (rand()%length);
                if(queue[tmp] == 0)
                {
                    queue[tmp] = 1;
                    break;
                }
            }
        }
        return queue;
    }
    */
    void Kmeans_Parallel::beginDataScan(Flexible_vector *records, size_t n_cluster)
    {
        srand((unsigned)time(NULL));
        int tmp = (rand()%(records->prob.l));
        mbasic_.centers.push_back(tmp);
        mbasic_.costX = 0;
    }
    
    void Kmeans_Parallel::endDataScan()
    {
        std::vector<int>().swap(mbasic_.centers);
        std::vector<double>().swap(mbasic_.dis);
        mbasic_.costX = 0.0;
    }
    
    Kmeans_Parallel* Kmeans_Parallel::processRecord(Model *prior, size_t rank){
        Kmeans_Parallel * tmp;
        return tmp;
    }
   
    /*
    bool Kmeans_Parallel::isConverged(Kmeans_Parallel* rhs, size_t feat_dim, double eps)
    {
        double diff = 0;
        std::cerr<<"fvkl.prob.l = "<<fvkl.prob.l<<" rhs.prob.l = "<<rhs->fvkl.prob.l<<std::endl;
        for(size_t i = 0; i < fvkl.prob.l; i ++)
        {
            double tmp = 0;
            std::vector<node> A;
            std::vector<node> B;
            A = this->fvkl.get_value_without_label(i);
            B = rhs->fvkl.get_value_without_label(i);
            //for(size_t j = 0;j < A.size();j++)
            //    std::cout<<" "<<A[j].value;
            //std::cout<<std::endl;
            //for(size_t jj = 0;jj < B.size();jj++)
            //    std::cout<<" "<<B[jj].value;
            //std::cout<<std::endl;
            tmp = euclidean_distance(A,B); 
            diff += tmp;
        }
        std::cout<<"diff = "<<diff<<std::endl;
        if (diff <= eps) return true;
        else return false;
    
    }
    */
    
    void Kmeans_Parallel::finish(Flexible_vector *records, size_t feat_dim)
    {
        std::ofstream file;
        file.open("./result_parallel.txt");	
        file <<"Candicated centers number is "<< mbasic_.centers.size() << std::endl;
        file <<"Candicated centers: "<< std::endl;
        for(size_t i = 0; i < mbasic_.centers.size(); ++i)
            file<<mbasic_.centers[i]<<"  ";
        file<<std::endl;
        file <<"Candicated centers weights number is "<< global_weights.size() << std::endl;
        file <<"Weights: "<< std::endl;
        for(size_t i = 0; i < global_weights.size(); ++i)
            file<<global_weights[i]<<"  ";
        file<<std::endl;
        file <<"Final centers: "<< std::endl;
        int j = 0;
        for(size_t i = 0; i < final_centers.size(); ++i){
            if(final_centers[i] == 1){
                file<<mbasic_.centers[i]<<" ";
                j++;
            }
        }
        file<<std::endl;
        file <<"Final centers number is "<< j << std::endl;
    }
    
    double Kmeans_Parallel::CountEuclideanDistance(std::vector<node> px,std::vector<node> py)
    {
    	double sum = 0;
    	double distances = 0;
        size_t ipx = 0;
        size_t ipy = 0;
        while((ipx <= (px.size()-1))||(ipy <= (py.size()-1)))
    	{
            if((ipx <= (px.size()-1))&&(ipy <= (py.size()-1)))
            {
    		    if(px[ipx].index == py[ipy].index)
    		    {
    		    	sum += (px[ipx].value-py[ipy].value)*(px[ipx].value-py[ipy].value);
    		    	++ipx;
    		    	++ipy;
    		    }
    		    else
    		    {
    		    	if(px[ipx].index > py[ipy].index)
                    {
                        sum += py[ipy].value*py[ipy].value;
    		    		++ipy;
                    }
    		    	else
                    {
                        sum += px[ipx].value*px[ipx].value;
    		    		++ipx;
                    }
    		    }
            }else if((ipx == px.size())&&(ipy <= (py.size()-1)))
            {
                sum += py[ipy].value*py[ipy].value;
    		    ++ipy;
            }else if((ipy == py.size())&&(ipx <= (px.size()-1)))
            {
                sum += px[ipx].value*px[ipx].value;
    	   		++ipx;
            }
    	}
        distances = sqrt(sum);
    	return distances;
    }

    Kmeans_Parallel *Kmeans_Parallel::ProcessDistanceSum(Flexible_vector *records, Model *prior, size_t loc){
        std::cout<<"process"<<loc<<"ProcessDistanceSum: prior centers size:"<<prior->centers.size()<<std::endl;
        std::cout<<"process"<<loc<<"ProcessDistanceSum: prior centers:"<<std::endl;
        for(size_t i = 0; i < prior->centers.size();++i)
            std::cout<<prior->centers[i]<<" ";
        std::cout<<std::endl;
        Kmeans_Parallel *model = new Kmeans_Parallel();
        size_t lim = std::min(loc*prior->members+prior->members,(size_t)records->prob.l);
        //std::cout<<"ProcessDistanceSum: processor"<<loc<<"  lim="<<lim<<" dis size="<<lim-loc*prior->members<<std::endl;
        std::vector<node> record_problem;
        std::vector<node> candidated_center;
        double sum_distances = 0.0;
        model->mbasic_.dis.resize(lim-loc*prior->members);
        //std::cout<<"ProcessDistanceSum: processor"<<loc<<"  lim="<<lim<<" dis size="<<model->mbasic_.dis.size()<<std::endl;
        //std::cout<<"ProcessDistanceSum: Initial dis:"<<std::endl;
        //for(size_t i = 0; i < model->mbasic_.dis.size();++i)
        //    std::cout<<model->mbasic_.dis[i]<<" ";
        //std::cout<<std::endl;
        for(size_t i = loc*prior->members;i < lim;i++){
            std::vector<int>::iterator f = find(prior->centers.begin(),prior->centers.end(),i);
            if(f != prior->centers.end()){
                model->mbasic_.dis[i-loc*prior->members] = 0.0;
                //std::cout<<"find"<<i<<" ";
                continue;
            }
            record_problem = records->get_value_without_label(i);
            double distance = (1L<<16)-1;
            //std::cout<<"distance = "<<distance<<std::endl;
            for(size_t j = 0;j < prior->centers.size();j++){
                candidated_center = records->get_value_without_label(prior->centers[j]);
                double distance_tmp = CountEuclideanDistance(record_problem,candidated_center);
                //std::cout<<"distance_tmp = "<<distance_tmp<<std::endl;
                if(distance_tmp < distance)
                    distance = distance_tmp;
            }
            model->mbasic_.dis[i-loc*prior->members] = distance*distance;
            sum_distances += model->mbasic_.dis[i-loc*prior->members];
        }
        model->mbasic_.costX = sum_distances;
        model->mbasic_.oversample = prior->oversample;
        model->mbasic_.members = prior->members;
        //std::cout<<"ProcessDistanceSum: model dis number "<<model->mbasic_.dis.size()<<" dis:"<<std::endl;
        //for(size_t i = 0; i < model->mbasic_.dis.size();++i)
        //    std::cout<<model->mbasic_.dis[i]<<" ";
        //std::cout<<std::endl;
        return model;
    }
    void Kmeans_Parallel::SampleC(Model *model, std::vector<int> centers, size_t loc){
        struct timeval tv;
        gettimeofday(&tv,NULL);
        srand(tv.tv_sec + tv.tv_usec);
        //srand((unsigned)time(NULL));
        for(size_t i = 0; i < model->dis.size(); i++){
            double px = model->oversample*model->dis[i]/model->costX;
            double r = rand()/double(RAND_MAX);
            //std::cout<<"px="<<px<<" r="<<r<<std::endl;
            std::vector<int>::iterator f = find(centers.begin(),centers.end(),loc*model->members+i);
            if((px >= r) && (f == centers.end())){
                model->centers.push_back(loc*model->members+i);
                //std::cout<<"px="<<px<<" r="<<r<<" i="<<loc*model->members+i<<std::endl;
            }
        }
    }
    std::vector<int> Kmeans_Parallel::SetWeight(Flexible_vector *records, Model *prior, size_t loc){
        std::cout<<"process"<<loc<<"SetWeight: prior centers size:"<<prior->centers.size()<<std::endl;
        //std::cout<<"process"<<loc<<"SetWeight: prior centers:"<<std::endl;
        //for(size_t i = 0; i < prior->centers.size();++i)
        //    std::cout<<prior->centers[i]<<" ";
        //std::cout<<std::endl;
        size_t l = prior->centers.size();
        std::vector<int> weights;
        weights.resize(l);
        size_t lim = std::min(loc*prior->members+prior->members,(size_t)records->prob.l);
        //std::cout<<"ProcessDistanceSum: processor"<<loc<<"  lim="<<lim<<" dis size="<<lim-loc*prior->members<<std::endl;
        std::vector<node> record_problem;
        std::vector<node> center;
        //double sum_distances = 0.0;
        //model->mbasic_.dis.resize(lim-loc*prior->members);
        //std::cout<<"ProcessDistanceSum: processor"<<loc<<"  lim="<<lim<<" dis size="<<model->mbasic_.dis.size()<<std::endl;
        //std::cout<<"ProcessDistanceSum: Initial dis:"<<std::endl;
        //for(size_t i = 0; i < model->mbasic_.dis.size();++i)
        //    std::cout<<model->mbasic_.dis[i]<<" ";
        //std::cout<<std::endl;
        for(size_t i = loc*prior->members;i < lim;i++){
            std::vector<int>::iterator f = find(prior->centers.begin(),prior->centers.end(),i);
            if(f != prior->centers.end()){
                int counter = 0;
                while(prior->centers[counter] != i)
                    counter++;
                weights[counter]++;
                //std::cout<<"find"<<i<<" ";
                continue;
            }
            record_problem = records->get_value_without_label(i);
            double distance = (1L<<16)-1;
            int center_num;
            //std::cout<<"distance = "<<distance<<std::endl;
            for(size_t j = 0;j < prior->centers.size();j++){
                center = records->get_value_without_label(prior->centers[j]);
                double distance_tmp = CountEuclideanDistance(record_problem,center);
                //std::cout<<"distance_tmp = "<<distance_tmp<<std::endl;
                if(distance_tmp < distance){
                    distance = distance_tmp;
                    center_num = j;
                }
            }
            weights[center_num]++;
            //sum_distances += model->mbasic_.dis[i-loc*prior->members];
        }
        //std::cout<<"ProcessDistanceSum: model dis number "<<model->mbasic_.dis.size()<<" dis:"<<std::endl;
        //for(size_t i = 0; i < model->mbasic_.dis.size();++i)
        //    std::cout<<model->mbasic_.dis[i]<<" ";
        //std::cout<<std::endl;
        return weights;
    }
    
    //test: output the centers in model
    void Kmeans_Parallel::OutputModel(){
        std::cout<<"center number: "<<this->mbasic_.centers.size()<<" "<<std::endl;
        for(size_t i = 0; i < this->mbasic_.centers.size();++i){
            std::cout<<this->mbasic_.centers[i]<<" ";
        }
        std::cout<<std::endl;
        //std::cout<<"oversample: "<<this->mbasic_.oversample<<std::endl;
        //std::cout<<"members: "<<this->mbasic_.members<<std::endl;
    }
}
