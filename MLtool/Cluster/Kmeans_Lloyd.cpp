/*
ID: neo-white 
LANG: C++
TASK: Kmeans_lloyd.cpp
*/

#include "Kmeans_Lloyd.h"
#include "../Core/Core.h"
#include <boost/serialization/vector.hpp>
#include <boost/serialization/queue.hpp>
#include <boost/serialization/export.hpp> 
#include <vector>
#include <queue>
#include <fstream>  
#include <time.h>

BOOST_CLASS_EXPORT(kmeans_lloyd::Kmeans_Lloyd)
namespace kmeans_lloyd{

    Kmeans_Lloyd* Kmeans_Lloyd::operator +(Kmeans_Lloyd &rhs)
    {
        Kmeans_Lloyd *pret = new Kmeans_Lloyd();
        pret->fvkl.init(fvkl.prob.l, fvkl.prob.max_feature);
        for(size_t i = 0; i< fvkl.prob.l ; ++i)
        {
            std::vector<node> tmp;
            
            if(rhs.fvkl.prob.y[i] != -1)
            {
                tmp = rhs.fvkl.get_value_without_label(i);
                pret->fvkl.add(tmp,i);
            }
            if(fvkl.prob.y[i] != -1)
            {
                tmp = fvkl.get_value_without_label(i);
                pret->fvkl.add(tmp,i);
            }
            if(pret->fvkl.prob.y[i] == 2)
               pret->fvkl.prob.y[i] = fvkl.prob.y[i] + rhs.fvkl.prob.y[i];
            else if(pret->fvkl.prob.y[i] == 1)
            {
                if(rhs.fvkl.prob.y[i] != -1)
                    pret->fvkl.prob.y[i] = rhs.fvkl.prob.y[i];
                else
                    pret->fvkl.prob.y[i] = fvkl.prob.y[i] ;
            }
        }
        return pret;
    }
    
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
    void Kmeans_Lloyd::beginDataScan(Flexible_vector *records, size_t n_cluster)
    {
        fvkl.init();
        size_t n_records = records->prob.l;
        std::vector<int> q = init_random(n_records,n_cluster);
        std::cout<<"Initial centroid: ";
        for (size_t i = 0; i < q.size(); ++i)
        {
            if(q[i]==1)
            {
                std::cout<<i<<",";
                fvkl.insert_end(*records,i);
            }
        }
    }
    
    
    void Kmeans_Lloyd::endDataScan()
    {
        for(size_t i = 0; i < this->fvkl.prob.l; ++i)
        {
            for(size_t j = 0; j < this->fvkl.prob.x_interval[i]; ++j)
            {
                this->fvkl.prob.x[this->fvkl.prob.x_ptr[i]+j].value /= this->fvkl.prob.y[i];
            }
            //data[i*ncol + ncol-1] = 1;
        } 
    }
    
    Kmeans_Lloyd* Kmeans_Lloyd::processRecord(Flexible_vector *records, size_t feat_dim)
    {
        Kmeans_Lloyd *pret = new Kmeans_Lloyd();
        pret->fvkl.init(this->fvkl.prob.l, this->fvkl.prob.max_feature);
        //std::cout<<"pret->fvkl.y = ";
        //for(size_t j = 0; j< pret->fvkl.prob.y.size();j++)
        //    std::cout<<pret->fvkl.prob.y[j]<<" ";
        //std::cout<<std::endl;
        //std::cout<<"pret->fvkl.x_ptr.size= "<< pret->fvkl.prob.x_ptr.size()<<std::endl;
        //for(size_t jj = 0; jj< pret->fvkl.prob.x_ptr.size();jj++)
        //    std::cout<<pret->fvkl.prob.x_ptr[jj]<<" ";
        //std::cout<<std::endl;
        //std::cout<<"l= "<<pret->fvkl.prob.l<<" max_feature= "<< pret->fvkl.prob.max_feature;
        //std::cout<<"After init in processRecord "<<std::endl;
        size_t n_records = records->prob.l;
        for (size_t i = 0 ; i < n_records ; i++)
        {
            std::vector<node> record;
            record.clear();
            record = records->get_value_without_label(i);
            
            double dis = (1L<<16)-1;
            int who = -1;
            //std::cout<<"bp"<<std::endl;
            for(size_t k = 0; k < this->fvkl.prob.l; k++)
            {
                std::vector<node> centroid = this->fvkl.get_value_without_label(k);
                double tmp = euclidean_distance(record,centroid);
    
                if (tmp < dis){
                    who = k;
                    dis = tmp;
                }
            }
            pret->fvkl.add(record, who);
        }
        return pret;
    }
    
    bool Kmeans_Lloyd::isConverged(Kmeans_Lloyd* rhs, size_t feat_dim, double eps)
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
    
    void Kmeans_Lloyd::finish(Flexible_vector *records, size_t feat_dim)
    {
        std::ofstream file;
        file.open("./result.txt");	
        size_t n_records = records->prob.l;
        for(size_t i = 0; i < n_records; ++i)
        {
            std::vector<node> record;
            record.clear();
            record = records->get_value_without_label(i);
            double dis = (1<<16)-1;
            int who = -1;
            for(size_t k = 0; k < this->fvkl.prob.l; k++)
            {
                std::vector<node> centroid = this->fvkl.get_value_without_label(k);
                double tmp = euclidean_distance(record,centroid);
    
                if (tmp < dis){
                    who = k;
                    dis = tmp;
                }
            }
            file << who << std::endl;
        }
    }
    double Kmeans_Lloyd::euclidean_distance(std::vector<node> px,std::vector<node> py)
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
}
