/*
ID: neo-white 
LANG: C++
TASK: Kmeans_PlusPlus.cpp
*/

#include "Kmeans_PlusPlus.h"
#include "../Core/Core.h"
#include <boost/serialization/vector.hpp>
#include <boost/serialization/queue.hpp>
#include <boost/serialization/export.hpp> 
#include <vector>
#include <queue>
#include <fstream>  
#include <time.h>

//BOOST_CLASS_EXPORT(kmeans_plusplus::Kmeans_PlusPlus)
namespace kmeans_plusplus{
        
    //Flexible_vector *kmeans_plusplus(Flexible_vector *problem,int k)
    std::vector<int> Kmeans_PlusPlus::kmeans_plusplus(Flexible_vector *problem,int k)
    {
    //	fprintf(stderr,"In kmeans_Lloyd \n");
        int l = problem->prob.l;
        int n_clusters = k;
        int max_index = problem->prob.max_feature;

    
        //choose initial center uniformly at random 
        std::vector<int> queue;
        queue.resize(l);
        std::fill(queue.begin(),queue.end(), 0);
        srand((unsigned)time(NULL));
        int index = (rand()%l);
        queue[index] = 1;
        int iter_counter = 0;
        std::cout<<"kmeans++:iteration "<< iter_counter<<". candidated center "<<index<<std::endl;
        
        //choose remaining centers
        int remainingInstances = l - 1;
        std::vector<double> distances;
        distances.resize(l);
        std::vector<double> cumProbs;
        cumProbs.resize(l);
        std::vector<double> weights;
        weights.resize(l);
        int ii,jj,kk;
        if(n_clusters > 1)
        {
            //proceed with selecting the rest
            //distances to the initial randomly chose center
            for(ii=0;ii<l;ii++)
                distances[ii] = euclidean_distance(problem->get_value_without_label(ii),problem->get_value_without_label(index));
            
            int center_candidate;
            //now choose the remaining cluster centers
            for(jj = 1; jj < n_clusters; jj++)
            {
                bool find = false;
                //distances converted to probabilities
                double sum_distances = 0;
                for(kk=0;kk<l;kk++)
                    sum_distances += distances[kk];
                for(kk=0;kk<l;kk++)
                    weights[kk] = distances[kk]/sum_distances;
                double sum_probs = 0;
                for(kk=0;kk<l;kk++)
                {
                    sum_probs += weights[kk];
                    cumProbs[kk] = sum_probs;
                }
                cumProbs[l-1] = 1.0;  //make sure there are no rounding issues
                //choose a random instance
                double prob = (double) rand()/RAND_MAX;  //random between 0 and 1
                for(ii=0;ii<l;ii++)
                {
                    if(prob < cumProbs[ii])
                    {
                        if(queue[ii] != 1)
                        {
                            queue[ii] = 1;
                            center_candidate = ii;
                            std::cout<<"kmeans++:iteration "<<iter_counter+1<<". candidated center "<<ii<<".prob "<<prob<<std::endl;
                            remainingInstances--;
                            find = true;
                            break;
                        }
                    }
                }
                iter_counter++;

                if(find == false)
                {
                    std::cout<<"kmeans++:iteration "<<iter_counter+1<<". No find center. prob "<<prob<<std::endl;
                    break;
                }
    
                if(remainingInstances == 0)
                    break;
    
                //prepares to choose the next cluster center.
                //check distances against the new cluster center to see if it is closer
                double newDist;
                for(ii = 0;ii < l;ii++)
                {
                    if(distances[ii] > 0)
                    {
                        newDist = euclidean_distance(problem->get_value_without_label(ii),problem->get_value_without_label(center_candidate));
                        if(newDist < distances[ii])
                        {
                            distances[ii] = newDist;
                        }
                    }
                }
            }
        }
        return queue;
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
*/    
    double Kmeans_PlusPlus::euclidean_distance(std::vector<node> px,std::vector<node> py)
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
