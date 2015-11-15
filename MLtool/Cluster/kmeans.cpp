#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <float.h>
#include <string.h>
#include <stdarg.h>
#include <limits.h>
#include <locale.h>
#include <time.h>
#include "mpi.h"

#include "kmeans-pmlp.h"
int kmeans_version = KMEANS_PMLP_VERSION;
typedef float Qfloat;
typedef signed char schar;
#ifndef min
template <class T> static inline T min(T x,T y) { return (x<y)?x:y; }
#endif
#ifndef max
template <class T> static inline T max(T x,T y) { return (x>y)?x:y; }
#endif
template <class T> static inline void swap(T& x, T& y) { T t=x; x=y; y=t; }
template <class S, class T> static inline void clone(T*& dst, S* src, int n)
{
	dst = new T[n];
	memcpy((void *)dst,(void *)src,sizeof(T)*n);
}
static inline double powi(double base, int times)
{
	double tmp = base, ret = 1.0;

	for(int t=times; t>0; t/=2)
	{
		if(t%2==1) ret*=tmp;
		tmp = tmp * tmp;
	}
	return ret;
}
#define INF HUGE_VAL
#define TAU 1e-12
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

static void print_string_stdout(const char *s)
{
	fputs(s,stdout);
	fflush(stdout);
}
static void (*kmeans_print_string) (const char *) = &print_string_stdout;
#if 1
static void info(const char *fmt,...)
{
	char buf[BUFSIZ];
	va_list ap;
	va_start(ap,fmt);
	vsprintf(buf,fmt,ap);
	va_end(ap);
	(*kmeans_print_string)(buf);
}
#else
static void info(const char *fmt,...) {}
#endif



double euclidean_distance(cluster_node *px,cluster_node *py)
{
	double sum = 0;
	double distances = 0;
	while(px->index != -1 && py->index != -1)
	{
		if(px->index == py->index)
		{
			sum += (px->value-py->value)*(px->value-py->value);
			++px;
			++py;
		}
		else
		{
			if(px->index > py->index)
				++py;
			else
				++px;
		}			
	}
    distances = sqrt(sum);
	return distances;
}

bool check_match(int *a,int *b,int length)
{
    for(int i=0;i<length;i++)
    {
        if(a[i] != b[i])
            return false;
    }
    return true;
}

//add sample px and py, result is px
void add_sample(cluster_node *px,cluster_node *py)
{
	while(px->index != -1 && py->index != -1)
	{
		if(px->index == py->index)
		{
			px->value += py->value;
			++px;
			++py;
		}
		else
		{
			if(px->index > py->index)
				++py;
			else
				++px;
		}			
	}
    return;
}
        
//average sample px with n
void average_sample(cluster_node *px,int n)
{
    if(n == 0)
    {
        printf("\ndivider can't be 0 in average_sample function!\n");
        return;
    }
    else
    {
        while(px->index != -1)
	    {
			px->value /= n;
			++px;
		}
	}
    return;
}
        
//copy sample src to sample dst
void copy_sample(cluster_node *dst,cluster_node *src)
{
	while(dst->index != -1 && src->index != -1)
	{
		if(dst->index == src->index)
		{
			dst->value = src->value;
			++dst;
			++src;
		}
		else
		{
			if(dst->index > src->index)
				++src;
			else
				++dst;
		}			
	}
    return;
}
        
//initialize sample dst using sample src
void init_sample(cluster_node *dst,cluster_node *src)
{
	while(src->index != -1)
	{
        dst->index = src->index;
	    dst->value = src->value;
		++dst;
		++src;
	}
    dst->index = -1;
    return;
}

void clustering_Lloyd(cluster_problem *problem,kmeans_model *model,cluster_node **cur_center)
{
    int l = problem->l;
    int n_clusters = model->n_clusters;
    cluster_node *kmeans_order = Malloc(cluster_node,l);
    model->kmeans_order = Malloc(int,l);
    double min_cost;
    double cost_inst;
    double total_cost = 0;

    fprintf(stderr,"In clustering_Lloyd\n");
    
    for(int i=0;i<l;i++)
    {
        min_cost = INF;
        for(int j=0;j<n_clusters;j++)
        {
            cost_inst = euclidean_distance(problem->x[i],cur_center[j]);
            if(cost_inst < min_cost)
            {
                kmeans_order[i].index = j;
                kmeans_order[i].value = cost_inst;
                min_cost = cost_inst;
            }
        }
        total_cost += min_cost*min_cost;    //kmeans
        //total_cost += min_cost;    //kmedians
    }

    model->total_cost = total_cost;
    for(int k=0;k<l;k++)
    {
        model->kmeans_order[k] = kmeans_order[k].index;

    }
    return;
}

void clustering_Lloyd_mpi(cluster_problem *problem,kmeans_model *model,cluster_node **cur_center)
{
    int l = problem->l;
    int n_clusters = model->n_clusters;
    cluster_node *kmeans_order = Malloc(cluster_node,l);
    int *kmeans_order_tmp = Malloc(int,l);
    model->kmeans_order = Malloc(int,l);
    double min_cost;
    double cost_inst;
    double total_cost = 0;
    double total_cost_tmp = 0;
    int i,j,k;

    int myid, numprocs;
    MPI_Comm_rank(MPI_COMM_WORLD,&myid);
    MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
    int block = l/numprocs;

    fprintf(stderr,"Thread %d: In clustering_Lloyd_mpi\n",myid);

    /*seq version. change to mpi version
    for(int i=0;i<l;i++)
    {
        min_cost = INF;
        for(int j=0;j<n_clusters;j++)
        {
            cost_inst = euclidean_distance(problem->x[i],cur_center[j]);
            if(cost_inst < min_cost)
            {
                kmeans_order[i].index = j;
                kmeans_order[i].value = cost_inst;
                min_cost = cost_inst;
            }
        }
        total_cost += min_cost*min_cost;    //kmeans
        //total_cost += min_cost;    //kmedians
    }
    */
    if(myid != numprocs-1)
    {
        for(i=myid*block;i<(myid+1)*block;i++)
        {
            min_cost = INF;
            for(j=0;j<n_clusters;j++)
            {
                cost_inst = euclidean_distance(problem->x[i],cur_center[j]);
                if(cost_inst < min_cost)
                {
                    kmeans_order[i].index = j;
                    kmeans_order[i].value = cost_inst;
                    min_cost = cost_inst;
                }
            }
            total_cost_tmp += min_cost*min_cost;    //kmeans
            //total_cost += min_cost*min_cost;    //kmeans
            //total_cost += min_cost;    //kmedians
        }
    }
    else
    {
        for(i=myid*block;i<l;i++)
        {
            min_cost = INF;
            for(j=0;j<n_clusters;j++)
            {
                cost_inst = euclidean_distance(problem->x[i],cur_center[j]);
                if(cost_inst < min_cost)
                {
                    kmeans_order[i].index = j;
                    kmeans_order[i].value = cost_inst;
                    min_cost = cost_inst;
                }
            }
            total_cost_tmp += min_cost*min_cost;    //kmeans
            //total_cost += min_cost*min_cost;    //kmeans
            //total_cost += min_cost;    //kmedians
        }
    }
    
    MPI_Allreduce(&total_cost_tmp,&total_cost,1,MPI_DOUBLE,MPI_SUM,MPI_COMM_WORLD);
    model->total_cost = total_cost;

    for(i = 0;i<l;i++)
    {
        kmeans_order_tmp[i] = 0;
    }

    if(myid != numprocs-1)
    {
        for(k=myid*block;k<(myid+1)*block;k++)
        {
            kmeans_order_tmp[k] = kmeans_order[k].index;
        }
    }
    else
    {
        for(k=myid*block;k<l;k++)
        {
            kmeans_order_tmp[k] = kmeans_order[k].index;
        }
    }
    MPI_Allreduce(kmeans_order_tmp,model->kmeans_order,l,MPI_INT,MPI_SUM,MPI_COMM_WORLD);
    return;
}

void clustering_new_center(cluster_problem *problem,kmeans_model *model,cluster_node **cur_center)
{
    int n_clusters = model->n_clusters;
    int max_index = model->max_index;
    int l = problem->l;
    fprintf(stderr,"in clustering_new_center!n_clusters = %d,max_index=%d \n",n_clusters,max_index);            
    //cluster_node **cur_center = Malloc(cluster_node *,param->n_clusters); 
    cluster_node *tmp_center_space = Malloc(cluster_node,n_clusters*(max_index+1));
    int i,j;
    cluster_node **tmp_center = Malloc(cluster_node*,n_clusters);
    fprintf(stderr,"in clustering_new_center!n_clusters = %d \n",n_clusters);            
    //cluster_node **tmp_center;
    int num_sample[n_clusters];
    for(i=0;i<n_clusters;i++)
        num_sample[i] = 0;
    int tmp_index;
    //sum
    for(i=0;i<l;i++)   
    {
        tmp_index = model->kmeans_order[i];
        if(num_sample[tmp_index] == 0)     //The sample number of this cluster is 0, tmp_center[tmp_index] is null
        {
            //tmp_center[tmp_index] = Malloc(cluster_node,max_index+1);
            tmp_center[tmp_index] = &tmp_center_space[tmp_index*(max_index+1)];
            init_sample(tmp_center[tmp_index],problem->x[i]);
            num_sample[tmp_index]++;
        }
        else
        {
            add_sample(tmp_center[tmp_index],problem->x[i]);
            num_sample[tmp_index]++;
        }

    }
    //average
    for(i=0;i<n_clusters;i++)
    {
        if(num_sample[i] == 0)
        {
            printf("cluster %d is null!\n",i);
        }
        else
        {
            average_sample(tmp_center[i],num_sample[i]);
        }
    }
    //copy tmp_center to cur_center
    for(i=0;i<n_clusters;i++)
    {
        copy_sample(cur_center[i],tmp_center[i]);
    }
 
    free(tmp_center);
    free(tmp_center_space);
    //delete[] num_sample;
    return;

}

void clustering_first_medoids(cluster_problem *problem,kmeans_model *model,int *medoids)
{
	//fprintf(stderr,"In clustering_first_medoids\n");
    //compute the belonging of each data point according to current medoids centers, and euclidean distances
    int l = problem->l;
    int n_medoids = 0;
    int i;
    int j;
    
    struct cluster_node *min_distances;                 //record which medoids non-medoids belong to 
    min_distances = Malloc(struct cluster_node,l);
    //initialization
    for(i=0;i<l;i++)
    {
        min_distances[i].index = -1;
        min_distances[i].value = INF;
        if(medoids[i] == 1)
            n_medoids++;
    }
    int medoids_index[n_medoids];
    int k=0;
    for(j=0;j<l;j++)
        if(medoids[j] == 1)
            medoids_index[k++] = j;
    
    //count distances of each pair of medoids and non-medoids
	//fprintf(stderr,"Before distances in clustering_first_medoids\n");
    struct cluster_node **distances;                    //distances matrix between non-medoids and medoids
    distances = Malloc(struct cluster_node *,l);
    struct cluster_node *distances_space;               //same with distances
    distances_space = Malloc(struct cluster_node,l*n_medoids);
    j=0;
    for(i=0;i<l;i++)
    {
        distances[i] = &distances_space[j];
        if(medoids[i] == 1)            //index i is medoid
        {
            distances_space[j].index = -1;
            distances_space[j].value = 0;
            min_distances[i].index = -1;
            min_distances[i].value = 0;
            j++;
        }
        else
        {
            for(k=0;k<n_medoids;k++)
            {
                distances_space[j].index = medoids_index[k];
                distances_space[j].value = euclidean_distance(problem->x[i],problem->x[medoids_index[k]]);
                if(distances_space[j].value <= min_distances[i].value)
                {
                    min_distances[i].index = distances_space[j].index;
                    min_distances[i].value = distances_space[j].value;
                }
                j++;
            }
        }
    }

    //build kmeans model
    model->n_clusters = n_medoids;
    model->total_cost = 0;
    model->kmeans_label = Malloc(double,l);
    
    for(i=0;i<l;i++)
    {
        if(min_distances[i].index == -1)
        {
            model->kmeans_label[i] = problem->y[i];
        }
        else
        {
            model->kmeans_label[i] = problem->y[min_distances[i].index];
            model->total_cost += min_distances[i].value;
        }
    }

    free(distances);
    free(distances_space);
    free(min_distances);
}

//
// A solution package for parallelization, include serial type and mpi.
//
class Solver {
public:
	Solver() {};
	virtual ~Solver() {};


    kmeans_model *kmeans_fit_seq(cluster_problem *prob, kmeans_parameter *param);
    kmeans_model *kmeans_fit_mpi(cluster_problem *prob, kmeans_parameter *param);

protected:
    int * init_user_certain(int l);
    int * init_random(int length, int random_seed);
    void init_kmeans_plusplus(cluster_problem *problem,kmeans_parameter *param,int *center_init);
    void kmedoids_pam_first_medoids(cluster_problem *problem,kmeans_parameter *param,int *center_init,kmeans_model *model);
    void kmeans_Lloyd(cluster_problem *problem,kmeans_parameter *param,int *center_init,kmeans_model *model);
    void kmeans_Lloyd_mpi(cluster_problem *problem,kmeans_parameter *param,int *center_init,kmeans_model *model);
private:
};

int * Solver::init_user_certain(int length)
{
    int tmp;
    int * queue = Malloc(int, length);
    for(int i=0;i<length;i++)
    {
        queue[i] = 0;
    }

    //certain the init center for test
    queue[25] = 1;
    queue[79] = 1;
    queue[136] = 1;
    
    return queue;
}

int * Solver::init_random(int length, int random_seed)
{
    int tmp;
    int * queue = Malloc(int, length);
    for(int i=0;i<length;i++)
    {
        queue[i] = 0;
    }
   
    srand((unsigned)time(NULL));
    for(int j=0;j<random_seed;j++)
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

void Solver::init_kmeans_plusplus(cluster_problem *problem,kmeans_parameter *param,int *center_init)
{
//	fprintf(stderr,"In kmeans_Lloyd \n");
    int l = problem->l;
    int ii,jj,kk;
    int k = 0;
    int n_clusters = param->n_clusters;
    int max_index = param->max_index;

    //choose initial center uniformly at random 
    int *queue = Malloc(int,l);
    for(ii=0;ii<l;ii++)
        queue[ii] = 0;

    srand((unsigned)time(NULL));
    int index = (rand()%l);
    queue[index] = 1;
    int iter_counter = 0;
    printf("kmeans++:iteration %d. candidated center %d randomly.\n",iter_counter,ii);
    int remainingInstances = l - 1;
    double * distances = new double[l];
    double * cumProbs = new double[l];
    double *weights = new double[l];
    if(n_clusters > 1)
    {
        //proceed with selecting the rest

        //distances to the initial randomly chose center
        for(ii=0;ii<l;ii++)
            distances[ii] = euclidean_distance(problem->x[ii],problem->x[iter_counter]);

        int center_candidate;
        //now choose the remaining cluster centers
        for(jj = 1; jj < n_clusters; jj++)
        {
            //distances converted to probabilities
            double sum_distances = 0;
            for(kk=0;kk<l;kk++)
            {
                weights[kk] = distances[kk];
                sum_distances += weights[kk];
            }
            for(kk=0;kk<l;kk++)
            {
                weights[kk] /= sum_distances;
            }
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
                        printf("kmeans++:iteration %d. candidated center %d.prob %f.\n",iter_counter+1,ii,prob);
                    }else
                    {
                        printf("In an error place!\nprob %f!",prob);
                    }
                    remainingInstances--;
                    break;
                }
            }
            iter_counter++;

            if(remainingInstances == 0)
                break;

            //prepares to choose the next cluster center.
            //check distances against the new cluster center to see if it is closer
            double newDist;
            for(ii = 0;ii < l;ii++)
            {
                if(distances[ii] > 0)
                {
                    newDist = euclidean_distance(problem->x[ii],problem->x[center_candidate]);
                    if(newDist < distances[ii])
                    {
                        distances[ii] = newDist;
                    }
                }
            }
        }
    }
    for(int i = 0; i < l;i++)
        center_init[i] = queue[i];

    free(queue);
    delete[] distances;
    delete[] cumProbs;
    delete[] weights;
}

void Solver::kmedoids_pam_first_medoids(cluster_problem *problem,kmeans_parameter *param,int *center_init,kmeans_model *model)
{
	fprintf(stderr,"In kmedoids_pam_first_medoids\n");
    int l = problem->l;
    int ii,jj;
    int *cur_medoids = Malloc(int,l); 
	fprintf(stderr,"Before clustering_first_medoids\n");
    //clone(cur_medoids,center_init,l);
    for(ii=0;ii<l;ii++)
        cur_medoids[ii] = center_init[ii];
    clustering_first_medoids(problem,model,cur_medoids);
	fprintf(stderr,"After clustering_first_medoids\n");
    double total_cost = model->total_cost;
    int *old_medoids = Malloc(int,l);
    int *best_medoids = Malloc(int,l);
    int *tmp_medoids = Malloc(int,l);
    int max_iter = param->max_iter;
    for(int k=0;k<l;k++)
        old_medoids[k] = 0;

    int iter_counter = 1;
    //stop if not improvement
    while(!check_match(old_medoids,cur_medoids,l))
    {
        printf("iteration counter : %d\n",iter_counter);
        if(iter_counter > max_iter)
        {
            printf("has reach max iteration!\nSearch is over!\n");
            break;
        }
        else
            iter_counter++;
        //memcpy(old_medoids,cur_medoids,l);
        //memcpy(tmp_medoids,cur_medoids,l);
        for(ii=0;ii<l;ii++)
        {
            old_medoids[ii] = cur_medoids[ii];
            tmp_medoids[ii] = cur_medoids[ii];
        }

        //iterate over all medoids and non-medoids
        for(int i=0;i<l;i++)
            for(int j=0;j<l;j++)
            {
                if((i!=j)&&(cur_medoids[j]==1)&&(cur_medoids[i]==0))
                {
                    tmp_medoids[j] = 0;
                    tmp_medoids[i] = 1;
                    
                    clustering_first_medoids(problem,model,tmp_medoids);
                    //tmp_cost = total_cost_first_medoids(problem,tmp_medoids);
                    if(total_cost > model->total_cost)
                    {
                        //memcpy(best_medoids,tmp_medoids,l);
                        for(ii=0;ii<l;ii++)
                            best_medoids[ii] = tmp_medoids[ii];
                        total_cost = model->total_cost;
                    }
                    tmp_medoids[j] = 1;
                    tmp_medoids[i] = 0;
                }
            }
        //memcpy(cur_medoids,best_medoids,l);
        for(ii=0;ii<l;ii++)
            cur_medoids[ii] = best_medoids[ii];
        printf("current total_cost : %f\n", total_cost);
    }
    model->iter_counter = iter_counter;
    model->center_medoids = Malloc(int,l);
    //memcpy(model->center,cur_medoids,l);
    for(ii=0;ii<l;ii++)
        model->center_medoids[ii] = cur_medoids[ii];
    //for(ii=0;ii<l;ii++)
    //    printf("center[%d] = %d  cur_medoids[%d] = %d\n",ii,model->center[ii],ii,cur_medoids[ii]);


    
    free(cur_medoids);
    free(old_medoids);
    free(best_medoids);
    free(tmp_medoids);
}


void Solver::kmeans_Lloyd(cluster_problem *problem,kmeans_parameter *param,int *center_init,kmeans_model *model)
{
	fprintf(stderr,"In kmeans_Lloyd \n");
    int l = problem->l;
    int ii,jj;
    int k = 0;
    int n_clusters = param->n_clusters;
    int max_index = param->max_index;
    model->max_index = param->max_index;
    cluster_node **cur_center = Malloc(cluster_node *,param->n_clusters); 
    cluster_node *cur_center_space = Malloc(cluster_node,n_clusters*(max_index+1));
	//fprintf(stderr,"Before clustering_first_medoids\n");
    //clone(cur_medoids,center_init,l);
    //initialize begining center
    jj = 0;
    int kk = 0;
    for(ii=0;ii<l;ii++)
    {        
        if( center_init[ii] == 1)
        {
            cur_center[jj] = &cur_center_space[kk];
            k = 0;
            while(1)
            {
                if(problem->x[ii][k].index != -1)
                {
                    cur_center_space[kk].index = problem->x[ii][k].index;
                    cur_center_space[kk].value = problem->x[ii][k].value;
                    k++;
                    kk++;
                }
                else
                {
                    cur_center_space[kk].index = -1;
                    kk++;
                    break;
                }
            }
            jj++;
        }
    }
	//fprintf(stderr,"After clustering_first_medoids\n");
    double total_cost = INF; 
    double differ = INF; 
    //int *old_medoids = Malloc(int,l);
    //int *best_medoids = Malloc(int,l);
    //int *tmp_medoids = Malloc(int,l);
    int max_iter = param->max_iter;
    //for(int k=0;k<l;k++)
    //    old_medoids[k] = 0;

    int iter_counter = 1;
    //stop if not improvement
    while(1)
    {
        printf("iteration counter : %d\n",iter_counter);
        if(iter_counter > max_iter)
        {
            printf("has reach max iteration!\nSearch is over!\n");
            break;
        }
        else
            iter_counter++;
        fprintf(stderr,"before clustering_Lloyd in kmeans_Lloyd\n");            
        clustering_Lloyd(problem,model,cur_center);
        differ = fabs(total_cost - model->total_cost);
        fprintf(stderr,"after clustering_Lloyd and differ in kmeans_Lloyd\n");            
        
        if(differ < TAU)
        {
            printf("current differ: %f\niteration is over!\n", differ);
            break;
        }
        
        total_cost = model->total_cost;
        fprintf(stderr,"before clustering_new_center in kmeans_Lloyd\n");            
        clustering_new_center(problem,model,cur_center);

        printf("current differ: %f\n", differ);
    }

    //build model

    model->iter_counter = iter_counter;
    model->center_count = Malloc(cluster_node *,param->n_clusters+1); 
    for(ii=0;ii<n_clusters;ii++)
    {
        model->center_count[ii] = Malloc(cluster_node,max_index+1);
        jj=0;
        while(1)
        {
            if(cur_center[ii][jj].index != -1)
            {
                model->center_count[ii][jj].index = cur_center[ii][jj].index;
                model->center_count[ii][jj].value = cur_center[ii][jj].value;
                jj++;
            }
            else
            {
                model->center_count[ii][jj].index = -1;
                break;
            }

        }
    }
    //model->center = Malloc(int,l);
    //memcpy(model->center,cur_medoids,l);
    //for(ii=0;ii<l;ii++)
    //    model->center[ii] = cur_medoids[ii];
    //for(ii=0;ii<l;ii++)
    //    printf("center[%d] = %d  cur_medoids[%d] = %d\n",ii,model->center[ii],ii,cur_medoids[ii]);


    
    free(cur_center);
    free(cur_center_space);
}

void Solver::kmeans_Lloyd_mpi(cluster_problem *problem,kmeans_parameter *param,int *center_init,kmeans_model *model)
{
    
    int myid, numprocs;
    MPI_Comm_rank(MPI_COMM_WORLD,&myid);
    MPI_Comm_size(MPI_COMM_WORLD,&numprocs);
	fprintf(stderr,"In kmeans_Lloyd_mpi in thread %d\n",myid);
    int l = problem->l;
    int ii,jj;
    int k = 0;
    int n_clusters = param->n_clusters;
    int max_index = param->max_index;
    model->max_index = param->max_index;
    cluster_node **cur_center = Malloc(cluster_node *,param->n_clusters); 
    cluster_node *cur_center_space = Malloc(cluster_node,n_clusters*(max_index+1));
	//fprintf(stderr,"Before clustering_first_medoids\n");
    //clone(cur_medoids,center_init,l);
    //initialize begining center
    jj = 0;
    int kk = 0;
    for(ii=0;ii<l;ii++)
    {        
        if( center_init[ii] == 1)
        {
            cur_center[jj] = &cur_center_space[kk];
            k = 0;
            while(1)
            {
                if(problem->x[ii][k].index != -1)
                {
                    cur_center_space[kk].index = problem->x[ii][k].index;
                    cur_center_space[kk].value = problem->x[ii][k].value;
                    k++;
                    kk++;
                }
                else
                {
                    cur_center_space[kk].index = -1;
                    kk++;
                    break;
                }
            }
            jj++;
        }
    }
	//fprintf(stderr,"After clustering_first_medoids\n");
    double total_cost = INF; 
    double differ = INF; 
    //int *old_medoids = Malloc(int,l);
    //int *best_medoids = Malloc(int,l);
    //int *tmp_medoids = Malloc(int,l);
    int max_iter = param->max_iter;
    //for(int k=0;k<l;k++)
    //    old_medoids[k] = 0;

    int iter_counter = 1;
    //stop if not improvement
    while(1)
    {
        printf("iteration counter : %d in thread %d\n",iter_counter,myid);
        if(iter_counter > max_iter)
        {
            printf("Thread %d has reach max iteration!\nSearch is over!\n",myid);
            break;
        }
        else
            iter_counter++;
        fprintf(stderr,"Thread %d: before clustering_Lloyd in kmeans_Lloyd\n",myid);            
        //clustering_Lloyd(problem,model,cur_center);
        clustering_Lloyd_mpi(problem,model,cur_center);
        differ = fabs(total_cost - model->total_cost);
        fprintf(stderr,"Thread %d : after clustering_Lloyd and differ in kmeans_Lloyd\n",myid);            
        
        if(differ < TAU)
        {
            printf("current differ: %f\niteration is over!\n", differ);
            break;
        }
        
        total_cost = model->total_cost;
        fprintf(stderr,"before clustering_new_center in kmeans_Lloyd\n");            
        clustering_new_center(problem,model,cur_center);

        printf("current differ: %f\n", differ);
    }

    //build model

    model->iter_counter = iter_counter;
    model->center_count = Malloc(cluster_node *,param->n_clusters+1); 
    for(ii=0;ii<n_clusters;ii++)
    {
        model->center_count[ii] = Malloc(cluster_node,max_index+1);
        jj=0;
        while(1)
        {
            if(cur_center[ii][jj].index != -1)
            {
                model->center_count[ii][jj].index = cur_center[ii][jj].index;
                model->center_count[ii][jj].value = cur_center[ii][jj].value;
                jj++;
            }
            else
            {
                model->center_count[ii][jj].index = -1;
                break;
            }

        }
    }
    //model->center = Malloc(int,l);
    //memcpy(model->center,cur_medoids,l);
    //for(ii=0;ii<l;ii++)
    //    model->center[ii] = cur_medoids[ii];
    //for(ii=0;ii<l;ii++)
    //    printf("center[%d] = %d  cur_medoids[%d] = %d\n",ii,model->center[ii],ii,cur_medoids[ii]);


    
    free(cur_center);
    free(cur_center_space);
}

kmeans_model *Solver::kmeans_fit_seq(cluster_problem *prob, kmeans_parameter *param)
{
	fprintf(stderr,"In kmeans_fit_seq()\n");
	kmeans_model *model = Malloc(kmeans_model,1);
	model->param = *param;
    model->n_clusters = param->n_clusters;
	//model->free_sv = 0;	// XXX
    int l = prob->l;
    int init_type = param->init_type;
    int n_clusters = param->n_clusters;
    int max_iter = param->max_iter;
    int count_type = param->count_type;
    int *center_init = Malloc(int,l);
    for(int i=0;i<l;i++)
    {
        center_init[i] = 0;
    }

    //initialize seed for clusters
    switch(init_type)
    {
    case RANDOM:
        center_init = init_random(l,n_clusters);
        break;
    case KMEANS_PP:
	    fprintf(stderr,"Before init_kmeans_plusplus()\n");
        init_kmeans_plusplus(prob,param,center_init);
        break;
    //case KMEANS_SS:
	//    fprintf(stderr,"Before init_kmeans_scablescalbe()\n");
    //    init_kmeans_scablescable(prob,param,center_init);
    //    break;
    case USER_CERTAIN:
	    fprintf(stderr,"Before init_user_certain()\n");
        center_init = init_user_certain(l);
        break;
	default:
		fprintf(stderr,"Unknown init_type: %d\n", init_type);
		//exit_with_help();
    }
    
	fprintf(stderr,"After initial random in kmeans_fit_seq()\n");
    model->center_init = Malloc(int,l);
    model->center_init = center_init;
    
    //clustering
    switch(count_type)
    {
    case LLOYD:
	    fprintf(stderr,"Before kmeans_Lloyd()\n");
        kmeans_Lloyd(prob,param,center_init,model);
        break;
    case PAM_FIRST_MEDOIDS:
	    fprintf(stderr,"Before kmedoids_pam_first_medoids()\n");
        kmedoids_pam_first_medoids(prob,param,center_init,model);
        break;
	default:
		fprintf(stderr,"Unknown count_type: %d\n", count_type);
		//exit_with_help();
    }

	fprintf(stderr,"After clustering\n");

//    test:copy input to output
    cluster_problem *test_tmp = Malloc(cluster_problem,1);
    test_tmp->l = prob->l;
    test_tmp->y = prob->y;
    test_tmp->x = prob->x;

    model->test_out.l = test_tmp->l;
    model->test_out.y = test_tmp->y;
    model->test_out.x = test_tmp->x;
    model->l = test_tmp->l;



	return model;
}

kmeans_model *Solver::kmeans_fit_mpi(cluster_problem *prob, kmeans_parameter *param)
{
	kmeans_model *model = Malloc(kmeans_model,1);
	model->param = *param;
    model->n_clusters = param->n_clusters;
	//model->free_sv = 0;	// XXX
    int l = prob->l;
    int init_type = param->init_type;
    int n_clusters = param->n_clusters;
    int max_iter = param->max_iter;
    int count_type = param->count_type;
    int *center_init = Malloc(int,l);

    int myid, numprocs;
    MPI_Comm_rank(MPI_COMM_WORLD,&myid);
    MPI_Comm_size(MPI_COMM_WORLD,&numprocs);

    //int block = l/numprocs;

	fprintf(stderr,"In kmeans_fit_mpi() in thread %d\n",myid);
    
    for(int i=0;i<l;i++)
    {
        center_init[i] = 0;
    }

    //initialize seed for clusters
    switch(init_type)
    {
    case RANDOM:
        if(myid == 0)
            center_init = init_random(l,n_clusters);
        MPI_Bcast(center_init,l,MPI_INT,0,MPI_COMM_WORLD);
        MPI_Barrier(MPI_COMM_WORLD);
        break;
    case KMEANS_PP:
	    fprintf(stderr,"Before init_kmeans_plusplus()\n");
        init_kmeans_plusplus(prob,param,center_init);
        break;
    //case KMEANS_SS:
	//    fprintf(stderr,"Before init_kmeans_scablescalbe()\n");
    //    init_kmeans_scablescable(prob,param,center_init);
    //    break;
    case USER_CERTAIN:
	    fprintf(stderr,"Before init_user_certain()\n");
        center_init = init_user_certain(l);
        break;
	default:
		fprintf(stderr,"Unknown init_type: %d\n", init_type);
		//exit_with_help();
    }
    


	fprintf(stderr,"After initial random in kmeans_fit_seq() in thread %d\n",myid);
    model->center_init = Malloc(int,l);
    model->center_init = center_init;
    
    //clustering
    switch(count_type)
    {
    case LLOYD:
	    fprintf(stderr,"Before kmeans_Lloyd_mpi() in thread %d\n",myid);
        //kmeans_Lloyd(prob,param,center_init,model);
        kmeans_Lloyd_mpi(prob,param,center_init,model);
        break;
    case PAM_FIRST_MEDOIDS:
	    fprintf(stderr,"Before kmedoids_pam_first_medoids()\n");
        kmedoids_pam_first_medoids(prob,param,center_init,model);
        break;
	default:
		fprintf(stderr,"Unknown count_type: %d\n", count_type);
		//exit_with_help();
    }

	fprintf(stderr,"After clustering in thread %d\n",myid);

//    test:copy input to output
    cluster_problem *test_tmp = Malloc(cluster_problem,1);
    test_tmp->l = prob->l;
    test_tmp->y = prob->y;
    test_tmp->x = prob->x;

    model->test_out.l = test_tmp->l;
    model->test_out.y = test_tmp->y;
    model->test_out.x = test_tmp->x;
    model->l = test_tmp->l;



	return model;
}


//void init_kmeans_scablescable(cluster_problem *,kmeans_parameter *,int *);
//
// Interface functions
//
kmeans_model *kmeans_fit(cluster_problem *prob, kmeans_parameter *param)
{
	kmeans_model *model = Malloc(kmeans_model,1);
	model->param = *param;

    Solver s;
    switch(param->parallel_type)
    {
        case SEQ:
            model = s.kmeans_fit_seq(prob,param);
            break;
        case MPI_CH:
	        fprintf(stderr,"Before kmeans_fit_mpi()\n");
            model = s.kmeans_fit_mpi(prob,param);
            break;
        //case KMEANS_SS:
	    //    fprintf(stderr,"Before init_kmeans_scablescalbe()\n");
        //    init_kmeans_scablescable(prob,param,center_init);
        //    break;
	    default:
		    fprintf(stderr,"Unknown parallel_type: %d\n", param->parallel_type);
		    //exit_with_help();
    }

    return model;
}


int kmeans_save_model_classification(char *model_file_name, kmeans_model *model)
{
	FILE *fp = fopen(model_file_name,"w");
	if(fp==NULL) return -1;

	char *old_locale = strdup(setlocale(LC_ALL, NULL));
	setlocale(LC_ALL, "C");

	kmeans_parameter& param = model->param;

    cluster_problem& out = model->test_out;
    int *center = model->center_medoids;
    int l = out.l;
    int k;	
    fprintf(fp, "initial center number is:\n ");
    for(k=0;k<l;k++)
    {
        if(model->center_init[k] == 1)
            fprintf(fp,"%d ",k);
    }
    
    fprintf(fp, "\nkmeans center number is:\n");
    for(k=0;k<l;k++)
    {
        if(center[k] == 1)
            fprintf(fp,"%d ",k);
    }

    fprintf(fp, "\nkmeans result is: \n");
    
    for(int i=0;i<l;i++)
	{
        int j =0;
        fprintf(fp, "%d,",center[i]);
        fprintf(fp, "%f,",model->kmeans_label[i]);
        fprintf(fp, "%f,",out.y[i]);
        while(1)
        {
            if(out.x[i][j+1].index != -1)
                fprintf(fp, "%f,",out.x[i][j].value);
            else
            {
                fprintf(fp, "%f",out.x[i][j].value);
                break;
            }
            ++j;            
        }
        fprintf(fp, "\n");
    }

	setlocale(LC_ALL, old_locale);
	free(old_locale);

	if (ferror(fp) != 0 || fclose(fp) != 0) return -1;
	else return 0;
}

int kmeans_save_model_cluster(char *model_file_name, kmeans_model *model)
{
	FILE *fp = fopen(model_file_name,"w");
	if(fp==NULL) return -1;

	char *old_locale = strdup(setlocale(LC_ALL, NULL));
	setlocale(LC_ALL, "C");

	kmeans_parameter& param = model->param;

    cluster_problem& out = model->test_out;
    int *center = model->center_medoids;
    int l = out.l;
    int k;	
    int n_clusters = model->n_clusters;

    fprintf(fp, "initial center number is:\n ");
    for(k=0;k<l;k++)
    {
        if(model->center_init[k] == 1)
            fprintf(fp,"%d ",k);
    }
    
    fprintf(fp, "\ntotal cost is: %f\n", model->total_cost);
    fprintf(fp, "iteration is: %d\n", model->iter_counter);
    fprintf(fp, "clusters result \n");
    int i,j;
    int order;
    for(k=0;k<n_clusters;k++)
    {
        fprintf(fp,"***********************************************************\n");
        fprintf(fp,"cluster %d center is\n",k);
        j = 0;
        while(1)
        {
            if(model->center_count[k][j+1].index != -1)
            {

                fprintf(fp,"%f,",model->center_count[k][j].value);
            }
            else
            {
                fprintf(fp,"%f\n",model->center_count[k][j].value);
                break;
            }
            ++j;
        }
        fprintf(fp,"cluster %d samples is\n",k);
        for(i=0;i<l;i++)
        {
            order = model->kmeans_order[i];
            if(order == k)
            {
                fprintf(fp,"%f,",out.y[i]);
                j = 0;
                while(1)
                {
                    if(out.x[i][j+1].index != -1)
                        fprintf(fp, "%f,",out.x[i][j].value);
                    else
                    {
                        fprintf(fp, "%f",out.x[i][j].value);
                        break;
                    }
                    ++j;
                }
                fprintf(fp, "\n");
            }
        }
    }

    fprintf(fp,"***********************************************************\n");
    fprintf(fp, "predict result is:(cluster,label,value) \n");
    
    for(i=0;i<l;i++)
	{
        j =0;
        //fprintf(fp, "%d,",center[i]);
        fprintf(fp, "%d,",model->kmeans_order[i]);
        fprintf(fp, "%f,",out.y[i]);
        while(1)
        {
            if(out.x[i][j+1].index != -1)
                fprintf(fp, "%f,",out.x[i][j].value);
            else
            {
                fprintf(fp, "%f",out.x[i][j].value);
                break;
            }
            ++j;            
        }
        fprintf(fp, "\n");
    }

	setlocale(LC_ALL, old_locale);
	free(old_locale);

	if (ferror(fp) != 0 || fclose(fp) != 0) return -1;
	else return 0;
}

void kmeans_set_print_string_function(void (*print_func)(const char *))
{
	if(print_func == NULL)
		kmeans_print_string = &print_string_stdout;
	else
		kmeans_print_string = print_func;

}

              



        

