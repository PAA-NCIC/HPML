#ifndef _KMEANS_PMLP_H
#define _KMEANS_PMLP_H

#define KMEANS_PMLP_VERSION 100

#ifdef __cplusplus
extern "C" {
#endif

extern int kmeans_pmlp_version;

struct cluster_node
{
	int index;
	double value;
};

struct cluster_problem
{
	int l;
	double *y;
	struct cluster_node **x;
};

enum { SEQ,  MPI_CH };	/* parallel_type */
enum { RANDOM, KMEANS_PP, KMEANS_SS, USER_CERTAIN };	/* initial_type */
enum { LLOYD, PAM_FIRST_MEDOIDS, PAM_SECOND_MEDOIDS, EM };	/* count_type */
enum { CLUSTER, CLASSIFICATION };	/* save_type */

struct kmeans_parameter
{
	int parallel_type;         //parallel type. {'SEQ', 'MPI'}, optional. Method for parallelization, default to 'SEQ':
	int init_type;         //initial type. {'k-means++', 'random', 'k-means||' or ndarray, or a callable}, optional. Method for initialization, default to 'k-means++':
 		          //'k-means++' : selects initial cluster centers for k-mean
 		          //clustering in a smart way to speed up convergence. See section
 			  // Notes in k_init for more details.
 		          //'random': generate k centroids from a Gaussian with mean and variance estimated from the data.
 		          //If an ndarray is passed, it should be of shape (n_clusters, n_featu     res) and gives the initial centers.
     	    		  //If a callable is passed, it should take arguments X, k and
			  //and a random state and return an initialization.
	//X : array-like or sparse matrix, shape (n_samples, n_features) 
	//The observations to cluster.
	int n_clusters;  // The number of clusters to form as well as the number of centroids to generate.
	int max_iter;    // optional, default 300. Maximum number of iterations of the k-means algorithm to run.
	int max_index;    // Maximum number of features of samples.
	int n_init;      //optional, default: 10. Number of time the k-means algorithm will be run with different centroid seeds. The final results will be the best output of n_init consecutive runs in terms of inertia.
    bool precompute_distances; // {'auto', True, False}
	         //Precompute distances (faster but takes more memory).
 	         //'auto' : do not precompute distances if n_samples * n_clusters > 12
         	 //million. This corresponds to about 100MB overhead per job using
         	 //double precision.
 	         //True : always precompute distances
 	         //False : never precompute distances
	double tol; // optional. The relative increment in the results before declaring convergence.
  	int count_type;   //the function of counting cluster
        //verbose : boolean, optional
        //  Verbosity mode.
  
        //random_state : integer or numpy.RandomState, optional
        //  The generator used to initialize the centers. If an integer is
        //  given, it fixes the seed. Defaults to the global numpy random
        //  number generator.
    int save_type; 
    double copy_x;
	  /*optional
          When pre-computing distances it is more numerically accurate to center
          the data first.  If copy_x is True, then the original data is not
          modified.  If False, the original data is modified, and put back before
          the function returns, but small numerical differences may be introduced
          by subtracting and then adding the data mean.*/
  
    int n_jobs;
          /*The number of jobs to use for the computation. This works by computing
          each of the n_init runs in parallel.  
          If -1 all CPUs are used. If 1 is given, no parallel computing code is
          used at all, which is useful for debugging. For n_jobs below -1,
          (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one
          are used.*/
  
     bool return_n_iter;   // optional. Whether or not to return the number of iterations.
 	 double centroid;  //array with shape (k, n_features). Centroids found at the last iteration of k-means.
	 int  label;      //integer ndarray with shape (n_samples,).label[i] is the code or index of the centroid the i'th observation is closest to.
 
     double inertia;
         /*The final value of the inertia criterion (sum of squared distances to
         the closest centroid for all observations in the training set).*/
 
     int best_n_iter;
         /*Number of iterations corresponding to the best results.
         Returned only if `return_n_iter` is set to True.*/
};

struct kmeans_model
{
	struct kmeans_parameter param;	/* parameter */
	int n_clusters;		/* number of classes, = 2 in regression/one class svm */
	int l;			/* total center number */
	int *center_medoids;		/* centers (center[l]) for kmedoids*/
	cluster_node **center_count;		/* count centers (center[l]) for kmeans*/
	int *center_init;		/* initial centers (center[l]) */
	double total_cost;  //cost of clustering
    //double **sv_coef;	/* coefficients for SVs in decision functions (sv_coef[k-1][l]) */
	//double *rho;		/* constants in decision functions (rho[k*(k-1)/2]) */
	//double *probA;		/* pariwise probability information */
	//double *probB;
	//int *sv_indices;        /* sv_indices[0,...,nSV-1] are values in [1,...,num_traning_data] to indicate SVs in the training set */

	/* for classification only */

	double *kmeans_label;		/* every sample's label after kmeans when label is valid */
	int *kmeans_order;		/* every sample's label after kmeans when label is useless*/
    int max_iter;
    int max_index;
    int iter_counter;
    //int *nSV;		/* number of SVs for each class (nSV[k]) */
				/* nSV[0] + nSV[1] + ... + nSV[k-1] = l */
	/* XXX */
	//int free_sv;		/* 1 if svm_model is created by svm_load_model*/
				/* 0 if svm_model is created by svm_train */
    //test
    struct cluster_problem test_out;
    int  l_test;
};
struct kmeans_model *kmeans_fit(struct cluster_problem *prob, struct kmeans_parameter *param);
int kmeans_save_model_cluster(char *model_file_name, struct kmeans_model *model);
int kmeans_save_model_classification(char *model_file_name, struct kmeans_model *model);
struct kmeans_model *kmeans_load_model(char *model_file_name);

int svm_get_svm_type(const struct svm_model *model);
int svm_get_nr_class(const struct svm_model *model);
void svm_get_labels(const struct svm_model *model, int *label);
void svm_get_sv_indices(const struct svm_model *model, int *sv_indices);
int svm_get_nr_sv(const struct svm_model *model);
double svm_get_svr_probability(const struct svm_model *model);


void kmeans_free_and_destroy_model(struct kmeans_model **model_ptr_ptr);
void kmeans_destroy_param(struct kmeans_parameter *param);

const char *kmeans_check_parameter(struct cluster_problem *prob,  struct kmeans_parameter *param);

void kmeans_set_print_string_function(void (*print_func)(const char *));
//enum { C_SVC, NU_SVC, ONE_CLASS, EPSILON_SVR, NU_SVR };	/* svm_type */
//enum { LINEAR, POLY, RBF, SIGMOID, PRECOMPUTED }; /* kernel_type */

struct svm_node
{
	int index;
	double value;
};

struct svm_problem
{
	int l;
	double *y;
	struct svm_node **x;
};
struct svm_parameter
{
	int svm_type;
	int kernel_type;
	int degree;	/* for poly */
	double gamma;	/* for poly/rbf/sigmoid */
	double coef0;	/* for poly/sigmoid */

	/* these are for training only */
	double cache_size; /* in MB */
	double eps;	/* stopping criteria */
	double C;	/* for C_SVC, EPSILON_SVR and NU_SVR */
	int nr_weight;		/* for C_SVC */
	int *weight_label;	/* for C_SVC */
	double* weight;		/* for C_SVC */
	double nu;	/* for NU_SVC, ONE_CLASS, and NU_SVR */
	double p;	/* for EPSILON_SVR */
	int shrinking;	/* use the shrinking heuristics */
	int probability; /* do probability estimates */
};

//
// svm_model
// 
struct svm_model
{
	struct svm_parameter param;	/* parameter */
	int nr_class;		/* number of classes, = 2 in regression/one class svm */
	int l;			/* total #SV */
	struct svm_node **SV;		/* SVs (SV[l]) */
	double **sv_coef;	/* coefficients for SVs in decision functions (sv_coef[k-1][l]) */
	double *rho;		/* constants in decision functions (rho[k*(k-1)/2]) */
	double *probA;		/* pariwise probability information */
	double *probB;
	int *sv_indices;        /* sv_indices[0,...,nSV-1] are values in [1,...,num_traning_data] to indicate SVs in the training set */

	/* for classification only */

	int *label;		/* label of each class (label[k]) */
	int *nSV;		/* number of SVs for each class (nSV[k]) */
				/* nSV[0] + nSV[1] + ... + nSV[k-1] = l */
	/* XXX */
	int free_sv;		/* 1 if svm_model is created by svm_load_model*/
				/* 0 if svm_model is created by svm_train */
};
/*
struct svm_model *svm_train(const struct svm_problem *prob, const struct svm_parameter *param);
void svm_cross_validation(const struct svm_problem *prob, const struct svm_parameter *param, int nr_fold, double *target);

int svm_save_model(const char *model_file_name, const struct svm_model *model);
struct svm_model *svm_load_model(const char *model_file_name);

int svm_get_svm_type(const struct svm_model *model);
int svm_get_nr_class(const struct svm_model *model);
void svm_get_labels(const struct svm_model *model, int *label);
void svm_get_sv_indices(const struct svm_model *model, int *sv_indices);
int svm_get_nr_sv(const struct svm_model *model);
double svm_get_svr_probability(const struct svm_model *model);

double svm_predict_values(const struct svm_model *model, const struct svm_node *x, double* dec_values);
double svm_predict(const struct svm_model *model, const struct svm_node *x);
double svm_predict_probability(const struct svm_model *model, const struct svm_node *x, double* prob_estimates);

void svm_free_model_content(struct svm_model *model_ptr);
void svm_free_and_destroy_model(struct svm_model **model_ptr_ptr);
void svm_destroy_param(struct svm_parameter *param);

const char *svm_check_parameter(const struct svm_problem *prob, const struct svm_parameter *param);
int svm_check_probability_model(const struct svm_model *model);

void svm_set_print_string_function(void (*print_func)(const char *));

//Yuan
int svm_save_problem(const char *problem_file_name, const struct svm_problem problem);
*/
#ifdef __cplusplus
}
#endif

#endif /* _LIBSVM_H */
