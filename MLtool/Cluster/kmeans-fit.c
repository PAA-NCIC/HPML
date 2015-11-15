#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>
#include <sys/time.h>
#include "kmeans-pmlp.h"
#include "mpi.h"

#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

void print_null(const char *s) {}

void exit_with_help()
{
	printf(
	"Usage: kmeans-fit [options] training_set_file [model_file]\n"
	"options:\n"
	"-p parallel_type : set type of parallelization (default 0)\n"
	"	0 -- SEQ		(serial type)\n"
	"	1 -- MPI_CH		\n"
	"-t initial_type : set type of initialization (default 1)\n"
	"	0 -- RANDOM		(random initialization)\n"
	"	1 -- kmeans ++		\n"
	"	2 -- kmeans || \n"
	"	3 -- user certain (certained by users, for test and compare) \n"
	"-c count_type : set type of count function (default 0)\n"
	"	0 -- LLOYD \n"
	"	1 -- PAM_FIRST_MEDOIDS: google's (gamma*u'*v + coef0)^degree\n"
	"	2 -- PAM_SECOND_MEDOIDS: google's (gamma*u'*v + coef0)^degree\n"
	"	3 -- EM: exp(-gamma*|u-v|^2)\n"
	"-s save_type : set type of save function (default 0)\n"
	"	0 -- save as cluster \n"
	"	1 -- save as classification\n"
	"-n n_clusters : set number of cluster (default 3)\n"
	"-m max_iter : set maxm number in cluster computing (default 300)\n"
	"-i n_init : set initial number in choosing  seeds (default 10)\n"
	"-j n_jobs : set the number of jobs (default 1)\n"
	"-e tol : set tolerance of termination criterion (default 0.001)\n"
    "other parameters perhaps\n"
 	"-p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)\n"
	"-m cachesize : set cache memory size in MB (default 100)\n"
	"-h shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)\n"
	"-b probability_estimates : whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)\n"
	"-wi weight : set the parameter C of class i to weight*C, for C-SVC (default 1)\n"
	"-v n: n-fold cross validation mode\n"
	"-q : quiet mode (no outputs)\n"
	);
	exit(1);
}

void exit_input_error(int line_num)
{
	fprintf(stderr,"Wrong input format at line %d\n", line_num);
	exit(1);
}

void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name);
//void read_problem(const char *filename);
void read_problem_without_index(const char *filename);
//void do_cross_validation();

struct kmeans_parameter param;		// set by parse_command_line
struct cluster_problem prob;		// set by read_problem
struct kmeans_model *model;
struct cluster_node *x_space;
//int cross_validation;
//int nr_fold;

static char *line = NULL;
static int max_line_len;

static char* readline(FILE *input)
{
	int len;
	
	if(fgets(line,max_line_len,input) == NULL)
		return NULL;

	while(strrchr(line,'\n') == NULL)
	{
		max_line_len *= 2;
		line = (char *) realloc(line,max_line_len);
		len = (int) strlen(line);
		if(fgets(line+len,max_line_len-len,input) == NULL)
			break;
	}
	return line;
}

int main(int argc, char **argv)
{
	char input_file_name[1024];
	char model_file_name[1024];
	const char *error_msg;

	struct timeval tv;
 	struct timezone tz;
	double timeStart;
	double timeEnd;
	double t_start_mpi;
	double t_end_mpi;
	MPI_Init(&argc,&argv);
	gettimeofday(&tv,&tz);
	timeStart = tv.tv_sec + tv.tv_usec*10e-6;
	t_start_mpi = MPI_Wtime();

	parse_command_line(argc, argv, input_file_name, model_file_name);
	
	//Yuan
	char test_file_name[1024];
    strcpy(test_file_name,"test-");
    strcat(test_file_name,model_file_name);
    FILE *fp=fopen(test_file_name,"a");
	if(fp==NULL)
	{
	    printf("File failed!");
	}
	printf("Y95:main in kmeans-train.c\n");
	printf("input file name is %s\n",input_file_name);
	fprintf(fp,"Y:input file name  %s\n",input_file_name);
	fprintf(fp,"Command: ");
	for(int i=0;i<argc;i++)
	{
		fprintf(fp," %s",argv[i]);
	}	
	fprintf(fp,"\n");
	
	//read_problem(input_file_name);
	//read_problem_with_index(input_file_name);
	fprintf(stderr,"Before read_problem_without_index\n");
	read_problem_without_index(input_file_name);
	fprintf(stderr,"After read_problem_without_index\n");
	//error_msg = svm_check_parameter(&prob,&param);

	//if(error_msg)
	//{
	//	fprintf(stderr,"ERROR: %s\n",error_msg);
	//	exit(1);
	//}
	model = kmeans_fit(&prob,&param);
	//if(kmeans_save_model_classification(model_file_name,model))
	if(kmeans_save_model_cluster(model_file_name,model))
	{
		fprintf(stderr, "can't save model to file %s\n", model_file_name);
		exit(1);
	}
	//kmeans_free_and_destroy_model(&model);

//	if(cross_validation)
//	{
//		do_cross_validation();
//	}
//	else
//	{
//		model = svm_train(&prob,&param);
//		if(svm_save_model(model_file_name,model))
//		{
//			fprintf(stderr, "can't save model to file %s\n", model_file_name);
//			exit(1);
//		}
//		svm_free_and_destroy_model(&model);
//	}
	//Yuan
	//fprintf(fp,"%d\n%d\n%d\n%f\n%f\n",param.svm_type,param.kernel_type,param.degree,param.gamma,param.coef0);
	//kmeans_destroy_param(&param);

	//time
	gettimeofday(&tv,&tz);
	timeEnd = tv.tv_sec + tv.tv_usec*10e-6;
	t_end_mpi = MPI_Wtime();
	printf("used C time:%.2fs\n",(timeEnd-timeStart));
	printf("used MPI time:%.2fs\n",(t_end_mpi-t_start_mpi));
	fprintf(fp,"used C time:%.2fs\n",(timeEnd-timeStart));
	fprintf(fp,"used MPI time:%.2fs\n",(t_end_mpi-t_start_mpi));
	MPI_Finalize();


	free(prob.y);
	free(prob.x);
	free(x_space);
	free(line);

	return 0;
}

/*
void do_cross_validation()
{
	int i;
	int total_correct = 0;
	double total_error = 0;
	double sumv = 0, sumy = 0, sumvv = 0, sumyy = 0, sumvy = 0;
	double *target = Malloc(double,prob.l);

	svm_cross_validation(&prob,&param,nr_fold,target);
	if(param.svm_type == EPSILON_SVR ||
	   param.svm_type == NU_SVR)
	{
		for(i=0;i<prob.l;i++)
		{
			double y = prob.y[i];
			double v = target[i];
			total_error += (v-y)*(v-y);
			sumv += v;
			sumy += y;
			sumvv += v*v;
			sumyy += y*y;
			sumvy += v*y;
		}
		printf("Cross Validation Mean squared error = %g\n",total_error/prob.l);
		printf("Cross Validation Squared correlation coefficient = %g\n",
			((prob.l*sumvy-sumv*sumy)*(prob.l*sumvy-sumv*sumy))/
			((prob.l*sumvv-sumv*sumv)*(prob.l*sumyy-sumy*sumy))
			);
	}
	else
	{
		for(i=0;i<prob.l;i++)
			if(target[i] == prob.y[i])
				++total_correct;
		printf("Cross Validation Accuracy = %g%%\n",100.0*total_correct/prob.l);
	}
	free(target);
}
*/

void parse_command_line(int argc, char **argv, char *input_file_name, char *model_file_name)
{
	int i;
	void (*print_func)(const char*) = NULL;	// default printing to stdout

	// default values
	//param.parallel_type = SEQ;  
	param.parallel_type = MPI_CH;  
	//param.init_type = RANDOM;
	param.init_type = USER_CERTAIN;  //for test and compare
	//param.count_type = PAM_FIRST_MEDOIDS;
	param.count_type = CLUSTER;
	param.save_type = CLASSIFICATION;
	param.n_clusters = 3;
	param.max_iter = 100;
	param.n_init = 10;
	param.tol = 1e-3;
    param.n_jobs =1;
	//param.cache_size = 100;
	//param.C = 1;
	//param.eps = 1e-3;
	//param.p = 0.1;
	//cross_validation = 0;

	// parse options
	for(i=1;i<argc;i++)
	{
		if(argv[i][0] != '-') break;
		if(++i>=argc)
			exit_with_help();
		switch(argv[i-1][1])
		{
			case 'p':
				param.parallel_type = atoi(argv[i]);
				printf("\nparallel_type is %d\n",param.parallel_type);
                break;
			case 't':
				param.init_type = atoi(argv[i]);
				break;
			case 'c':
				param.count_type = atoi(argv[i]);
				break;
			case 's':
				param.save_type = atoi(argv[i]);
				break;
			case 'n':
				param.n_clusters = atoi(argv[i]);
				break;
			case 'm':
				param.max_iter = atoi(argv[i]);
				break;
			case 'i':
				param.n_init = atoi(argv[i]);
				break;
			case 'j':
				param.n_jobs = atoi(argv[i]);
				break;
			case 'e':
				param.tol = atof(argv[i]);
				break;
			//case 'v':
			//	cross_validation = 1;
			//	nr_fold = atoi(argv[i]);
				//YUAN
			//	if(nr_fold < 2)
			//	{
			//		fprintf(stderr,"n-fold cross validation: n must >= 2\n");
			//		exit_with_help();
			//	}
			//	break;
			//case 'w':
			//	++param.nr_weight;
			//	param.weight_label = (int *)realloc(param.weight_label,sizeof(int)*param.nr_weight);
			//	param.weight = (double *)realloc(param.weight,sizeof(double)*param.nr_weight);
			//	param.weight_label[param.nr_weight-1] = atoi(&argv[i-1][2]);
			//	param.weight[param.nr_weight-1] = atof(argv[i]);
			//	break;
			default:
				fprintf(stderr,"Unknown option: -%c\n", argv[i-1][1]);
				exit_with_help();
		}
	}

	kmeans_set_print_string_function(print_func);

	// determine filenames

	if(i>=argc)
		exit_with_help();

	strcpy(input_file_name, argv[i]);

	if(i<argc-1)
		strcpy(model_file_name,argv[i+1]);
	else
	{
		char *p = strrchr(argv[i],'/');
		if(p==NULL)
			p = argv[i];
		else
			++p;
		sprintf(model_file_name,"%s.model",p);
	}
	if(param.parallel_type == MPI_CH)
	{
		//MPI_Init(&argc,&argv);
		int threadid;
		MPI_Comm_rank(MPI_COMM_WORLD,&threadid);
		char model_name_tmp[1024];
		//itoa(threadid,model_name_tmp,10);
		//strcat(model_name_tmp,"-");
		//strcat(model_name_tmp,model_file_name);
		sprintf(model_name_tmp,"%d-%s",threadid,model_file_name);
		strcpy(model_file_name,model_name_tmp);
		printf("\nparallel_type is %d\n",param.parallel_type);
	}

}

// read in a problem (in svmlight format)
// for kmeans
void read_problem_without_index(const char *filename)
{
	int elements, max_index, inst_max_index, i, j;
	FILE *fp = fopen(filename,"r");
	char *endptr;
	char *idx, *val, *label;

	if(fp == NULL)
	{
		fprintf(stderr,"can't open input file %s\n",filename);
		exit(1);
	}

	prob.l = 0;
	elements = 0;

	max_line_len = 1024;
	line = Malloc(char,max_line_len);
	while(readline(fp)!=NULL)
	{
		char *p = strtok(line," ,"); // label

		// features
		while(1)
		{
			p = strtok(NULL," ,");
			if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
				break;
			++elements;
		}
		++elements;
		++prob.l;
	}
	rewind(fp);

	prob.y = Malloc(double,prob.l);
	prob.x = Malloc(struct cluster_node *,prob.l);
	x_space = Malloc(struct cluster_node,elements);

	max_index = 0;
	j=0;
    //int tmp_j = 0;
	for(i=0;i<prob.l;i++)
	{
		inst_max_index = 0; // 
		readline(fp);
		prob.x[i] = &x_space[j];
		label = strtok(line," ,");
		if(label == NULL) // empty line
			exit_input_error(i+1);

		prob.y[i] = strtod(label,&endptr);
		if(endptr == label || *endptr != '\0')
			exit_input_error(i+1);

		while(1)
		{
			//idx = strtok(NULL,":");
			val = strtok(NULL," ,");

			if(val == NULL)
				break;


			errno = 0;
			x_space[j].value = strtod(val,&endptr);
	//		fprintf(stderr,"%f,",prob.x[i][j-tmp_j]);
            if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error(i+1);

            ++inst_max_index;
			x_space[j].index = inst_max_index;
			
            ++j;
		}
	//	fprintf(stderr,"\n");
       	if(inst_max_index > max_index)
			max_index = inst_max_index;
		x_space[j++].index = -1;
   //   tmp_j = j;
	}
    
    param.max_index = max_index;

	fclose(fp);
}


//void read_problem_with_index(const char *filename)
//{
//	int elements, max_index, inst_max_index, i, j;
//	FILE *fp = fopen(filename,"r");
//	char *endptr;
//	char *idx, *val, *label;
//
//	if(fp == NULL)
//	{
//		fprintf(stderr,"can't open input file %s\n",filename);
//		exit(1);
//	}
//
//	prob.l = 0;
//	elements = 0;
//
//	max_line_len = 1024;
//	line = Malloc(char,max_line_len);
//	while(readline(fp)!=NULL)
//	{
//		char *p = strtok(line," \t"); // label
//
//		// features
//		while(1)
//		{
//			p = strtok(NULL," \t");
//			if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
//				break;
//			++elements;
//		}
//		++elements;
//		++prob.l;
//	}
//	rewind(fp);
//
//	prob.y = Malloc(double,prob.l);
//	prob.x = Malloc(struct cluster_node *,prob.l);
//	x_space = Malloc(struct cluster_node,elements);
//
//	max_index = 0;
//	j=0;
//	for(i=0;i<prob.l;i++)
//	{
//		inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0
//		readline(fp);
//		prob.x[i] = &x_space[j];
//		label = strtok(line," \t\n");
//		if(label == NULL) // empty line
//			exit_input_error(i+1);
//
//		prob.y[i] = strtod(label,&endptr);
//		if(endptr == label || *endptr != '\0')
//			exit_input_error(i+1);
//
//		while(1)
//		{
//			idx = strtok(NULL,":");
//			val = strtok(NULL," \t");
//
//			if(val == NULL)
//				break;
//
//			errno = 0;
//			x_space[j].index = (int) strtol(idx,&endptr,10);
//			if(endptr == idx || errno != 0 || *endptr != '\0' || x_space[j].index <= inst_max_index)
//				exit_input_error(i+1);
//			else
//				inst_max_index = x_space[j].index;
//
//			errno = 0;
//			x_space[j].value = strtod(val,&endptr);
//			if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
//				exit_input_error(i+1);
//
//			++j;
//		}
//
//		if(inst_max_index > max_index)
//			max_index = inst_max_index;
//		x_space[j++].index = -1;
//	}
//
//	if(param.gamma == 0 && max_index > 0)
//		param.gamma = 1.0/max_index;
//
//	if(param.kernel_type == PRECOMPUTED)
//		for(i=0;i<prob.l;i++)
//		{
//			if (prob.x[i][0].index != 0)
//			{
//				fprintf(stderr,"Wrong input format: first column must be 0:sample_serial_number\n");
//				exit(1);
//			}
//			if ((int)prob.x[i][0].value <= 0 || (int)prob.x[i][0].value > max_index)
//			{
//				fprintf(stderr,"Wrong input format: sample_serial_number out of range\n");
//				exit(1);
//			}
//		}
//
//	fclose(fp);
//}

//for svm
/*
void read_problem(const char *filename)
{
	int elements, max_index, inst_max_index, i, j;
	FILE *fp = fopen(filename,"r");
	char *endptr;
	char *idx, *val, *label;

	if(fp == NULL)
	{
		fprintf(stderr,"can't open input file %s\n",filename);
		exit(1);
	}

	prob.l = 0;
	elements = 0;

	max_line_len = 1024;
	line = Malloc(char,max_line_len);
	while(readline(fp)!=NULL)
	{
		char *p = strtok(line," \t"); // label

		// features
		while(1)
		{
			p = strtok(NULL," \t");
			if(p == NULL || *p == '\n') // check '\n' as ' ' may be after the last feature
				break;
			++elements;
		}
		++elements;
		++prob.l;
	}
	rewind(fp);

	prob.y = Malloc(double,prob.l);
	prob.x = Malloc(struct svm_node *,prob.l);
	x_space = Malloc(struct svm_node,elements);

	max_index = 0;
	j=0;
	for(i=0;i<prob.l;i++)
	{
		inst_max_index = -1; // strtol gives 0 if wrong format, and precomputed kernel has <index> start from 0
		readline(fp);
		prob.x[i] = &x_space[j];
		label = strtok(line," \t\n");
		if(label == NULL) // empty line
			exit_input_error(i+1);

		prob.y[i] = strtod(label,&endptr);
		if(endptr == label || *endptr != '\0')
			exit_input_error(i+1);

		while(1)
		{
			idx = strtok(NULL,":");
			val = strtok(NULL," \t");

			if(val == NULL)
				break;

			errno = 0;
			x_space[j].index = (int) strtol(idx,&endptr,10);
			if(endptr == idx || errno != 0 || *endptr != '\0' || x_space[j].index <= inst_max_index)
				exit_input_error(i+1);
			else
				inst_max_index = x_space[j].index;

			errno = 0;
			x_space[j].value = strtod(val,&endptr);
			if(endptr == val || errno != 0 || (*endptr != '\0' && !isspace(*endptr)))
				exit_input_error(i+1);

			++j;
		}

		if(inst_max_index > max_index)
			max_index = inst_max_index;
		x_space[j++].index = -1;
	}

	if(param.gamma == 0 && max_index > 0)
		param.gamma = 1.0/max_index;

	if(param.kernel_type == PRECOMPUTED)
		for(i=0;i<prob.l;i++)
		{
			if (prob.x[i][0].index != 0)
			{
				fprintf(stderr,"Wrong input format: first column must be 0:sample_serial_number\n");
				exit(1);
			}
			if ((int)prob.x[i][0].value <= 0 || (int)prob.x[i][0].value > max_index)
			{
				fprintf(stderr,"Wrong input format: sample_serial_number out of range\n");
				exit(1);
			}
		}

	fclose(fp);
}
*/
