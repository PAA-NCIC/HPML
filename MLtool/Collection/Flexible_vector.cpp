/*
ID: neo-white 
LANG: C++
TASK: Flexible_vector.cpp
*/

#include "Flexible_vector.h"


void exit_input_error(int line_num)
{
	fprintf(stderr,"Wrong input format at line %d\n", line_num);
	exit(1);
}

// read in a problem (in svmlight format)

problem Flexible_vector::Read_file_without_index(std::string filename)
{
    std::ifstream file_in(filename.c_str());
    std::string e,line;
    problem prob;
    int elements, max_index, inst_max_index;
	
    char *endptr;

	prob.l = 0;
	elements = 0;
    max_index = 0;
    while (std::getline(file_in, line))
    {
        inst_max_index = 0;
        std::stringstream  lineStream(line);

        std::getline(lineStream, e,',');   //label
        if(e.empty())  //empty line
            exit_input_error(prob.l+1);
        prob.y.push_back(stringToNum<double>(e));
        //prob.y.push_back(atof(e));
        //if(endptr == e || *endptr != '\0')
        //    exit_input_error(prob.l+1);

        prob.x_ptr.push_back(elements);
        //feature
        while(std::getline(lineStream, e,','))
        {
            inst_max_index++;
            node x_tmp;
            x_tmp.index = inst_max_index;
            x_tmp.value = stringToNum<double>(e);
            //x_tmp.value = atof(e);
            prob.x.push_back(x_tmp);
            elements++;
        }
        prob.x_interval.push_back(inst_max_index);
        if(inst_max_index > max_index)
            max_index = inst_max_index;
        prob.l++;
    }
    return prob;
}

void Flexible_vector::output_problem(std::string filename, problem prob)
{
    std::ofstream fout(filename.c_str());
    int i;
    
    for(i=0;i<prob.l;i++)
    {
        //fout.unsetf(std::ios::fixed);
        fout<<prob.y[i]<<',';
        //std::cout<<prob.y[i]<<',';
        for(int j=prob.x_ptr[i];j<prob.x_ptr[i]+prob.x_interval[i];j++)
        {
            if(j == (prob.x_ptr[i]+prob.x_interval[i]-1))
            {
                //fout<<std::fixed<<std::setprecision(1)<<prob.x[j].value<< std::endl;
                fout<<prob.x[j].value<< std::endl;
                //std::cout<<prob.x[j].value<< std::endl;
            }else
            {
                //fout<<std::fixed<<std::setprecision(1)<<prob.x[j].value<<',';
                fout<<prob.x[j].value<<',';
                //std::cout<<prob.x[j].value<<',';
            }
        }
    }
}

