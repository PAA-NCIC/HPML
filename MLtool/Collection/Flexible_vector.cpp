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

void Flexible_vector::Load(std::string filename)
{
    prob = Read_file_without_index(filename);
}

//Collection* partition(size_t number, size_t rank);
Flexible_vector* Flexible_vector::partition(Flexible_vector *origin, size_t number, size_t loc, size_t feat_dim)
{

    Flexible_vector *ret = new Flexible_vector();
    ret->init();
    //std::cout<<"origin.l= "<<origin->prob.l<<" origin.max_feature= "<<origin->prob.max_feature<<std::endl;
    size_t n_records = origin->prob.l;
    int members = n_records/number;
    if(number*members < n_records) members += 1;
    if(loc*members+members < n_records)
        ret->prob.l = members;
    else
        ret->prob.l = n_records-loc*members; 
    size_t lim = std::min(loc*members+members, n_records);
    int tmp_xptr = 0;
    for (size_t i = loc*members; i < lim ; i++)
    {
        ret->prob.y.push_back(origin->prob.y[i]);
        ret->prob.x_interval.push_back(origin->prob.x_interval[i]);
        ret->prob.x_ptr.push_back(tmp_xptr);
        tmp_xptr += origin->prob.x_interval[i];
        ret->prob.max_feature = origin->prob.max_feature;
        std::vector<node> tmp = origin->get_value_without_label(i);
        for(size_t j = 0 ; j < tmp.size(); j ++)
        {
            node tmp_x;
            tmp_x.index = tmp[j].index;
            tmp_x.value = tmp[j].value;
            ret->prob.x.push_back(tmp_x);
        }
    }
    
    std::cout<<"ret.l= "<<ret->prob.l<<" ret.max_feature= "<<ret->prob.max_feature<<std::endl;
    return ret;
}

std::vector<double> Flexible_vector::operator [](size_t loc) const 
{
    std::vector<double> data;
    for(size_t i = prob.x_ptr[loc];i < prob.x_ptr[loc]+prob.x_interval[loc];i++)
    {
        data.push_back(prob.x[i].value);

    }
    return data;
}

inline size_t Flexible_vector::size()
{
    return prob.l;
}

void Flexible_vector::init()
{
    this->prob.l = 0;
    this->prob.max_feature = 0;
}

void Flexible_vector::init(size_t l, size_t max_feature)
{
    //std::cout<<"In init(l,max_feature)"<<std::endl;
    this->prob.l = l;
    this->prob.max_feature = max_feature;
    
    prob.y.resize(l);
    std::fill(prob.y.begin(),prob.y.end(),-1);
    //std::cout<<"fvkl.y.size= "<< prob.y.size()<<std::endl;
    //std::cout<<"l= "<< prob.l<<std::endl;
    //for(size_t i = 0;i<prob.y.size();i++)
    //    std::cout<<i<<":"<<prob.y[i]<<std::endl;
    
    prob.x_ptr.resize(l);
    for(size_t i = 0;i < l;i++)
    {
    //    std::cout<<i*max_feature<<std::endl;
        prob.x_ptr[i] = i*max_feature;
    }
    //std::cout<<"fvkl.x_ptr.size= "<< prob.x_ptr.size()<<std::endl;

    prob.x_interval.resize(l);
    prob.x.resize(l*max_feature);
}

void Flexible_vector::insert_end(Flexible_vector src, size_t loc)
{
    prob.max_feature = src.prob.max_feature;
    prob.y.push_back(src.prob.y[loc]);
    prob.x_ptr.push_back((prob.y.size()-1)*prob.max_feature);
    prob.x_interval.push_back(src.prob.x_interval[loc]);
    node tmp;
    for(size_t i = 0;i < src.prob.x_interval[loc];i++)
    {
        size_t start = src.prob.x_ptr[loc];
        tmp.index = src.prob.x[start+i].index;
        tmp.value = src.prob.x[start+i].value;
        prob.x.push_back(tmp);
    }
    prob.l++;
}
std::vector<node> Flexible_vector::get_value_without_label(size_t loc)
{
    std::vector<node> data;
    node tmp;
    for(size_t i = prob.x_ptr[loc];i < prob.x_ptr[loc]+prob.x_interval[loc];i++)
    {
        tmp.index = prob.x[i].index;
        tmp.value = prob.x[i].value;
        data.push_back(tmp);

    }
    return data;
}
        
//add src to the loc-th sample of prob in fvkl 
void Flexible_vector::add(std::vector<node> src, size_t loc)
{
    std::vector<node> tmp_v;
    tmp_v.clear();
    node tmp;
    if(this->prob.y[loc] == -1)
    {
        for(size_t i = 0;i<src.size();i++)
        {
            prob.x[prob.x_ptr[loc]+i].index = src[i].index;
            prob.x[prob.x_ptr[loc]+i].value = src[i].value;
        }
        this->prob.y[loc] = 1;
        this->prob.x_interval[loc] = src.size();
    }else{
        size_t ip_fv = 0;
        size_t ip_src = 0;
        while((ip_fv <= (this->prob.x_interval[loc]-1))||(ip_src <= (src.size()-1)))
    	{
            if((ip_fv <= (this->prob.x_interval[loc]-1))&&(ip_src <= (src.size()-1)))
            {
    		    if(prob.x[prob.x_ptr[loc]+ip_fv].index == src[ip_src].index)
    		    {
                    tmp.index = src[ip_src].index;
    		        tmp.value = prob.x[prob.x_ptr[loc]+ip_fv].value + src[ip_src].value;
    		    	tmp_v.push_back(tmp);
                    ++ip_fv;
    		    	++ip_src;
    		    }
    		    else
    		    {
    		        if(prob.x[prob.x_ptr[loc]+ip_fv].index > src[ip_src].index)
                    {
                        tmp.index = src[ip_src].index;
                        tmp.value = src[ip_src].value;
                        tmp_v.push_back(tmp);
    		    		++ip_src;
                    }
    		    	else
                    {
                        tmp.index = prob.x[prob.x_ptr[loc]+ip_fv].index;
                        tmp.value = prob.x[prob.x_ptr[loc]+ip_fv].value;
                        tmp_v.push_back(tmp);
    		    		++ip_fv;
                    }
    		    }
            }else if((ip_fv == this->prob.x_interval[loc])&&(ip_src <= (src.size()-1)))
            {
                tmp.index = src[ip_src].index;
                tmp.value = src[ip_src].value;
    		    tmp_v.push_back(tmp);
                ++ip_src;
            }else if((ip_src == src.size())&&(ip_fv <= (this->prob.x_interval[loc]-1)))
            {
                tmp.index = prob.x[prob.x_ptr[loc]+ip_fv].index;
                tmp.value = prob.x[prob.x_ptr[loc]+ip_fv].value;
    		    tmp_v.push_back(tmp);
                ++ip_fv;
            }
    	}
        for(size_t i = 0;i < tmp_v.size();i++)
        {
            prob.x[prob.x_ptr[loc]+i].index = tmp_v[i].index;
            prob.x[prob.x_ptr[loc]+i].value = tmp_v[i].value;
        }
        this->prob.y[loc] += 1;
        this->prob.x_interval[loc] = tmp_v.size();
    }
}

std::vector<double> Flexible_vector::FvtoVector() const
{
    std::vector<double> ret;
    for(size_t i = 0;i < prob.l;i++)
    {
        ret.push_back(prob.y[i]);
        size_t j = prob.x_ptr[i];
        size_t k = 1;
        while(k <= prob.max_feature)
        {
            if((k == prob.x[j].index)&&(j < prob.x_ptr[i]+prob.x_interval[i]))
            {
                ret.push_back(prob.x[j].value);
                j++;
                k++;
            }else
            {
                ret.push_back((double)0.0);
                k++;
            }
        }
    }
    return ret;
}

std::vector<std::string> Flexible_vector::Vector_DtoS(std::vector<double> data) const
{
    std::vector<std::string> data_s;
    for(size_t i = 0;i <= data.size();i++)
    {
        std::stringstream s;
        std::string ss;
        s<<data[i];
        s>>ss;
        data_s.push_back(ss);
    }
    return data_s;
}

// read in a problem (in svmlight format)

problem Flexible_vector::Read_file_without_index(std::string filename)
{
    std::cout<<"IN Read_file_without_index"<<std::endl;
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
        //std::cout<<std::endl;
        inst_max_index = 0;
        std::stringstream  lineStream(line);

        std::getline(lineStream, e,',');   //label
        if(e.empty())  //empty line
            exit_input_error(prob.l+1);
        prob.y.push_back(stringToNum<double>(e));
        //std::cout<<prob.y[prob.y.size()-1]<<',';
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
            //std::cout<<prob.x[prob.x.size()-1].value<<',';
            elements++;
        }
        prob.x_interval.push_back(inst_max_index);
        if(inst_max_index > max_index)
            max_index = inst_max_index;
        prob.l++;
    }
    prob.max_feature = max_index;
    return prob;
}

void Flexible_vector::output_problem(std::string filename, problem prob)
{
    std::ofstream fout(filename.c_str());
    int i;
    std::cout<<"IN output_problem"<<std::endl;
    
    for(i=0;i<prob.l;i++)
    {
        //fout.unsetf(std::ios::fixed);
        fout<<prob.y[i]<<',';
        std::cout<<prob.y[i]<<',';
        for(int j=prob.x_ptr[i];j<prob.x_ptr[i]+prob.x_interval[i];j++)
        {
            if(j == (prob.x_ptr[i]+prob.x_interval[i]-1))
            {
                //fout<<std::fixed<<std::setprecision(1)<<prob.x[j].value<< std::endl;
                fout<<prob.x[j].value<< std::endl;
                std::cout<<prob.x[j].value<< std::endl;
            }else
            {
                //fout<<std::fixed<<std::setprecision(1)<<prob.x[j].value<<',';
                fout<<prob.x[j].value<<',';
                std::cout<<prob.x[j].value<<',';
            }
        }
    }
}

void Flexible_vector::print_fv()
{
    //std::ofstream fout(filename.c_str());
    int i;
    //std::cout<<"IN print_fv"<<std::endl;
    if(this == NULL)
    {
        std::cout<<"Flexible_vector is NULL"<<std::endl;
    }else
    {
        for(i=0;i<prob.l;i++)
        {
            //fout.unsetf(std::ios::fixed);
            //fout<<prob.y[i]<<',';
            std::cout<<prob.y[i]<<' ';
            if(prob.y[i] == -1)
                std::cout<<std::endl;
            else{
                for(int j=prob.x_ptr[i];j<prob.x_ptr[i]+prob.x_interval[i];j++)
                {
                    if(j == (prob.x_ptr[i]+prob.x_interval[i]-1))
                    {
                        //fout<<std::fixed<<std::setprecision(1)<<prob.x[j].value<< std::endl;
                        //fout<<prob.x[j].value<< std::endl;
                        std::cout<<prob.x[j].value<< std::endl;
                    }else
                    {
                        //fout<<std::fixed<<std::setprecision(1)<<prob.x[j].value<<',';
                        //fout<<prob.x[j].value<<',';
                        std::cout<<prob.x[j].value<<' ';
                    }
                }
            }
        }
        std::cout<<std::endl;
    }
}
