#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>

inline std::vector<std::string> ReadCSV(std::string pwd){
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

int main(){
    std::vector<std::string> data = ReadCSV("./data/iris.csv");
    std::vector<float> real;
    for (size_t i = 0; i < data.size(); ++i)
        real.push_back(atof(data[i].c_str()));
    std::vector<float> mi;
    mi.resize(4);
    std::fill(mi.begin(), mi.end(), 100000);
    std::vector<float> ma;
    ma.resize(4);
    std::fill(ma.begin(), ma.end(),0);
    for (size_t i = 0; i < real.size(); ++i){
        mi[(i+1)%4] = std::min(mi[(i+1)%4], real[i]);
        ma[(i+1)%4] = std::max(ma[(i+1)%4], real[i]);
    }

    for (size_t i = 0; i < real.size(); ++i){
        real[i] = (real[i] - mi[(i+1)%4]) / (ma[(i+1)%4] - mi[(i+1)%4]);
    }

    std::ofstream file;
    file.open("./data/iris_n.csv");	
    for (size_t i = 0; i < real.size(); ++i){
        char ch = (i+1)%4 == 0 ? '\n' : ',';
        file << real[i]<< ch;
    }
    file.close();
}
