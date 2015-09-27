#include<vector>
#include<tuple>
namespace MLtool{

class element{
  private:
    int ith;
    double x,y;
    int num;
  public:
    element(int ith, double x, double y, int num):ith(ith),x(x),y(y),num(num){}
    bool operator == (element &rhs){
        return ith == rhs.ith;
    }
    element operator + (element &rhs){
        return element(ith, x+rhs.x, y+rhs.y, num+rhs.num);
    }
    std::tuple<int, int> centroids(){
        return std::make_tuple(x/num,y/num);
    }
};


class Config{
  public:
    double eps;
    int cluster_number;
    Config(double eps, int cluster_number):eps(eps),cluster_number(cluster_number){}
};

class Kmeans{
  public:
    void init();
    bool beginDataScan();
    void serialize(int mode);
    void readObject();
    void processRecord();
    void mergeResults();
    void endDataScan();
    Config conf;
  private: 
    std::vector<element> monoid;
    void add(Kmeans rhs);
};

}
