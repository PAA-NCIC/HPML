#include <Kmeans.h>

namespace MLtool{
    void Kmeans:: add(Kmeans rhs){
        for (auto element: rhs.monoid){
            for(auto &base: this->monoid){
                if(base == element)
                    base = base + element;
                else
                    this->monoid.push_back(element);
            }
        }
    }

    void Kmeans:: init(){
        monoid.clear();
        conf = Config(1e-3, 10);
    }

    bool Kmeans:: beginDataScan(){
    }
}
