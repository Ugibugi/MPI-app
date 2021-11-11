#include <string>
#include <iostream>
#include <iomanip>
#include <mpi.h>
#include "mpihelper.h"
#include "matrix.h"
#include <chrono>
#include <random>
#include <numeric>
using Clock = std::chrono::steady_clock;
auto mili = [](auto a){return std::chrono::duration_cast<std::chrono::milliseconds>(a).count();};
auto dist(double min, double max)
{
    static std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_real_distribution<double> dis(min,max);
    return [dis,gen]()mutable{
        return dis(gen);
    };
}
auto dist(int min, int max)
{
    static std::random_device rd;  // Will be used to obtain a seed for the random number engine
    std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    std::uniform_int_distribution<int> dis(min,max);
    return [dis,gen]()mutable{
        return dis(gen);
    };
}

int measure(const char* label){
    static auto lastT = Clock::now();
    auto now = Clock::now();
    auto T = mili(now - lastT);
    std::cerr << label << T <<"ms\n";
    lastT = Clock::now();
    return T;
}
int master_main(const Mpi& mpi);
int slave_main(const Mpi& mpi);
int main()
{
    Mpi mpi;
    if(mpi.info().worldSize < 2)
    {
        std::cout << "At least 2 workers are needed for this program\n";
        return -1;
    }
    if(mpi.info().worldRank == 0) return master_main(mpi);
    else return slave_main(mpi);
}
int master_main(const Mpi& mpi)
{
    std::cerr << std::setprecision(3) << std::boolalpha;
    measure("Start of the main program");
    Mat<double> m1(1000),m2(1000),m3(1000),m4(1000);
    std::generate(m1.e.begin(),m1.e.end(),dist(0,5));
    std::generate(m2.e.begin(),m2.e.end(),dist(0,5));
    measure("Initializing took: ");
    //calculate M1 * M2
    matmul(m3,m1,m2);
    auto t1 = measure("Matrix multiplication(1 core) took: ");
    auto rows = m1.getRows();
    auto cols = m2.getCols();
    m1.free();
    m2.free();
    struct Computation{
        Mat<double>* a,*b;
        int row,col;
    };
    std::vector<Computation> pairs;
    for(int i=0;i<rows.size();i++){
        for (int j=0;j<cols.size();j++){
            pairs.push_back(Computation{&rows[i],&cols[j],i,j});
        }
    }

    std::vector<MulSum<double>> workers;
    for(int i=mpi.info().nextAddr();i != mpi.info().worldSize; i++ ){
        //std::cerr <<" Creating worker handler "  << i <<" \n";
        workers.push_back(MulSum<double>{&mpi,i});
    }
    measure("Work decomposition took: ");

    for(int i=0;i<pairs.size();){
        const auto items_left = pairs.size() - i;
        const auto n = std::min(workers.size(),items_left);
        for(int k=0;k<n;k++){
            auto pair = pairs[i+k];
            workers[k].begin(View<double>{pair.a->e.data(),pair.a->e.size()},
                             View<double>{pair.b->e.data(),pair.b->e.size()});
        }
        for(int k=0;k<n;k++){
            m4.at(pairs[i+k].row,pairs[i+k].col) = workers[k].await();
        }
        i+=n;
    }
    auto t2 = measure("Matrix multiplication(8 cores) took: ");
    std::cerr << "M1 == M2: " << (m4==m3) <<'\n';
    measure("Matrix comparison took: ");
    std::cerr << "Speedup: " << (double)t1/(double)t2 << " \n";
    mpi.sendAll<int>(KILL_SELF,INFO_TAG);
    return 0;
}
int slave_main(const Mpi& mpi)
{
    Info msg = NO_INFO;
    int util = 0;
    while(msg == NO_INFO){
        msg = static_cast<Info>(mpi.recv<int>(MPI_ANY_SOURCE,INFO_TAG));
        if(msg == KILL_SELF)
        {
            break;
        }
        else if(msg == BEGIN_WORK)
        {
            util++;
            msg=NO_INFO;
            auto v1 = mpi.recvv<double>(MPI_ANY_SOURCE,DATA_TAG);
            auto v2 = mpi.recvv<double>(MPI_ANY_SOURCE,DATA_TAG);
            auto result = std::inner_product(v1.begin(),v1.end(),v2.begin(),0.0);
            mpi.send(result,0,DATA_TAG);
        }
    }
    std::cout << "Process " << mpi.info() << " was utilised " << util <<" times\n";
    return 0;
}
