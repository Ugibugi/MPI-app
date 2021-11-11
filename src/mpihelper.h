#ifndef MPIHELPER_H
#define MPIHELPER_H
#include <mpi.h>
#include <iostream>
#include <vector>
#include <type_traits>
struct MpiInfo {
    int worldSize=0;
    int worldRank=0;
    std::string processorName;
    int nextAddr()const{
        //modulo plus
        return (worldRank+1)%worldSize;
    }
    int prevAddr() const{
        //modulo minus
        return (worldRank + (worldSize-1))%worldSize;
    }
};
enum Tag {
    DATA_TAG,
    INFO_TAG
};
enum Info{
    NO_INFO,
    BEGIN_WORK,
    KILL_SELF
};
template<typename T>
struct View{
    const T* _data;
    size_t _size;
    int size() const {
        return _size;
    }
    const T* data() const {
        return _data;
    }
};

std::ostream& operator<<(std::ostream& os, const MpiInfo& i)
{
    os << i.processorName << "(rank=" << i.worldRank <<" of size="<< i.worldSize << ')';
    return os;
}
template<typename T>
std::ostream& operator<<(std::ostream& os, const std::vector<T>& i)
{
    os << '{';
    for(const auto& e : i) os << e <<", ";
    return os << '}';
}
template<typename T>
constexpr MPI_Datatype MpiType()=delete;

#define TypeMapMpi(mpitype, ctype)                  \
    template<>                                          \
    constexpr MPI_Datatype MpiType<ctype>(){                              \
        return mpitype;  \
    }                                                   \

TypeMapMpi(MPI_INT,int);
TypeMapMpi(MPI_UNSIGNED_LONG,unsigned);
TypeMapMpi(MPI_FLOAT,float);
TypeMapMpi(MPI_DOUBLE,double);
TypeMapMpi(MPI_CHAR,char);


#undef TypeMapMpi

class Mpi {
public:
    Mpi(const int* argc=nullptr, char*** argv=nullptr){
        MPI_Init(argc,argv);
        MPI_Comm_size(MPI_COMM_WORLD,&_info.worldSize);
        MPI_Comm_rank(MPI_COMM_WORLD,&_info.worldRank);
        char processor_name[MPI_MAX_PROCESSOR_NAME];
        int name_len;
        MPI_Get_processor_name(processor_name, &name_len);
        _info.processorName = std::string(processor_name,name_len);
    }
    template<typename T>
    void send(const T& item,int dest, int tag=0) const
    {
        MPI_Send(&item,1,MpiType<T>(),dest,tag,MPI_COMM_WORLD);
    }
    template<typename T>
    void sendAll(const T& item, int tag=0) const
    {
        for(int i=_info.nextAddr();i!=_info.worldRank;i=(i+1)%_info.worldSize)
        {
            send(item,i,tag);
        }
    }


    template<typename T>
    void sendv(const T& container, int dest, int tag=0) const
    {
        const auto* begin = std::data(container);
        auto el = *begin;
        MPI_Send(begin,
                 static_cast<int>(std::size(container)),
                 MpiType<decltype (el)>(),dest,tag,MPI_COMM_WORLD);
    }
    template<typename T>
    T recv(int source,int tag=MPI_ANY_TAG) const
    {
        T item;
        MPI_Recv(&item,1,MpiType<T>(),source,tag,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        return item;
    }
    template<typename T>
    std::vector<T> recvv(int source,int tag=MPI_ANY_TAG) const
    {
        MPI_Status status;
        auto mpiType = MpiType<T>();
        MPI_Probe(source,tag,MPI_COMM_WORLD,&status);
        int count;
        MPI_Get_count(&status,mpiType,&count);

        std::vector<T> vec(count);
        MPI_Recv(vec.data(),count,MpiType<T>(),source,tag,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        return vec;
    }
    ~Mpi(){
         MPI_Finalize();
    }
    const MpiInfo& info() const {return _info;}
private:
    MpiInfo _info;
};

template <typename T>
class MulSum{
    const Mpi* _mpi;
    int _remoteRank;
public:
    MulSum(const Mpi* mpi, int remoteRank) : _mpi(mpi), _remoteRank(remoteRank){}
    void begin(const View<T>& v1, const View<T>& v2){
        _mpi->send<int>(BEGIN_WORK,_remoteRank,INFO_TAG);
        _mpi->sendv(v1,_remoteRank,DATA_TAG);
        _mpi->sendv(v2,_remoteRank,DATA_TAG);
    }
    T await(){
        return _mpi->recv<T>(_remoteRank,DATA_TAG);
    }
};

#endif // MPIHELPER_H
