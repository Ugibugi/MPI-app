#ifndef MATRIX_H
#define MATRIX_H
#include <vector>
#include <assert.h>
template<typename T>
struct Mat{
    typedef Mat<T> this_t;
    typedef Mat<T> col_t;
    typedef Mat<T> row_t;
    int Rows;
    int Cols;
    std::vector<T> e;
    Mat(): Mat(0,0){}
    Mat(int N): Mat(N,N){}
    Mat(int rows, int cols): Rows(rows), Cols(cols),e(cols*rows){e.resize(cols*rows,0.0);}
    T& at(int i,int j){return e[i+(Rows*j)];}
    const T& at(int i,int j) const {return e[i+(Rows*j)];}
    bool operator==(const this_t& m2) const {return std::equal(e.cbegin(),e.cend(),m2.e.cbegin());}
    std::vector<row_t> getRows() const {
        std::vector<row_t> ret(Rows,row_t(1,Cols));
        for(int i=0;i<Rows;i++){
            for(int j=0;j<Cols;j++){
                ret[i].at(0,j) = this->at(i,j);
            }
        }
        return ret;
    }
    std::vector<col_t> getCols() const {
        std::vector<col_t> ret(Cols,col_t(Rows,1));
        for(int j=0;j<Cols;j++){
            for(int i=0;i<Rows;i++){
                ret[j].at(i,0) = this->at(i,j);
            }
        }
        return ret;
    }
    void free(){
        e.clear();
        e.shrink_to_fit();
    }
};
template<typename T>
void matmul(Mat<T>& dest, const Mat<T>& a, const Mat<T>& b)
{
    for(int i=0;i<dest.Rows;i++)
    {
        for(int j=0;j<dest.Cols;j++)
        {
            dest.at(i,j)=0;
            for(int k=0;k<b.Rows;k++)
            {
                dest.at(i,j) += a.at(i,k)*b.at(k,j);
            }
        }
    }
}


#endif // MATRIX_H
