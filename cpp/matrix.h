#ifndef MATRIX_H_INCLUDE_
#define MATRIX_H_INCLUDE_

#include <stdexcept>
#include <iostream>
#include "utils.h"

namespace myccg {

#define IGNORE -1


// just wrap a pointer to 2d matrix
// does not own the pointer and not delete it
// in the deconstructor
template<typename T>
class Matrix
{
public:
    Matrix(T* data, int row, int column)
        : data_(data), row_(row), column_(column),
          size_(row * column), own_(false) {}

    Matrix(T* begin, T* end)
        : data_(begin), column_(1) {
        T* beg = begin;
        size_ = 0;
        while (beg++ != end) size_++;
        row_ = size_;
    }

    Matrix(int row, int column)
        : data_(new T[row * column]), row_(row), column_(column),
          size_(row * column), own_(true) {
              // for (int i = 0; i < size_; i++)
              //     data_[i] = T(0);
          }

    ~Matrix() {
        if (own_) delete[] data_;
    }

    friend std::ostream& operator<<(std::ostream& out, Matrix<T> m) {
        for (int i = 0; i < m.row_; i++) {
            for (int j = 0; j < m.column_; j++) {
                out << (j ? ", " : "") << m.data_[i * m.column_ + j];
            }
            out << std::endl;
        }
        out << std::endl;
        return out;
    }

    Matrix<T> Reshaped(int row, int column) {
        row_ = row;
        column_ = column;
        return *this;
    }

    T* Ptr() {
        return data_;
    }

    T& operator() (int row, int column) const {
        return data_[row * column_ + column];
    }

    int ArgMax() const {
        return utils::ArgMax(data_, data_ + size_);
    }

    int ArgMax(int row) const {
        return utils::ArgMax(data_ + (row * column_),
                data_ + (row * column_ + column_));
    }

    T Max() const {
        int idx = ArgMax();
        return data_[idx];
    }

    int Max(int row) const {
        int idx = ArgMax(row);
        return data_[row * row_ + idx];
    }

    int Size() const { return size_; }
    int Column() const { return column_; }
    int Row() const { return row_; }

private:
    T* data_;
    int row_;
    int column_;
    int size_;
    bool own_;
};

} // namespace myccg

#endif
