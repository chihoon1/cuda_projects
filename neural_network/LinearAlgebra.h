#include <iostream>
using namespace std;
#pragma once


class Matrix {
    // row-major matrix
    public:
        float* elem;  // 1D representation of Matrix element
        int num_row;
        int num_col;
        Matrix(int num_row, int num_col) {
            this->num_row = num_row;
            this->num_col = num_col;
            this->elem = (float*) malloc(num_row * num_col * sizeof(float));
        }
        float* get_ptr(int row, int col);
};