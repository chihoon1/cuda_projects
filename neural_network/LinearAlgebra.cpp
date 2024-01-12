#include <cstdio>
#include <stdlib.h>
#include <cmath>
#include "LinearAlgebra.h"
using namespace std;

float* Matrix::get_ptr(int row, int col) {
    // return the pointer of the given position in the matrix
    return &this->elem[row * this->num_row + col];
}