#include <cstdio>
#include <stdlib.h>
#include <cmath>
#include "Activations.h"
using namespace std;
#include "LinearAlgebra.h"

/*
class Softmax {
    public:
        int input_shape;
        Softmax(int input_shape = 0) {
            this->input_shape = input_shape;
        }
        void setInputShape(int input_shape) {
            this->input_shape = input_shape;
        }
        float apply(float *input_vector, int input_shape);

};
*/



float* Softmax::apply(float *input_vector, int input_shape) {
    // Apply softmax function on input_vector
    // param: input_vector(a ptr to float array) is a input vector where softmax will be applied
    // param: input_shape(int) is a shape of input vector
    // return the same shape vector with value equal to softmax(input)

    float softmax_denom = 0;
    for (int i = 0; i < input_shape; i++) {
        softmax_denom = softmax_denom + exp(input_vector[i]);
    }
    float *softmaxed = (float*) malloc(input_shape * sizeof(float));
    for (int i = 0; i < input_shape; i++) {
        softmaxed[i] = exp(input_vector[i]) / softmax_denom;
    }
    return softmaxed;
}


//float* Softmax::derivative(float *softmax_output, int input_shape, int output_shape) {
Matrix* Softmax::derivative(float *softmax_output, int input_shape, int output_shape) {
    // param: softmax_output(ptr to a vector processed by softmax function)
    // param: input_shape is the shape of input vector of the softmax_output
    // param: output_shape is the shape of the softmax_output vector
    // softmax: R^d -> R^d
    // Hence, jacobian of softmax will be in the shape of output_shape * output_shape
    // Our jacobian(transposed. col-major, left-multiplied): row = input variable x_j, col = function f_i
    //float *jacobian[output_shape][input_shape];
    //float *jacobian;
    Matrix jacobian_t = Matrix(output_shape, output_shape);
    for (int i = 0; i < jacobian_t.num_row; i++) {
        for (int j = 0; j < jacobian_t.num_col; j++) {
            if (i == j) {  // diagonal entries
                jacobian_t.elem[i * jacobian_t.num_row + j] = softmax_output[j]*(1-softmax_output[i]);
            } else {  // non-diagonal entries
                jacobian_t.elem[i * jacobian_t.num_row + j] = -softmax_output[j]*softmax_output[i];
            }
        }
    }
    return &jacobian_t;
}