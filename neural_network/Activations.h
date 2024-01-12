#include <iostream>
using namespace std;
#pragma once

// Define classes of Activation Functions
class Softmax {
    public:
        int input_shape;
        Softmax(int input_shape = 0) {
            this->input_shape = input_shape;
        }
        void setInputShape(int input_shape) {
            this->input_shape = input_shape;
        }
        float* apply(float *input_vector, int input_shape);
        //float* derivative(float *softmax_output, int input_shape, int output_shape);
        Matrix* derivative(float *softmax_output, int input_shape, int output_shape);
};
