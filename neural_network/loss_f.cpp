#include <cstdio>
#include <stdlib.h>
#include <cmath>
#include "Activations.h"
using namespace std;


class CrossEntropy {
    public:
        float value;
        CrossEntropy(float value = 0) {  // Constructor with default value
            this->value = value;
        }
        void set_value(float new_val = 0) {
            this->value = new_val;
        }
        float compute_loss(float* y, float* pred, int output_shape);
        float derivative(float *avg_preds);
};

float CrossEntropy::compute_loss(float* y, float* pred, int output_shape) {
    // Compute Cross Entropy loss with the given target label and prediction
    // Here, CrossEntropy formula follows pytorch. i.e., softmax will be applied to pred before taking log of it
    // param: y(array aka vector) is the array of encoded target label provided in the dataset
    // param: pred(array) is model predicted value (i-th index == probability of falling in class i)
    // param: output_shape is the the shape output layer (1==scalar output. else, vector output)
    // y and preds must be the same shape
    // return computed entropy value(float)
    /*
    float softmax_pred[output_shape];
    float softmax_denom = 0;
    float ent_val = 0;
    if (output_shape > 1) {  // multiclass
        for (int i = 0; i < output_shape; i++) {
            softmax_denom = softmax_denom + exp(pred[i]);
        }
        for (int i = 0; i < output_shape; i++) {
            ent_val = ent_val - (y[i] * log( exp(pred[i]) / softmax_denom));
        }
    } else {  // binary crossentropy
        softmax_denom = exp(*pred) + exp(1.0 - *pred);
        ent_val = -( *y * log( exp(*pred)/softmax_denom ))\
                    - ( (1.0 - *y) * log( exp( (1.0 - *pred)/softmax_denom )));
    }
    return ent_val;
    */
   Softmax activation = Softmax();
   float ent_val = 0;
   float *softmax_pred = activation.apply(pred, output_shape);
   for (int i = 0; i < output_shape; i++) {
        printf("softmax %d: %.3f\n", i, softmax_pred[i]);
   }
   for (int i = 0; i < output_shape; i++) {
        ent_val = ent_val - (y[i] * log(softmax_pred[i]));
    }
    return ent_val;
}

float CrossEntropy::derivative(float *avg_preds) {
    float avg_loss = this->value;
    //exp(x)*((sum exp) - 1) / (sum exp)^2
    // Note: each preds[j] contribute to the softmax(preds[i]) for all i because of the denominator
    // So, need to for all i sum all partial derivatives of sofatmax(preds[i]) w.r.t preds[j]
}

/*
float process_batch(float* y, float* preds, int batch_size, int output_shape) {
    // Forward propagate neural network to compute cross entropy loss with given data and model prediction
    // param: y(array aka vector) is the array of target labels provided in the dataset
    // param: preds(array) is model predicted value (preds[i] derived using max of ouput vector)
    // param: batch_size(int) is the number of data in the current batch of data
    // y and preds must be the same shape
    // if y and preds are 2D, then multiclass classification
    float res_val = 0;
    for (int i = 0; i < batch_size; i++) {
        res_val = res_val + entropy(y, preds, output_shape);
    }
    return res_val;
}
*/




int main() {
    CrossEntropy l;
    printf("l value: %.3f\n", l.value);
    /* working
    float *a, *b;
    a = (float*) malloc(9*sizeof(float));
    b = (float*) malloc(9*sizeof(float));
    for (int i = 0; i < 9; i++) {
        a[i] = 0.1*9;
        if (i%2) {
            b[i] = 1;
        }
    }
    */
    int batch_size = 9;
    float a[9] = {0.77, 0.08, 0.94, 0.84, 0.68, 0.97, 0.02, 0.33, 0.24};
    float b[9] = {0, 0, 0, 0, 0, 0, 0, 1, 0};
    
    float res = l.compute_loss(b, a, 9);
    printf("res %.3f\n", res);


}