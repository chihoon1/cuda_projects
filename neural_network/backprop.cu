#include <cuda.h>
#include <studio.h>
#include <stdlib.h>
#include <math.h>

// For feedforward NN
// list of tensors(2d) where each tensor represents a layer from the first to the last layer
// also, list includes activation layer but with weight zero.
// tensor serves as weight to a neuron in the prev layer to a neuron in the next layer
// and also as a adjacency matrix of a graph where non-zero weigth represents connection
//      row: outgoing neuron(prev).   col: incoming neuron(next)

// compute loss. Outer loop == epoch
//  In a middle loop where looping over layers
//      In one iteration, each weight is handled by one processor in a SM in GPU (P)
//      But if more processors than all num_weights in a layer,
//               use remaining processors to compute loss with the same weights but different data (P)
//  Once loss is computed, compute gradient with another middle loop (P)
//  use trainable weights as adjacency matrix to compute gradient (don't forget activation derivative)
//  Apply gradients parallel way by using one processor in a SM per one weight (P)

// Key: if more blocks needed than num_blocks limit, then split kernel function calls in multiple times
//          to use less than limited block amount per a kernel call while utilizing all processors

class 