# The Single Layer Perceptron Problem And Its Solution

It is better known under the name XOR problem and it is the problem of a classical single layer perceptron
being unable to solve a problem whose data is not linear separable. 

Usually the solution is a hidden layer and it is said that a hidden layer has to be trained with something
called back propagation. However for this an activation function is required which is differentiable. 

However there is a much simpler solution to that no one actually talks about. 

The idea is to instead of waiting for a hidden layer to somehow magically emerge by backpropagation
the hidden layer is replaced what I call **prefilters**. 

Prefilters consists of filters which filter part of the input. In the case of the digit recognition
there are five such filters. All take 2 x 2 inputs.

```
   *.  .*
   .*  *.    diagonal lines
   
   ** ..
   .. **     horizontal lines
   
   *.  .*
   *.  .*    vertical lines
   
   *.  .*  ..  ..
   ..  ..  *.  .*    single dots
   
   **  **  *.  .*
   *.  .*  **  **    corners
```

The **prefilters** convert the 15 input neurons into 30 prefiltered output neurons and suddenly the digit recognition
becomes linear separable. 

All of these prefilters aren't linear separable either so every filter has to be broken up into two or four (for single dots and corners)
sub-filters that again have to be connected. However this way they are solvable like the XOR problem.  


# Implementations

The implementantions and solutions are found in the following directories or folders:

   * **xorproblem** contains an implementation and explanation of the XOR problem
   * **linearSeparable** contains implmentations that proof that the digit recognition problem is not solvable by linear separation
   * **prefiltered** contains implementations of the **prefilters** and the solution to the digit recoginition problem 
   
   The digit recognition program is further available for trying out on  https://codepen.io/mahagugu/full/xbxOvwB .
   
   

