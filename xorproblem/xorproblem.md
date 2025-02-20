### Solving the XOR problem of the original single layer perceptron

The idea is to break the problem into bits to be solvable by linear regression. 
I suppose that is already done in the hidden layer of neural networks but that 
comes with the need of more complex learning algorithms like back propagation 
and more complicate activation functions like sigmoid. 

## Very simple explanation 

The input is forwareded to two different neurons that work and are trained independently.
One neurons looks if there are two zeros and the other one if there are two ones. The output
of both is then forwarded to yet another neuron that has to figure out if any of the previous 
neurons fired or detected a pair the result is wrong. Actually, that is what a XOR does -- 
detecting if two equal inputs or a pair. The simple original perceptron from
Rosenblatt could not do that but that's a simple work around or solution to the problem. 

## Background to the solution

The idea came when writing the truth table for XOR. Usually a logic formula can 
be derived from the truth table by linking the entries with or. As this might 
be a bit too abstract for those not familiar with I give a sample.

# Truth table for XOR

|a | b|   |            |
|--|--|---|------------|
|0 | 0| 0 |            |
|0 | 1| 1 | = !a && b  |
|1 | 0| 1 | =  a && !b |
|1 | 1| 0 |            |

Thus the logical formula derived from the truth table for XOR is: (!a && b) || (a && !b)

As it turned out that !a && b and a && !b are difficult to learn for the simple perceptron
I decided to use:  !((!a && b) || (a && b)) which is equal to 
   (a nor b) nor (a and b)

## Conclusion

So I decided to create two simple perceptrons which perform the operations NOR and AND. The output
of these two is  the **filtered input** for yet another simple perceptron which is trained on XOR
on the **unfiltered** or **original input**.

As expected the new perceptron performed NOR and thus all three perceptrons together performed XOR.

It seems to me there are similarities with what a hidden layer in neural networks does. However here
this was done by *prefiltering* the input. I hope to generalize this concept to make it applicable 
for more complex tasks and I see potential in that it might reduce training of neural networks because
simple pretrained perceptrons can be used to filter inputs which might over several steps make 
the problem solvable by linear regression. 