'''
1. Actiavtion Function
'''

'''
2. Shape(D)/Shape(D, H)
shape simply means the dimensions (size) of an array/vector/matrix
Vector = 1D array or 1d Tensor
Matrix = 2D array
1D Array:
x = [10, 20, 30, 40]
┌────┬────┬────┬────┐
│ 10 │ 20 │ 30 │ 40 │
└────┴────┴────┴────┘
(4,) i.e. 4 elements, 1 dimension
2D Array (Matrix):
A = [
      [1, 2, 3],
      [4, 5, 6]
    ]
┌───┬───┬───┐
│ 1 │ 2 │ 3 │
├───┼───┼───┤
│ 4 │ 5 │ 6 │
└───┴───┴───┘
(2,3) i.e. 2 rows, 3 columns, 2 dimension
3D Array:
B = [
      [
        [1,2],
        [3,4]
      ],

      [
        [5,6],
        [7,8]
      ]
    ]
Layer 1
┌───┬───┐
│ 1 │ 2 │
├───┼───┤
│ 3 │ 4 │
└───┴───┘
Layer 2
┌───┬───┐
│ 5 │ 6 │
├───┼───┤
│ 7 │ 8 │
└───┴───┘
(2,2,2) i.e. 2 matrices, each matrix has 2 rows, each row has 2 columns
[1,2,3]                    # 1D array
[[1,2],[3,4]]             # 2D array
[[[1],[2]],[[3],[4]]]     # 3D array
'''

'''
3. RNN internal layers
'''

'''
4. Vanilla RNN
A Vanilla RNN is the basic, original RNN architecture. The word vanilla in machine learning means:
"The simplest standard version without any special modifications."
'''

'''
5. How to batch?
Batching means: instead of processing one input at a time, you process multiple inputs together for efficiency
Single time-step (NO batching)

You currently have:
x_t: (D,)
h_prev: (H,)

Example:
one sentence → one word at a time

So at time t:
x_t = [1.0, 2.0]   # one sample

What is batching?
Instead of 1 input, you take B inputs together.

So shape becomes: (B, D)

Where:
B = batch size (number of samples)
D = features

Example
Suppose batch size = 3

x_t =
[
 [1, 2],   # sample 1
 [3, 4],   # sample 2
 [5, 6]    # sample 3
]

Shape: (3, 2)
'''

'''
6. Single time_step vector
"Single time-step vector" means the function should process only one input vector at one time step, not multiple inputs at once.
'''

'''
7. why Pass in End?
In RNN problems, when you see something like “Pass in End” or “Pass-in End condition”, it usually means:
This function handles only ONE step of the RNN, not the whole sequence.
'''

'''
8. * vs @
* (Star operator)
Meaning: element-wise multiplication OR unpacking

Case 1: Element-wise multiplication (NumPy arrays):

import numpy as np
a = np.array([1, 2, 3])
b = np.array([10, 20, 30])
print(a * b)

Output: [10 40 90]

Each element multiplies individually: 1×10, 2×20, 3×30

Case 2: Scalar multiplication

a * 2

Output: [2 4 6]

Case 3: Python lists (NOT math)

[1,2,3] * 2

Output: [1,2,3,1,2,3]

This is repetition, not multiplication.

@ (Matrix multiplication operator)
Meaning: Linear algebra multiplication (dot product / matrix multiply) #Used in NumPy, PyTorch, ML.

Example: vector × matrix

import numpy as np

x = np.array([1, 2])
W = np.array([[3, 4],
              [5, 6]])

print(x @ W)

Output: [13 16]

Because:
1×3 + 2×5 = 13
1×4 + 2×6 = 16


Example: matrix multiplication

A = np.array([[1,2],
              [3,4]])

B = np.array([[5,6],
              [7,8]])

print(A @ B)
'''

'''
9. np.tanh()
np.tanh() is a NumPy function that applies the hyperbolic tangent (tanh) activation to numbers
It transforms any number into a value between [-1, +1]
np.tanh() (NumPy version)
import numpy as np
np.tanh(x)
tanh() only (Python / other libraries)
import math
math.tanh(1)
'''

'''
10. how to take input as matrix shape(D) when shape is unknown?
What you actually do in practice

You always receive input first:
x = np.array([1.0, 2.0, 3.0, 4.0])

Now NumPy automatically sets:
x.shape = (4,)
D = 4
So D is discovered at runtime, not fixed beforehand.

How to handle unknown D safely
Step 1: take input
x = np.array(input_data)
Step 2: infer D
D = x.shape[0]
(((Why index 0?
Because Python indexing starts from 0.
So:
Expression	Meaning
shape[0]	first dimension
shape[1]	second dimension
shape[2]	third dimension)))

RNN case (important)
Your function should NOT assume D.

Instead:
def rnn_step(x_t, h_prev, Wx, Wh, b):
    #D is already embedded in Wx shape here
    (((it means: You don’t need to manually know or pass D, because it is already determined by the weight matrix Wx
        We have:
        x_t: (D,)
        Wx  : (D, H)
        So Wx already contains D.)))

You never manually need D
so basically in this question shape is already declared
'''

'''
11. how to initiate weight with that shape?
Answer given above
'''

'''
12. how can I declare h_t as that shape only when 1<= D,H<=256?
Answer given above
'''

#Answer:
import numpy as np

def rnn_step_forward(x_t, h_prev, Wx, Wh, b):
    """
    Returns: h_t of shape (H,)
    """
    pre_act = (x_t@Wx)+(h_prev@Wh)+b
    h_t = np.tanh(pre_act)
    return h_t
    pass
