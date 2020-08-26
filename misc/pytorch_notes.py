import torch
import torch.nn as nn
import torch.optim as optim
import math


# Pytorch's Cross-entropy loss
"""
Prediction shape required: Num_Samples, Prediction Scores across K classes
Target shape required: Num_Samples

In case of Translation Model: Num_Samples = Batch_size * Seq_len 
"""
loss = nn.CrossEntropyLoss()
# Scores for 5 classes
input = torch.randn(1, 5, requires_grad=True)
# >> tensor([[ 0.0326,  0.3639,  0.0207, -0.6089, -0.1124]], requires_grad=True)
target = torch.empty(1, dtype=torch.long).random_(5)  # >> 3
output = loss(input, target)  # >> tensor(2.2044, grad_fn=<NllLossBackward>)

"""
Inside implementation of nn.CrossEntropyLoss  
>>> m = nn.LogSoftmax(dim=1)
>>> loss = nn.NLLLoss()
>>> # input is of size N x C = 3 x 5
>>> input = torch.randn(3, 5, requires_grad=True)
>>> target = torch.tensor([1, 0, 4])
>>> output = loss(m(input), target)
"""
# Loss = -log(exp(input[target])/sum(exp(iter(input)))
_output = -math.log(math.exp(-0.6089)/sum([math.exp(t) for t in input.tolist()[0]]))  # 2.2043837815118175


# Pytorch's torch.nn.utils.clip_grad_norm_
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
"""
grad_mat = x.grad
if norm(grad_mat) > max_norm:
    clipped_grad = max_norm * grad_mat/norm(grad_mat)
else:
    continue

Gradient clipping ensures the gradient_mat has norm <= max_norm               
"""


# Comparison of output of Dropout layer b/w model.train() & model.eval()
"""
Since dropout has different behavior during training and test, you have to scale the activations sometime.
Imagine a very simple model with two linear layers of size 10 and 1, respectively.
If you don’t use dropout, and all activations are approx. 1, your expected value in the output layer would be 10.
Now using dropout with p=0.5, we will lose half of these activations, 
so that during training our expected value would be 5. 
As we deactivate dropout during test, the values will have the original expected value of 10 again 
and the model will most likely output a lot of garbage.
One way to tackle this issue is to scale down the activations during test simply by multiplying with p.
Since we prefer to have as little work as possible during test, 
we can also scale the activations during training with 1/p, which is exactly what you observe.
"""
drop = nn.Dropout()
x = torch.ones(1, 10)

# Train mode
drop.train()
print(drop(x))
# tensor([[0., 2., 2., 0., 0., 2., 2., 0., 0., 0.]])

# Eval mode
drop.eval()
print(drop(x))
# tensor([[1., 1., 1., 1., 1., 1., 1., 1., 1., 1.]])


