import torch
import torch.nn.functional as F

a=torch.randn(10,7,12,8)
b=torch.randn(10,9,12,8)
c=torch.cat((a,b),1)
a=torch.randn(10,32,12,10)
b=torch.randn(32,32,3,1)
c=torch.nn.functional.conv2d(a,b,padding=[1,0])

print(c.shape)