import torch
import torch.nn as nn
import torch.nn.functional as F

initializer = nn.init.xavier_uniform_


class Gconv(nn.Module):
    def __init__(self, n_route, Ks, c_in, c_out):
        super(Gconv, self).__init__()
        self.n_route = n_route
        self.Ks = Ks
        self.kernel = torch.ones(self.n_route, self.Ks*self.n_route)
        self.theta = nn.Parameter(initializer(torch.randn(self.Ks*c_in, c_out)))

    def forward(self, x, theta, Ks, c_in, c_out):
        n = self.kernel.shape[0]
        x_tmp = x
        x_tmp = x_tmp.reshape(-1,n)
        x_mul=torch.matmul(x_tmp,self.kernel)
        #print(x_mul.shape, self.kernel.shape)
        x_mul=x_mul.reshape(-1,c_in,Ks,n)
        x_mul=torch.transpose(x_mul,1,3)
        x_mul=torch.transpose(x_mul,2,3)
        x_ker=x_mul.reshape(-1,c_in*Ks)
        x_gconv=torch.matmul(x_ker,self.theta)
        x_gconv=x_gconv.reshape(-1,n,c_out)
        x_gconv=torch.transpose(x_gconv,2,1)
        return x_gconv


class Temporal_conv_layer(nn.Module):
    def __init__(self,c_in,c_out,Kt,act_func):
        super(Temporal_conv_layer,self).__init__()
        self.wt_input=nn.Parameter(initializer(torch.randn(c_out,c_in,1,1)))
        self.wt_glu=nn.Parameter(initializer(torch.randn(2*c_out,c_in,Kt,1)))
        self.wt=nn.Parameter(initializer(torch.randn(c_out,c_in,Kt,1)))
        self.c_in=c_in
        self.c_out=c_out
        self.Kt=Kt
        self.sigmoid=torch.nn.Sigmoid()
        self.relu=nn.ReLU()
        self.act_func=act_func

    def forward(self,x):
        '''
        :param x: batch*in_channel*H*W
        :param c_in:
        :param c_out:
        :return:
        '''
        B,c,T,n=x.shape
        if self.c_in>self.c_out:
            x_input=F.conv2d(x,self.wt_input)
        elif self.c_in<self.c_out:
            temp_zero=torch.zeros(B,self.c_out-self.c_in,T,n)
            x_input=torch.cat((x,temp_zero),1)

        else:
            x_input=x

        x_input=x_input[:,:,self.Kt-1:T,:]


        if self.act_func=='glu':
            x_conv=F.conv2d(x,self.wt_glu)

            return (x_conv[:,0:self.c_out,:,:]+x_input)*self.sigmoid(x_conv[:,-self.c_out:,:,:])
        else:
            x_conv=F.conv2d(x,self.wt)
            if self.act_func=='linear':
                return x_conv
            elif self.act_func=='relu':
                #return torch.relu(x_conv+x_input)
                return F.relu(x_conv+x_input)
            elif self.act_func=='sigmoid':
                #return torch.sigmoid(x_conv)
                return F.sigmoid(x_conv)
            else:
                raise ValueError(f'ERROR: activation function"{self.act_func}" is not defined.')


class Spatio_conv_layer(nn.Module):
    def __init__(self,c_in,c_out,Ks,n_route):
        super(Spatio_conv_layer,self).__init__()
        self.ws_input = nn.Parameter(initializer(torch.randn(c_out, c_in, 1, 1)))
        self.ws = nn.Parameter(initializer(torch.randn(Ks*c_in,c_out)))
        self.c_in = c_in
        self.c_out = c_out
        self.Ks = Ks
        self.sigmoid = torch.nn.Sigmoid()
        self.relu = nn.ReLU()
        self.gconv = Gconv(n_route,Ks,c_in,c_out)

    def forward(self,x):
        B, c, T, n = x.shape
        if self.c_in > self.c_out:
            x_input = F.conv2d(x, self.ws_input)
        elif self.c_in < self.c_out:
            temp_zero = torch.zeros(B, self.c_out - self.c_in, T, n)
            x_input = torch.cat((x, temp_zero), 1)
        else:
            x_input = x
        x_input = x_input[:, :, self.Ks - 1:T, :]

        x_input2=torch.transpose(x_input,2,1)

        #x_input=torch.transpose(x_input,2,3)
        x_input3=x_input2.reshape(-1,c,n)

        x_gconv=self.gconv(x_input3,self.ws,self.Ks,self.c_in,self.c_out)

        x_gconv=x_gconv.reshape(B,-1,c,n)

        x_gconv=torch.transpose(x_gconv,2,1)

        return self.relu(x_gconv+x_input)


class St_conv_block(nn.Module):
    def __init__(self,n_route,Kt,Ks,channels,scope,keep_prob,act_func='glu',channel=3,num_class=5):
        super(St_conv_block,self).__init__()
        c_si,c_t,c_oo = channels # 7,16,32
        self.bn_t1 = nn.BatchNorm2d(c_t)
        self.bn_t2 = nn.BatchNorm2d(c_t)
        #self.bn_t3 = nn.BatchNorm2d(c_oo)

        self.temp1 = Temporal_conv_layer(c_si,c_t,Kt,act_func) # 7,16
        self.spat1 = Spatio_conv_layer(c_t,c_t,Ks,n_route)  # 16,16

        self.spat2 = Spatio_conv_layer(c_t,c_t, Ks, n_route)  # 16,16
        self.temp2 = Temporal_conv_layer(c_t,c_t,Kt,act_func) # 16,16

        self.spat3 = Spatio_conv_layer(c_t, c_t, Ks, n_route)  # 16, 16
        self.temp3 = Temporal_conv_layer(c_t, c_oo, Kt, act_func)  # 16, 32

        #self.spat4 = Spatio_conv_layer(c_oo, c_oo, Ks, n_route)  # 32, 32
        #output = (32, 6, 8)
        self.mlp1 = nn.Linear(c_oo * 48, 100)
        self.mlp2 = nn.Linear(100, 100)
        self.mlp=nn.Linear(100, num_class)
        self.drop=nn.Dropout2d(p=keep_prob)

    def forward(self,x):
        #print(x.shape)
        x = self.temp1(x)
        x = self.bn_t1(x)



        x = self.temp2(x)
        x = self.bn_t2(x)
        #x = self.spat2(x)
        #print(x.shape)
        #x = self.spat3(x)

        x = self.temp3(x)
        x = self.spat1(x)
        #print(x.shape)
        #x = self.bn_t3(x)

        #x = self.spat4(x)

        bs ,_ , _ , _ = x.shape
        #x=F.normalize(x,p=2,dim=1)  ##normalize 잘 되는지 확인 필요
        x = x.reshape(bs,-1)
        x = self.mlp1(x)
        x = F.relu(x)
        x = self.mlp2(x)
        x = F.relu(x)
        x = self.mlp(x)

        return F.log_softmax(x, dim=1)

class Fully_conn_layer(nn.Module):
    def __init__(self, channel, scope):
        super(Fully_conn_layer, self).__init__()
        self.channel=channel
        self.scope=scope
        self.weight=nn.Parameter(initializer(torch.randn(1,channel,1,1)))

    def forward(self, x):
        return F.conv2d(x, self.weight)

class Output_layer(nn.Module):
    def __init__(self, t, scope, act_func, channel):
        super(Output_layer,self).__init__()
        self.T = t
        self.act_func=act_func
        self.scope=scope
        self.temp1 = Temporal_conv_layer(channel,channel,t,act_func)
        self.temp2 = Temporal_conv_layer(channel,channel,1,'relu')
        self.full = Fully_conn_layer(channel,scope)
    def forward(self,x):
        x=self.temp1(x)
        x=self.temp2(x)
        x=self.full(x)
        return x

if __name__=='__main__':
    a=torch.randn(12,7,8,12)
    model=St_conv_block(8,2,2,[7,16,32],'scope',0.2,'glu',1,7)
    first=list(model.parameters())[5].clone()
    last=torch.nn.Linear(8,4)
    last.requires_grad=True
    for i in range(1):
        model.train()
        b=model(a)

        answer=torch.randn(12,7)

        #answer=answer.to(torch.float32)
        #answer.requires_grad=True
        #answer=answer.reshape(-1,1,1,1)
        #print(answer.shape)
        #b=b.reshape(1,12)
        #answer=answer.reshape(1,12)
        #print(b.shape,answer.shape)
        optim=torch.optim.Adam(model.parameters(),lr=2)
        loss=torch.nn.MSELoss()
        losses=loss(b,answer)

        losses.backward()
        optim.step()

    count=0
    for name,param in model.named_parameters():
        count+=1
        print(name)
        if param.grad is not None:
            print("not None bro",count)
    #optim.step()
    b=list(model.parameters())[5].clone()
    print(torch.equal(first.data,b.data))























