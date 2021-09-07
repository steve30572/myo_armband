import torch
import torch.nn as nn
import torch.nn.functional as F

adj = torch.zeros(8,8)
for i in range(8):
    devider= i+8
    adj[i][devider % 8] = 0
    adj[i][(devider + 1) % 8] = 1
    adj[i][(devider - 1) % 8] = 1
    #adj[i][(devider + 4) % 8] = -1


class GCN(nn.Module):
    def __init__(self, feat_dim, hidden_dim1, hidden_dim2, dropout):
        super(GCN, self).__init__()
        self.W1 = nn.Parameter(torch.FloatTensor(feat_dim, hidden_dim1))
        self.W2 = nn.Parameter(torch.FloatTensor(hidden_dim1, hidden_dim2))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

        nn.init.xavier_uniform_(self.W1.data)
        nn.init.xavier_uniform_(self.W2.data)

    def forward(self, x):
        bs, feat, n_node, timestamp = x.shape  # 10 7 8 12
        #x = torch.transpose(x, 1, 3)  # 10 12 8 7   #TODO
        x = x.reshape(-1,feat)  #10*12*8, 7
        x = torch.mm(x, self.W1) # 10*12*8, 32(hidden_dim)
        x = x.reshape(bs, timestamp, n_node, -1) # 10 12 8 32
        x = torch.transpose(x, 2, 3)  # 10 12 32 8
        x = x.reshape(-1, n_node)  #10*12*32 8
        x = torch.transpose(x, 0, 1)  # 8, 10*12*32
        x = torch.mm(adj, x)          # 8, 10*12*32
        x = torch.transpose(x, 0, 1)  #10*12*32 8
        x = x.reshape(bs, timestamp, -1, n_node) # 10, 12, 32, 8
        x = self.dropout(x)
        x = self.relu(x)
        x = torch.transpose(x, 2, 3)   # 10 12 8 32
        # ##1st layer
        # x = torch.transpose(x, 1, 3)
        # return x



        ## 2nd layer
        x = x.reshape(bs*n_node*timestamp, -1)  # 10*12*8 32
        x = torch.mm(x, self.W2)   # 10*12*8 32
        x = x.reshape(bs, timestamp, n_node, -1)  #10 12 8 32
        x = torch.transpose(x, 2, 3)   #10 12 32 8
        x = x.reshape(-1, n_node)     #10*12*32 8
        x = torch.transpose(x, 0, 1)  #8 10*12*32
        x = torch.mm(adj, x)  #8 10*12*32
        x = torch.transpose(x, 0, 1) # 10*12*32 8
        x = x.reshape(bs, timestamp, -1, n_node) # 10 12 32 8
        x = torch.transpose(x, 1, 3)  # 10 8 32 12
        x = torch.transpose(x, 1, 2)  # 10 32 8 12
        return x

class Temporal_layer(nn.Module):
    def __init__(self, kernel_size, in_channel, out_channel, dropout):
        super(Temporal_layer, self).__init__()
        self.cnn = nn.Conv2d(in_channel//2, out_channel//2, kernel_size)
        self.bn_1st = nn.BatchNorm2d(out_channel//2)
        self.bn_2nd = nn.BatchNorm2d(out_channel//2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p = dropout)
        self.in_channel = in_channel
        self.out_channel = out_channel
    def forward(self, x):
        bs, feat, n_node, time= x.shape
        x = torch.transpose(x, 1, 3)
        for i in range(2):
            temp_result = x[:, 0+i*self.in_channel//2:(i+1)*self.in_channel//2, :, :]
            temp_result = self.cnn(temp_result)
            if i == 0:
                temp_result = self.bn_1st(temp_result)
            else:
                temp_result = self.bn_2nd(temp_result)
            temp_result = self.relu(temp_result)
            temp_result = self.dropout(temp_result)
            if i == 0:
                result = temp_result
            else:
                result = torch.cat((result, temp_result), 1)
        result = torch.transpose(result, 1, 3)
        return result


# class Temporal_layer(nn.Module):
#     def __init__(self, kernel_size, in_channel, out_channel, dropout):
#         super(Temporal_layer, self).__init__()
#         self.cnn = nn.Conv1d(in_channel, out_channel, kernel_size)
#         self.bn = nn.BatchNorm1d(out_channel)
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(p = dropout)
#         self.in_channel = in_channel
#         self.out_channel = out_channel
#
#     def forward(self, x):
#         x = torch.transpose(x, 2, 3)
#         x = torch.transpose(x, 1, 3)
#         bs, n_node, timestamp, feat = x.shape
#         for i in range(bs):
#             temp_result = self.cnn(x[i])
#             temp_result = self.bn(temp_result)
#             temp_result = self.relu(temp_result)
#             temp_result = self.dropout(temp_result)
#             if i == 0:
#                 result = temp_result
#                 result = result.reshape(1, n_node, self.out_channel, -1)
#             else:
#                 temp_result = temp_result.reshape(1, n_node, self.out_channel, -1)
#                 result = torch.cat((result, temp_result), 0)
#         result = torch.transpose(result, 1, 2)
#         result = torch.transpose(result, 1, 3)
#         return result


class MYOGCN(nn.Module):
    def __init__(self, feature_dim, hidden_dim1, output_dim1, feature_dim2, hidden_dim2, output_dim2,\
                 dropout, num_of_class):
        super(MYOGCN, self).__init__()
        self.output_dim2 = output_dim2
        self.gcn_1st = GCN(feature_dim, hidden_dim1, output_dim1, dropout)
        self.temp1 = Temporal_layer(3, 12, 12, dropout)
        self.bn_1 = nn.BatchNorm2d(30)
        # temporal 만들기  __feature 개수 달라져야함
        self.gcn_2nd = GCN(feature_dim2, hidden_dim2, output_dim2, dropout)
        # temporal 만들기
        self.temp2 = Temporal_layer(3, 12, 24, dropout)
        self.bn_2 = nn.BatchNorm2d(28)
        self.temp3 = Temporal_layer(3, 24, 12, dropout)
        self.bn_3 = nn.BatchNorm2d(26)
        self.temp4 = Temporal_layer(2, 48, 24, dropout)
        # shape  bs*7*8*2
        self.FC1 = nn.Linear(26*2*12, 100)#16*output_dim2, 100)
        # Fully --1st
        self.relu1 = nn.ReLU()
        self.Fc2 = nn.Linear(100,100)
        # Fully --2nd
        self.relu2 = nn.ReLU()
        self.Fc3 = nn.Linear(100, num_of_class)
        # Fully --3rd
        self.relu3 = nn.ReLU()
        self.drop = nn.Dropout(p=0.5)
        self.drop2 = nn.Dropout(p=0.5)
        self.drop3 = nn.Dropout(p=0.5)

    def forward(self, x):
        x = self.gcn_1st(x)
        #x = torch.transpose(x, 1, 3)
        #x = self.gcn_2nd(x)
        #print(x.shape)
        #print(x.shape)
        x = self.temp1(x)
        #print(x.shape)
        x = self.bn_1(x)
        x = F.relu(x)
        #x = self.gcn_2nd(x)
        x = self.temp2(x)
        x = self.bn_2(x)
        x = F.relu(x)
        x = self.temp3(x)
        x = self.bn_3(x)
        x = F.relu(x)
        #print(x.shape)
        #x = self.temp4(x)
        x = x.reshape(-1, 26*2*12)#16*self.output_dim2)ㅌ
        x = self.FC1(x)
        x = self.drop(x)
        x = self.relu1(x)
        # x = self.Fc2(x)
        # x = self.drop2(x)
        # x = self.relu2(x)
        x = self.Fc3(x)
        #x = self.drop3(x)
        return F.log_softmax(x, dim=1)


if __name__ == "__main__":
    a = torch.randn(128, 7, 8, 12)
    model = MYOGCN(7, 32, 32, 32, 16, 7, 0.2, 7)
    for i in range(1):
        model.train()
        b = model(a)
        #print(b.shape)
        answer = torch.randn(128, 7)
        optim = torch.optim.Adam(model.parameters(), lr=2)
        loss = torch.nn.MSELoss()
        losses = loss(b, answer)

        losses.backward()
        optim.step()





