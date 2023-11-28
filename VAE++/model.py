import torch
#11.8 13:41 把axis改成dim 效果：无效


class VAE(torch.nn.Module):
    def __init__(self, item_num : int, hidden_size : int,batch_size : int, dropout_rate : float = 0.5, lr : float = 1e-3):
        super(VAE,self).__init__()
        self.KL_target = None
        self.logits_target = None
        print(item_num)
        self.device = torch.device('cuda')
        self.layer_u_E = torch.nn.Linear(item_num,hidden_size,device='cuda',dtype=torch.double)
        self.init_weights(self.layer_u_E)
        self.layer_u_PUE = torch.nn.Linear(item_num,hidden_size,device='cuda',dtype=torch.double)
        self.init_weights(self.layer_u_PUE)
        self.layer_u_P = torch.nn.Linear(item_num,2*hidden_size,device='cuda',dtype=torch.double)
        self.init_weights(self.layer_u_P)
        self.layer_mlp = torch.nn.Linear(2*hidden_size,item_num,device='cuda',dtype=torch.double)
        self.init_weights(self.layer_mlp)
        self.layer_g = torch.nn.Linear(2*hidden_size,1,device='cuda',dtype=torch.double)
        self.init_weights(self.layer_g)

        self.dropout_rate = dropout_rate
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.item_num = item_num
        self.is_training = 1
        self.lr = lr
        self.anneal = 0.2
        self.input_exa = torch.empty((self.batch_size,self.item_num),requires_grad=False,dtype=torch.double,device='cuda')
        self.input_mix = torch.empty((self.batch_size, self.item_num),requires_grad=False,dtype=torch.double,device='cuda')
        self.input_pur = torch.empty((self.batch_size, self.item_num),requires_grad=False,dtype=torch.double,device='cuda')
        self.prediction_top_k = torch.empty((self.batch_size,self.item_num),requires_grad=False,dtype=torch.double,device='cuda')
        self.cri = torch.nn.CrossEntropyLoss()

    def init_weights(self, layer):
        if type(layer) == torch.nn.Linear:
            torch.nn.init.xavier_normal_(layer.weight)
            torch.nn.init.normal_(layer.bias, 0, 0.001)
    def Data_Collect(self, input_pur : torch.TensorType,input_exa : torch.TensorType, input_mix : torch.TensorType ):
        #input_size = [batch_size, hidden_size]
        self.input_pur = torch.tensor(input_pur,dtype=torch.double,device='cuda')
        self.input_exa = torch.tensor(input_exa,dtype=torch.double,device='cuda')
        self.input_mix = torch.tensor(input_mix,dtype=torch.double,device='cuda')

    def get_u_E(self):
        a = torch.nn.functional.normalize(self.input_exa,dim=1,p=2)
        a = torch.nn.functional.dropout(a ,self.dropout_rate,training=self.training)
        #h = torch.tensor(h, dtype=torch.float).to(self.device)
        # print(h.dtype)
        # print(self.layer_u_E.weight.dtype)
        # print(self.W_uP.dtype)
        h = self.layer_u_E(a)
        h.retain_grad = True
        r = h[:,:self.item_num]
        return r

    def get_u_PUE(self):
        h = torch.nn.functional.normalize(self.input_mix, dim=1,p=2)
        h = torch.nn.functional.dropout(h, self.dropout_rate,training=self.training)
        #h = torch.tensor(h, dtype=torch.float).to(self.device)
        h = self.layer_u_PUE(h)
        h = h[:, :self.item_num]
        return h

    def through_mlp(self, z_Pe, u_E):
        h = torch.cat((z_Pe,u_E),1)
        #h = torch.tensor(h, dtype=torch.float).to(self.device)
        # print(h.shape)
        # print(self.W_mlp.shape)
        h = self.layer_mlp(h)

        return h.to(self.device)

    def TransferGatingNetWork(self,u_PUE):

        h = torch.nn.functional.normalize(self.input_pur,dim=1,p=2)
        h = torch.nn.functional.dropout(h, self.dropout_rate,training=self.training)
        #h = torch.tensor(h, dtype=torch.float).to(self.device)
        h = self.layer_u_P(h)

        u_P =  h[:,:self.hidden_size]
        #500*200   19244*1

        vec_con = torch.cat((u_P,u_PUE),1)

        g = self.layer_g(vec_con)
        g = torch.sigmoid(g)

        u_Pe = u_P * g + u_PUE * (1-g)

        sigma_u_P = h[:,self.hidden_size:]

        KL = torch.mean(torch.sum(0.5 * (-sigma_u_P + torch.exp(sigma_u_P) + u_Pe ** 2 - 1),dim=1))


        sigma_u_P = torch.exp(sigma_u_P / 2)

        return u_Pe,sigma_u_P,KL

    def process(self):
        u_E = self.get_u_E()
        u_PUE = self.get_u_PUE()

        u_Pe, sigma_P, KL = self.TransferGatingNetWork(u_PUE)
        epsilon = torch.randn(sigma_P.shape,device='cuda')
        z_Pe = u_Pe + self.is_training * epsilon * sigma_P

        self.logits_target = self.through_mlp(z_Pe, u_E)
        self.KL_target = KL


    def create_loss(self):
        log_softmax_var_target = torch.log_softmax(self.logits_target,dim=-1)

        self.neg_ll_target = - torch.mean(
           torch.sum(log_softmax_var_target * self.input_pur, dim=-1)
        )
        # * self.input_pur
        self.loss = self.anneal * self.KL_target + self.neg_ll_target
        #self.logits_target = torch.sigmoid(self.logits_target)
        #self.loss = torch.nn.functional.binary_cross_entropy(self.logits_target, self.input_pur.to(self.device))
        #self.loss = self.KL_target + self.neg_ll_target
        #for param in self.parameters(): self.loss += 0.01 * torch.norm(param)

    def build_graph(self):
        self.process()
        self.create_loss()

    def get_topk(self):
        self.list = torch.topk(self.prediction_top_k, k=5)
        return self.list

    def forward(self, input_pur : torch.TensorType,input_exa : torch.TensorType, input_mix : torch.TensorType,anneal : float ,is_training : int):
        self.is_training = is_training
        self.anneal = anneal

        self.Data_Collect(input_pur,input_exa,input_mix)

        self.process()
        self.create_loss()
        self.get_topk()

        return self.loss,self.logits_target,self.list



