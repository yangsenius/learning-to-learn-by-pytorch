import torch
import torch.nn as nn
from cuda import USE_CUDA

#####################     LSTM 优化器的模型  ##########################
class LSTM_optimizer_Model(torch.nn.Module):
    """LSTM优化器"""
    
    def __init__(self,input_size,output_size, hidden_size, num_stacks, batchsize, preprocess = True ,p = 10 ,output_scale = 1):
        super(LSTM_optimizer_Model,self).__init__()
        self.preprocess_flag = preprocess
        self.p = p
        self.input_flag = 2
        if preprocess != True:
             self.input_flag = 1
        self.output_scale = output_scale #论文
        self.lstm = torch.nn.LSTM(input_size*self.input_flag, hidden_size, num_stacks)
        self.Linear = torch.nn.Linear(hidden_size,output_size) #1-> output_size
        self.Layers = num_stacks
        self.batchsize = batchsize
        self.Hidden_nums =  hidden_size
        
    def LogAndSign_Preprocess_Gradient(self,gradients):
        """
        Args:
          gradients: `Tensor` of gradients with shape `[d_1, ..., d_n]`.
          p       : `p` > 0 is a parameter controlling how small gradients are disregarded 
        Returns:
          `Tensor` with shape `[d_1, ..., d_n-1, 2 * d_n]`. The first `d_n` elements
          along the nth dimension correspond to the `log output` \in [-1,1] and the remaining
          `d_n` elements to the `sign output`.
        """
        p  = self.p
        log = torch.log(torch.abs(gradients))
        clamp_log = torch.clamp(log/p , min = -1.0,max = 1.0)
        clamp_sign = torch.clamp(torch.exp(torch.Tensor(p))*gradients, min = -1.0, max =1.0)
        return torch.cat((clamp_log,clamp_sign),dim = -1) #在gradients的最后一维input_dims拼接
    
    def Output_Gradient_Increment_And_Update_LSTM_Hidden_State(self, input_gradients, prev_state):
        """LSTM的核心操作 coordinate-wise LSTM """
        
        Layers,batchsize,Hidden_nums = self.Layers, self.batchsize, self.Hidden_nums
        
        if prev_state is None: #init_state
            prev_state = (torch.zeros(Layers,batchsize,Hidden_nums),
                            torch.zeros(Layers,batchsize,Hidden_nums))
            if USE_CUDA :
                 prev_state = (torch.zeros(Layers,batchsize,Hidden_nums).cuda(),
                            torch.zeros(Layers,batchsize,Hidden_nums).cuda())
         			
        update , next_state = self.lstm(input_gradients, prev_state)
        update = self.Linear(update) * self.output_scale #因为LSTM的输出是当前步的Hidden，需要变换到output的相同形状上 
        return update, next_state
    
    def forward(self,input_gradients, prev_state):
        if USE_CUDA:
            input_gradients = input_gradients.cuda()
        #LSTM的输入为梯度，pytorch要求torch.nn.lstm的输入为（1，batchsize,input_dim）
        #原gradient.size()=torch.size[5] ->[1,1,5]
        gradients = input_gradients.unsqueeze(0)
      
        if self.preprocess_flag == True:
            gradients = self.LogAndSign_Preprocess_Gradient(gradients)
       
        update , next_state = self.Output_Gradient_Increment_And_Update_LSTM_Hidden_State(gradients , prev_state)
        # Squeeze to make it a single batch again.[1,1,5]->[5]
        update = update.squeeze().squeeze()
      
        return update , next_state
