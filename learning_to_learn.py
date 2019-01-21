# coding: utf-8

# Learning to learn by gradient descent by gradient descent
# =========================#

# https://arxiv.org/abs/1611.03824
# https://yangsenius.github.io/blog/LSTM_Meta/
# author：yangsen
# #### “通过梯度下降来学习如何通过梯度下降学习”
# #### "learning to learn by gradient descent by gradient descent"
# #### 要让优化器学会这样   "为了更好地得到，要先去舍弃"  这样类似的知识！
# #### make the optimizer to learn the knowledge of "sometimes, in order to get it better, you have to give up first. "

import torch
import torch.nn as nn
from timeit import default_timer as timer
#####################      优化问题   ##########################
#####################      optimization  ##########################
USE_CUDA = False
DIM = 10
batchsize = 128

if torch.cuda.is_available():
    USE_CUDA = True  
USE_CUDA = False  # for your need


print('\n\nUSE_CUDA = {}\n\n'.format(USE_CUDA))
 

def f(W,Y,x):
    """quadratic function : f(\theta) = \|W\theta - y\|_2^2"""
    if USE_CUDA:
        W = W.cuda()
        Y = Y.cuda()
        x = x.cuda()

    return ((torch.matmul(W,x.unsqueeze(-1)).squeeze()-Y)**2).sum(dim=1).mean(dim=0)

###############################################################

######################    手工的优化器   ###################
######################    hand-craft optimizer   ###################

def SGD(gradients, state, learning_rate=0.001):
   
    return -gradients*learning_rate, state

def RMS(gradients, state, learning_rate=0.01, decay_rate=0.9):
    if state is None:
        state = torch.zeros(DIM)
        if USE_CUDA == True:
            state = state.cuda()
            
    state = decay_rate*state + (1-decay_rate)*torch.pow(gradients, 2)
    update = -learning_rate*gradients / (torch.sqrt(state+1e-5))
    return update, state

def adam():
    return torch.optim.Adam()

##########################################################


#####################    自动 LSTM 优化器模型  ##########################
#####################    auto LSTM optimizer model  ##########################
class LSTM_Optimizee_Model(torch.nn.Module):
    """LSTM优化器
       LSTM optimizer"""
    
    def __init__(self,input_size,output_size, hidden_size, num_stacks, batchsize, preprocess = True ,p = 10 ,output_scale = 1):
        super(LSTM_Optimizee_Model,self).__init__()
        self.preprocess_flag = preprocess
        self.p = p
        self.input_flag = 2
        if preprocess != True:
             self.input_flag = 1
        self.output_scale = output_scale 
        self.lstm = torch.nn.LSTM(input_size*self.input_flag, hidden_size, num_stacks)
        self.Linear = torch.nn.Linear(hidden_size,output_size) #1-> output_size
        
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
        return torch.cat((clamp_log,clamp_sign),dim = -1) #在gradients的最后一维input_dims拼接 # concatenate in final dim
    
    def Output_Gradient_Increment_And_Update_LSTM_Hidden_State(self, input_gradients, prev_state):
        """LSTM的核心操作  core operation
        coordinate-wise LSTM """
        if prev_state is None: #init_state
            prev_state = (torch.zeros(Layers,batchsize,Hidden_nums),
                            torch.zeros(Layers,batchsize,Hidden_nums))
            if USE_CUDA :
                 prev_state = (torch.zeros(Layers,batchsize,Hidden_nums).cuda(),
                            torch.zeros(Layers,batchsize,Hidden_nums).cuda())
         			
        update , next_state = self.lstm(input_gradients, prev_state)
        update = self.Linear(update) * self.output_scale # transform the LSTM output to the target output dim 
        return update, next_state
    
    def forward(self,input_gradients, prev_state):
        if USE_CUDA:
            input_gradients = input_gradients.cuda()
        #pytorch requires the `torch.nn.lstm`'s input as（1，batchsize,input_dim）
        # original gradient.size()=torch.size[5] ->[1,1,5]
        gradients = input_gradients.unsqueeze(0)
      
        if self.preprocess_flag == True:
            gradients = self.LogAndSign_Preprocess_Gradient(gradients)
      
        update , next_state = self.Output_Gradient_Increment_And_Update_LSTM_Hidden_State(gradients , prev_state)
        # Squeeze to make it a single batch again.[1,1,5]->[5]
        update = update.squeeze().squeeze()
       
        return update , next_state
    
#################   优化器模型参数  ##############################
#################   Parameters of optimizer ##############################
Layers = 2
Hidden_nums = 20
Input_DIM = DIM
Output_DIM = DIM
output_scale_value=1

#######   构造一个优化器  #######
#######  construct a optimizer  #######
LSTM_Optimizee = LSTM_Optimizee_Model(Input_DIM, Output_DIM, Hidden_nums ,Layers , batchsize=batchsize,\
                preprocess=False,output_scale=output_scale_value)
print(LSTM_Optimizee)

if USE_CUDA:
    LSTM_Optimizee = LSTM_Optimizee.cuda()
   

######################  优化问题目标函数的学习过程   ###############
######################  the learning process of optimizing the target function   ###############

class Learner( object ):
    """
    Args :
        `f` : 要学习的问题 the learning problem
        `optimizee` : 使用的优化器 the used optimizer 
        `train_steps` : 对于其他SGD,Adam等是训练周期，对于LSTM训练时的展开周期  training steps for SGD and ADAM, unfolded step for LSTM train
        `retain_graph_flag=False`  : 默认每次loss_backward后 释放动态图 default: free the dynamic graph after the loss backward
        `reset_theta = False `  :  默认每次学习前 不随机初始化参数 default: do not initialize the theta
        `reset_function_from_IID_distirbution = True` : 默认从分布中随机采样函数  default: random sample from  distribution

    Return :
        `losses` : reserves each loss value in each iteration
        `global_loss_graph` : constructs the graph of all Unroll steps for LSTM's BPTT 
    """
    def __init__(self,    f ,   optimizee,  train_steps ,  
                                            eval_flag = False,
                                            retain_graph_flag=False,
                                            reset_theta = False ,
                                            reset_function_from_IID_distirbution = True):
        self.f = f
        self.optimizee = optimizee
        self.train_steps = train_steps
        #self.num_roll=num_roll
        self.eval_flag = eval_flag
        self.retain_graph_flag = retain_graph_flag
        self.reset_theta = reset_theta
        self.reset_function_from_IID_distirbution = reset_function_from_IID_distirbution  
        self.init_theta_of_f()
        self.state = None

        self.global_loss_graph = 0 # global loss for optimizing LSTM
        self.losses = []   # KEEP each loss of all epoches

    def init_theta_of_f(self,):  
        ''' 初始化 优化问题 f 的参数 
            initialize the theta of optimization f '''
        self.DIM = 10
        self.batchsize = 128
        self.W = torch.randn(batchsize,DIM,DIM) # represents IID
        self.Y = torch.randn(batchsize,DIM)
        self.x = torch.zeros(self.batchsize,self.DIM)
        self.x.requires_grad = True
        if USE_CUDA:
            self.W = self.W.cuda()
            self.Y = self.Y.cuda()
            self.x = self.x.cuda()
        
            
    def Reset_Or_Reuse(self , x , W , Y , state, num_roll):
        ''' re-initialize the `W, Y, x , state`  at the begining of each global training
            IF `num_roll` == 0    '''

        reset_theta =self.reset_theta
        reset_function_from_IID_distirbution = self.reset_function_from_IID_distirbution

       
        if num_roll == 0 and reset_theta == True:
            theta = torch.zeros(batchsize,DIM)
           
            theta_init_new = torch.tensor(theta,dtype=torch.float32,requires_grad=True)
            x = theta_init_new
            
            
        ################   每次全局训练迭代，从独立同分布的Normal Gaussian采样函数     ##################
        ################   at the first iteration , sample from IID Normal Gaussian    ##################
        if num_roll == 0 and reset_function_from_IID_distirbution == True :
            W = torch.randn(batchsize,DIM,DIM) # represents IID
            Y = torch.randn(batchsize,DIM)     # represents IID
         
            
        if num_roll == 0:
            state = None
            print('reset W, x , Y, state ')
            
        if USE_CUDA:
            W = W.cuda()
            Y = Y.cuda()
            x = x.cuda()
            x.retain_grad()
          
            
        return  x , W , Y , state

    def __call__(self, num_roll=0) : 
        '''
        Total Training steps = Unroll_Train_Steps * the times of  `Learner` been called
        
        SGD,RMS,LSTM FROM defination above
         but Adam is adopted by pytorch~ This can be improved later'''
        f  = self.f 
        x , W , Y , state =  self.Reset_Or_Reuse(self.x , self.W , self.Y , self.state , num_roll )
        self.global_loss_graph = 0   #at the beginning of unroll, reset to 0 
        optimizee = self.optimizee
        print('state is None = {}'.format(state == None))
     
        if optimizee!='Adam':
            
            for i in range(self.train_steps):     
                loss = f(W,Y,x)
                #self.global_loss_graph += (0.8*torch.log10(torch.Tensor([i+1]))+1)*loss
                self.global_loss_graph += loss
              
                loss.backward(retain_graph=self.retain_graph_flag) # default as False,set to True for LSTMS
              
                update, state = optimizee(x.grad, state)
              
                self.losses.append(loss)
             
                x = x + update  
                x.retain_grad()
                update.retain_grad()
                
            if state is not None:
                self.state = (state[0].detach(),state[1].detach())
                
            return self.losses ,self.global_loss_graph 

        else: #Pytorch Adam

            x.detach_()
            x.requires_grad = True
            optimizee= torch.optim.Adam( [x],lr=0.1 )
            
            for i in range(self.train_steps):
                
                optimizee.zero_grad()
                loss = f(W,Y,x)
                
                self.global_loss_graph += loss
                
                loss.backward(retain_graph=self.retain_graph_flag)
                optimizee.step()
                self.losses.append(loss.detach_())
                
            return self.losses, self.global_loss_graph


#######   LSTM 优化器的训练过程 Learning to learn   ###############
#######   LSTM training Learning to learn   ###############

def Learning_to_learn_global_training(optimizee, global_taining_steps, Optimizee_Train_Steps, UnRoll_STEPS, Evaluate_period ,optimizer_lr=0.1):
    """ Training the LSTM optimizee . Learning to learn

    Args:   
        `optimizee` : DeepLSTMCoordinateWise optimizee model
        `global_taining_steps` : how many steps for optimizer training o可以ptimizee
        `Optimizee_Train_Steps` : how many step for optimizee opimitzing each function sampled from IID.
        `UnRoll_STEPS` :: how many steps for LSTM optimizee being unrolled to construct a computing graph to BPTT.
    """
    global_loss_list = []
    Total_Num_Unroll = Optimizee_Train_Steps // UnRoll_STEPS
    adam_global_optimizer = torch.optim.Adam(optimizee.parameters(),lr = optimizer_lr)

    LSTM_Learner = Learner(f, optimizee, UnRoll_STEPS, retain_graph_flag=True, reset_theta=True,)
  #这里考虑Batchsize代表IID的话，那么就可以不需要每次都重新IID采样
  # If regarding `Batchsize` as `IID` ,there is no need for reset the theta
  #That is ，reset_function_from_IID_distirbution = False else it is True

    best_sum_loss = 999999
    best_final_loss = 999999
    best_flag = False
    for i in range(Global_Train_Steps): 

        print('\n=======> global training steps: {}'.format(i))

        for num in range(Total_Num_Unroll):
            
            start = timer()
            _,global_loss = LSTM_Learner(num)   

            adam_global_optimizer.zero_grad()
            global_loss.backward() 
       
            adam_global_optimizer.step()
            # print('xxx',[(z.grad,z.requires_grad) for z in optimizee.lstm.parameters()  ])
            global_loss_list.append(global_loss.detach_())
            time = timer() - start
            #if i % 10 == 0:
            print('-> time consuming [{:.1f}s] optimizee train steps :  [{}] | Global_Loss = [{:.1f}] '\
                  .format(time,(num +1)* UnRoll_STEPS,global_loss,))

        if (i + 1) % Evaluate_period == 0:
            
            best_sum_loss, best_final_loss, best_flag  = evaluate(best_sum_loss,best_final_loss,best_flag , optimizer_lr)

    return global_loss_list,best_flag


def evaluate(best_sum_loss,best_final_loss, best_flag,lr):
    print('\n --> evalute the model')
    STEPS = 100
    LSTM_learner = Learner(f , LSTM_Optimizee, STEPS, eval_flag=True,reset_theta=True, retain_graph_flag=True)
    lstm_losses, sum_loss = LSTM_learner()
    try:
        best = torch.load('best_loss.txt')
    except IOError:
        print ('can not find best_loss.txt')
        pass
    else:
        best_sum_loss = best[0]
        best_final_loss = best[1]
        print("load_best_final_loss and sum_loss")
    if lstm_losses[-1] < best_final_loss and  sum_loss < best_sum_loss:
        best_final_loss = lstm_losses[-1]
        best_sum_loss =  sum_loss
        
        print('\n\n===> update new best of final LOSS[{}]: =  {}, best_sum_loss ={}'.format(STEPS, best_final_loss,best_sum_loss))
        torch.save(LSTM_Optimizee.state_dict(),'best_LSTM_optimizer.pth')
        torch.save([best_sum_loss ,best_final_loss,lr ],'best_loss.txt')
        best_flag = True
        
    return best_sum_loss, best_final_loss, best_flag 

    
##########################   before learning LSTM optimizee ###############################
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

STEPS = 100
x = np.arange(STEPS)

Adam = 'Adam' # Adam in Pytorch

for _ in range(1): 
   
    SGD_Learner = Learner(f , SGD, STEPS, eval_flag=True,reset_theta=True,)
    RMS_Learner = Learner(f , RMS, STEPS, eval_flag=True,reset_theta=True,)
    Adam_Learner = Learner(f , Adam, STEPS, eval_flag=True,reset_theta=True,)
    LSTM_learner = Learner(f , LSTM_Optimizee, STEPS, eval_flag=True,reset_theta=True,retain_graph_flag=True)

    sgd_losses, sgd_sum_loss = SGD_Learner()
    rms_losses, rms_sum_loss = RMS_Learner()
    adam_losses, adam_sum_loss = Adam_Learner()
    lstm_losses, lstm_sum_loss = LSTM_learner()

    p1, = plt.plot(x, sgd_losses, label='SGD')
    p2, = plt.plot(x, rms_losses, label='RMS')
    p3, = plt.plot(x, adam_losses, label='Adam')
    p4, = plt.plot(x, lstm_losses, label='LSTM')
    p1.set_dashes([2, 2, 2, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
    p2.set_dashes([4, 2, 8, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
    p3.set_dashes([3, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
    plt.yscale('log')
    plt.legend(handles=[p1, p2, p3, p4])
    plt.title('Losses')
    plt.show()
    print("\n\nsum_loss:sgd={},rms={},adam={},lstm={}".format(sgd_sum_loss,rms_sum_loss,adam_sum_loss,lstm_sum_loss ))



#################### Learning to learn (optimizing optimizee) ######################
Global_Train_Steps = 1000 #可修改 changeable
Optimizee_Train_Steps = 100
UnRoll_STEPS = 20
Evaluate_period = 1 #可修改 changeable
optimizer_lr = 0.1 #可修改 changeable
global_loss_list ,flag = Learning_to_learn_global_training(   LSTM_Optimizee,
                                                        Global_Train_Steps,
                                                        Optimizee_Train_Steps,
                                                        UnRoll_STEPS,
                                                        Evaluate_period,
                                                          optimizer_lr)


######################################################################3#
##########################   show learning process results 
#torch.load('best_LSTM_optimizer.pth'))
#import numpy as np
#import matplotlib
#import matplotlib.pyplot as plt

#Global_T = np.arange(len(global_loss_list))
#p1, = plt.plot(Global_T, global_loss_list, label='Global_graph_loss')
#plt.legend(handles=[p1])
#plt.title('Training LSTM optimizee by gradient descent ')
#plt.show()


######################################################################3#
##########################   show contrast results SGD,ADAM, RMS ,LSTM ###############################
import copy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

if flag ==True :
    print('\n==== > load best LSTM model')
    last_state_dict = copy.deepcopy(LSTM_Optimizee.state_dict())
    torch.save(LSTM_Optimizee.state_dict(),'final_LSTM_optimizer.pth')
    LSTM_Optimizee.load_state_dict( torch.load('best_LSTM_optimizer.pth'))
    
LSTM_Optimizee.load_state_dict(torch.load('best_LSTM_optimizer.pth'))
#LSTM_Optimizee.load_state_dict(torch.load('final_LSTM_optimizer.pth'))
STEPS = 100
x = np.arange(STEPS)

Adam = 'Adam'

for _ in range(3): #可以多试几次测试实验，LSTM不稳定 for several test,  the trained LSTM is not stable?
    
    SGD_Learner = Learner(f , SGD, STEPS, eval_flag=True,reset_theta=True,)
    RMS_Learner = Learner(f , RMS, STEPS, eval_flag=True,reset_theta=True,)
    Adam_Learner = Learner(f , Adam, STEPS, eval_flag=True,reset_theta=True,)
    LSTM_learner = Learner(f , LSTM_Optimizee, STEPS, eval_flag=True,reset_theta=True,retain_graph_flag=True)

    
    sgd_losses, sgd_sum_loss = SGD_Learner()
    rms_losses, rms_sum_loss = RMS_Learner()
    adam_losses, adam_sum_loss = Adam_Learner()
    lstm_losses, lstm_sum_loss = LSTM_learner()

    p1, = plt.plot(x, sgd_losses, label='SGD')
    p2, = plt.plot(x, rms_losses, label='RMS')
    p3, = plt.plot(x, adam_losses, label='Adam')
    p4, = plt.plot(x, lstm_losses, label='LSTM')
    p1.set_dashes([2, 2, 2, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
    p2.set_dashes([4, 2, 8, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
    p3.set_dashes([3, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
    #p4.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
    plt.yscale('log')
    plt.legend(handles=[p1, p2, p3, p4])
    plt.title('Losses')
    plt.show()
    print("\n\nsum_loss:sgd={},rms={},adam={},lstm={}".format(sgd_sum_loss,rms_sum_loss,adam_sum_loss,lstm_sum_loss ))
