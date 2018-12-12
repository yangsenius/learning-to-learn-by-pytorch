import torch
import torch.nn as nn
from timeit import default_timer as timer
from optim import SGD, RMS, adam
from learner import Learner
from CoordinateWiseLSTM import LSTM_Optimizee_Model
from learning_to_learn import Learning_to_learn_global_training
from cuda import USE_CUDA 
#####################      优化问题   ##########################
def f(W,Y,x):
    """quadratic function : f(\theta) = \|W\theta - y\|_2^2"""
    if USE_CUDA:
        W = W.cuda()
        Y = Y.cuda()
        x = x.cuda()

    return ((torch.matmul(W,x.unsqueeze(-1)).squeeze()-Y)**2).sum(dim=1).mean(dim=0)


USE_CUDA = USE_CUDA
DIM = 10
batchsize = 128

print('\n\nUSE_CUDA = {}\n\n'.format(USE_CUDA))

#################   优化器模型参数  ##############################
Layers = 2
Hidden_nums = 20
Input_DIM = DIM
Output_DIM = DIM
output_scale_value=1

#######   构造一个优化器  #######
LSTM_Optimizee = LSTM_Optimizee_Model(Input_DIM, Output_DIM, Hidden_nums ,Layers , batchsize=batchsize,\
                preprocess=False,output_scale=output_scale_value)
print(LSTM_Optimizee)

if USE_CUDA:
    LSTM_Optimizee = LSTM_Optimizee.cuda()



#################### Learning to learn (优化optimizee) ######################
Global_Train_Steps = 2000
Optimizee_Train_Steps = 100
UnRoll_STEPS = 20
Evaluate_period = 1
optimizer_lr = 0.1

global_loss_list ,flag = Learning_to_learn_global_training(  f, LSTM_Optimizee,
                                                        Global_Train_Steps,
                                                        Optimizee_Train_Steps,
                                                        UnRoll_STEPS,
                                                        Evaluate_period,
                                                        optimizer_lr)

if flag ==True :
    print('\n=== > load best LSTM model')
    torch.save(LSTM_Optimizee.state_dict(),'final_LSTM_optimizer.pth')
    LSTM_Optimizee.load_state_dict( torch.load('best_LSTM_optimizer.pth'))



######################################################################3#
##########################   show results ###############################

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#import seaborn as sns; #sns.set(color_codes=True)
#sns.set_style("white")
#Global_T = np.arange(len(global_loss_list))
#p1, = plt.plot(Global_T, global_loss_list, label='Global_graph_loss')
#plt.legend(handles=[p1])
#plt.title('Training LSTM optimizee by gradient descent ')
#plt.show()


STEPS = 100
x = np.arange(STEPS)

Adam = 'Adam' #因为这里Adam使用Pytorch

for _ in range(3): 
   
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
    #plt.yscale('log')
    #plt.legend(handles=[p1, p2, p3, p4])
    #plt.title('Losses')
    #plt.show()
    #print("\n\nsum_loss:sgd={},rms={},adam={},lstm={}".format(sgd_sum_loss,rms_sum_loss,adam_sum_loss,lstm_sum_loss ))