import torch
from cuda import USE_CUDA
import torch.nn as nn
from timeit import default_timer as timer
from  learner import Learner

#######   LSTM 优化器的训练过程 Learning to learn   ###############

def Learning_to_learn_global_training(f,optimizer, global_taining_steps, optimizer_Train_Steps, UnRoll_STEPS, Evaluate_period ,optimizer_lr):
    """ Training the LSTM optimizer . Learning to learn

    Args:   
        `optimizer` : DeepLSTMCoordinateWise optimizer model
        `global_taining_steps` : how many steps for optimizer training optimizer
        `optimizer_Train_Steps` : how many step for optimizer opimitzing each function sampled from IID.
        `UnRoll_STEPS` :: how many steps for LSTM optimizer being unrolled to construct a computing graph to BPTT.
    """
    global_loss_list = []
    Total_Num_Unroll = optimizer_Train_Steps // UnRoll_STEPS
    adam_global_optimizer = torch.optim.Adam(optimizer.parameters(),lr = optimizer_lr)

    LSTM_Learner = Learner(f, optimizer, UnRoll_STEPS, retain_graph_flag=True, reset_theta=True,reset_function_from_IID_distirbution = False)
  #这里考虑Batchsize代表IID的化，那么就可以不需要每次都重新IID采样

    best_sum_loss = 999999
    best_final_loss = 999999
    best_flag = False
    for i in range(global_taining_steps): 

        print('\n=============> global training steps: {}'.format(i))

        for num in range(Total_Num_Unroll):
            
            start = timer()
            _,global_loss = LSTM_Learner(num)   

            adam_global_optimizer.zero_grad()
     
            global_loss.backward() 
       
            adam_global_optimizer.step()
            
            global_loss_list.append(global_loss.detach_())
            time = timer() - start
            print('--> time consuming [{:.4f}s] optimizer train steps :  [{}] | Global_Loss = [{:.1f}]'.format(time,(num +1)* UnRoll_STEPS,global_loss))

        if (i + 1) % Evaluate_period == 0:
            
            best_sum_loss, best_final_loss, best_flag  = evaluate(f, optimizer,best_sum_loss,best_final_loss,best_flag,optimizer_lr)
            
    
    return global_loss_list,best_flag


def evaluate(f, optimizer, best_sum_loss,best_final_loss, best_flag,lr):
    print('\n --> evalute the model')
    STEPS = 100
    LSTM_learner = Learner(f , optimizer, STEPS, eval_flag=True,reset_theta=True, retain_graph_flag=True)
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
        
        print('\n\n===> best of final LOSS[{}]: =  {}, best_sum_loss ={}'.format(STEPS, best_final_loss,best_sum_loss))
        torch.save(optimizer.state_dict(),'best_LSTM_optimizer.pth')
        torch.save([best_sum_loss ,best_final_loss,lr ],'best_loss.txt')
        best_flag = True
        
    return best_sum_loss, best_final_loss, best_flag 
