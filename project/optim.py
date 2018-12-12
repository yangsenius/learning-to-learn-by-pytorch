import torch
USE_CUDA = torch.cuda.is_available()


###############################################################

######################    手工的优化器   ###################

def SGD(gradients, state, learning_rate=0.001):
   
    return -gradients*learning_rate, state

def RMS(gradients, state, learning_rate=0.01, decay_rate=0.9):
    if state is None:
        state = torch.zeros(gradients.size()[-1])
        if USE_CUDA == True:
            state = state.cuda()
            
    state = decay_rate*state + (1-decay_rate)*torch.pow(gradients, 2)
    update = -learning_rate*gradients / (torch.sqrt(state+1e-5))
    return update, state

def adam():
    return torch.optim.Adam()

##########################################################