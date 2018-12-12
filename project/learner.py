import torch
from cuda import USE_CUDA

class Learner( object ):
    """
    Args :
        `f` : 要学习的问题
        `optimizee` : 使用的优化器
        `train_steps` : 对于其他SGD,Adam等是训练周期，对于LSTM训练时的展开周期
        `retain_graph_flag=False`  : 默认每次loss_backward后 释放动态图
        `reset_theta = False `  :  默认每次学习前 不随机初始化参数
        `reset_function_from_IID_distirbution = True` : 默认从分布中随机采样函数 

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

        self.global_loss_graph = 0 #这个是为LSTM优化器求所有loss相加产生计算图准备的
        self.losses = []   # 保存每个训练周期的loss值

    def init_theta_of_f(self,):  
        ''' 初始化 优化问题 f 的参数 '''
        self.DIM = 10
        self.batchsize = 128
        self.W = torch.randn(self.batchsize,self.DIM,self.DIM) #代表 已知的数据 # 独立同分布的标准正太分布
        self.Y = torch.randn(self.batchsize,self.DIM)
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

        #torch.manual_seed(2)
        if num_roll == 0 and reset_theta == True:
            theta = torch.zeros(self.batchsize,self.DIM)
            #torch.manual_seed(0) ##独立同分布的 标准正太分布
            theta_init_new = torch.tensor(theta,dtype=torch.float32,requires_grad=True)
            x = theta_init_new
            
            print('Reset x to zero')
        ################   每次全局训练迭代，从独立同分布的Normal Gaussian采样函数     ##################
        if num_roll == 0 and reset_function_from_IID_distirbution == True :
            W = torch.randn(self.batchsize,self.DIM,self.DIM) #代表 已知的数据 # 独立同分布的标准正太分布
            Y = torch.randn(self.batchsize,self.DIM)     #代表 数据的标签 #  独立同分布的标准正太分布
            #torch.nn.init.normal_(W)
            #torch.nn.init.normal_(Y)
            print('reset W and Y ')
        if num_roll == 0:
            state = None
            print('reset state to None')
            
        if USE_CUDA:
            W = W.cuda()
            Y = Y.cuda()
            x = x.cuda()
            x.retain_grad()
            #print(x.requires_grad)
            
        return  x , W , Y , state

    #x , W , Y , state = Reset(num_roll,reset_theta,reset_function_from_IID_distirbution)

    def __call__(self, num_roll=0) : 
        '''
        Total Training steps = Unroll_Train_Steps * the times of  `Learner` been called
        
        SGD,RMS,LSTM 用上述定义的
         Adam优化器直接使用pytorch里的，所以代码上有区分 后面可以完善！'''
        f  = self.f 
        x , W , Y , state =  self.Reset_Or_Reuse(self.x , self.W , self.Y , self.state , num_roll )
        self.global_loss_graph = 0   #每个unroll的开始需要 重新置零
        optimizee = self.optimizee
        #x.requires_grad = True
        if optimizee!='Adam':
            
            for i in range(self.train_steps):     
                loss = f(W,Y,x)    
                #global_loss_graph += torch.exp(torch.Tensor([-i/20]))*loss
                self.global_loss_graph += (i/20+1)*loss
                ###self.global_loss_graph += loss
                #print('############ epoch {} ###############:\n'.format(i))
                loss.backward(retain_graph=self.retain_graph_flag) # 默认为False,当优化LSTM设置为True
                #print(i,'qw',x,x.grad,x.requires_grad)
                update, state = optimizee(x.grad, state)
                #print(update)
                self.losses.append(loss)
                #print(i,'x',x,'u', update)
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