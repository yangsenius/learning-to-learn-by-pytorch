class Simple_LSTM_Optimizer(torch.nn.Module):

    def __init__(self, input_dim, hidden_dim, num_stacks, output_dim ,batchsize):
        super(LSTM_Optimizer,self).__init__()
        self.batchsize = batchsize
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_stacks = num_stacks
	### LSTM模块 ##
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, num_stacks)
        self.Linear = torch.nn.Linear(hidden_dim, output_dim)

    def Output_And_Update(self, input_gradients, prev_state):
    
        if prev_state is None: #init_state
            cell = torch.zeros(self.num_stacks,self.batchsize,self.hidden_dim)
            hidden = torch.zeros(self.num_stacks,self.batchsize,self.hidden_dim)
	## LSTM 更新cell和hidden，并输出hidden  ##
        update , (cell,hidden) = self.lstm(input_gradients, (cell, hidden))
        update = self.Linear(update) 
        return update, (cell, hidden)
    	
    def forward(self,input_gradients, prev_state):
	##输入梯度，输出预测梯度 并更新隐状态
        update , next_state = self.Output_And_Update (input_gradients , prev_state)  
        return update , next_state
