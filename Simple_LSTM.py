class Simple_LSTM_Optimizer(torch.nn.Module):
    
    def __init__(self, args):
        super(LSTM_Optimizer,self).__init__()
        input_dim, hidden_dim, num_stacks, output_dim = args()
	
	# LSTM 模块 
        self.lstm = torch.nn.LSTM(input_dim, hidden_dim, num_stacks)
        self.Linear = torch.nn.Linear(hidden_dim, output_dim)

    def Output_And_Update(self, input_gradients, prev_state):
    
        if prev_state is None: #init_state
           (cell , hidden) = prev_state.init()
	
	# LSTM 更新cell和hidden，并输出hidden 
        update , (cell,hidden) = self.lstm(input_gradients, (cell, hidden))
        update = self.Linear(update) 
	
        return update, (cell, hidden)
    	
    def forward(self,input_gradients, prev_state):
	
	# 输入梯度，输出预测梯度 并更新隐状态
        update , next_state = self.Output_And_Update (input_gradients , prev_state) 
	
	return update , next_state
