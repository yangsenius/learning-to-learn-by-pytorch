def learning_to_learn(f, x, LSTM_Optimizer, Adam_Optimizer ,T ,TT, unrolled_step):

    for tt in range(TT):
    
        L = 0

        for t in range(T):
            
            loss = f(x)

            x_grad = loss.backward(retain_graph=True)

            x += LSTM_Optimizer(x_grad)

            L += loss
            
            if (t+1)% unrolled_step ==0:

                LSTM_grad = L.backward()

                LSTM_Optimizer.parameters += Adam_Optimizer( LSTM_grad)
                
                L = 0

    return LSTM_Optimizer
