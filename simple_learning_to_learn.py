def learning_to_learn(f, x, LSTM_Optimizer, Adam_Optimizer ,T ,TT):

    for tt in range(TT):
    
        L = 0

        for t in range(T):
            
            loss = f(x)

            x_grad = loss.backward()

            x += LSTM_Optimizer(x_grad)

            L += loss
            

        LSTM_grad = L.backward()

        LSTM_Optimizer.parameters += Adam_Optimizer( LSTM_grad)

    return LSTM_Optimizer
