#LSTM Source code
import torch
import torch.nn as nn


class CustomLSTMCell(nn.Module):
    def __init__(self,input_size, hidden_size,hidden_layer=False):
        super(CustomLSTMCell,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_layer = hidden_layer

        #gate's Fully connected network
        if hidden_layer:
            self.combined_linear = nn.Linear(self.input_size+self.hidden_size,128)
            self.combined_linear2 = nn.Linear(128,4*self.hidden_size)
        else:
            self.combined_linear = nn.Linear(self.input_size+self.hidden_size,4*self.hidden_size)
            
        self.layer_norm = nn.LayerNorm(hidden_size)
        

    def forward(self, x, hidden_state, cell_state):
        input_hidden = torch.concat([x,hidden_state],dim=1)
        
        if self.hidden_layer:
            x = self.combined_linear(input_hidden)
            combined_linear = self.combined_linear2(torch.relu(x))
        else:
            combined_linear = self.combined_linear(input_hidden)

        f_t,i_t,ic_t,o_t = torch.chunk(combined_linear,4,dim=1)
        # forget gate 
        f_t = torch.sigmoid(f_t)
        #input gate
        i_t = torch.sigmoid(i_t)
        ic_t = torch.tanh(ic_t)

        #output_gate
        o_t = torch.sigmoid(o_t)

        c_t = cell_state * f_t  + i_t * ic_t  #cell state of current lstm

        h_t = o_t * torch.tanh(c_t)   #hidden state of current lstm [batch, hidden_size]

        return h_t, c_t


class CustomLSTM(nn.Module):
    def __init__(self,input_size, hidden_size,hidden_layer=False):
        super(CustomLSTM,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_layer = hidden_layer

        self.LSTM_cell = CustomLSTMCell(self.input_size,self.hidden_size,self.hidden_layer)

    def forward(self,x):
        batch,padded_size,embedding_dim = x.shape
        h_t = torch.zeros(batch,self.hidden_size).to(x.device)  #initializing hidden state
        c_t = torch.zeros(batch,self.hidden_size).to(x.device)   #initializing cell state

        outputs =  torch.zeros(batch,padded_size,self.hidden_size,device=x.device)
        
        #loop through each word in sequence to their find hidden_state
        for t in range(padded_size):
            x_t = x[:, t, :]
            h_t, c_t = self.LSTM_cell(x_t, h_t, c_t)
            outputs[:, t, :] = h_t  #[batch,seq_len,hidden_size]
        return outputs


if __name__ == "__main__":
    lstm = CustomLSTM(8, 4)
    example_input = torch.randint(0,100,(1,4,8))
    aa = lstm(example_input)
    print("Shape of LSTM Output: ",aa.shape)
    print("LSTM output: \n",aa)



