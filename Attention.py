import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self,input_size):
        super(Attention,self).__init__()
        self.hidden_size = input_size
        self.attention = nn.Linear(self.hidden_size,1)
    
    def forward(self, x, mask=None):
        attention_score = self.attention(x).squeeze(-1)  #[batch,seq_len,1]
        
        if mask is not None:
            attention_score = attention_score.masked_fill(mask == 0, -1e9)

        attention_weight = torch.softmax(attention_score, dim=1)  #softmax on sequence to calculate probability [batch,seq_len]
        
        attn_weights_expanded = attention_weight.unsqueeze(-1)  #[batch,seq_len] --> [batch,seq_len, 1]  [[[1],[2],[3]]]=[1,3,1]
        weighted_output = x * attn_weights_expanded     #hidden state multiply with atten_weight => [batch,seq_len,hidden_state] * 
                                                        #[batch,seq_len,hidden_state] * [batch,seq_len,1] = 
                                                        # [batch,seq_len,hidden_state] this determine the importance of each word in sequence
        context_vector = weighted_output.sum(dim=1)     #find the summary of sequence [batch,hidden_state]

        return context_vector, attention_weight
    

if __name__ == "__main__":
    att = Attention(input_size=8)
    input = torch.rand(1,4,8)
    context_vector,attn_score = att(input)
    print(attn_score)
    print(context_vector)
