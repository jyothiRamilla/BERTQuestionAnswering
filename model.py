import config
import transformers
import torch.nn as nn

class BERTBaseUncased(nn.Module):
    def __init__(self):
        super(BERTBaseUncased, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH)
        self.bert_drop = nn.Dropout(0.3)
        self.l0 = nn.Linear(768, 2)
    
    def forward(self,ids,mask,token_type_ids):
        sequence_output, pooled_output = self.bert(
            ids,
            attention_mask=mask,
            token_type_ids=token_type_ids
        ) 
        #(batch_size, num_tokens, 768)
        logits = self.l0(sequence_output)#(batch_size, num_tokens, 2)
        #we are going to split like (batch_size, num_tokens,1), (batch_size,num_tokens,1)
        start_logits,end_logits = logits.split(1,dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        #(batch_size,num_tokens), (batch_size, num_tokens)

        bo = self.bert_drop(o2)
        output = self.out(bo)
        return output
