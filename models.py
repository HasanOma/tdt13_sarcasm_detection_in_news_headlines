import torch
from torch import nn
from transformers import DistilBertModel

class LSTMSarcasmClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, n_classes):
        super(LSTMSarcasmClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            n_layers,
            batch_first=True,
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, n_classes) 

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        lstm_out = lstm_out[:, -1, :]
        out = self.fc(lstm_out)
        return out

class SarcasmClassifier(nn.Module):
    def __init__(self, n_classes):
        super(SarcasmClassifier, self).__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.drop = nn.Dropout(p=0.3)
        self.pre_classifier = nn.Linear(self.distilbert.config.hidden_size, self.distilbert.config.hidden_size)
        self.relu = nn.ReLU()
        self.classifier = nn.Linear(self.distilbert.config.hidden_size + 1, n_classes)
    
    def forward(self, input_ids, attention_mask, sentiment):
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        hidden_state = outputs.last_hidden_state  
        pooled_output = hidden_state[:, 0]        
        pooled_output = self.pre_classifier(pooled_output)
        pooled_output = self.relu(pooled_output)
        pooled_output = self.drop(pooled_output)
        sentiment = sentiment.unsqueeze(1)        
        pooled_output = torch.cat((pooled_output, sentiment), dim=1) 
        output = self.classifier(pooled_output)
        return output
    

