import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import DistilBertTokenizer
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.metrics import classification_report
import pickle
import joblib
from models import LSTMSarcasmClassifier, SarcasmClassifier
from collections import Counter
from itertools import chain
from torch.nn.utils.rnn import pad_sequence
from transformers import AdamW, get_linear_schedule_with_warmup

MODEL_TYPE = 'lstm'  # 'distilbert', 'lstm', 'logistic_regression'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

df = pd.read_json('Sarcasm_Headlines_Dataset.json', lines=True)

df = df.sample(n=23000, random_state=42).reset_index(drop=True)

analyzer = SentimentIntensityAnalyzer()
df['sentiment'] = df['headline'].apply(lambda x: analyzer.polarity_scores(x)['compound'])

X = df[['headline', 'sentiment']]
y = df['is_sarcastic']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test = y_test.reset_index(drop=True)

train_df = X_train.copy()
train_df['is_sarcastic'] = y_train

test_df = X_test.copy()
test_df['is_sarcastic'] = y_test

print(f"Train DataFrame columns: {train_df.columns}")


# -----------------------------
# Logistic Regression Model
# -----------------------------
if MODEL_TYPE == 'logistic_regression':

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    tfidf = TfidfVectorizer(max_features=8192)
    X_train_tfidf = tfidf.fit_transform(train_df['headline'])
    X_test_tfidf = tfidf.transform(test_df['headline'])

    clf = LogisticRegression(max_iter=1024)
    clf.fit(X_train_tfidf, train_df['is_sarcastic'])

    y_pred = clf.predict(X_test_tfidf)
    print("Classification Report:")
    print(classification_report(test_df['is_sarcastic'], y_pred))

    joblib.dump(clf, 'logistic_regression_model.pkl')
    joblib.dump(tfidf, 'tfidf_vectorizer.pkl')



# -----------------------------
# LSTM Model
# -----------------------------
elif MODEL_TYPE == 'lstm':

    tokenizer = lambda x: x.split()

    token_counts = Counter(chain.from_iterable(map(tokenizer, train_df['headline'])))
    vocab = {token: idx + 2 for idx, (token, _) in enumerate(token_counts.items())}
    vocab['<unk>'] = 0
    vocab['<pad>'] = 1

    def text_to_sequence(text):
        return [vocab.get(token, vocab['<unk>']) for token in tokenizer(text)]

    train_sequences = [text_to_sequence(text) for text in train_df['headline']]
    test_sequences = [text_to_sequence(text) for text in test_df['headline']]

    def pad_sequences(sequences, max_len=50):
        sequences = [torch.tensor(seq[:max_len]) for seq in sequences]
        sequences = pad_sequence(sequences, batch_first=True, padding_value=vocab['<pad>'])
        return sequences

    X_train_seq = pad_sequences(train_sequences)
    X_test_seq = pad_sequences(test_sequences)

    y_train_tensor = torch.tensor(train_df['is_sarcastic'].values, dtype=torch.long)
    y_test_tensor = torch.tensor(test_df['is_sarcastic'].values, dtype=torch.long)

    train_dataset = torch.utils.data.TensorDataset(X_train_seq, y_train_tensor)
    test_dataset = torch.utils.data.TensorDataset(X_test_seq, y_test_tensor)

    batch_size = 32
    train_data_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(test_dataset, batch_size=batch_size)

    vocab_size = len(vocab)
    embedding_dim = 100
    hidden_dim = 128
    n_layers = 2
    n_classes = 2

    class LSTMSarcasmClassifier(nn.Module):
        def __init__(self, vocab_size, embedding_dim, hidden_dim, n_layers, n_classes):
            super(LSTMSarcasmClassifier, self).__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=vocab['<pad>'])
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
            
            lengths = (x != vocab['<pad>']).sum(dim=1)
            packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False)
            packed_output, (hidden, cell) = self.lstm(packed_embedded)
            hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)
            out = self.fc(hidden)
            return out

    model = LSTMSarcasmClassifier(vocab_size, embedding_dim, hidden_dim, n_layers, n_classes)
    model = model.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    epochs = 5
    best_accuracy = 0

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct_predictions = 0

        for X_batch, y_batch in train_data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = loss_fn(outputs, y_batch)
            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == y_batch)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        train_acc = correct_predictions.double() / len(train_dataset)
        train_loss = total_loss / len(train_data_loader)

        model.eval()
        total_loss = 0
        correct_predictions = 0
        preds_all = []
        labels_all = []

        with torch.no_grad():
            for X_batch, y_batch in test_data_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                outputs = model(X_batch)
                loss = loss_fn(outputs, y_batch)
                _, preds = torch.max(outputs, dim=1)
                correct_predictions += torch.sum(preds == y_batch)
                total_loss += loss.item()
                preds_all.extend(preds.cpu().numpy())
                labels_all.extend(y_batch.cpu().numpy())

        val_acc = correct_predictions.double() / len(test_dataset)
        val_loss = total_loss / len(test_data_loader)

        print(f'Epoch {epoch + 1}/{epochs}')
        print(f'Train loss: {train_loss:.4f} accuracy: {train_acc:.4f}')
        print(f'Validation loss: {val_loss:.4f} accuracy: {val_acc:.4f}')

        if val_acc > best_accuracy:
            torch.save(model.state_dict(), 'lstm_model_state.bin')
            best_accuracy = val_acc

    print("Classification Report:")
    print(classification_report(labels_all, preds_all))

    joblib.dump(vocab, 'vocab.pkl')



# -----------------------------
# DistilBERT Model
# -----------------------------
elif MODEL_TYPE == 'distilbert':

    class SarcasmDataset(Dataset):
        def __init__(self, headlines, sentiments, labels, tokenizer, max_len=50):
            self.headlines = headlines
            self.sentiments = sentiments
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_len = max_len
            
        def __len__(self):
            return len(self.headlines)
        
        def __getitem__(self, idx):
            headline = str(self.headlines.iloc[idx])
            sentiment = self.sentiments.iloc[idx]
            label = self.labels.iloc[idx]
            
            encoding = self.tokenizer.encode_plus(
                headline,
                add_special_tokens=True,
                max_length=self.max_len,
                return_token_type_ids=False,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
            )
            
            return {
                'headline_text': headline,
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'sentiment': torch.tensor(sentiment, dtype=torch.float),
                'labels': torch.tensor(label, dtype=torch.long)
            }

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    with open('tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def create_data_loader(df, tokenizer, batch_size):
        ds = SarcasmDataset(
            headlines=df['headline'],
            sentiments=df['sentiment'],
            labels=df['is_sarcastic'],
            tokenizer=tokenizer
        )
        return DataLoader(ds, batch_size=batch_size, num_workers=0)

    batch_size = 16
    train_data_loader = create_data_loader(train_df, tokenizer, batch_size)
    test_data_loader = create_data_loader(test_df, tokenizer, batch_size)

    model = SarcasmClassifier(n_classes=2)
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_data_loader) * 2 
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=0,
        num_training_steps=total_steps
    )
    loss_fn = nn.CrossEntropyLoss().to(device)

    epochs = 5
    best_accuracy = 0

    for epoch in range(epochs):
        print(f'Epoch {epoch + 1}/{epochs}')
        print('-' * 10)

        model.train()
        losses = []
        correct_predictions = 0

        for d in train_data_loader:
            input_ids = d["input_ids"].to(device)
            attention_mask = d["attention_mask"].to(device)
            sentiment = d["sentiment"].to(device)
            labels = d["labels"].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                sentiment=sentiment
            )

            loss = loss_fn(outputs, labels)
            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        train_acc = correct_predictions.double() / len(train_df)
        train_loss = np.mean(losses)

        model.eval()
        losses = []
        correct_predictions = 0
        preds_all = []
        labels_all = []

        with torch.no_grad():
            for d in test_data_loader:
                input_ids = d["input_ids"].to(device)
                attention_mask = d["attention_mask"].to(device)
                sentiment = d["sentiment"].to(device)
                labels = d["labels"].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    sentiment=sentiment
                )

                loss = loss_fn(outputs, labels)
                _, preds = torch.max(outputs, dim=1)

                correct_predictions += torch.sum(preds == labels)
                losses.append(loss.item())
                preds_all.extend(preds.cpu().numpy())
                labels_all.extend(labels.cpu().numpy())

        val_acc = correct_predictions.double() / len(test_df)
        val_loss = np.mean(losses)

        print(f'Train loss {train_loss:.4f} accuracy {train_acc:.4f}')
        print(f'Validation loss {val_loss:.4f} accuracy {val_acc:.4f}')

        if val_acc > best_accuracy:
            torch.save(model.state_dict(), 'best_model_state.bin')
            best_accuracy = val_acc

    print("Classification Report:")
    print(classification_report(labels_all, preds_all))

    torch.save(model.state_dict(), 'sarcasm_detection_model.bin')

else:
    print("Invalid MODEL_TYPE selected.")
