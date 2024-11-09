from flask import Flask, render_template, request
import torch
import pickle
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import joblib
from models import LSTMSarcasmClassifier, SarcasmClassifier

app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

analyzer = SentimentIntensityAnalyzer()

# Load All Models
clf_lr = joblib.load('results/logistic_regression_model.pkl')
tfidf = joblib.load('results/tfidf_vectorizer.pkl')
vocab = joblib.load('results/vocab.pkl')

def tokenizer(text):
    return text.lower().split()

def text_to_sequence(text):
    return [vocab.get(token, vocab['<unk>']) for token in tokenizer(text)]

vocab_size = len(vocab)
embedding_dim = 100
hidden_dim = 128
n_layers = 2
n_classes = 2

model_lstm = LSTMSarcasmClassifier(vocab_size, embedding_dim, hidden_dim, n_layers, n_classes)
model_lstm.load_state_dict(torch.load('results/lstm_model_state.bin', map_location=device))
model_lstm = model_lstm.to(device)
model_lstm.eval()

with open('results/tokenizer.pickle', 'rb') as handle:
    tokenizer_distilbert = pickle.load(handle)

model_distilbert = SarcasmClassifier(n_classes=2)
model_distilbert.load_state_dict(torch.load('results/best_model_state.bin', map_location=device))
model_distilbert = model_distilbert.to(device)
model_distilbert.eval()

def preprocess_text(text, model_type, max_len=50):
    if model_type == 'logistic_regression':
        return text
    elif model_type == 'lstm':
        sequence = text_to_sequence(text)
        sequence = torch.tensor(sequence).unsqueeze(0).to(device)
        return sequence
    elif model_type == 'distilbert':
        encoding = tokenizer_distilbert.encode_plus(
            text,
            add_special_tokens=True,
            max_length=max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        sentiment_score = analyzer.polarity_scores(text)['compound']
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        sentiment = torch.tensor([sentiment_score], dtype=torch.float).to(device)
        return input_ids, attention_mask, sentiment

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        headline = request.form['headline']
        
        results = {}
        
        # Logistic Regression Prediction
        X_input = tfidf.transform([headline])
        y_prob = clf_lr.predict_proba(X_input)[0]
        confidence_lr = max(y_prob) * 100
        predicted_class_lr = clf_lr.predict(X_input)[0]
        pred_label_lr = 'Sarcastic' if predicted_class_lr == 1 else 'Not Sarcastic'
        results['lr_prediction'] = pred_label_lr
        results['lr_confidence'] = round(float(confidence_lr), 2)
        
        # LSTM Prediction
        sequence = preprocess_text(headline, 'lstm')
        with torch.no_grad():
            outputs = model_lstm(sequence)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            confidence_lstm, predicted_class_lstm = torch.max(probs, dim=1)
            pred_label_lstm = 'Sarcastic' if predicted_class_lstm.item() == 1 else 'Not Sarcastic'
            confidence_lstm = confidence_lstm.item() * 100
        results['lstm_prediction'] = pred_label_lstm
        results['lstm_confidence'] = round(float(confidence_lstm), 2)
        
        # DistilBERT Prediction
        input_ids, attention_mask, sentiment = preprocess_text(headline, 'distilbert')
        with torch.no_grad():
            outputs = model_distilbert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                sentiment=sentiment
            )
            logits = outputs
            probs = torch.nn.functional.softmax(logits, dim=1)
            confidence_db, predicted_class_db = torch.max(probs, dim=1)
            pred_label_db = 'Sarcastic' if predicted_class_db.cpu().numpy()[0] == 1 else 'Not Sarcastic'
            confidence_db = confidence_db.cpu().numpy()[0] * 100
        results['distilbert_prediction'] = pred_label_db
        results['distilbert_confidence'] = round(float(confidence_db), 2)
        
        return render_template(
            'index.html',
            headline=headline,
            lr_prediction=results['lr_prediction'],
            lr_confidence=results['lr_confidence'],
            lstm_prediction=results['lstm_prediction'],
            lstm_confidence=results['lstm_confidence'],
            distilbert_prediction=results['distilbert_prediction'],
            distilbert_confidence=results['distilbert_confidence']
        )
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
