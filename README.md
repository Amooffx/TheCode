import requests
import pandas as pd
import torch
import torch.nn as nn
from flask import Flask, request, jsonify
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Data Collection & Preprocessing

def fetch_gas_price(api_key):
    url = f"https://api.etherscan.io/api?module=gastracker&action=gasoracle&apikey={api_key}"
    response = requests.get(url)
    data = response.json()
    return data

def preprocess_data(data):
    df = pd.DataFrame(data['result'], index=[0])
    df['timestamp'] = pd.Timestamp.now()
    df = df[['timestamp', 'SafeGasPrice', 'ProposeGasPrice', 'FastGasPrice']]
    df = df.apply(pd.to_numeric, errors='ignore')
    return df

# Model Definition

class GasPricePredictor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(GasPricePredictor, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# Model Initialization
input_size = 1
hidden_size = 50
num_layers = 2
output_size = 1

model = GasPricePredictor(input_size, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Evaluation Function

def evaluate_model(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    return mae, mse

# API Deployment

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    input_data = torch.tensor(data['input']).float().unsqueeze(0).unsqueeze(2)
    prediction = model(input_data)
    return jsonify({'prediction': prediction.item()})

if __name__ == '__main__':
    # Example data collection and processing
    api_key = "your_api_key_here"
    raw_data = fetch_gas_price(api_key)
    processed_data = preprocess_data(raw_data)
    processed_data.to_csv("gas_price_data.csv", index=False)

    # Example API run
    app.run(debug=True)
