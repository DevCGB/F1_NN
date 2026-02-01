import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader

imputer = KNNImputer(n_neighbors=1)
scaler = StandardScaler()

columns = ['lap', 'laps_remaining', 'gap_to_p2', 'gap_trend_3', 'tire_age', 
           'air_temp', 'track_temp', 'tire_compound']

columns_scaler = ['lap', 'laps_remaining', 'gap_to_p2', 'gap_trend_3', 'tire_age', 
           'air_temp', 'track_temp']

compound_to_num = {'ULTRASOFT': 0, 'SOFT': 1, 'SUPERSOFT': 2, "MEDIUM": 3,
                'HARD': 4, "WET": 5, 'INTERMEDIATE': 6, 'UNKNOWN': 7}

num_to_compound = {0: 'ULTRASOFT', 1: 'SOFT', 2: 'SUPERSOFT', 3: 'MEDIUM',
                   4: 'HARD', 5: 'WET', 6: 'INTERMEDIATE', 7: 'UNKNOWN'}

df = pd.read_csv('f1_data.csv')

df['tire_compound'] = df['tire_compound'].map(compound_to_num)

split_num = int(0.8 * len(df['tire_compound']))

train_df = df.iloc[:split_num, :].copy().reset_index(drop=True)
test_df = df.iloc[split_num:, :].copy().reset_index(drop=True)

print(test_df.head(90))

new_train_df = pd.DataFrame(imputer.fit_transform(train_df[columns]), columns=columns)
new_test_df = pd.DataFrame(imputer.fit_transform(test_df[columns]), columns=columns)


for column in columns:

    train_df[column] = new_train_df[column]

for column in columns:

    test_df[column] = new_test_df[column]


train_df['tire_compound'] = new_train_df['tire_compound'].map(num_to_compound)
test_df['tire_compound'] = new_test_df['tire_compound'].map(num_to_compound)


train_df = pd.get_dummies(train_df, columns=['track', 'tire_compound'], dtype=int)
test_df = pd.get_dummies(test_df, columns=['track', 'tire_compound'], dtype=int)

#print(train_df)

scale_train_df = pd.DataFrame(scaler.fit_transform(train_df[columns_scaler]), columns=columns_scaler)
scale_test_df = pd.DataFrame(scaler.fit_transform(test_df[columns_scaler]), columns=columns_scaler)

for column in columns_scaler:

    train_df[column] = scale_train_df[column]

for column in columns_scaler:

    test_df[column] = scale_test_df[column]

test_df = test_df.reindex(columns=train_df.columns, fill_value=0)

train_df = train_df.drop(columns=['Unnamed: 0'])
test_df = test_df.drop(columns=['Unnamed: 0'])

# Double check
#print(train_df)
print(test_df)


x_train = torch.tensor((train_df.loc[:, train_df.columns != 'label']).to_numpy()).float()
y_train = torch.tensor((train_df['label']).to_numpy()).float()

x_test = torch.tensor((test_df.loc[:, test_df.columns != 'label']).to_numpy()).float()
y_test = torch.tensor((test_df['label']).to_numpy()).float()

model = nn.Sequential(
    nn.Linear(45, 64),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
    nn.Sigmoid()
)

num_epochs = 100

loss_fn = nn.MSELoss()
optimiser = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

train_dataset = TensorDataset(x_train, y_train) 
train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

for epoch in range(num_epochs):
    epoch_loss = 0
    for batch_features, batch_targets in train_dataloader:
        model.train()
        optimiser.zero_grad()
        pred = model(batch_features)

        loss = loss_fn(pred, batch_targets.view(-1, 1))
        loss.backward()
        optimiser.step()
        epoch_loss += loss.item()

    if epoch % 10 == 0:
        print(f"Epoch {epoch} | Avg Loss: {epoch_loss/len(train_dataloader):.4f}")

model.eval()
with torch.no_grad():
    test_preds = model(x_test)
    test_loss = loss_fn(test_preds, y_test.view(-1, 1))
print("\nTest MSEL", float(test_loss.item()))


# Test the model with real data

# This data is a loss
data = {'track': 'United Arab Emirates', 'lap': 24, 'laps_remaining': 34, 'gap_to_p2': 18.823, 'gap_trend_3': 5.522666666666667, 'tire_compound': 'HARD', 'tire_age': np.float64(24.0), 'safety_car': 0, 'air_temp': np.float64(26.8), 'track_temp': np.float64(31.3), 'rain': 0}
#9624,United Arab Emirates,24,34,18.823,5.522666666666667,HARD,24.0,0,26.8,31.3,0,0

data_df = pd.DataFrame([data])

data_df = pd.get_dummies(data_df, columns=['track', 'tire_compound'], dtype=int)

scale_data_df = pd.DataFrame(scaler.fit_transform(data_df[columns_scaler]), columns=columns_scaler)

for column in columns_scaler:

    data_df[column] = scale_data_df[column]

data_df = data_df.reindex(columns=train_df.columns, fill_value=0)
data_df = data_df.drop(columns=['label'])

tensor_data = torch.tensor(data_df.to_numpy()).float()

with torch.no_grad():
    final_pred = model(tensor_data)
print(f"The driver is {float(final_pred * 100):.4f}% likely to win")

# TO DO: Make it so you can input data and the model makes a guess (e.g. with the data below)

