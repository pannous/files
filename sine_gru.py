import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from math import pi

# RNN = Recurrent Neural Network trivial, nur für kleine Probleme wegen Vanishing Gradient
# LSTM = Long Short Term Memory  komplexes Rechenmodell, verhindert Vanishing Gradient
# GRU = Gated Recurrent Unit     einfacheres Rechenmodell, genauso mächtig

# Training loop
epochs = 150
seq_length = 50 # ok can be arbitrary large with GRU
input_size = 1
hidden_size = 128
output_size = 1
num_layers = 4 # not helpful
learning_rate = 0.01
weight_decay = 0 #.05 not helpful
dropout = 0 # 0.2 # not helpful

# Define the sine function and generate data points
x = torch.linspace(-2 * pi, 2 * pi, 200)
x = x.reshape(-1, 1)  # 200 rows, 1 column turn vector into matrix
def func(x):
	# return torch.sin(x)
	return x**3 + 2 * x**2 + 3 * x + 4
y = func(x)

# Convert to sequences
# Wir erzeugen Sequenzen von 10 Werten, für verschiedene Intervalle der Sinusfunktion
# Die Ausgabe des Netzwerkes ist dann der nächste Wert der Sinusfunktion auf der y Achse
# zB. x [0, 1/100, 2/100, 3/100, 4/100, 5/100, 6/100, 7/100, 8/100, 9/100] -> 10/100
# mit Sinus Aktivierung y in [0,0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008] -> 0.009
def create_sequences(data, seq_length):
	xs, ys = [], []
	for i in range(len(data) - seq_length):
		x_seq = data[i:i + seq_length]
		y_seq = data[i + seq_length] # next value
		xs.append(x_seq)
		ys.append(y_seq)
	return torch.stack(xs), torch.stack(ys)

# kleine Ausschnitte der Sinusfunktion 10 samples lang ( mit x-Abstand Intervalllänge / steps )
x_seq, y_seq = create_sequences(y, seq_length)

class GRUModel(nn.Module):
	def __init__(self, input_size, hidden_size, output_size, num_layers):
		super(GRUModel, self).__init__()
		self.hidden_size = hidden_size
		self.num_layers = num_layers
		self.gru = nn.RNN(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout,  nonlinearity="tanh" ) # RNN same signature:
		# self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
		self.fc = nn.Linear(hidden_size, output_size)

	def forward(self, x):
		h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size)
		out, _ = self.gru(x, h0)
		out = self.fc(out[:, -1, :])
		return out



model = GRUModel(input_size, hidden_size, output_size, num_layers)

# Define loss function and optimizer
criterion = nn.MSELoss()  # + Punish > 1 + punish wild predictions
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

for epoch in range(epochs):
	if epoch % 10 == 0:
		print(f'Epoch {epoch}/{epochs}')
	model.train()
	optimizer.zero_grad()
	outputs = model(x_seq)
	loss = criterion(outputs, y_seq)
	loss.backward()
	optimizer.step()

# To make predictions
model.eval()
with torch.no_grad():
	x_test = torch.linspace(-8 * pi, 8 * pi, 400).reshape(-1, 1)
	y_test = func(x_test).reshape(-1, 1)
	x_test_seq, _ = create_sequences(y_test, seq_length)
	y_pred = model(x_test_seq)

# Align x_test for plotting
x_test_plot = x_test[seq_length:]

# Plotting
plt.plot(x.numpy(), y.numpy(), label='Original Sine Function')
plt.plot(x_test_plot.numpy(), y_pred.detach().numpy(), label='Predicted Sine Function')
plt.legend()
plt.show()
