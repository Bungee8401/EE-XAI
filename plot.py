import re

# Your data as a multiline string
data = """
Epoch [1/50], Train Loss: 2.2662, Val Loss: 2.0644
Epoch [2/50], Train Loss: 1.9107, Val Loss: 1.7685
Epoch [3/50], Train Loss: 1.6817, Val Loss: 1.6006
Epoch [4/50], Train Loss: 1.4933, Val Loss: 1.3679
Epoch [5/50], Train Loss: 1.3558, Val Loss: 1.2421
Epoch [6/50], Train Loss: 1.2151, Val Loss: 1.0871
Epoch [7/50], Train Loss: 1.1147, Val Loss: 0.9970
Epoch [8/50], Train Loss: 1.0333, Val Loss: 0.9848
Epoch [9/50], Train Loss: 0.9615, Val Loss: 0.8754
Epoch [10/50], Train Loss: 0.9040, Val Loss: 0.8739
Epoch [11/50], Train Loss: 0.8498, Val Loss: 0.8830
Epoch [12/50], Train Loss: 0.8008, Val Loss: 0.7765
Epoch [13/50], Train Loss: 0.7633, Val Loss: 0.7077
Epoch [14/50], Train Loss: 0.7148, Val Loss: 0.7039
Epoch [15/50], Train Loss: 0.6846, Val Loss: 0.6966
Epoch [16/50], Train Loss: 0.6508, Val Loss: 0.6507
Epoch [17/50], Train Loss: 0.6241, Val Loss: 0.6225
Epoch [18/50], Train Loss: 0.6035, Val Loss: 0.5817
Epoch [19/50], Train Loss: 0.5729, Val Loss: 0.6110
Epoch [20/50], Train Loss: 0.5513, Val Loss: 0.6374
Epoch [21/50], Train Loss: 0.5335, Val Loss: 0.6065
Epoch [22/50], Train Loss: 0.5168, Val Loss: 0.5331
Epoch [23/50], Train Loss: 0.4989, Val Loss: 0.6122
Epoch [24/50], Train Loss: 0.4756, Val Loss: 0.5653
Epoch [25/50], Train Loss: 0.4718, Val Loss: 0.5389
Epoch [26/50], Train Loss: 0.4483, Val Loss: 0.5280
Epoch [27/50], Train Loss: 0.4340, Val Loss: 0.5205
Epoch [28/50], Train Loss: 0.4211, Val Loss: 0.5242
Epoch [29/50], Train Loss: 0.4092, Val Loss: 0.5139
Epoch [30/50], Train Loss: 0.3948, Val Loss: 0.5024
Epoch [31/50], Train Loss: 0.3883, Val Loss: 0.5348
Epoch [32/50], Train Loss: 0.3675, Val Loss: 0.5065
Epoch [33/50], Train Loss: 0.3609, Val Loss: 0.4767
Epoch [34/50], Train Loss: 0.3512, Val Loss: 0.4900
Epoch [35/50], Train Loss: 0.3379, Val Loss: 0.4716
Epoch [36/50], Train Loss: 0.3329, Val Loss: 0.4810
Epoch [37/50], Train Loss: 0.3219, Val Loss: 0.4885
Epoch [38/50], Train Loss: 0.3141, Val Loss: 0.4417
Epoch [39/50], Train Loss: 0.3018, Val Loss: 0.4608
Epoch [40/50], Train Loss: 0.2942, Val Loss: 0.4762
Epoch [41/50], Train Loss: 0.2800, Val Loss: 0.4674
Epoch [42/50], Train Loss: 0.2759, Val Loss: 0.4466
Epoch [43/50], Train Loss: 0.2671, Val Loss: 0.4667
Epoch [44/50], Train Loss: 0.2631, Val Loss: 0.4489
Epoch [45/50], Train Loss: 0.2555, Val Loss: 0.4820
Epoch [46/50], Train Loss: 0.2439, Val Loss: 0.4608
Epoch [47/50], Train Loss: 0.2386, Val Loss: 0.4697
"""

# Regular expression to extract the numbers
pattern = r"Epoch \[(\d+)/50\], Train Loss: ([\d\.]+), Val Loss: ([\d\.]+)"

epochs = []
train_losses = []
val_losses = []

for match in re.findall(pattern, data):
    epoch, train_loss, val_loss = match
    epochs.append(int(epoch))
    train_losses.append(float(train_loss))
    val_losses.append(float(val_loss))

# Print the extracted values to verify
print("Epochs:", epochs)
print("Training Losses:", train_losses)
print("Validation Losses:", val_losses)

import matplotlib.pyplot as plt

# Plot Training and Validation Loss
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_losses, label='Training Loss')
plt.plot(epochs, val_losses, label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.show()
