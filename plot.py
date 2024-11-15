import json
import matplotlib.pyplot as plt

# Load data from JSON file
with open('D:/Study/Module/Master Thesis/trained_models/losses.json', 'r') as f:
    data = json.load(f)

# Ensure the JSON file contains the required keys
if 'train_losses' in data and 'val_losses' in data:
    train_losses = data['train_losses']
    val_losses = data['val_losses']
else:
    raise KeyError("The JSON file does not contain the required keys: 'train_losses' and 'val_losses'")

# Determine the number of epochs based on the loaded data
num_epochs = len(train_losses)

# Plotting the losses
plt.figure(figsize=(12, 5))

plt.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
plt.plot(range(1, num_epochs + 1), val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Losses')
plt.legend()

plt.tight_layout()
plt.show()