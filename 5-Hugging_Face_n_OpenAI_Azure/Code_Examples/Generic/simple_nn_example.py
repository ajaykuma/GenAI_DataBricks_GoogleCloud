# ================================
# 1. IMPORT LIBRARIES
# ================================
import torch
import torch.nn as nn
import torch.optim as optim

# ================================
# 2. CREATE DATASET
# ================================

# Training sentences and labels
sentences = [
    "tigers are running in forest",
    "birds are flying in sky", 
    "river flows in forest", 
    "mountains are still", 
    "humans are walking in park",
    "people are eating food"
]

# Labels:
# 0 = animals in motion
# 1 = static nature
# 2 = human activity
labels = [0, 0, 1, 1, 2, 2]

# ================================
# 3. BUILD VOCABULARY
# ================================
vocab = set()
for sentence in sentences:
    for word in sentence.split():
        vocab.add(word)

vocab = list(vocab)
word_to_idx = {word: i for i, word in enumerate(vocab)}

# ================================
# 4. BAG OF WORDS FUNCTION
# ================================
def sentence_to_vector(sentence):
    vector = torch.zeros(len(vocab))
    for word in sentence.split():
        if word in word_to_idx:
            vector[word_to_idx[word]] = 1
    return vector

# Convert dataset into tensors
X = torch.stack([sentence_to_vector(s) for s in sentences])
y = torch.tensor(labels)

# ================================
# 5. DEFINE SIMPLE NEURAL NETWORK
# ================================
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        
        # Input → Hidden layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        
        # Activation function
        self.relu = nn.ReLU()
        
        # Hidden → Output layer
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Pass through first layer
        out = self.fc1(x)
        
        # Apply activation
        out = self.relu(out)
        
        # Output layer
        out = self.fc2(out)
        
        return out

# Initialize model
model = SimpleNN(input_size=len(vocab), hidden_size=8, output_size=3)

# ================================
# 6. LOSS + OPTIMIZER
# ================================
criterion = nn.CrossEntropyLoss()   # For classification
optimizer = optim.Adam(model.parameters(), lr=0.01)

# ================================
# 7. TRAINING LOOP
# ================================
print("=== TRAINING START ===")

for epoch in range(100):
    
    # Forward pass
    outputs = model(X)
    
    # Compute loss
    loss = criterion(outputs, y)
    
    # Backward pass (compute gradients)
    optimizer.zero_grad()
    loss.backward()
    
    # Update weights
    optimizer.step()
    
    if (epoch+1) % 20 == 0:
        print(f"Epoch [{epoch+1}/100], Loss: {loss.item():.4f}")

print("=== TRAINING COMPLETE ===\n")


# ================================
# 8. TESTING / INFERENCE
# ================================

def predict(sentence):
    model.eval()  # Set model to evaluation mode
    
    vector = sentence_to_vector(sentence)
    
    with torch.no_grad():
        output = model(vector)
        predicted_class = torch.argmax(output).item()
    
    return predicted_class

# ================================
# 9. TEST WITH NEW INPUT
# ================================
test_sentence = "lions are running in jungle"

prediction = predict(test_sentence)

label_map = {
    0: "Animals in motion",
    1: "Static nature",
    2: "Human activity"
}

print("Test Sentence:", test_sentence)
print("Predicted Class:", prediction)
print("Meaning:", label_map[prediction])
