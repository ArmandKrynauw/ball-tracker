# %%==================== SIMULATION ====================%%
import numpy as np
from random import randrange

NUM_POINTS = 10
TIME_RANGE = (0, 50)

def generate_timeline():
    # Random initial conditions
    x0 = np.random.uniform(0, 1920)
    y0 = np.random.uniform(0, 1080)
    vx = np.random.uniform(-50, 50)  # Initial velocity in x
    vy = np.random.uniform(-20, 20)  # Initial velocity in y
    dt = 1  # Time step for simulation
    drag = 0.95  # Drag factor
    gravity = randrange(0, 1000)/100 # Gravity

    # Initialize state
    timeline = []
    for t in np.arange(TIME_RANGE[0], TIME_RANGE[1], dt):
        # Update position based on current velocity
        x0 += vx * dt
        y0 += vy * dt
        
        # Apply gravity
        vy += gravity * dt
        
        # Apply drag
        vx *= drag
        vy *= drag

        # Append new state
        timeline.append((x0, y0, t))

        # Randomly apply a strong force
        if np.random.rand() < 0.1:
            force_x = np.random.uniform(-5, 5)
            force_y = np.random.uniform(-5, 5)
            vx += force_x
            vy += force_y

    # Sample NUM_POINTS + 1 entries
    sampled_indices = np.random.choice(len(timeline), NUM_POINTS + 1, replace=False)
    sampled_points = [timeline[i] for i in sampled_indices]
    sampled_points = sorted(sampled_points, key=lambda x: x[2])

    return sampled_points, timeline


# %%==================== DEFINE NN ====================%%
import torch
import torch.nn as nn
import torch.optim as optim

class PositionPredictor(nn.Module):
    def __init__(self):
        super(PositionPredictor, self).__init__()
        self.fc1 = nn.Linear(3 * NUM_POINTS + 1, 64)  # 3 coordinates for each point + tPred
        self.fc2 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 2)  # Output (xPred, yPred)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# %%==================== CREATE DATASET ====================%%
def create_dataset(num_samples):
    X = []
    y = []
    for _ in range(num_samples):
        sampled_points, _ = generate_timeline()

        # Flatten sampled points and separate tPred
        targetIdx = np.random.randint(1, NUM_POINTS + 1)
        tPred = sampled_points[targetIdx][2]  # t of the last point
        target = sampled_points[targetIdx][:2]  # The (x, y) of the last point

        del sampled_points[targetIdx]

        points = np.array(sampled_points).flatten() # All points except the target
        features = np.concatenate([points, [tPred]])

        # print(f"Features: {features}\tTarget: {target}")
        X.append(features)
        y.append(target)
    return np.array(X), np.array(y)

# %%==================== TRAIN ====================%%
import os

REUSE_MODEL = False
model_save_file = 'position_predictor.pth'
model = PositionPredictor()

# Load previous model if exists
if REUSE_MODEL and os.path.exists(model_save_file):
    print(f"Reusing saved model: {model_save_file}")
    model.load_state_dict(torch.load(model_save_file))

else:
    # Create model, define loss function and optimizer
    print("Training model")
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 50000
    for epoch in range(num_epochs):
        # Training parameters
        X, y = create_dataset(200)

        # Convert to PyTorch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        # Train
        model.train()
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()

        if epoch > num_epochs - 50:
            print(f'Target: {y[0]}\tPrediction: {outputs[0].detach().numpy()}')

        if (epoch + 1) % 50 == 0:
            print(f'[{epoch + 1}/{num_epochs}]\tLoss: {loss.item():.4f}')

    print("Training complete.")

    # Save the model
    torch.save(model.state_dict(), model_save_file)

# %%==================== PREDICT ====================%%
import matplotlib.pyplot as plt

def predict_and_plot(model):
    model.eval()
    # Generate a random timeline using the generate_timeline function
    sampled_points, timeline = generate_timeline()

    # Training parameters
    # X, y = create_dataset(1)

    # Convert to PyTorch tensors
    # X_tensor = torch.tensor(X, dtype=torch.float32)
    # y_tensor = torch.tensor(y, dtype=torch.float32)
    
    # Convert to np arrays for plotting
    sampled_points_np = np.array(sampled_points)
    timeline_np = np.array(timeline)

    # Plotting the object path in blue
    plt.figure(figsize=(12, 6))
    plt.plot(timeline_np[:, 0], timeline_np[:, 1], color='blue', label='Object Path')
    # plt.scatter(timeline_np[:, 0], timeline_np[:, 1], color='blue', s=10)
    plt.scatter(sampled_points_np[:, 0], sampled_points_np[:, 1], color='blue', s=20)

    points = np.array(sampled_points[:-1]).flatten()

    print("SAMPLED_POINTS_NP:")
    for p in sampled_points_np:
        print(p)

    print("POINTS:")
    for p in points:
        print(p)
    
    print()
    print("PREDS:")
    for _ in range(10):
        t_pred = randrange(TIME_RANGE[0], TIME_RANGE[1])
        # Flatten the sampled points and add the target timestamp
        features = np.concatenate([points, [t_pred]])
        features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0)

        # Predict the position
        with torch.no_grad():
            prediction = model(features_tensor).numpy().squeeze()

        print(f"{t_pred}:\t{prediction}")

        # Plot the predicted point in green
        plt.scatter(prediction[0], prediction[1], color='green', marker='o', s=50)

    # Customize the plot
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.legend(['Object Path', 'Sampled Inputs', 'Predicted Positions'])
    plt.title('Object Path and Predicted Positions')
    plt.show()

predict_and_plot(model)