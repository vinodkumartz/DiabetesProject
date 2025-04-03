from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix, f1_score
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam


# 6. Reinforcement Learning with Deep Q-Network (DQN) - Simulation
def simulate_rl_model(X, y, cat_features, num_features):
    """
    Simulate a Reinforcement Learning approach for diabetes readmission

    Note: This is a simplified simulation as real RL requires an environment to interact with.
    """
    print("\n=== Reinforcement Learning Simulation ===")
    print("Note: This is a simulated approach as real RL requires an environment to interact with")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Create preprocessing pipeline
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, num_features),
            ('cat', categorical_transformer, cat_features)
        ])

    # Preprocess data
    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    # Define a simple Deep Q-Network
    def create_dqn_model(input_dim, output_dim=2):
        model = Sequential([
            Dense(128, input_dim=input_dim, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(output_dim, activation='linear')  # Q-values for each action
        ])

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='mse'
        )

        return model

    # Create DQN model
    dqn_model = create_dqn_model(X_train_processed.shape[1])

    # Simulate Q-learning training
    # In a real RL scenario, we would interact with an environment
    # Here, we'll use the historical data to simulate this process

    # Define a simple reward function
    def get_reward(y_true, y_pred):
        # Higher reward for correct predictions, especially for positive class
        if y_true == 1 and y_pred == 1:  # True positive
            return 10
        elif y_true == 0 and y_pred == 0:  # True negative
            return 5
        elif y_true == 1 and y_pred == 0:  # False negative (worse)
            return -10
        else:  # False positive
            return -5

    # Hyperparameters
    epsilon = 1.0  # Exploration rate
    epsilon_min = 0.01
    epsilon_decay = 0.995
    gamma = 0.95  # Discount factor
    batch_size = 32
    epochs = 10

    # Memory buffer for experience replay
    memory = []

    # Training process simulation
    for epoch in range(epochs):
        total_reward = 0

        # Shuffle data for each epoch
        indices = np.arange(X_train_processed.shape[0])
        np.random.shuffle(indices)
        X_shuffled = X_train_processed[indices]
        y_shuffled = y_train.iloc[indices].values

        for i in range(X_shuffled.shape[0]):
            state = X_shuffled[i:i+1]

            # Epsilon-greedy action selection
            if np.random.rand() <= epsilon:
                action = np.random.randint(0, 2)
            else:
                q_values = dqn_model.predict(state, verbose=0)[0]
                action = np.argmax(q_values)

            # Get reward
            reward = get_reward(y_shuffled[i], action)
            total_reward += reward

            # Store experience in memory
            memory.append((state, action, reward, state, y_shuffled[i]))

            # Experience replay
            if len(memory) > batch_size:
                batch_indices = np.random.choice(len(memory), batch_size, replace=False)
                batch = [memory[i] for i in batch_indices]

                states = np.vstack([experience[0] for experience in batch])
                actions = np.array([experience[1] for experience in batch])
                rewards = np.array([experience[2] for experience in batch])
                next_states = np.vstack([experience[3] for experience in batch])

                # Current Q-values
                current_q = dqn_model.predict(states, verbose=0)

                # Target Q-values
                target_q = current_q.copy()

                for j in range(batch_size):
                    target_q[j, actions[j]] = rewards[j]

                # Train the model
                dqn_model.train_on_batch(states, target_q)

        # Decay epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        print(f"Epoch {epoch+1}/{epochs}, Total Reward: {total_reward}, Epsilon: {epsilon:.4f}")

    # Evaluate on test set
    q_values = dqn_model.predict(X_test_processed, verbose=0)
    y_pred_rl = np.argmax(q_values, axis=1)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred_rl)
    f1 = f1_score(y_test, y_pred_rl)

    # For ROC AUC, we need probabilities - we'll use a softmax on Q-values
    # This is an approximation for demonstration
    def softmax(x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)

    probs = softmax(q_values)
    roc_auc = roc_auc_score(y_test, probs[:, 1])

    print(f"Simulated RL Model - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, ROC AUC: {roc_auc:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred_rl))

    return {
        'accuracy': accuracy,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'confusion_matrix': confusion_matrix(y_test, y_pred_rl),
        'classification_report': classification_report(y_test, y_pred_rl)
    }

