import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt

# Load historical match data
# The dataset should have columns like 'HomeTeam', 'AwayTeam', 'HomeGoals', 'AwayGoals', 'HomeWin', 'AwayWin', 'Draw'
# For simplicity, we are generating synthetic data here

np.random.seed(42)
teams = ['TeamA', 'TeamB', 'TeamC', 'TeamD', 'TeamE']
data = {
    'HomeTeam': np.random.choice(teams, 1000),
    'AwayTeam': np.random.choice(teams, 1000),
    'HomeGoals': np.random.randint(0, 5, 1000),
    'AwayGoals': np.random.randint(0, 5, 1000)
}
df = pd.DataFrame(data)
df['HomeWin'] = (df['HomeGoals'] > df['AwayGoals']).astype(int)
df['AwayWin'] = (df['HomeGoals'] < df['AwayGoals']).astype(int)
df['Draw'] = (df['HomeGoals'] == df['AwayGoals']).astype(int)

# Feature engineering
df['GoalDifference'] = df['HomeGoals'] - df['AwayGoals']

# Prepare features and target variable
X = df[['GoalDifference']]
y = df['HomeWin']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'ROC AUC Score: {roc_auc}')

# Plot the ROC curve
from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()

# Betting strategy: Bet on home team if predicted probability > 0.5
df_test = X_test.copy()
df_test['Actual'] = y_test
df_test['Predicted'] = y_pred
df_test['Predicted_Proba'] = y_pred_proba

# Assuming a fixed bet amount and odds for simplicity
bet_amount = 100
odds = 2.0  # Assuming even odds for simplicity
df_test['Bet'] = np.where(df_test['Predicted_Proba'] > 0.5, bet_amount, 0)
df_test['Win'] = np.where(df_test['Actual'] == df_test['Predicted'], odds * df_test['Bet'], 0)
df_test['Profit'] = df_test['Win'] - df_test['Bet']

total_bet = df_test['Bet'].sum()
total_profit = df_test['Profit'].sum()
roi = (total_profit / total_bet) * 100 if total_bet != 0 else 0

print(f'Total Bet Amount: {total_bet}')
print(f'Total Profit: {total_profit}')
print(f'Return on Investment (ROI): {roi:.2f}%')
