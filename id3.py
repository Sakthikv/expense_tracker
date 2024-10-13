import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder

# Sample weather data
data = {
    "Outlook": ["Sunny", "Sunny", "Overcast", "Rainy", "Rainy", "Rainy", "Overcast", "Sunny", "Sunny", "Rainy"],
    "Temperature": ["Hot", "Hot", "Hot", "Mild", "Mild", "Cool", "Cool", "Mild", "Cool", "Mild"],
    "Humidity": ["High", "High", "High", "Normal", "Normal", "Normal", "High", "Normal", "Normal", "High"],
    "Windy": ["False", "True", "False", "False", "False", "True", "True", "False", "True", "True"],
    "PlayTennis": ["No", "No", "Yes", "Yes", "Yes", "No", "Yes", "No", "Yes", "Yes"]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Function to calculate entropy
def calculate_entropy(data, target_column):
    total_rows = len(data)
    target_values = data[target_column].unique()
    entropy = 0
    for value in target_values:
        value_count = len(data[data[target_column] == value])
        proportion = value_count / total_rows
        entropy -= proportion * math.log2(proportion) if proportion != 0 else 0
    return entropy

# Calculate overall entropy for the outcome
entropy_outcome = calculate_entropy(df, 'PlayTennis')
print(f"Entropy of the dataset: {entropy_outcome:.3f}")

# Function to calculate information gain
def calculate_information_gain(data, feature, target_column):
    unique_values = data[feature].unique()
    weighted_entropy = 0
    for value in unique_values:
        subset = data[data[feature] == value]
        proportion = len(subset) / len(data)
        weighted_entropy += proportion * calculate_entropy(subset, target_column)
    information_gain = entropy_outcome - weighted_entropy
    return information_gain

# Calculate and print entropy and information gain for each feature
for column in df.columns[:-1]:
    entropy = calculate_entropy(df, column)
    information_gain = calculate_information_gain(df, column, 'PlayTennis')
    print(f"{column} - Entropy: {entropy:.3f}, Information Gain: {information_gain:.3f}")

# Encode categorical features into numerical values
label_encoders = {}
for column in df.columns:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column])
    label_encoders[column] = le

# Prepare features and target
features = df.columns[:-1]
X = df[features]
y = df['PlayTennis']

# Create and fit the decision tree classifier
clf = DecisionTreeClassifier(criterion='entropy', max_depth=3)
clf.fit(X, y)

# Visualize the decision tree
plt.figure(figsize=(10, 8))
plot_tree(clf, feature_names=features, class_names=label_encoders['PlayTennis'].classes_, filled=True, rounded=True)
plt.show()

# Function to implement the ID3 algorithm
def id3(data, target_column, features):
    # If all target values are the same, return that value
    if len(data[target_column].unique()) == 1:
        return data[target_column].iloc[0]

    # If no features are left, return the most common target value
    if len(features) == 0:
        return data[target_column].mode().iloc[0]

    # Select the best feature to split on
    best_feature = max(features, key=lambda x: calculate_information_gain(data, x, target_column))
    tree = {best_feature: {}}

    # Remove the best feature from the list of features
    features = [f for f in features if f != best_feature]

    # Recursively build the tree
    for value in data[best_feature].unique():
        subset = data[data[best_feature] == value]
        tree[best_feature][value] = id3(subset, target_column, features)

    return tree

# Running ID3 algorithm
features = df.columns[:-1].tolist()  # Exclude the target column
decision_tree = id3(df, 'PlayTennis', features)
print("Decision Tree Structure:\n", decision_tree)