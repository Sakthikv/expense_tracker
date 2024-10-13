from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import re

app = Flask(__name__)

# Global months list
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]

# Global variables to track update progress
expense_columns = ['Rent', 'Utilities', 'Groceries', 'Transportation', 'Entertainment', 'Healthcare', 'Savings', 'Misc']
update_index = 0  # To track the progress of updating expenses

# Load dataset and process it
def load_salary_data(file_path):
    df = pd.read_excel(file_path)
    df = calculate_total_usage(df)
    return df

# Function to calculate 'Total_Usage' and 'Savings_Percentage'
def calculate_total_usage(df):
    expense_columns = ['Rent', 'Utilities', 'Groceries', 'Transportation', 'Entertainment', 'Healthcare', 'Savings', 'Misc']

    if 'Total_Usage' not in df.columns:
        df['Total_Usage'] = df[expense_columns].sum(axis=1)

    # Calculate 'Savings_Percentage' if not present
    if 'Savings_Percentage' not in df.columns:
        df['Savings_Percentage'] = (df['Savings'] / df['Total_Usage']) * 100

    return df

# Predict usage and suggest savings
def predict_usage(df):
    df['Previous_Month_Usage'] = df['Total_Usage'].shift(1)
    df['2_Months_Ago_Usage'] = df['Total_Usage'].shift(2)
    df.dropna(inplace=True)

    X = df[['Previous_Month_Usage', '2_Months_Ago_Usage']]
    y = df['Total_Usage']

    model = LinearRegression()
    model.fit(X, y)

    recent_usage = df.tail(1)[['Previous_Month_Usage', '2_Months_Ago_Usage']]
    predicted_usage = model.predict(recent_usage)

    return predicted_usage[0]

# Predict expenses by category based on proportions
def predict_expenses_by_category(df, predicted_total_usage):
    expense_columns = ['Rent', 'Utilities', 'Groceries', 'Transportation', 'Entertainment', 'Healthcare', 'Misc']

    total_usage_last_month = df['Total_Usage'].iloc[-1]
    expense_proportions = df[expense_columns].iloc[-1] / total_usage_last_month

    predicted_expenses = expense_proportions * predicted_total_usage
    return predicted_expenses

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_chart_data')
def get_chart_data():
    df = load_salary_data('Dataset.xlsx')

    # Ensure the dataset has a 'Month' column
    if 'Month' not in df.columns:
        raise ValueError("Dataset must contain a 'Month' column")

    # Sort dataframe by the 'Month' column to ensure correct order
    df['Month'] = pd.Categorical(df['Month'], categories=MONTHS, ordered=True)
    df = df.sort_values('Month').reset_index(drop=True)

    predicted_usage = predict_usage(df)

    # Extract months and usage for the chart
    months = df['Month'].tolist()
    total_usage = df['Total_Usage'].tolist()

    chart_data = {
        'months': months,
        'usage': total_usage
    }

    return jsonify(chart_data)

@app.route('/chat', methods=['POST'])
def chat():
    global update_index

    user_message = request.json.get('message')
    df = load_salary_data('Dataset.xlsx')
    predicted_usage = predict_usage(df)

    # Handle "add expense" or "update expense"
    if "update expenses" in user_message.lower() or "add expense" in user_message.lower():
        update_index = 0  # Reset the index to start updating from the first category
        return jsonify({"response": "Let's update your next month Expenses one by one. Please type 'ok' to proceed."})

    # Handle the update confirmation
    if "ok" in user_message.lower():
        if update_index < len(expense_columns):
            return jsonify({"response": f"We'll do it one by one. Please enter the amount for {expense_columns[update_index]}."})

    # Handle predicted total usage queries
    if "predicted usage" in user_message.lower() or "predicted expenses" in user_message.lower():
        return jsonify({"response": f"The predicted total usage for the next month is ${predicted_usage:.2f}."})

    # Handle "give me all predicted expenses"
    if "individual" in user_message.lower():
        predicted_expenses = predict_expenses_by_category(df, predicted_usage)
        predicted_expenses_response = "\n".join([f"{category}: ${amount:.2f}" for category, amount in zip(expense_columns, predicted_expenses)])
        return jsonify({"response": f"The predicted expenses for the next month are:\n{predicted_expenses_response}\nTotal: ${predicted_usage:.2f}"})

    # Check if the message contains both amount and category for updating
    amount_match = re.search(r'\$?(\d+\.?\d*)', user_message)
    if amount_match and update_index < len(expense_columns):
        amount_found = float(amount_match.group(1))

        category_found = expense_columns[update_index]
        average_spending = np.mean(df[category_found])
        latest_savings_percentage = df['Savings_Percentage'].tail(1).values[0]

        if amount_found <= average_spending:
            response = f"Actually, it is good to spend ${amount_found} on {category_found} next month and it will increase your savings percentage up to {latest_savings_percentage:.2f}%."
        else:
            response = f"Sorry for your inconvenience, actually it is not suggested for you to spend ${amount_found} on {category_found} next month because typically it will reduce your savings percentage to {latest_savings_percentage:.2f}%."

        # Move to the next expense
        update_index += 1

        if update_index < len(expense_columns):
            return jsonify({"response": f"{response} Now, please enter the amount for {expense_columns[update_index]}."})
        else:
            return jsonify({"response": f"{response} All the expenses are updated. Now you can ask for suggestions or predictions."})

    return jsonify({"response": "Could not understand the query. Please ask about spending suitability, predicted usage, income update, or expense details."})

if __name__ == '__main__':
    app.run(debug=True)
