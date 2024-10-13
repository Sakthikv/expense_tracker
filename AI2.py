import numpy as np
from sklearn.linear_model import LinearRegression

# Historical salary and expenses data
months = np.array([1, 2, 3, 4, 5, 6]).reshape(-1, 1)
expenses = np.array([1200, 1500, 1300, 1600, 1400, 1700])

# Create and train the model
model = LinearRegression()
model.fit(months, expenses)

# Predict future expenses
next_month_expense = model.predict(np.array([[7]]))
print("Predicted Expense for Month 7: $", next_month_expense)
