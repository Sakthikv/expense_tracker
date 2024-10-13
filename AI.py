import pandas as pd

class SalaryManagementAI:
    def __init__(self, file_path):
        # Load dataset from the CSV file
        self.file_path = file_path
        self.data = pd.read_csv(file_path)

    def add_expense(self, month, expense_name, amount):
        # Add the new expense to the dataset
        new_expense = {'Month': month, 'Salary': self.get_salary(month), 'Savings Percentage': self.get_savings_percentage(month),
                       'Expense Name': expense_name, 'Expense Amount': amount}
        self.data = self.data.append(new_expense, ignore_index=True)
        print(f"Added expense '{expense_name}' of ${amount} for {month}")

    def get_salary(self, month):
        if month in self.data['Month'].values:
            return self.data[self.data['Month'] == month]['Salary'].iloc[0]
        else:
            return float(input(f"Enter the salary for {month}: "))

    def get_savings_percentage(self, month):
        if month in self.data['Month'].values:
            return self.data[self.data['Month'] == month]['Savings Percentage'].iloc[0]
        else:
            return float(input(f"Enter the savings percentage for {month}: "))

    def calculate_savings(self, month):
        salary = self.get_salary(month)
        savings_percentage = self.get_savings_percentage(month)
        total_expenses = self.data[self.data['Month'] == month]['Expense Amount'].sum()
        savings = (salary * savings_percentage) / 100
        remaining_salary = salary - total_expenses - savings
        print(f"\n--- {month} Summary ---")
        print(f"Salary: ${salary}")
        print(f"Savings: ${savings}")
        print(f"Total Expenses: ${total_expenses}")
        print(f"Remaining Salary: ${remaining_salary}\n")

    def save_to_csv(self):
        # Save the updated dataset back to the CSV file
        self.data.to_csv(self.file_path, index=False)
        print(f"Data saved to {self.file_path}")

    def show_overall_summary(self):
        # Display the overall dataset
        print("\n--- Overall Data ---")
        print(self.data)

# Example usage
file_path = 'salary_data.csv'  # Path to the external dataset
salary_manager = SalaryManagementAI(file_path)

while True:
    print("\n1. Add an expense")
    print("2. Calculate savings for a month")
    print("3. Show overall summary")
    print("4. Save data to CSV")
    print("5. Exit")
    choice = int(input("\nChoose an option: "))

    if choice == 1:
        month = input("Enter the month (e.g., January): ")
        expense_name = input("Enter expense name: ")
        amount = float(input("Enter expense amount: "))
        salary_manager.add_expense(month, expense_name, amount)

    elif choice == 2:
        month = input("Enter the month to calculate savings (e.g., January): ")
        salary_manager.calculate_savings(month)

    elif choice == 3:
        salary_manager.show_overall_summary()

    elif choice == 4:
        salary_manager.save_to_csv()

    elif choice == 5:
        print("Exiting...")
        break

    else:
        print("Invalid option. Please try again.")
