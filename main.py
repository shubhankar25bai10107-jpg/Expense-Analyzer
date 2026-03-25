import pandas as pd
import matplotlib.pyplot as plt
from preprocess import clean_text
from model import train_model

# Load dataset
data = pd.read_csv("data/expenses.csv")

# Preprocess
data['description'] = data['description'].apply(clean_text)

# Train model
model, vectorizer = train_model(data['description'], data['category'])

# User input
desc = input("Enter expense description: ")
amount = float(input("Enter amount: "))

cleaned = clean_text(desc)
vector = vectorizer.transform([cleaned])

predicted_category = model.predict(vector)[0]

print(f"\nCategory: {predicted_category}")

# Add to dataset
new_entry = pd.DataFrame([[cleaned, amount, predicted_category]],
                         columns=["description", "amount", "category"])

data = pd.concat([data, new_entry], ignore_index=True)

# Analysis
summary = data.groupby("category")["amount"].sum()

print("\nExpense Summary:")
print(summary)

# Plot
summary.plot(kind="bar")
plt.title("Expense Distribution")
plt.xlabel("Category")
plt.ylabel("Amount")
plt.show()
