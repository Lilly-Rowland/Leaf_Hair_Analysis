import pandas as pd
import matplotlib.pyplot as plt

# Load the Excel file into a DataFrame
file_path = 'results/ablation_results.xlsx'  # Replace with your actual file path
df = pd.read_excel(file_path)

# Select numeric columns for calculating averages
numeric_columns = ['Average Test Loss', 'Average IOU', 'Average Weighted IOU',
                   'Average Dice Coefficient', 'Precision', 'Recall', 'F1', 'Elapsed Time (seconds)']

# Convert relevant columns to numeric
df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric, errors='coerce')

# Group by Architecture and calculate averages
archictecture_averages = df.groupby('Architecture')[numeric_columns].mean().reset_index()
archictecture_averages.to_excel("results/architecture_averages.xlsx", index=False)

# Group by loss and calculate averages
loss_averages = df.groupby('Loss')[numeric_columns].mean().reset_index()
loss_averages.to_excel("results/loss_averages.xlsx", index=False)

# Group by balance and calculate averages
balance_averages = df.groupby('Balance')[numeric_columns].mean().reset_index()

# Group by batch size and calculate averages
balance_averages = df.groupby('Batch Size')[numeric_columns].mean().reset_index()


# Plotting the bar graph for Average Test Loss
plt.figure(figsize=(10, 5))
plt.bar(archictecture_averages['Architecture'], archictecture_averages['Average Test Loss'], color='skyblue')
plt.xlabel('Architecture')
plt.ylabel('Average Test Loss')
plt.title('Average Test Loss by Architecture')
plt.show()

# Plotting the bar graph for Elapsed Time
plt.figure(figsize=(10, 5))
plt.bar(df['Architecture'], df['Elapsed Time (seconds)'], color='lightgreen')
plt.xlabel('Architecture')
plt.ylabel('Elapsed Time (seconds)')
plt.title('Elapsed Time by Architecture')
plt.show()