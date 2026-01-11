import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load CSV files (make sure they exist in the same folder)
df1 = pd.read_csv('df1.csv', index_col=0)
df2 = pd.read_csv('df2.csv')

# Inspect the data
print(df1.head())
print(df1.info())

# Plot as bar chart (only numeric columns will be plotted)
df2.plot.bar()
df1.plot.scatter(x ='A', y ='B')
df1.plot(style=['-', '--', '-.', ':'], title='Line Plot with Different Styles', xlabel='Index', ylabel='Values', grid=True)
# Show the plot
plt.show()
