import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the UFO sightings dataset (replace with your actual file path)
df = pd.read_csv("https://raw.githubusercontent.com/justmarkham/pandas-videos/master/data/ufo.csv")

# Convert the 'Time' column to datetime format
df['Time'] = pd.to_datetime(df['Time'], errors='coerce')

# Extract the hour of the day from the 'Time' column
df['Hour'] = df['Time'].dt.hour

# Create a histogram to visualize UFO sightings by hour
plt.figure(figsize=(10, 6))
plt.hist(df['Hour'].dropna(), bins=24, range=(0, 24), edgecolor='black', alpha=0.7)
plt.xlabel('Hour of the Day')
plt.ylabel('Number of UFO Sightings')
plt.title('UFO Sightings by Hour of the Day')
plt.xticks(range(0, 25))
plt.grid(True)
plt.show()