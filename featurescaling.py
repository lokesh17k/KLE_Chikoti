import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
#import matplotlib inline
matplotlib.style.use('fivethirtyeight')
x = pd.DataFrame({
    'x1': np.concatenate([np.random.normal(20, 2, 1000), np.random.normal(1, 2, 25)]),
    'x2': np.concatenate([np.random.normal(30, 2, 1000), np.random.normal(50, 2, 25)]),
})
np.random.normal


scaler = preprocessing.RobustScaler()
robust_df = scaler.fit_transform(x)
robust_df = pd.DataFrame(robust_df, columns=['x1', 'x2'])


scaler = preprocessing.StandardScaler()
standard_df = scaler.fit_transform(x)
standard_df = pd.DataFrame(standard_df, columns=['x1', 'x2'])


scaler = preprocessing.MinMaxScaler()
minmax_df = scaler.fit_transform(x)
minmax_df = pd.DataFrame(minmax_df, columns=['x1', 'x2'])


fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4, figsize=(20, 5))
ax1.set_title('Before Scaling')
sns.kdeplot(x['x1'], ax=ax1, color='r')
sns.kdeplot(x['x2'], ax=ax1, color='b')
ax2.set_title('After Robust Scaling')
sns.kdeplot(robust_df['x1'], ax=ax2, color='red')
sns.kdeplot(robust_df['x2'], ax=ax2, color='blue')
ax3.set_title('After Standard Scaling')
sns.kdeplot(standard_df['x1'], ax=ax3, color='black')
sns.kdeplot(standard_df['x2'], ax=ax3, color='g')
ax4.set_title('After Min-Max Scaling')
sns.kdeplot(minmax_df['x1'], ax=ax4, color='black')
sns.kdeplot(minmax_df['x2'], ax=ax4, color='g')
plt.show()