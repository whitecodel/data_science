# Example: Using the Titanic dataset
import pandas as pd
df = pd.read_csv('data/titanic.csv') # You can also use url to read data directly from the web (e.g. 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')

# Display first few rows
df.head()

# Check shape and column information
# print(df.shape)
# print(df.columns)
# print(df.info())
# print(df.describe())

# Drop columns that are not useful
df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1) # axis=0 for rows, axis=1 for columns

# Check for missing values
# print(df.isnull().sum()) # Count missing values in each column

# Fill missing values in 'Age' with the mean age
df['Age'] = df['Age'].fillna(df['Age'].mean()) # Fill missing values with the mean of the column

# Fill missing values in 'Embarked' with the most common port
df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0]) # Fill first value from most frequent values series

# Check for missing values again
# print(df.isnull().sum())

# Convert categorical variables to numerical 
# Method 1: Using pandas get_dummies
# df = pd.get_dummies(df, columns=['Sex', 'Embarked'], drop_first=True) # drop_first=True to avoid multicollinearity

# Method 2: Using map function [Recommended for binary variables]
df['Sex'] = df['Sex'].map({'female': 0, 'male': 1})
df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})

# Method 3: Using LabelEncoder
# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# df['Sex'] = le.fit_transform(df['Sex'])
# df['Embarked'] = le.fit_transform(df['Embarked'])

# Display first few rows after encoding
# print(df.head())

# Split the data into features and target
