import json

# Completing the notebook content with final corrections



# Mastering Feature Engineering with the Titanic Dataset: Theory, Coding, and Visualisations
# Feature engineering is a crucial step in the machine learning process, involving the creation, transformation, and selection of features (variables) to enhance model performance. In this blog, we'll use the famous Titanic dataset to explore feature engineering in detail, covering theoretical concepts, practical coding, and visualisation techniques.
# Table of Contents
# Introduction to Feature Engineering
# Data Preprocessing
# Handling Missing Values
# Encoding Categorical Variables
# Creating New Features
# Feature Scaling
# Feature Selection
# Data Visualization
# Putting It All Together
# Conclusion

# 1. Introduction to Feature Engineering
# Feature engineering is the process of transforming raw data into meaningful features that improve the performance of machine learning models. It is one of the most critical steps in the data science pipeline and often determines the success of a model. The process of feature engineering involves several key steps:
# Step 1: Data Preprocessing
# Purpose: Prepare the raw data for analysis by cleaning, formatting, and structuring it.
# Tasks:
# Removing or imputing missing values.
# Dropping irrelevant or redundant columns.
# Formatting and normalising data for consistency.

# Step 2: Handling Missing Values
# Purpose: Address missing or incomplete data to maintain the integrity of the dataset.
# Strategies:
# Dropping Rows/Columns: If missing values are excessive or the data is not essential.
# Imputation: Filling missing values with statistical measures (mean, median, mode) or using more sophisticated techniques.

# Step 3: Encoding Categorical Variables
# Purpose: Convert categorical variables into a numerical format that machine learning models can understand.
# Techniques:
# Label Encoding: Assigning each category a unique integer.
# One-Hot Encoding: Creating binary columns for each category to avoid ordinal relationships.

# Step 4: Creating New Features
# Purpose: Enhance the dataset by creating features that capture more information and improve model accuracy.
# Examples:
# Combining existing features (e.g., family size = siblings + parents).
# Extracting new features from date and time, such as day of the week or hour of the day.
# Creating interaction terms to capture relationships between features.

# Step 5: Feature Scaling
# Purpose: Standardise or normalise features to ensure they are on a similar scale, which is crucial for algorithms sensitive to feature scaling.
# Methods:
# Standardisation: Centering the data by subtracting the mean and dividing by the standard deviation.
# Normalisation: Scaling the data to a range of [0, 1] or [-1, 1].

# Step 6: Feature Selection
# Purpose: Select the most relevant features to reduce dimensionality, improve model performance, and prevent overfitting.
# Techniques:
# Correlation Analysis: Identifying relationships between features and the target variable.
# Feature Importance: Using models like Random Forest or Lasso Regression to rank feature importance.
# Principal Component Analysis (PCA): Reducing dimensionality while retaining variance.

# Step 7: Data Visualisation
# Purpose: Visualise data and features to gain insights and uncover hidden patterns.
# Tools:
# Histograms and Bar Plots: To visualise distributions and counts.
# Box Plots: To identify outliers and understand data spread.
# Heatmaps: To visualise correlations between features.

# Step 8: Putting It All Together
# Purpose: Combine all the steps to create a clean, enriched, and well-structured dataset ready for model training and evaluation.
# Tasks:
# Implementing feature engineering techniques iteratively.
# Saving the final processed data for model training.

# By mastering these steps, you can significantly enhance your data's quality and make your machine learning models more effective. In the following sections, we will apply each of these steps to the Titanic dataset with detailed coding examples and visualisations.
# 2. Data Preprocessing
# Before diving into feature engineering, let's start with basic data preprocessing.
# # Importing Libraries
# import pandas as pd

# # Loading the Titanic dataset
# df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')

# # Display first few rows to understand the dataset
# print(df.head())
# Output:
# Key Observations:
# The dataset contains information about passengers, such as their age, gender, ticket class, and whether they survived.
# Columns like PassengerId, Name, Ticket, and Cabin are not useful for prediction and will be dropped.

# 3. Handling Missing Values
# Handling missing values is essential for data integrity. In the Titanic dataset, certain columns have missing values, as seen in the info() method output:
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load the Titanic dataset
# df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')

# # Check for missing values
# missing_values = df.isnull().sum()

# # Plot missing values
# plt.figure(figsize=(10, 6))
# barplot = sns.barplot(x=missing_values.index, y=missing_values.values, palette='viridis')
# plt.title('Missing Values in Each Column')
# plt.xlabel('Columns')
# plt.ylabel('Number of Missing Values')
# plt.xticks(rotation=20)

# # Add text annotations below each bar
# for index, value in enumerate(missing_values.values):
#     barplot.text(index, value + 5, str(value), ha='center', va='bottom', color='black', fontsize=10)

# plt.show()
# Output:
# PassengerId 0
# Survived 0
# Pclass 0
# Name 0
# Sex 0
# Age 177
# SibSp 0
# Parch 0
# Ticket 0
# Fare 0
# Cabin 687
# Embarked 2
# Observation:
# The Age column has 177 missing values, which can affect model predictions.
# The Embarked column has 2 missing values, which need to be filled for proper model training.
# The Cabin column has 687 missing values, so it is dropped due to the high proportion of missing data.

# Solution: Filling Missing Values
# 1. Filling 'Age' with the Mean Age:
# # Fill missing values in 'Age' with the mean age
# df['Age'] = df['Age'].fillna(df['Age'].mean())
# Reason: Age is a continuous variable, and using the mean value is a reasonable approach to handle missing data.
# 2. Filling 'Embarked' with the Most Frequent Value:
# # Fill missing values in 'Embarked' with the most common port
# df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
# Reason: 'Embarked' is a categorical variable, and filling missing values with the most frequent category ensures consistency.
# 3. Dropping Irrelevant Columns:
# # Drop irrelevant columns
# df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)
# These columns do not contribute to predicting survival and are dropped to reduce dimensionality.
# 4. Encoding Categorical Variables
# Machine learning models require numerical inputs, so categorical variables need to be converted.
# Theory:
# One-Hot Encoding: Converts categories into binary columns (0 or 1). Useful for variables with no ordinal relationship.
# Label Encoding: Converts categories into integers. Useful for ordinal categories.

# Coding Example:
# # Encoding 'Sex' and 'Embarked'
# df['Sex'] = df['Sex'].map({'female': 0, 'male': 1})
# df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
# Explanation:
# Sex: Female is encoded as 0, and Male as 1.
# Embarked: Ports S, C, and Q are encoded as 0, 1, and 2 respectively.

# 5. Creating New Features
# Creating features that add more information can significantly improve model performance.
# Feature 1: Family Size
# # Create a new feature 'FamilySize'
# df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
# Explanation: Combines the number of siblings/spouses (SibSp) and parents/children (Parch) with the passenger itself, providing insight into family size.
# Feature 2: Is Alone
# # Create a new feature 'IsAlone'
# df['IsAlone'] = 1  # Initialize to 1 (Alone)
# df.loc[df['FamilySize'] > 1, 'IsAlone'] = 0  # Set to 0 if not alone
# Explanation: Determines if a passenger is traveling alone or with family.
# 6. Feature Scaling
# Scaling features ensures that they are on the same scale, which is crucial for algorithms like KNN and SVM.
# Theory:
# Standardisation: Subtract the mean and divide by the standard deviation.
# Normalisation: Scale features to a range of [0, 1].

# Coding Example:
# from sklearn.preprocessing import StandardScaler

# # Standardize the 'Age' and 'Fare' columns
# scaler = StandardScaler()
# df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])
# Explanation:
# The Age and Fare columns are standardised to have a mean of 0 and a standard deviation of 1.

# 7. Feature Selection
# Selecting the most relevant features improves model performance and reduces complexity.
# Theory:
# Correlation Matrix: Shows the correlation between features and the target variable.
# Feature Importance: Use models like Random Forest to get feature importance scores.

# # Correlation Matrix
# corr_matrix = df.corr()
# print('Correlation Matrix:')
# print(corr_matrix['Survived'].sort_values(ascending=False))

# print('\n')

# # Feature Selection using Random Forest
# from sklearn.ensemble import RandomForestClassifier

# # Define the model
# model = RandomForestClassifier()
# X = df.drop('Survived', axis=1)
# y = df['Survived']
# model.fit(X, y)

# # Get feature importances
# feature_importances = pd.Series(model.feature_importances_, index=X.columns)
# print('Feature Importances:')
# print(feature_importances.sort_values(ascending=False))
# Output:
# Observations:
# Based on the feature importances from the Random Forest model, you should consider selecting the features with the highest importance scores for your predictive model. In this case, the top features are:
# Sex (0.263628)
# Age (0.261210)
# Fare (0.257230)
# Pclass (0.081137)

# These four features contribute significantly to the model's predictive power and should be included in your final model.
# You may also consider including the following features based on their importance scores, although they have less impact compared to the top four:
# FamilySize (0.049350)
# Embarked (0.030583)

# These features still provide some value but are not as critical. The remaining features have relatively low importance and might not contribute much to the model's performance. However, if you aim for a balance between model simplicity and performance, you can exclude:
# SibSp (0.028613)
# Parch (0.021102)
# IsAlone (0.007148)

# In summary, you should primarily select the top four features (Sex, Age, Fare, Pclass) and optionally include FamilySize and Embarked based on your needs for model complexity and interpretability.
# Explanation:
# The correlation matrix and feature importance scores help identify which features are most relevant for predicting survival.

# 8. Data Visualisation
# Visualising data can reveal patterns that are not obvious in raw data.
# Example 1: Distribution of Age by Survival
# import seaborn as sns
# import matplotlib.pyplot as plt

# # Visualize age distribution by survival status
# sns.histplot(data=df, x='Age', hue='Survived', multiple='stack')
# plt.title('Age Distribution by Survival Status')
# plt.show()

# Output:
# Observation:
# The histogram visualises the age distribution of passengers on the Titanic, separated by their survival status (0 = did not survive, 1 = survived). Here are some observations based on the graph:
# 1. Age Distribution: The majority of passengers fall within the 20–40 age range, with a particularly high concentration around age 30.
# 2. Survival Rate by Age:
# For children (ages 0–10), the survival rate appears relatively higher compared to older age groups.
# For young adults (ages 20–30), there are more non-survivors, indicating a lower survival rate in this age group.
# Middle-aged passengers (ages 30–50) show a more balanced distribution between survivors and non-survivors.
# Elderly passengers (ages 60 and above) are fewer, but they show a higher proportion of non-survivors.

# 3. Peak at Age 30: There is a noticeable spike at age 30, indicating a large number of passengers in this age group. However, the majority of them did not survive.
# 4. Overall Survival Trend: Generally, there are more non-survivors across most age groups, but specific age ranges (such as children and older adults) show different survival dynamics.
# These observations provide insights into how age influenced the likelihood of survival on the Titanic.
# Example 2: Family Size vs. Survival Rate
# # Visualize family size vs. survival rate
# sns.barplot(data=df, x='FamilySize', y='Survived', ci=None)
# plt.title('Family Size vs. Survival Rate')
# plt.show()
# Output:
# Observation:
# The bar plot shows the relationship between family size and the survival rate on the Titanic. Here are some observations:
# Single Passengers (Family Size = 1): The survival rate is relatively low, indicating that passengers traveling alone had a lower chance of survival.
# Small Families (Family Size = 2–4): Passengers with small family sizes (2–4 members) had the highest survival rates. This suggests that traveling with a small group might have improved their chances of receiving help or making it to lifeboats.
# Large Families (Family Size = 5–7): For larger families, the survival rate drops significantly. This could be due to difficulties in managing large groups during the evacuation or the availability of fewer lifeboat spaces for large families.

# These insights suggest that traveling in a small group may have provided a survival advantage on the Titanic.
# Explanation:
# Age Distribution: Shows how age impacts survival.
# Family Size: Visualises the effect of family size on the likelihood of survival.

# 9. Putting It All Together
# After all the transformations and feature engineering steps, the final processed dataset is saved for model training:
# # Save the processed data
# df.to_csv('output/titanic_processed.csv', index=False)
# 10. Conclusion
# In this blog, we've covered the complete feature engineering process using the Titanic dataset. We explored:
# Data preprocessing and handling missing values.
# Encoding categorical variables.
# Creating new features like Family Size and IsAlone.
# Scaling and selecting features for better model performance.
# Visualising data to uncover patterns.

# Feature engineering is an iterative process that combines domain knowledge with technical skills. By mastering these techniques, you can significantly improve your machine learning models.
# Final Thoughts
# If you have any questions or feedback, feel free to leave a comment! If you're interested in learning Data Science, Flutter, Node.js, Web Development, or App Development, consider joining WhiteCodel for an internship. Visit WhiteCodel Career to explore opportunities. Happy coding!



notebook_content_final_completed = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Mastering Feature Engineering with the Titanic Dataset\n",
    "In this notebook, we will go through the complete feature engineering process using the Titanic dataset, including handling missing values, encoding, creating new features, feature scaling, feature selection, and data visualization."
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 1. Importing Necessary Libraries\n",
    "We start by importing the necessary libraries for data manipulation, visualization, and machine learning."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import necessary libraries\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.ensemble import RandomForestClassifier\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 2. Loading the Titanic Dataset\n",
    "Let's load the Titanic dataset directly from the provided URL. We'll also display the first few rows to get an understanding of the dataset structure."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load Titanic dataset\n",
    "df = pd.read_csv('https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv')\n",
    "\n",
    "# Display the first few rows of the dataset\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 3. Data Preprocessing\n",
    "We will drop irrelevant columns that do not contribute to predicting the target variable (Survived)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Drop irrelevant columns\n",
    "df = df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1)\n",
    "\n",
    "# Display the dataset after dropping columns\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 4. Handling Missing Values\n",
    "Handling missing values is crucial for maintaining data integrity. We'll fill missing values for the 'Age' and 'Embarked' columns."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Fill missing values in 'Age' with the mean age\n",
    "df['Age'] = df['Age'].fillna(df['Age'].mean())\n",
    "\n",
    "# Fill missing values in 'Embarked' with the most common port\n",
    "df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])\n",
    "\n",
    "# Check for any remaining missing values\n",
    "df.isnull().sum()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 5. Encoding Categorical Variables\n",
    "We need to convert categorical variables into a numerical format that machine learning models can understand."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Encoding 'Sex' and 'Embarked' columns\n",
    "df['Sex'] = df['Sex'].map({'female': 0, 'male': 1})\n",
    "df['Embarked'] = df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})\n",
    "\n",
    "# Display the dataset after encoding\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 6. Creating New Features\n",
    "Creating new features can help improve model performance. We'll create two new features: `FamilySize` and `IsAlone`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Create a new feature 'FamilySize'\n",
    "df['FamilySize'] = df['SibSp'] + df['Parch'] + 1\n",
    "\n",
    "# Create a new feature 'IsAlone'\n",
    "df['IsAlone'] = 1  # Initialize to 1 (Alone)\n",
    "df.loc[df['FamilySize'] > 1, 'IsAlone'] = 0  # Set to 0 if not alone\n",
    "\n",
    "# Display the dataset with new features\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 7. Feature Scaling\n",
    "Feature scaling ensures that numerical features are on the same scale. We will standardize the `Age` and `Fare` columns."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Standardize the 'Age' and 'Fare' columns\n",
    "scaler = StandardScaler()\n",
    "df[['Age', 'Fare']] = scaler.fit_transform(df[['Age', 'Fare']])\n",
    "\n",
    "# Display the dataset after scaling\n",
    "df.head()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 8. Feature Selection\n",
    "We will use a Random Forest model to identify the most important features for predicting survival."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Define the features and target variable\n",
    "X = df.drop('Survived', axis=1)\n",
    "y = df['Survived']\n",
    "\n",
    "# Train a Random Forest model\n",
    "model = RandomForestClassifier()\n",
    "model.fit(X, y)\n",
    "\n",
    "# Get feature importances\n",
    "feature_importances = pd.Series(model.feature_importances_, index=X.columns)\n",
    "feature_importances.sort_values(ascending=False)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 9. Data Visualization\n",
    "Visualizing data helps in understanding patterns and relationships. We will create a few plots to visualize the dataset."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Correlation Matrix\n",
    "plt.figure(figsize=(10, 8))\n",
    "sns.heatmap(df.corr(), annot=True, cmap='coolwarm')\n",
    "plt.title('Correlation Matrix')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Age Distribution by Survival Status\n",
    "plt.figure(figsize=(10, 6))\n",
    "sns.histplot(data=df, x='Age', hue='Survived', multiple='stack')\n",
    "plt.title('Age Distribution by Survival Status')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Family Size vs. Survival Rate\n",
    "plt.figure(figsize=(10, 6))\n",
    "sns.barplot(data=df, x='FamilySize', y='Survived', ci=None)\n",
    "plt.title('Family Size vs. Survival Rate')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 10. Saving the Processed Dataset\n",
    "We will save the processed dataset for future use in model training and evaluation."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Save the processed dataset\n",
    "df.to_csv('titanic_processed.csv', index=False)"
    ]
}
 ],
   "metadata": {
    "kernelspec": {
      "display_name": "Python 3",
      "language": "python",
      "name": "python3"
    },
    "language_info": {
      "name": "python",
      "version": "3.7.0"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 4

}

# Convert the notebook content into a ipynb file
with open('feature_engineering.ipynb', 'w') as json_file:
    json.dump(notebook_content_final_completed, json_file)

