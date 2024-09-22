#!/usr/bin/env python
# coding: utf-8

# IMPORTING LIBRARIES

# In[2]:


import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm


# LOADING DATASET

# In[3]:


df = pd.read_excel("student_data.xlsx")


# In[4]:


df.head()


# DATA CLEANING AND PRE-PROCESSING

# In[5]:


duplicate_count = df.duplicated().sum()
print(duplicate_count)


# In[6]:


df.columns = df.columns.str.replace(' ', '_')


# In[7]:


null_counts = df.isnull().sum()
print(null_counts)


# In[8]:


df_cleaned = df.dropna()


# In[9]:


print(f"Number of rows after removing null values: {df_cleaned.shape[0]}")


# In[10]:


df.fillna(method='ffill', inplace=True)


# In[11]:


df['First_Name'] = df['First_Name'].str.capitalize()


# In[12]:


df['Year_of_Graduation'] = pd.to_datetime(df['Year_of_Graduation'], format='%Y')


# In[13]:


df['CGPA'] = df['CGPA'].astype(float)


# In[14]:


#df['Leadership-_skills'] = df['Leadership-_skills'].map({'Yes': 1, 'No': 0})


# In[15]:


# EDA
print(df.describe())


# In[16]:


# 1. How many unique students are included in the dataset?
# Number of unique student records based on Email_ID and First_Name  
unique_student_records = df[['Email_ID', 'First_Name']].drop_duplicates().shape[0]  
print("Number of unique student records:", unique_student_records)


# In[17]:


# 2. What is the average CGPA of the students?
average_gpa = df['CGPA'].mean()
print(f"Average CGPA: {average_gpa:.2f}")


# In[18]:


# 3. What is the distribution of students across different graduation years?
# Converting the 'Year of graduation' to just the year as it's in datetime format
# Checking the data type of 'Year_of_Graduation'
print(df['Year_of_Graduation'].dtype)

# If it's not an integer or string, converting it to integer
if df['Year_of_Graduation'].dtype == 'object':
    df['Year_of_Graduation'] = df['Year_of_Graduation'].astype(int)

# Calculating the distribution of students across different graduation years
graduation_year_distribution = df['Year_of_Graduation'].value_counts().sort_index()

# Plotting the distribution
graduation_year_distribution.plot(kind='bar', title='Graduation Year Distribution')
plt.xlabel('Graduation Year')
plt.ylabel('Number of Students')
plt.show()


# Obs: Overall, the chart illustrates a trend of decreasing graduation numbers as the years progress, particularly notable in 2025 and 2026

# In[19]:


# 4. What is the distribution of students’ experience with Python programming?
plt.figure(figsize=(5, 5))

python_exp_distribution = df['Experience_with_python_(Months)'].describe()
sns.histplot(df['Experience_with_python_(Months)'], kde=True, bins=10)
plt.title('Distribution of Python Programming Experience')
plt.xlabel('Python Experience Level(Months)')
plt.ylabel('Number of Students')
plt.tight_layout()

plt.show()


# Obs: A blue line overlays the bars, illustrating the trend in student numbers across the experience levels. The line fluctuates, indicating peaks and troughs in student distribution.The highest number of students appears at experience level 5, followed by levels 4 and 6.
# Levels 3 and 8 have lower student counts compared to the others.

# In[20]:


# 5. What is the average family income of the student
average_family_income = df['Family_Income2'].mean()

print(f"The average family income of the students is: {average_family_income:.2f} lakhs")


# In[21]:


# 6. How does the GPA vary among different colleges? (Show top 5 results only)
avg_cgpa_by_college = df.groupby('College_Name')['CGPA'].mean().sort_values(ascending=False).head(5)
print("Top 5 Colleges by Average CGPA:\n", avg_cgpa_by_college)

# Plot a bar chart to visualize the variation in CGPA among the top 5 colleges

avg_cgpa_by_college.plot(kind='bar', title='Top 5 Colleges by Average GPA')
plt.xlabel('College Name')
plt.ylabel('Average GPA')
plt.xticks(rotation=45, ha='right')  # Rotate college names for better readability
plt.show()


# Obs: Overall, this boxplot suggests that the majority of individuals completed exactly one course, with no significant variation or outliers in the data.

# In[22]:


# 8. What is the average GPA for students from each city?

# Group data by the "City" column
grouped_data = df.groupby('City')

# Replace 'Column_Name' with the name of the column you want to find the average of
column_name = 'CGPA'

# Calculate the average for each city
for city, city_group in grouped_data:
    city_average = city_group[column_name].mean()
    print(f'Average {column_name} in {city}: {city_average}')


# In[25]:


# 9. Can we identitfy any relationship between family income and CGPA?
# Aggregate average GPA by Family Income
avg_gpa_by_income = df.groupby('Family_Income2')['CGPA'].mean().reset_index()

# Plot the results
plt.figure(figsize=(5, 4))
sns.barplot(x='Family_Income2', y='CGPA', data=avg_gpa_by_income)
plt.xlabel('Family Income (in Lakhs)')
plt.ylabel('Average GPA')
plt.title('Average GPA by Family Income Category')
plt.show()


# Obs: The bar chart shows that the average GPA remains consistent across various family income categories, all hovering around 8. This suggests that family income does not significantly impact academic performance in this dataset

# In[38]:


# 10. How many students from various cities? (Solve using data visualization tool)
students_by_city = df['City'].value_counts().head(20)
students_by_city.plot(kind='bar', title='Number of Students by City')
plt.show()


# Obs: The bar chart shows a relatively even distribution of students across various cities, with most cities having around 50 to 60 students. This indicates a balanced representation of students from different locations.

# In[22]:


# 11. How does the expected salary vary based on factors like ‘GPA’, ‘Family income’, ‘Experience with Python (Months)’?


# Scatter plot between CGPA and Salary Expectations
plt.scatter(df['CGPA'], df['Expected_salary_(Lac)'])
plt.xlabel('CGPA')
plt.ylabel('Expected salary (Lac)')
plt.title('Relationship between CGPA and Expected salary (Lac)')
plt.show()

# Scatter plot between Family Income and Salary Expectations
plt.scatter(df['Family_Income2'], df['Expected_salary_(Lac)'])
plt.xlabel('Family Income')
plt.ylabel('Expected salary (Lac)')
plt.title('Relationship between Family Income and Expected salary (Lac)')
plt.show()

# Scatter plot between Experience with python (Months) and Salary Expectations
plt.scatter(df['Experience_with_python_(Months)'], df['Expected_salary_(Lac)'])
plt.xlabel('Experience with python (Months)')
plt.ylabel('Expected salary (Lac)')
plt.title('Relationship between Experience with python (Months) and Expected salary (Lac)')
plt.show()


# Obs: 1. The scatter plot indicates that expected salary remains relatively stable across different months of experience with Python, primarily clustering around 30 to 35 lakhs. This suggests that additional experience does not significantly increase expected salary in this dataset.
# 
# 2. The scatter plot reveals that expected salary varies significantly across different family income levels, with clusters primarily around 10 to 35 lakhs.
# 
# 3. The scatter plot shows that expected salary remains consistent around 30 to 35 lakhs, regardless of months of experience with Python.

# In[23]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer


# Defining the features (independent variables) and target (dependent variable)
X = df[['CGPA', 'Family_Income2', 'Experience_with_python_(Months)']]
y = df['Expected_salary_(Lac)']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the SimpleImputer
imputer = SimpleImputer(strategy='mean')

# Fit the imputer on the training data and transform both training and test data
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# Create a linear regression model
model = LinearRegression()

# Fit the model to the training data
model.fit(X_train, y_train)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')


# In[24]:


# 12. Which event tends to attract more students from specific fields of study?

# Assuming df is already defined and contains the 'Events' column

column_name = 'Events'  # Replace with your column name

# Find the most common values in the specified column
most_common_values = df[column_name].value_counts()

N = 10

# Convert to DataFrame for better readability
top_n_values = most_common_values.head(N).reset_index()
top_n_values.columns = [column_name, 'Count']

# Print the DataFrame
print(f"Top {N} most common values in the '{column_name}' column:")
print(top_n_values)

# Optional: Display the results using matplotlib if needed
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.bar(top_n_values[column_name], top_n_values['Count'], color='skyblue')
plt.xlabel(column_name)
plt.ylabel('Count')
plt.title(f'Top {N} Most Common Values in {column_name}')
plt.xticks(rotation=45, ha='right')
plt.show()



# Obs: The bar chart highlights that "Product Design & Full Stack" and "Internship Program(S) Success Conclave" are the most attended events, each exceeding 800 participants.
# 

# In[25]:


# 13. Do students in leadership positions during their college years tend to have higher GPAs or better expected salary?
leadership_effect = df.groupby('Leadership-_skills').agg({'CGPA': 'mean', 'Expected_salary_(Lac)': 'mean'})
print("Leadership Effect on GPA and Expected Salary:\n", leadership_effect)


# In[26]:


# 14. How many students are graduating by the end of 2024?
graduation_dates = df['Year_of_Graduation']

graduation_dates = pd.to_datetime(graduation_dates, errors='coerce')

# Define the end date for 2024
end_of_2024 = pd.to_datetime('2024')

# Filter the data for students graduating by the end of 2024
graduating_by_2024 = df[graduation_dates <= end_of_2024]

# Count the number of students meeting the criteria
num_students_graduating_by_2024 = len(graduating_by_2024)

# Print the result
print(f"Number of students graduating by the end of 2024: {num_students_graduating_by_2024}")


# In[272]:


# 15. Which promotion channel brings in more student participation for the event?
# Assuming you have a 'Promotion Channel' and 'Event Participation' columns
promotion_effectiveness = df.groupby('Events')['Quantity'].sum().sort_values(ascending=False)
promotion_effectiveness.plot(kind='bar', title='Effectiveness of Promotion Channels')
plt.show()
# Get the Events with the highest participation
most_participated_channel = channel_participation.iloc[0]['Events']
print(f"The Events that brings in the most student participation is: {most_participated_channel}")


# Obs: The bar chart indicates that "Product Design & Full Stack" and "Internship Program(S) Success Conclave" are the most effective promotion channels, each attracting over 800 participants.
# 

# In[27]:


# 16. Find the total number of students who attended events related to Data Science? (From all Data Science related courses.)

# Sample DataFrame setup (replace with actual DataFrame)
# df = pd.read_csv('your_dataset.csv')  # If reading from a CSV or similar source

# Define a list of keywords related to Data Science events
data_science_keywords = ['Data Science', 'Machine Learning', 'Artificial Intelligence','Data Visualization using Power BI', 'Deep Learning', 'Analytics']

# Filter the DataFrame for events related to Data Science
data_science_events_df = df[df['Events'].str.contains('|'.join(data_science_keywords), case=False, na=False)]

# Sum up the number of students attending these Data Science events
total_attendance = data_science_events_df['Email_ID'].nunique()

# Print the result
print(f"Total number of students attending Data Science events: {total_attendance}")


# In[28]:


# 17. Do students with high CGPA and more experience in languages have higher salary expectations? (Avg)
from sklearn.impute import SimpleImputer

# Define the features (independent variables) and target (dependent variable)
X = df[['CGPA', 'Family_Income2', 'Experience_with_python_(Months)']]
y = df['Expected_salary_(Lac)']

# Handle missing values in X using imputation
imputer = SimpleImputer(strategy='mean')  # You can also try 'median' or 'most_frequent'
X = imputer.fit_transform(X)

# Fit the linear regression model again
model = LinearRegression()
model.fit(X, y)

# Predict the expected salary using the model
df['Expected_salary_(Lac)'] = model.predict(X)

# Filter students with high CGPA and high experience
high_gpa_exp = df[(df['CGPA'] > df['CGPA'].mean()) & (df['Experience_with_python_(Months)'] > df['Experience_with_python_(Months)'].mean())]

# Calculate the average salary expectation for these students
average_salary_expectation = high_gpa_exp['Expected_salary_(Lac)'].mean()

print(f"Average Salary Expectation for High GPA and Experience: {average_salary_expectation:.2f}")


# In[ ]:




