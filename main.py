# Imports libraries to use to visualize data
# Pandas builds on matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Reads the dataset
df = pd.read_csv('data.csv')

# Cleans the dataset
df = df.dropna(subset=['ANNUAL SALARY'])
df.rename(columns=lambda x: x.strip().lower().replace(' ', '_'), inplace=True)
df.drop_duplicates(inplace=True)

# Adds hiring year from hire date
if 'current_hire_date' in df.columns:
    df['current_hire_date'] = pd.to_datetime(df['current_hire_date'], errors='coerce')
    df['hiring_year'] = df['current_hire_date'].dt.year

# Distribution of Salaries
# Creates Graph
plt.figure(figsize=(10, 6))
sns.histplot(df['annual_salary'], bins=20, kde=True, color='skyblue')
plt.title('Distribution of Employee Salaries')
plt.xlabel('Annual Salary')
plt.ylabel('Frequency')
plt.savefig('salary_distribution.png')
# Shows Graph
plt.show()

# Scheduled Hours vs Salary
if 'scheduled_hours' in df.columns and 'annual_salary' in df.columns:
    # Creates Graph
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='scheduled_hours', y='annual_salary', data=df, color='green')
    plt.title('Correlation Between Scheduled Hours and Salary')
    plt.xlabel('Scheduled Hours')
    plt.ylabel('Annual Salary')
    plt.savefig('scheduled_hours_vs_salary.png')
    # Shows Graph
    plt.show()

# Years of Service vs Salary
if 'years_of_service' in df.columns and 'annual_salary' in df.columns:
    # Creates Graph
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='years_of_service', y='annual_salary', data=df, color='orange')
    plt.title('Does Years of Service Play a Role in Salary?')
    plt.xlabel('Years of Service')
    plt.ylabel('Annual Salary')
    plt.savefig('years_of_service_vs_salary.png')
    # Shows Graph
    plt.show()

# Pay Location Code vs Salary
if 'pay_location_code' in df.columns and 'annual_salary' in df.columns:
    # Creates Graph
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='pay_location_code', y='annual_salary', data=df)
    plt.xticks(rotation=45)
    plt.title('Does Pay Location Code Affect Salary?')
    plt.xlabel('Pay Location Code')
    plt.ylabel('Annual Salary')
    plt.savefig('pay_location_code_vs_salary.png')
    # Shows Graph
    plt.show()

# Heatmap for Numerical Correlations
numerical_df = df.select_dtypes(include=['float64', 'int64'])
if not numerical_df.empty:
    # Creates Graph
    plt.figure(figsize=(10, 8))
    sns.heatmap(numerical_df.corr(), annot=True, cmap='coolwarm')
    plt.title('Correlation Heatmap')
    plt.savefig('correlation_heatmap.png')
    # Shows Graph
    plt.show()

# Hiring Trends Over Time
if 'hiring_year' in df.columns:
    # Creates Graph
    plt.figure(figsize=(10, 6))
    df['hiring_year'].value_counts().sort_index().plot(kind='line', marker='o', color='purple')
    plt.title('Hiring Trends Over the Years')
    plt.xlabel('Year')
    plt.ylabel('Number of Hires')
    plt.savefig('hiring_trends.png')
    # Shows Graph
    plt.show()

# Salary Change Over Time
if 'hiring_year' in df.columns and 'annual_salary' in df.columns:
    # Finds salary over time by finding average of hiring year and annual salary
    salary_over_time = df.groupby('hiring_year')['annual_salary'].mean()
    # Creates Graph
    plt.figure(figsize=(10, 6))
    salary_over_time.plot(kind='line', marker='o', color='blue')
    plt.title('How Has Salary Changed Over Time?')
    plt.xlabel('Year')
    plt.ylabel('Average Annual Salary')
    plt.savefig('salary_change_over_time.png')
    # Shows Graph
    plt.show()

# Locations with Least Scheduled Hours
if 'pay_location_code' in df.columns and 'scheduled_hours' in df.columns:
    # Finds smallest 5 possible location codes
    least_hours = df.groupby('pay_location_code')['scheduled_hours'].mean().nsmallest(5)
    # Creates Graph
    plt.figure(figsize=(10, 6))
    least_hours.plot(kind='bar', color='red')
    plt.title('Locations with Least Scheduled Hours')
    plt.xlabel('Pay Location Code')
    plt.ylabel('Average Scheduled Hours')
    plt.savefig('least_hours_locations.png')
    # Shows Graph
    plt.show()

# Locations with Most Scheduled Hours
if 'pay_location_code' in df.columns and 'scheduled_hours' in df.columns:
    # Finds 5 highest pay location codes
    most_hours = df.groupby('pay_location_code')['scheduled_hours'].mean().nlargest(5)
    # Creates Graph
    plt.figure(figsize=(10, 6))
    most_hours.plot(kind='bar', color='green')
    plt.title('Locations with Most Scheduled Hours')
    plt.xlabel('Pay Location Code')
    plt.ylabel('Average Scheduled Hours')
    plt.savefig('most_hours_locations.png')
    # Shows Graph
    plt.show()

# Locations with Higher Percentile Salaries
if 'pay_location_code' in df.columns and 'annual_salary' in df.columns:
    # Finds highest percentile (top 25%)
    salary_percentile = df.groupby('pay_location_code')['annual_salary'].quantile(0.75).sort_values(ascending=False)
    # Creates graph
    plt.figure(figsize=(10, 6))
    salary_percentile.head(5).plot(kind='bar', color='gold')
    plt.title('Locations with Highest Percentile Salaries')
    plt.xlabel('Pay Location Code')
    plt.ylabel('75th Percentile Salary')
    plt.savefig('high_percentile_salaries.png')
    # Shows Graph
    plt.show()

# Save processed data
df.to_csv('processed_data.csv', index=False)
