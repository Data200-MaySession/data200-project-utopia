import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style for plots
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

# --- 1. Load Data and Initial Inspection ---
print("--- Loading Data ---")

# Load the dataset
try:
    df = pd.read_csv('dataset/moviesDb.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: moviesDb.csv not found. Please ensure the file is in the 'dataset/' directory.")
    df = None

if df is not None:
    print("\n--- Initial Data Inspection ---")
    print("\nFirst 5 rows of the dataset:")
    print(df.head())
    print("\nDataset Info:")
    df.info()
    print("\nMissing values per column:")
    print(df.isnull().sum())

    # --- 2. Descriptive Statistics ---
    print("\n--- Descriptive Statistics ---")
    print("\nDescriptive statistics for numerical columns:")
    print(df.describe())

    print("\nValue counts for key categorical columns:")
    print("Genre counts:")
    print(df['genre'].value_counts().head(10))
    print("\nCountry counts:")
    print(df['country'].value_counts().head(10))
    print("\nCertification counts:")
    print(df['certification_US'].value_counts())
    print("\nSuccess counts:")
    print(df['success'].value_counts())

    # --- 3. Univariate Analysis ---
    print("\n--- Univariate Analysis ---")

    print("\n### Categorical Variables")
    # Bar plot for 'genre'
    plt.figure(figsize=(12, 6))
    sns.countplot(y='genre', data=df, order=df['genre'].value_counts().index[:10], palette='viridis')
    plt.title('Top 10 Movie Genres')
    plt.xlabel('Number of Movies')
    plt.ylabel('Genre')
    plt.show()

    # Bar plot for 'country'
    plt.figure(figsize=(12, 6))
    sns.countplot(y='country', data=df, order=df['country'].value_counts().index[:10], palette='magma')
    plt.title('Top 10 Production Countries')
    plt.xlabel('Number of Movies')
    plt.ylabel('Country')
    plt.show()

    # Bar plot for 'certification_US'
    plt.figure(figsize=(10, 5))
    sns.countplot(x='certification_US', data=df, order=df['certification_US'].value_counts().index, palette='cividis')
    plt.title('Distribution of US Movie Certifications')
    plt.xlabel('Certification')
    plt.ylabel('Number of Movies')
    plt.show()

    # Pie chart for 'success'
    plt.figure(figsize=(8, 8))
    df['success'].value_counts().plot.pie(autopct='%1.1f%%', colors=['lightcoral', 'lightskyblue'], startangle=90)
    plt.title('Distribution of Movie Success')
    plt.ylabel('')
    plt.show()

    print("\n### Numerical Variables")
    numerical_cols = ['runtime', 'budget', 'revenue', 'vote_average', 'vote_count', 'year']
    for col in numerical_cols:
        plt.figure(figsize=(10, 5))
        sns.histplot(df[col].dropna(), kde=True, bins=30, color='skyblue')
        plt.title(f'Distribution of {col.replace("_", " ").title()}')
        plt.xlabel(col.replace("_", " ").title())
        plt.ylabel('Frequency')
        plt.show()

    # Log transformation for highly skewed data like budget and revenue for better visualization
    for col in ['budget', 'revenue']:
        plt.figure(figsize=(10, 5))
        # Only apply log to positive values
        sns.histplot(np.log1p(df[df[col] > 0][col].dropna()), kde=True, bins=30, color='lightgreen')
        plt.title(f'Distribution of Log-transformed {col.replace("_", " ").title()} (for values > 0)')
        plt.xlabel(f'Log({col.replace("_", " ").title()})')
        plt.ylabel('Frequency')
        plt.show()

    # --- 4. Bivariate Analysis ---
    print("\n--- Bivariate Analysis ---")

    print("\n### Numerical vs. Numerical (Correlation)")
    # Scatter plots of key numerical features against revenue and budget
    sns.pairplot(df[['revenue', 'budget', 'runtime', 'vote_average', 'vote_count']].dropna())
    plt.suptitle('Pair Plot of Key Numerical Features', y=1.02)
    plt.show()

    # Correlation Matrix Heatmap
    plt.figure(figsize=(12, 10))
    correlation_matrix = df[numerical_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix of Numerical Features')
    plt.show()

    print("\n### Categorical vs. Numerical")
    # Box plots for Revenue by Genre (Top 10)
    top_genres = df['genre'].value_counts().index[:10]
    plt.figure(figsize=(14, 7))
    sns.boxplot(y='genre', x='revenue', data=df[df['genre'].isin(top_genres)], order=top_genres, palette='pastel')
    plt.title('Revenue Distribution by Top 10 Genres')
    plt.xlabel('Revenue')
    plt.ylabel('Genre')
    plt.xscale('log') # Use log scale for better visualization due to skewed data
    plt.show()

    # Box plots for Vote Average by Certification
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='certification_US', y='vote_average', data=df, order=df['certification_US'].value_counts().index, palette='viridis')
    plt.title('Vote Average Distribution by US Certification')
    plt.xlabel('US Certification')
    plt.ylabel('Vote Average')
    plt.show()

    # Box plots for Runtime by Success
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='success', y='runtime', data=df, palette='coolwarm')
    plt.title('Runtime Distribution by Movie Success')
    plt.xlabel('Success (True/False)')
    plt.ylabel('Runtime (minutes)')
    plt.show()

    print("\n### Categorical vs. Categorical")
    # Stacked bar chart for Success by Genre (Top 5)
    top_5_genres = df['genre'].value_counts().index[:5]
    genre_success_crosstab = pd.crosstab(df[df['genre'].isin(top_5_genres)]['genre'], df['success'], normalize='index') * 100
    genre_success_crosstab.plot(kind='bar', stacked=True, figsize=(12, 7), colormap='plasma')
    plt.title('Percentage of Success by Top 5 Genres')
    plt.xlabel('Genre')
    plt.ylabel('Percentage')
    plt.xticks(rotation=45, ha='right')
    plt.legend(title='Success', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

    # --- 5. Target Variable Analysis (Movie Success) ---
    print("\n--- Target Variable Analysis (Movie Success) ---")

    # Distribution of success over years
    plt.figure(figsize=(14, 7))
    success_by_year = df.groupby('year')['success'].value_counts(normalize=True).unstack()
    success_by_year.plot(kind='area', stacked=True, alpha=0.7, colormap='Accent', figsize=(14, 7))
    plt.title('Proportion of Successful vs. Unsuccessful Movies Over Years')
    plt.xlabel('Year')
    plt.ylabel('Proportion')
    plt.legend(title='Success')
    plt.grid(True)
    plt.show()

    # Revenue and Budget for Successful vs. Unsuccessful Movies
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    sns.boxplot(x='success', y='revenue', data=df, ax=axes[0], palette='viridis')
    axes[0].set_title('Revenue Distribution by Success')
    axes[0].set_xlabel('Success')
    axes[0].set_ylabel('Revenue')
    axes[0].set_yscale('log') # Use log scale for better visualization

    sns.boxplot(x='success', y='budget', data=df, ax=axes[1], palette='viridis')
    axes[1].set_title('Budget Distribution by Success')
    axes[1].set_xlabel('Success')
    axes[1].set_ylabel('Budget')
    axes[1].set_yscale('log') # Use log scale for better visualization
    plt.tight_layout()
    plt.show()

    # --- 6. Conclusion and Key Insights ---
    print("\n--- Conclusion and Key Insights ---")
    print("Summarize key findings from the EDA here.")
else:
    print("Could not perform EDA because the dataset was not loaded.") 