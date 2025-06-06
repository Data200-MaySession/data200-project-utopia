import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

# Import functions for effect size calculation (need to implement or use a library like pingouin if available)
# For now, we'll calculate manually or use scipy/numpy where possible.

# --- Configuration ---
# Define the significance level (alpha) for hypothesis testing.
# This is the probability of rejecting the null hypothesis when it is true (Type I error).
# A commonly used value is 0.05.
alpha = 0.05

# --- Data Loading ---
# Load the preprocessed movie dataset.
try:
    df = pd.read_csv('dataset/moviesDb.csv')
    print("Dataset loaded successfully for detailed hypothesis testing!")
except FileNotFoundError:
    # Handle the case where the dataset file is not found.
    print("Error: moviesDb.csv not found. Cannot perform detailed hypothesis testing.")
    df = None

# --- Hypothesis Testing Analysis ---
# Proceed with hypothesis testing only if the dataset was loaded successfully.
if df is not None:
    print("\n---\n--- Performing Detailed Hypothesis Tests ---\n---")
    print(f"Using a significance level (alpha) of: {alpha}")
    print("\nInterpretation Guide:\n")
    print(f"- If p-value < {alpha}: Reject the Null Hypothesis (H0). This indicates statistically significant evidence for the Alternative Hypothesis (Ha).")
    print(f"- If p-value >= {alpha}: Fail to Reject the Null Hypothesis (H0). There is not enough statistically significant evidence for the Alternative Hypothesis (Ha) at the chosen alpha level.")
    print("\nEffect size measures the magnitude of the effect, independent of sample size. Interpretations of effect size (e.g., Cohen's d: Small=0.2, Medium=0.5, Large=0.8; Cramer's V: Small=0.1, Medium=0.3, Large=0.5) are general guidelines and can vary by field.\n---")

    # --- Hypothesis Test 1: Budget vs. Success ---
    # Objective: Investigate if movies with higher budgets are more likely to be successful.
    # Null Hypothesis (H0): There is no significant difference in the mean budget between successful and unsuccessful movies (\(\mu_{\text{successful}} = \mu_{\text{unsuccessful}}\)).
    # Alternative Hypothesis (Ha): The mean budget of successful movies is significantly higher than that of unsuccessful movies (\(\mu_{\text{successful}} > \mu_{\text{unsuccessful}}\)). This is a one-tailed test.
    # Test Used: Independent samples t-test (specifically, Welch's t-test).
    # Rationale for Welch's t-test: Used because it is robust to the assumption of equal variances, which might be violated with financial data.
    # Assumptions for Independent Samples t-test:
    # 1. Independence of observations: Assumed.
    # 2. Normality: Approximately normal distribution of the dependent variable (budget) within each group. Robust for large samples.
    # 3. Homogeneity of Variances: Not assumed by Welch's t-test.
    print("\n--- Hypothesis Test 1: Budget vs. Success ---")
    print("H0: Mean budget is the same for successful and unsuccessful movies (μ_successful = μ_unsuccessful).")
    print("Ha: Mean budget is higher for successful movies (μ_successful > μ_unsuccessful).")

    # Separate budget data for successful and unsuccessful movies, dropping any potential NaN values.
    successful_budgets = df[df['success'] == True]['budget'].dropna()
    unsuccessful_budgets = df[df['success'] == False]['budget'].dropna()

    # Print descriptive statistics for comparison to understand the data distribution and central tendency.
    print("\nDescriptive Statistics for Budget:")
    print("Successful Movies Budget:\n", successful_budgets.describe())
    print("\nUnsuccessful Movies Budget:\n", unsuccessful_budgets.describe())

    # Check if there are enough data points in both groups to perform the t-test (minimum of 2 in each group).
    if len(successful_budgets) > 1 and len(unsuccessful_budgets) > 1:
        # Perform independent samples t-test (Welch's t-test is the default when equal_var=False).
        # alternative='greater' specifies a one-tailed test as per Ha.
        ttest_budget = stats.ttest_ind(successful_budgets, unsuccessful_budgets, equal_var=False, alternative='greater')
        print(f"\nIndependent t-test Results (Budget vs. Success):\n  Test Statistic = {ttest_budget.statistic:.4f}, p-value = {ttest_budget.pvalue:.4f}")

        # Calculate Cohen's d for effect size.
        # Pooled standard deviation is typically used for Cohen's d, but for Welch's t-test (unequal variances),
        # a modified Cohen's d or reporting mean difference and standard deviations is common.
        # We'll calculate a version of Cohen's d using the pooled standard deviation for simplicity, but note this limitation.
        n1, n2 = len(successful_budgets), len(unsuccessful_budgets)
        mean1, mean2 = successful_budgets.mean(), unsuccessful_budgets.mean()
        std1, std2 = successful_budgets.std(), unsuccessful_budgets.std()
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        cohen_d_budget = (mean1 - mean2) / pooled_std
        print(f"  Effect Size (Cohen's d): {cohen_d_budget:.4f}")

        # Conclusion based on the p-value compared to the significance level (alpha).
        if ttest_budget.pvalue < alpha:
            print("  Conclusion: Reject the null hypothesis (H0).")
            print(f"  Interpretation: At the {alpha} significance level, there is statistically significant evidence to support the claim that successful movies have a higher mean budget than unsuccessful movies. The effect size (Cohen's d = {cohen_d_budget:.4f}) suggests a [Interpret Cohen's d magnitude - e.g., small, medium, large] difference.")
        else:
            print("  Conclusion: Fail to reject the null hypothesis (H0).")
            print(f"  Interpretation: At the {alpha} significance level, there is not enough statistically significant evidence to support the claim that successful movies have a higher mean budget than unsuccessful movies.")
    else:
        print("  Insufficient data in one or both groups to perform independent samples t-test for Budget.")

    # --- Hypothesis Test 2: Genre vs. Success ---
    # Objective: Determine if there is a statistically significant association between movie genre and success.
    # Null Hypothesis (H0): Movie genre and success are independent.
    # Alternative Hypothesis (Ha): Movie genre and success are dependent (i.e., the proportion of successful movies varies across different genres).
    # Test Used: Chi-squared test of independence.
    # Rationale: Appropriate for examining the association between two categorical variables.
    # Assumptions for Chi-squared Test of Independence:
    # 1. Independence of observations: Assumed.
    # 2. Expected frequencies: Generally, expected counts should be >= 5. Violations can affect the validity of the test. Handling involves grouping categories or using exact tests.
    print("\n\n--- Hypothesis Test 2: Genre vs. Success ---")
    print("H0: Genre and Success are independent.")
    print("Ha: Genre and Success are dependent.")

    # To ensure sufficient data for the Chi-squared test, focus on the top N genres.
    top_genres_count = 15  # Number of top genres to include.
    top_genres = df['genre'].value_counts().nlargest(top_genres_count).index.tolist()
    # Filter the DataFrame to include only movies within the top genres.
    df_top_genres = df[df['genre'].isin(top_genres)].copy()

    if not df_top_genres.empty:
        # Create a contingency table (cross-tabulation) of Genre and Success for the filtered data.
        genre_success_crosstab = pd.crosstab(df_top_genres['genre'], df_top_genres['success'])

        # Check if the contingency table has enough dimensions (at least 2x2) and data to perform the test.
        if genre_success_crosstab.shape[0] > 1 and genre_success_crosstab.shape[1] > 1:
             # Perform Chi-squared test of independence.
             chi2, p, dof, expected = stats.chi2_contingency(genre_success_crosstab)
             print(f"\nContingency table for top {top_genres_count} Genres vs. Success:\n{genre_success_crosstab}")
             print(f"\nExpected frequencies table (for assumption check):\n{pd.DataFrame(expected, index=genre_success_crosstab.index, columns=genre_success_crosstab.columns)}")
             print(f"\nChi-squared test Results (Genre vs. Success for Top {top_genres_count} Genres):\n  Test Statistic = {chi2:.4f}, p-value = {p:.4f}, Degrees of Freedom = {dof}")

             # Check for low expected counts and provide a warning if the assumption is potentially violated.
             low_expected_counts = (expected < 5).sum()
             total_cells = expected.size
             if low_expected_counts > 0:
                 print(f"  Warning: {low_expected_counts} out of {total_cells} cells in the expected frequencies table have values less than 5. Chi-squared test results may be unreliable, especially if a large proportion of cells (< 20% is a common guideline for proceeding) are affected or if any expected count is zero. Consider grouping genres or using alternative tests if appropriate.")

             # Calculate Cramer's V for effect size (for tables larger than 2x2).
             # Phi coefficient is typically used for 2x2 tables, Cramer's V for larger.
             # Cramer's V is calculated as sqrt(chi2 / (n * min(rows - 1, columns - 1)))
             n = genre_success_crosstab.sum().sum()
             min_dim = min(genre_success_crosstab.shape) - 1
             cramers_v = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else float('nan') # Avoid division by zero
             print(f"  Effect Size (Cramer's V): {cramers_v:.4f}")

             # Conclusion based on the p-value and alpha.
             if p < alpha:
                 print("  Conclusion: Reject the null hypothesis (H0).")
                 print(f"  Interpretation: At the {alpha} significance level, there is statistically significant evidence to suggest an association between movie genre (among the top {top_genres_count}) and success. The effect size (Cramer's V = {cramers_v:.4f}) suggests a [Interpret Cramer's V magnitude - e.g., small, medium, large] association. This implies that the success rate is not the same across these genres.")
             else:
                 print("  Conclusion: Fail to reject the null hypothesis (H0).")
                 print(f"  Interpretation: At the {alpha} significance level, there is no statistically significant evidence to suggest an association between movie genre (among the top {top_genres_count}) and success.")
        else:
             print("  Insufficient data or categories in the filtered genres to perform Chi-squared test.")

    else:
        print("  No data available for top genres to perform Chi-squared test.")

    # --- Hypothesis Test 3: Vote Average vs. Success ---
    # Objective: Investigate if movies with higher vote averages are more likely to be successful.
    # Null Hypothesis (H0): There is no significant difference in the mean vote average between successful and unsuccessful movies (\(\mu_{\text{successful}} = \mu_{\text{unsuccessful}}\)).
    # Alternative Hypothesis (Ha): The mean vote average of successful movies is significantly higher than that of unsuccessful movies (\(\mu_{\text{successful}} > \mu_{\text{unsuccessful}}\)). This is a one-tailed test.
    # Test Used: Independent samples t-test (Welch's t-test).
    # Rationale and Assumptions: Same as Hypothesis Test 1, applied to 'vote_average'.
    print("\n\n--- Hypothesis Test 3: Vote Average vs. Success ---")
    print("H0: Mean vote average is the same for successful and unsuccessful movies (μ_successful = μ_unsuccessful).")
    print("Ha: Mean vote average is higher for successful movies (μ_successful > μ_unsuccessful).")

    # Separate vote average data for successful and unsuccessful movies, dropping any potential NaN values.
    successful_votes = df[df['success'] == True]['vote_average'].dropna()
    unsuccessful_votes = df[df['success'] == False]['vote_average'].dropna()

    # Print descriptive statistics for comparison.
    print("\nDescriptive Statistics for Vote Average:")
    print("Successful Movies Vote Average:\n", successful_votes.describe())
    print("\nUnsuccessful Movies Vote Average:\n", unsuccessful_votes.describe())

    # Check if there are enough data points in both groups.
    if len(successful_votes) > 1 and len(unsuccessful_votes) > 1:
        # Perform independent samples t-test (Welch's t-test).
        # alternative='greater' specifies a one-tailed test.
        ttest_votes = stats.ttest_ind(successful_votes, unsuccessful_votes, equal_var=False, alternative='greater')
        print(f"\nIndependent t-test Results (Vote Average vs. Success):\n  Test Statistic = {ttest_votes.statistic:.4f}, p-value = {ttest_votes.pvalue:.4f}")

        # Calculate Cohen's d for effect size.
        n1, n2 = len(successful_votes), len(unsuccessful_votes)
        mean1, mean2 = successful_votes.mean(), unsuccessful_votes.mean()
        std1, std2 = successful_votes.std(), unsuccessful_votes.std()
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        cohen_d_votes = (mean1 - mean2) / pooled_std
        print(f"  Effect Size (Cohen's d): {cohen_d_votes:.4f}")

        # Conclusion based on the p-value and alpha.
        if ttest_votes.pvalue < alpha:
            print("  Conclusion: Reject the null hypothesis (H0).")
            print(f"  Interpretation: At the {alpha} significance level, there is statistically significant evidence to support the claim that successful movies have a higher mean vote average than unsuccessful movies. The effect size (Cohen's d = {cohen_d_votes:.4f}) suggests a [Interpret Cohen's d magnitude] difference.")
        else:
            print("  Conclusion: Fail to reject the null hypothesis (H0).")
            print(f"  Interpretation: At the {alpha} significance level, there is no statistically significant evidence to support the claim that successful movies have a higher mean vote average than unsuccessful movies.")
    else:
        print("  Insufficient data in one or both groups to perform independent samples t-test for Vote Average.")

    # --- Hypothesis Test 4: Runtime vs. Success ---
    # Objective: Investigate if there is a significant difference in the mean runtime between successful and unsuccessful movies.
    # Null Hypothesis (H0): There is no significant difference in the mean runtime between successful and unsuccessful movies (\(\mu_{\text{successful}} = \mu_{\text{unsuccessful}}\)).
    # Alternative Hypothesis (Ha): The mean runtime differs between successful and unsuccessful movies (\(\mu_{\text{successful}} \neq \mu_{\text{unsuccessful}}\)). This is a two-tailed test.
    # Test Used: Independent samples t-test (Welch's t-test).
    # Rationale and Assumptions: Same as Hypothesis Test 1, applied to 'runtime'. Using a two-tailed test as we are checking for any difference, not a specific direction.
    print("\n\n--- Hypothesis Test 4: Runtime vs. Success ---")
    print("H0: Mean runtime is the same for successful and unsuccessful movies (μ_successful = μ_unsuccessful).")
    print("Ha: Mean runtime differs between successful and unsuccessful movies (μ_successful ≠ μ_unsuccessful).")

    # Separate runtime data for successful and unsuccessful movies, dropping any potential NaN values.
    successful_runtime = df[df['success'] == True]['runtime'].dropna()
    unsuccessful_runtime = df[df['success'] == False]['runtime'].dropna()

    # Print descriptive statistics for comparison.
    print("\nDescriptive Statistics for Runtime:")
    print("Successful Movies Runtime:\n", successful_runtime.describe())
    print("\nUnsuccessful Movies Runtime:\n", unsuccessful_runtime.describe())

    # Check if there are enough data points in both groups.
    if len(successful_runtime) > 1 and len(unsuccessful_runtime) > 1:
        # Perform independent samples t-test (Welch's t-test).
        # Using default alternative='two-sided' for a two-tailed test.
        ttest_runtime = stats.ttest_ind(successful_runtime, unsuccessful_runtime, equal_var=False)
        print(f"\nIndependent t-test Results (Runtime vs. Success):\n  Test Statistic = {ttest_runtime.statistic:.4f}, p-value = {ttest_runtime.pvalue:.4f}")

        # Calculate Cohen's d for effect size.
        n1, n2 = len(successful_runtime), len(unsuccessful_runtime)
        mean1, mean2 = successful_runtime.mean(), unsuccessful_runtime.mean()
        std1, std2 = successful_runtime.std(), unsuccessful_runtime.std()
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        cohen_d_runtime = (mean1 - mean2) / pooled_std
        print(f"  Effect Size (Cohen's d): {cohen_d_runtime:.4f}")

        # Conclusion based on the p-value and alpha.
        if ttest_runtime.pvalue < alpha:
            print("  Conclusion: Reject the null hypothesis (H0).")
            print(f"  Interpretation: At the {alpha} significance level, there is statistically significant evidence to suggest a difference in mean runtime between successful and unsuccessful movies. The effect size (Cohen's d = {cohen_d_runtime:.4f}) suggests a [Interpret Cohen's d magnitude] difference.")
        else:
            print("  Conclusion: Fail to reject the null hypothesis (H0).")
            print(f"  Interpretation: At the {alpha} significance level, there is no statistically significant evidence to suggest a difference in mean runtime between successful and unsuccessful movies.")
    else:
        print("  Insufficient data in one or both groups to perform independent samples t-test for Runtime.")

    # --- Hypothesis Test 5: Vote Count vs. Success ---
    # Objective: Investigate if movies with higher vote counts are more likely to be successful.
    # Null Hypothesis (H0): There is no significant difference in the mean vote count between successful and unsuccessful movies (\(\mu_{\text{successful}} = \mu_{\text{unsuccessful}}\)).
    # Alternative Hypothesis (Ha): The mean vote count of successful movies is significantly higher than that of unsuccessful movies (\(\mu_{\text{successful}} > \mu_{\text{unsuccessful}}\)). This is a one-tailed test.
    # Test Used: Independent samples t-test (Welch's t-test).
    # Rationale and Assumptions: Same as Hypothesis Test 1, applied to 'vote_count'.
    print("\n\n--- Hypothesis Test 5: Vote Count vs. Success ---")
    print("H0: Mean vote count is the same for successful and unsuccessful movies (μ_successful = μ_unsuccessful).")
    print("Ha: Mean vote count is higher for successful movies (μ_successful > μ_unsuccessful).")

    # Separate vote count data for successful and unsuccessful movies, dropping any potential NaN values.
    successful_votecount = df[df['success'] == True]['vote_count'].dropna()
    unsuccessful_votecount = df[df['success'] == False]['vote_count'].dropna()

    # Print descriptive statistics for comparison.
    print("\nDescriptive Statistics for Vote Count:")
    print("Successful Movies Vote Count:\n", successful_votecount.describe())
    print("\nUnsuccessful Movies Vote Count:\n", unsuccessful_votecount.describe())

    # Check if there are enough data points in both groups.
    if len(successful_votecount) > 1 and len(unsuccessful_votecount) > 1:
        # Perform independent samples t-test (Welch's t-test).
        # alternative='greater' specifies a one-tailed test.
        ttest_votecount = stats.ttest_ind(successful_votecount, unsuccessful_votecount, equal_var=False, alternative='greater')
        print(f"\nIndependent t-test Results (Vote Count vs. Success):\n  Test Statistic = {ttest_votecount.statistic:.4f}, p-value = {ttest_votecount.pvalue:.4f}")

        # Calculate Cohen's d for effect size.
        n1, n2 = len(successful_votecount), len(unsuccessful_votecount)
        mean1, mean2 = successful_votecount.mean(), unsuccessful_votecount.mean()
        std1, std2 = successful_votecount.std(), unsuccessful_votecount.std()
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        cohen_d_votecount = (mean1 - mean2) / pooled_std
        print(f"  Effect Size (Cohen's d): {cohen_d_votecount:.4f}")

        # Conclusion based on the p-value and alpha.
        if ttest_votecount.pvalue < alpha:
            print("  Conclusion: Reject the null hypothesis (H0).")
            print(f"  Interpretation: At the {alpha} significance level, there is statistically significant evidence to support the claim that successful movies have a higher mean vote count than unsuccessful movies. The effect size (Cohen's d = {cohen_d_votecount:.4f}) suggests a [Interpret Cohen's d magnitude] difference.")
        else:
            print("  Conclusion: Fail to reject the null hypothesis (H0).")
            print(f"  Interpretation: At the {alpha} significance level, there is no statistically significant evidence to support the claim that successful movies have a higher mean vote count than unsuccessful movies.")
    else:
        print("  Insufficient data in one or both groups to perform independent samples t-test for Vote Count.")

    # --- Hypothesis Test 6: Certification vs. Success ---
    # Objective: Investigate if there is a statistically significant association between movie certification (US) and success.
    # Null Hypothesis (H0): Movie certification and success are independent.\n    # Alternative Hypothesis (Ha): Movie certification and success are dependent.
    # Test Used: Chi-squared test of independence.
    # Rationale: Appropriate for examining the association between two categorical variables.
    # Assumptions for Chi-squared Test of Independence:
    # 1. Independence of observations: Assumed.
    # 2. Expected frequencies: Generally, expected counts should be >= 5. Violations can affect the validity of the test.
    print("\n\n--- Hypothesis Test 6: Certification vs. Success ---")
    print("H0: Certification and Success are independent.")
    print("Ha: Certification and Success are dependent.")

    # Create a contingency table (cross-tabulation) of Certification and Success.
    certification_success_crosstab = pd.crosstab(df['certification_US'], df['success'])

    # Check if the contingency table has enough dimensions (at least 2x2) and data.
    if certification_success_crosstab.shape[0] > 1 and certification_success_crosstab.shape[1] > 1:
        # Perform Chi-squared test of independence.
        chi2, p, dof, expected = stats.chi2_contingency(certification_success_crosstab)
        print(f"\nContingency table for Certification vs. Success:\n{certification_success_crosstab}")
        print(f"\nExpected frequencies table (for assumption check):\n{pd.DataFrame(expected, index=certification_success_crosstab.index, columns=certification_success_crosstab.columns)}")
        print(f"\nChi-squared test Results (Certification vs. Success):\n  Test Statistic = {chi2:.4f}, p-value = {p:.4f}, Degrees of Freedom = {dof}")

        # Check for low expected counts and provide a warning.
        low_expected_counts = (expected < 5).sum()
        total_cells = expected.size
        if low_expected_counts > 0:
            print(f"  Warning: {low_expected_counts} out of {total_cells} cells in the expected frequencies table have values less than 5. Chi-squared test results may be unreliable, especially if a large proportion of cells (< 20%) are affected or if any expected count is zero. Consider grouping certifications or using alternative tests.")

        # Calculate Cramer's V for effect size.
        n = certification_success_crosstab.sum().sum()
        min_dim = min(certification_success_crosstab.shape) - 1
        cramers_v_cert = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else float('nan')
        print(f"  Effect Size (Cramer's V): {cramers_v_cert:.4f}")

        # Conclusion based on the p-value and alpha.
        if p < alpha:
            print("  Conclusion: Reject the null hypothesis (H0).")
            print(f"  Interpretation: At the {alpha} significance level, there is statistically significant evidence to suggest an association between movie US certification and success. The effect size (Cramer's V = {cramers_v_cert:.4f}) suggests a [Interpret Cramer's V magnitude] association. This implies that the success rate is not the same across different certifications.")
        else:
            print("  Conclusion: Fail to reject the null hypothesis (H0).")
            print(f"  Interpretation: At the {alpha} significance level, there is no statistically significant evidence to suggest an association between movie US certification and success.")
    else:
         print("  Insufficient data or categories to perform Chi-squared test for Certification.")

    # --- Hypothesis Test 7: Country vs. Success ---
    # Objective: Investigate if the production country is associated with movie success.
    # Null Hypothesis (H0): Production country and success are independent.\n    # Alternative Hypothesis (Ha): Production country and success are dependent.
    # Test Used: Chi-squared test of independence.
    # Rationale: Appropriate for examining the association between two categorical variables.
    # Assumptions for Chi-squared Test of Independence:
    # 1. Independence of observations: Assumed.
    # 2. Expected frequencies: Generally, expected counts should be >= 5. Violations can affect the validity of the test.
    print("\n\n--- Hypothesis Test 7: Country vs. Success ---")
    print("H0: Country and Success are independent.")
    print("Ha: Country and Success are dependent.")

    # To ensure sufficient data for the Chi-squared test, focus on the top N countries.
    top_countries_count = 10 # Number of top countries to include.
    top_countries = df['country'].value_counts().nlargest(top_countries_count).index.tolist()
    # Filter the DataFrame to include only movies within the top countries.
    df_top_countries = df[df['country'].isin(top_countries)].copy()

    if not df_top_countries.empty:
        # Create a contingency table (cross-tabulation) of Country and Success for the filtered data.
        country_success_crosstab = pd.crosstab(df_top_countries['country'], df_top_countries['success'])

        # Check if the contingency table has enough dimensions (at least 2x2) and data.
        if country_success_crosstab.shape[0] > 1 and country_success_crosstab.shape[1] > 1:
            # Perform Chi-squared test of independence.
            chi2, p, dof, expected = stats.chi2_contingency(country_success_crosstab)
            print(f"\nContingency table for top {top_countries_count} Countries vs. Success:\n{country_success_crosstab}")
            print(f"\nExpected frequencies table (for assumption check):\n{pd.DataFrame(expected, index=country_success_crosstab.index, columns=country_success_crosstab.columns)}")
            print(f"\nChi-squared test Results (Country vs. Success for Top {top_countries_count} Countries):\n  Test Statistic = {chi2:.4f}, p-value = {p:.4f}, Degrees of Freedom = {dof}")

            # Check for low expected counts and provide a warning.
            low_expected_counts = (expected < 5).sum()
            total_cells = expected.size
            if low_expected_counts > 0:
                 print(f"  Warning: {low_expected_counts} out of {total_cells} cells in the expected frequencies table have values less than 5. Chi-squared test results may be unreliable, especially if a large proportion of cells (< 20%) are affected or if any expected count is zero. Consider grouping countries or using alternative tests if appropriate.")

            # Calculate Cramer's V for effect size.
            n = country_success_crosstab.sum().sum()
            min_dim = min(country_success_crosstab.shape) - 1
            cramers_v_country = np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else float('nan')
            print(f"  Effect Size (Cramer's V): {cramers_v_country:.4f}")

            # Conclusion based on the p-value and alpha.
            if p < alpha:
                print("  Conclusion: Reject the null hypothesis (H0).")
                print(f"  Interpretation: At the {alpha} significance level, there is statistically significant evidence to suggest an association between production country (among the top {top_countries_count}) and success. The effect size (Cramer's V = {cramers_v_country:.4f}) suggests a [Interpret Cramer's V magnitude] association. This implies that the success rate is not the same across these countries.")
            else:
                print("  Conclusion: Fail to reject the null hypothesis (H0).")
                print(f"  Interpretation: At the {alpha} significance level, there is no statistically significant evidence to suggest an association between production country (among the top {top_countries_count}) and success.")
        else:
             print("  Insufficient data or categories in the filtered countries to perform Chi-squared test.")
    else:
        print("  No data available for top countries to perform Chi-squared test.")

    # --- Overall Summary of Hypothesis Testing Findings ---
    print("\n\n---\n--- Overall Summary of Detailed Hypothesis Testing Findings ---\n---")
    print(f"This section provides a summary of the conclusions drawn from the detailed hypothesis tests performed at a significance level of alpha = {alpha}.\n")
    print("Key to Interpreting Results:\n")
    print(f"- **Statistical Significance (p-value vs. alpha)**:\n")
    print(f"  - If p-value < {alpha}: The result is statistically significant. We reject the null hypothesis (H0), supporting the alternative hypothesis (Ha). This suggests the observed relationship or difference is unlikely due to random chance.\n")
    print(f"  - If p-value \(\ge\) {alpha}: The result is not statistically significant at this alpha level. We fail to reject the null hypothesis (H0). We do not have sufficient evidence to support the alternative hypothesis (Ha).\n")
    print("\n- **Practical Significance (Effect Size)**:\n")
    print("  - Effect size measures the magnitude of the relationship or difference, independent of sample size.\n")
    print("  - **Cohen's d (for t-tests)**: Small \(\approx 0.2\), Medium \(\approx 0.5\), Large \(\approx 0.8\).\n")
    print("  - **Cramer's V (for Chi-squared)**: Small \(\approx 0.1\), Medium \(\approx 0.3\), Large \(\approx 0.5\) (These are general guidelines and depend on the table size).\n")
    print("  - A statistically significant result with a small effect size might be less practically important than a non-significant result with a large effect size (though non-significant large effects could be due to low power/small sample size).\n")

    print("\nSummary of Test Conclusions (Refer to the detailed output above for specific test statistics, p-values, and effect sizes):\n")

    # Note: The specific conclusions (Reject/Fail to Reject) and interpretations for each test will depend on the actual data and the calculated values when the script is run.
    # The interpretations below are general guidance.

    print("- **Hypothesis 1 (Budget vs. Success)**: Check the t-test results (Statistic, p-value, Cohen's d) and interpretation above to determine if there is statistically significant evidence for successful movies having higher mean budgets and the magnitude of this difference.")
    print("- **Hypothesis 2 (Genre vs. Success)**: Check the Chi-squared test results (Statistic, p-value, Cramer's V) and interpretation above to determine if there is a statistically significant association between genre (among the top 15) and success and the strength of this association.\n  *Review the Expected Frequencies table to assess if the Chi-squared test assumptions are met.\n")
    print("- **Hypothesis 3 (Vote Average vs. Success)**: Check the t-test results (Statistic, p-value, Cohen's d) and interpretation above to determine if there is statistically significant evidence for successful movies having higher mean vote averages and the magnitude of this difference.")
    print("- **Hypothesis 4 (Runtime vs. Success)**: Check the t-test results (Statistic, p-value, Cohen's d) and interpretation above to determine if there is a statistically significant difference in mean runtime between successful and unsuccessful movies and the magnitude of this difference.")
    print("- **Hypothesis 5 (Vote Count vs. Success)**: Check the t-test results (Statistic, p-value, Cohen's d) and interpretation above to determine if there is statistically significant evidence for successful movies having higher mean vote counts and the magnitude of this difference.")
    print("- **Hypothesis 6 (Certification vs. Success)**: Check the Chi-squared test results (Statistic, p-value, Cramer's V) and interpretation above to determine if there is a statistically significant association between movie US certification and success and the strength of this association.\n  *Review the Expected Frequencies table to assess if the Chi-squared test assumptions are met.\n")
    print("- **Hypothesis 7 (Country vs. Success)**: Check the Chi-squared test results (Statistic, p-value, Cramer's V) and interpretation above to determine if there is a statistically significant association between production country (among the top 10) and success and the strength of this association.\n  *Review the Expected Frequencies table to assess if the Chi-squared test assumptions are met.\n")

    print("\nGeneral Considerations When Interpreting:\n")
    print("- **Causation**: Remember that statistical association does not imply causation. These tests show relationships, but not necessarily cause and effect.\n")
    print("- **Assumptions**: Always consider if the assumptions of the tests are met. Violations can invalidate the results.\n")
    print("- **Multiple Comparisons**: When performing multiple tests, the probability of finding a statistically significant result purely by chance increases. For a formal analysis, consider using methods to adjust for multiple comparisons (e.g., Bonferroni correction, Benjamini-Hochberg). This script does not include such adjustments.\n")
    print("- **Data Quality**: Ensure the data is clean and preprocessed correctly, as errors in the data can affect the results.\n")

else:
    print("\nSkipping hypothesis testing due to data loading error. Please ensure 'dataset/moviesDb.csv' exists.") 