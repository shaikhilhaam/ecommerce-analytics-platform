import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# --- Global Style Configuration ---
# You can set a consistent theme and color palette for all plots here.
sns.set_theme(style="whitegrid", palette="viridis")

# ==============================================================================
# UNIVARIATE ANALYSIS (Analyzing single variables)
# ==============================================================================

def plot_histogram(data: pd.DataFrame, column: str, title: str):
    """
    Plots a histogram for a single numerical column.
    USE WHEN: You want to see the distribution (shape, center, spread) of a numerical variable.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(data[column], kde=True)
    plt.title(title, fontsize=16)
    plt.xlabel(column, fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.show()

def plot_countplot(data: pd.DataFrame, column: str, title: str, order_by_value=False):
    """
    Plots a count plot (bar chart) for a single categorical column.
    USE WHEN: You want to see the frequency of each category in a categorical variable.
    """
    plt.figure(figsize=(12, 7))
    if order_by_value:
        order = data[column].value_counts().index
        sns.countplot(y=data[column], order=order)
    else:
        sns.countplot(y=data[column])
    plt.title(title, fontsize=16)
    plt.xlabel('Count', fontsize=12)
    plt.ylabel(column, fontsize=12)
    plt.show()


# ==============================================================================
# BIVARIATE ANALYSIS (Analyzing relationships between two variables)
# ==============================================================================

def plot_scatterplot(data: pd.DataFrame, x_col: str, y_col: str, title: str, hue_col: str = None):
    """
    Plots a scatter plot to compare two numerical variables.
    USE WHEN: You want to see the relationship (correlation, patterns) between two numerical variables.
              Optionally use a 'hue' for a third categorical variable.
    """
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=data[x_col], y=data[y_col], hue=data.get(hue_col))
    plt.title(title, fontsize=16)
    plt.xlabel(x_col, fontsize=12)
    plt.ylabel(y_col, fontsize=12)
    plt.show()


def plot_boxplot(data: pd.DataFrame, x_col: str, y_col: str, title: str):
    """
    Plots a box plot to compare the distribution of a numerical variable across different categories.
    USE WHEN: You have a numerical variable (y_col) and a categorical variable (x_col) and
              want to compare the distribution (median, quartiles, outliers) of the numerical
              variable for each category.
    """
    plt.figure(figsize=(12, 7))
    sns.boxplot(x=data[x_col], y=data[y_col])
    plt.title(title, fontsize=16)
    plt.xlabel(x_col, fontsize=12)
    plt.ylabel(y_col, fontsize=12)
    plt.xticks(rotation=45)
    plt.show()

def plot_heatmap(data: pd.DataFrame, title: str):
    """
    Plots a heatmap of the correlation matrix for numerical columns.
    USE WHEN: You want to visualize the correlation between many numerical variables at once.
    """
    plt.figure(figsize=(10, 8))
    corr_matrix = data.corr(numeric_only=True)
    sns.heatmap(corr_matrix, annot=True, cmap='viridis', fmt=".2f")
    plt.title(title, fontsize=16)
    plt.show()