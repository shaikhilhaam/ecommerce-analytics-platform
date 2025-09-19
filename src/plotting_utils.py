import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import plotly.express as px
import json

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

def plot_scatterplot(data: pd.DataFrame, x_col: str, y_col: str, title: str, hue_col: str = None, size_col: str = None):
    """
    Plots a scatter plot to compare two numerical variables.
    """
    plt.figure(figsize=(12, 8))
    sns.scatterplot(x=data[x_col], y=data[y_col], hue=data.get(hue_col), size=data.get(size_col), sizes=(20, 200), alpha=0.7)
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
    
    
def plot_bar(data: pd.DataFrame, x_col: str, y_col: str, title: str, top_n: int = None):
    """
    Plots a bar chart for pre-aggregated data.
    USE WHEN: You have already calculated the values you want to plot (e.g., after a groupby).
    """
    if top_n:
        data = data.head(top_n)
    plt.figure(figsize=(12, 7))
    sns.barplot(x=x_col, y=y_col, data=data)
    plt.title(title, fontsize=16)
    plt.xlabel(x_col, fontsize=12)
    plt.ylabel(y_col, fontsize=12)
    plt.xticks(rotation=45)
    plt.show()
    
def plot_choropleth(data: pd.DataFrame, geojson_path: str, locations_col: str, color_col: str, title: str):
    """
    Plots an interactive choropleth map.
    USE WHEN: You want to visualize geographic data on a map.
    """

    with open(geojson_path) as f:
        geojson_data = json.load(f)

    fig = px.choropleth(
        data,
        geojson=geojson_data,
        locations=locations_col,
        featureidkey="properties.sigla", # Key in GeoJSON to match with locations_col
        color=color_col,
        color_continuous_scale="Viridis",
        scope="south america",
        title=title
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.show()
    
def plot_line(data: pd.DataFrame, x_col: str, y_col: str, title: str):
    """
    Plots a line chart, ideal for time series data.
    USE WHEN: You want to show a trend of a numerical variable over time.
    """
    plt.figure(figsize=(14, 7))
    sns.lineplot(x=x_col, y=y_col, data=data, marker='o')
    plt.title(title, fontsize=16)
    plt.xlabel(x_col, fontsize=12)
    plt.ylabel(y_col, fontsize=12)
    plt.xticks(rotation=45)
    plt.show()
    
def plot_pivot_heatmap(data: pd.DataFrame, title: str):
    """
    Plots a heatmap from a pivot table.
    USE WHEN: You want to visualize the relationship between two categorical variables,
              showing the intensity of their intersections.
    """
    plt.figure(figsize=(12, 8))
    sns.heatmap(data, cmap="viridis", annot=True, fmt=".0f")
    plt.title(title, fontsize=16)
    plt.xlabel("Day of Week", fontsize=12)
    plt.ylabel("Hour of Day", fontsize=12)
    plt.show()
    
def plot_pie_chart(data: pd.DataFrame, column: str, title: str):
    """
    Plots a pie chart for a categorical column.
    USE WHEN: You want to show the proportion of categories in a whole.
              Best for a small number of categories (2-5).
    """
    plt.figure(figsize=(8, 8))
    counts = data[column].value_counts()
    plt.pie(counts, labels=counts.index, autopct='%1.1f%%', startangle=140,
            colors=sns.color_palette('viridis', len(counts)))
    plt.title(title, fontsize=16)
    plt.ylabel('') # Hide the y-label which is 'None' by default
    plt.show()