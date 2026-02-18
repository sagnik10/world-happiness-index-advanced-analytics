# World Happiness Index — Advanced Analytics and Forecasting System

## Overview

This project is a production-grade data analytics and machine learning system designed to analyze, model, and forecast global happiness trends using the World Happiness Index dataset.

The system performs automated data ingestion, cleaning, statistical analysis, regression modeling, anomaly detection, dimensionality reduction, time-series analysis, visualization, and executive report generation.

This project is intended for portfolio, research, and business intelligence applications.

---

## Business Objective

The goal of this project is to analyze global happiness metrics across multiple years, identify key influencing factors, detect anomalies, and forecast future trends using unbiased regression models and statistical techniques.

This system helps answer important analytical questions such as:

* How has global happiness changed over time
* Which factors most strongly influence happiness score
* Are there abnormal or unusual country trends
* What are the future projected trends based on historical data
* How data-driven insights can support decision-making

---

## Dataset Information

Source: World Happiness Index

Files used in this project:

```
2015.csv
2016.csv
2017.csv
2018.csv
2019.csv
```

Each file contains country-level data including metrics such as:

* Happiness Score
* GDP per capita
* Social support
* Healthy life expectancy
* Freedom to make life choices
* Generosity
* Perceptions of corruption

The system automatically merges and processes all files into a unified master dataset.

---

## Core Features

### Automated Data Processing

* Automatic CSV ingestion
* Multi-year dataset merging
* Duplicate removal
* Missing value handling
* Clean master dataset generation
* Structured output storage

---

### Statistical Analysis

The system performs comprehensive statistical analysis including:

* Pearson correlation analysis
* Spearman correlation analysis
* Kendall correlation analysis
* Covariance matrix analysis
* Distribution analysis

These methods identify relationships between variables and important influencing factors.

---

### Time Series Analysis

The system analyzes trends across years, including:

* Year-wise metric aggregation
* Time-series trend visualization
* Temporal pattern identification
* Regression-based forecasting

---

### Machine Learning Models

This system uses unbiased and appropriate regression models:

#### Linear Regression

Captures linear relationships and overall trends.

#### Ridge Regression

Reduces overfitting and improves generalization.

#### Polynomial Regression

Captures nonlinear relationships in data trends.

#### Random Forest Regression

Captures complex nonlinear patterns using ensemble learning.

These models provide reliable and unbiased trend modeling and forecasting.

---

### Dimensionality Reduction

Principal Component Analysis (PCA) is used for:

* Feature reduction
* Pattern identification
* Structural visualization
* High-dimensional data simplification

---

### Anomaly Detection

Isolation Forest is used to detect anomalies such as:

* Statistical outliers
* Unusual country behavior
* Structural deviations

This helps identify irregular patterns in the dataset.

---

### Visualization System

The system generates professional-grade visualizations including:

* Time series trend charts
* Regression forecast charts
* Correlation heatmaps
* Histograms
* Density plots
* Box plots
* PCA projection charts
* Anomaly detection charts

All charts are generated automatically.

---

### Executive Intelligence Report

The system generates a professional executive report:

```
Executive_Report.pdf
```

This report includes:

* Key analytics visualizations
* Trend analysis charts
* Statistical insights
* Machine learning outputs

This report is suitable for portfolio presentation or business use.

---

## Project Structure

```
world-happiness-index-advanced-analytics/

│
├── 2015.csv
├── 2016.csv
├── 2017.csv
├── 2018.csv
├── 2019.csv
│
├── Data_Analyser.py
│
├── Output/
│   ├── charts/
│   ├── data/
│   ├── models/
│   └── Executive_Report.pdf
│
└── README.md
```

---

## Technologies Used

Programming Language:

* Python 3.11+

Libraries:

* pandas
* numpy
* matplotlib
* seaborn
* scikit-learn
* tqdm
* reportlab
* openpyxl

Machine Learning Algorithms:

* Linear Regression
* Ridge Regression
* Polynomial Regression
* Random Forest Regression
* Isolation Forest
* Principal Component Analysis

---

## Installation

Clone the repository:

```
git clone https://github.com/yourusername/world-happiness-index-advanced-analytics.git
```

Navigate into the folder:

```
cd world-happiness-index-advanced-analytics
```

Install required dependencies:

```
pip install pandas numpy matplotlib seaborn scikit-learn tqdm reportlab openpyxl
```

---

## Usage

Run the analysis system:

```
python Data_Analyser.py
```

The system will automatically generate:

```
Output/
charts/
data/
Executive_Report.pdf
```

---

## System Capabilities Summary

This system performs complete analytics pipeline operations including:

* Data ingestion
* Data cleaning
* Statistical analysis
* Machine learning regression
* Time-series analysis
* Anomaly detection
* Visualization generation
* Executive reporting

Fully automated workflow.

---

## Machine Learning Methodology

The regression models are selected to ensure unbiased, accurate, and robust analysis.

Linear Regression captures global trends.

Ridge Regression improves model stability.

Polynomial Regression captures nonlinear patterns.

Random Forest Regression captures complex relationships.

Isolation Forest identifies anomalies.

PCA reduces dimensional complexity.

---

## Output Examples

The system produces:

* Correlation matrices
* Time-series trend charts
* Regression forecast charts
* PCA visualizations
* Anomaly detection plots
* Executive intelligence report

---

## Portfolio Value

This project demonstrates professional skills in:

* Data Analysis
* Machine Learning
* Statistical Modeling
* Time Series Analysis
* Data Visualization
* Business Intelligence
* Automated Reporting

Suitable for:

* Data Analyst portfolio
* Machine Learning portfolio
* Business Intelligence portfolio

---

## Use Cases

This system can be used for:

* Economic analysis
* Policy research
* Social science research
* Machine learning research
* Business intelligence reporting
* Data analytics portfolio projects

---

## Author

Sagnik Sen

Data Analyst
Machine Learning Engineer
Business Intelligence Developer

---

## License

MIT License

---

## Future Improvements

Potential enhancements include:

* Forecasting future years
* ARIMA time-series modeling
* Interactive dashboards
* Deep learning forecasting
* Web-based analytics platform

---

## Conclusion

This project represents a complete end-to-end analytics and machine learning system capable of transforming raw multi-year data into actionable insights and executive intelligence.
