# Week 02 - Exploratory Data Analysis & Correlation in ML

## Key Topics
- Variable types
- Exploratory Data Analysis (EDA)
- Correlation Analysis
- Multicollinearity

To start an EDA in pandas: `describe()` method

### Variable Types
- Quantitative variables (weight, height)
       - Continuous
       - Discrete
- Qualitative variables (name, gender)
       - Ordinal
       - Nominal

### Describe Variables
Patterns that need to be identified in the dataset.

Exploratory Data Analysis (EDA):
       1. Univariate Analysis
       2. Bi-variate Analysis
       3. Multi-variate Analysis
       4. Correlation Analysis


#### Univariate Analysis
Analyzing one variable at a time to understand its distribution.

- Skewness - having a tail (Right Skewed / Left Skewed distribution)
- Outliers - extreme values that violate the common pattern
       * exceptional cases, we have to decide whether to use that data for training
- Boxplot is used to detect outliers and distribution
- Histogram is used to visualize the distribution shape


#### Bi-variate Analysis
- Checking the connection between two variables of a dataset, and how strong that connection is.
- Scatter plot is used to visualize the relationship between two numerical variables.


#### Multi-variate Analysis
- Analyzing the relationships among three or more variables at the same time.
- Pair plots and correlation heatmaps are commonly used.


#### Correlation Analysis

Pearson correlation - a statistical approach to measure the linear correlation between two variables. Often visualized using a heatmap.

$$-1 \leq Corr(x,y) \leq 1$$


| Correlation Value | Relationship Type | Visual Meaning | Example |
| --- | --- | --- | --- |
| $+1.0$ | Strong Positive | As $X$ increases, $Y$ increases at a constant rate. | House size vs. House price |
| $0.0$ | No Correlation | $X$ and $Y$ have no linear relationship. | Shoe size vs. IQ score |
| $-1.0$ | Strong Negative | As $X$ increases, $Y$ decreases at a constant rate. | Car age vs. Resale value |

```txt

r = Corr(x,y)

Positive Correlation (r ≈ +0.95)         Negative Correlation (r ≈ -0.95)
  Y │                                      Y │
    │           *                            │ *
    │         *                              │   *
    │       *                                │     *
    │     *                                  │       *
    │   *                                    │         *
    └─────────────────────── X               └─────────────────────── X
      As X goes UP, Y goes UP                  As X goes UP, Y goes DOWN


Standard Rules of Thumb for Correlation Strength
------------------------------------------------
  |r| < 0.3  ──► Weak Correlation (Low linear dependency)
  |r| ≈ 0.5  ──► Moderate Correlation
  |r| ≥ 0.7  ──► STRONG Correlation (Benchmark for high dependency & feature redundancy)
  |r| ≥ 0.9  ──► Severe Multicollinearity (High redundancy in regression models)
  
 ```     

##### Pearson Correlation Matrix Heatmap

A correlation heatmap displays pairwise Pearson coefficients between multiple features using color gradients (typically cool-to-warm tones):

| Feature | Feature_A | Feature_B | Feature_C | Feature_D | Color Interpretation |
| --- | --- | --- | --- | --- | --- |
| Feature_A | 1.00 | 0.98 | -0.95 | -0.17 | 1.00: Perfect identity (Diagonal) |
| Feature_B | 0.98 | 1.00 | -0.93 | -0.18 | +0.98: Dark Blue / Warm Red (Strong Positive) |
| Feature_C | -0.95 | -0.93 | 1.00 | 0.17 | -0.95: Dark Red / Cool Blue (Strong Negative) |
| Feature_D | -0.17 | -0.18 | 0.17 | 1.00 | ~0.00: White / Neutral (No Relationship) |


### Multicollinearity
When two or more X (predictor) variables are correlated with each other.

*Assumptions to apply Linear Regression*

1. Normally distributed features (X, Y)
2. X variables (predictors) should be independent (no correlation) ---> No multicollinearity
3. Linear relationship between X and Y
4. Homoscedasticity (constant variance of residuals)

*Regularization methods to prevent Multicollinearity*
1. Lasso (L1)
2. Ridge (L2)
3. Elastic Net (L1 + L2)


