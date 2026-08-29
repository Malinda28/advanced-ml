# Week 01 - Introduction to AI/ML - Part 2

## Key Topics
- What is Machine Learning - Components
- ML Subdomains types
- Septs of ML workflow
- Linear Regression
-


### 1. What is Machine Learning - Components

Actual input and result
x₁ ---> y₁
x₂ ---> y₂  

### Decision Process
- The calculation or algorithm the model uses to make a prediction or guess based on input data, 
**also known froward propagation in deep leaning**

```text 
           ┌────────┐
    x₁ --> │        │-->  ŷ₁  
           └────────┘
```

### Error Function
- Calculates the model's "mistake score" by comparing its guessed Results against the true Results
```text
        ε₁ = y₁ - ŷ  // diff of actual result and predicted output

        L = ƒ(ε) // Lost function
```

### Optimization Process
-It looks at the mistake calculated by the Error Function and automatically adjusts the Decision Process so the model gets smarter and makes better guesses next time.
  *L min value is 0*

***Trained modal -The final artifact or learned program generated after running input data through the decision, error, and optimization processes.***


## ML Subdomains

* **Supervised:** Trains on **labeled data** (Inputs + True Answers). The model learns to map inputs to outputs (e.g., Spam Detection, House Price Prediction).

* **Unsupervised:** Trains on **unlabeled data** (Inputs only). The model finds hidden patterns, groupings, or structures on its own (e.g., Customer Segmentation, Clustering).

* **Semi-Supervised:** Trains on a **small amount of labeled data combined with a large amount of unlabeled data**. It balances the accuracy of supervised learning with the cost-efficiency of unsupervised data collection.

* **Reinforcement:** Trains an **agent through trial and error in an environment** using rewards and penalties. The goal is to learn an optimal strategy/action sequence (e.g., Self-Driving Cars, LLM fin-tune, Game AI like AlphaGo).

<!-- ** Diff of Error and lost function  -->

## Septs of ML workflow

1. Data Collection
2. Data Preprocessing
3. EDA : Behavior of data
4. Feature Engineering :Choose, transform, or create the most relevant input variables (features) for the model. selecting the model
5. Model Selection 
6. Training
7. Evaluation 
8. Deployment

*Training and Evaluation iterative process, Test the model's accuracy until it's gets Optimized*

## Linear Regression

A supervised learning algorithm used to predict continuous numeric values by finding the best-fitting straight line through data points.

Equation: Y = β₁X + β₀

when 
X=0 -> Y = β₀
X=1 -> Y = + β₁ + β₀

** *ML Linear regression model always assume that the data pattern is linear pattern or hyperplane.*

Y= β₁X₁+ β₂X₂ + β₃X₃ + ε

### Training Process

Imagine a dataset with 3 points ($n = 3$) where you want to predict house prices.

| Data Point ($i$) | Actual Price ($Y$) | Predicted Price ($\hat{Y}$) | 1. Error ($Y - \hat{Y}$) | 2. Squared Error $(Y - \hat{Y})^2$ | 3. Absolute Error $\vert{}Y - \hat{Y}\vert{}$
| --- | --- | --- | --- | --- | --- |
| **Point 1** | 100 | 110 | $100 - 110 = -10$ | $(-10)^2 = 100$ | $\vert{}-10\vert{} = 10$
| **Point 2** | 150 | 140 | $150 - 140 = 10$ | $10^2 = 100$ | $\vert{}10\vert{} = 10$
| **Point 3** | 200 | 195 | $200 - 195 = 5$ | $5^2 = 25$ | $\vert{}5\vert{} = 5$

 **Mean Squared Error (MSE) Formula**

$$MSE = \frac{1}{n} \sum_{i=1}^{n} (Y_i - \hat{Y}_i)^2$$

* **$n$**: Total number of data points
* **$Y_i$**: The actual (true) target value for a specific data point


* **$\hat{Y}_i$**: The model's predicted value for that same data point (e.g., from the decision process $\hat{Y} = wX + b$)


* **$\Sigma$**: Summation (add up the results for all data points)

* ***Step-by-Step Calculation Process***

  1. **Predict:** Use your model to calculate a prediction ($\hat{Y}$) for each input feature ($X$).

  2. **Calculate the Error (Residual):** Subtract the predicted value from the actual value ($Y - \hat{Y}$) to find the raw mistake for each point.

  3. **Square the Error:** Square each individual error ($(Y - \hat{Y})^2$). This serves two purposes: it makes all negative errors positive (so they don't cancel out positive errors) and it heavily penalizes larger mistakes.

  4. **Sum the Squared Errors:** Add all the squared errors together.

  5. **Calculate the Mean:** Divide the total sum by the number of data points ($n$) to get the final average loss.

* ***Worked Example Calculation***

  * **Step 4 (Sum of Squared Error):** $100 + 100 + 25 = 225$
  * **Step 5 (Mean):** $225 / 3 = 75$

**Final Output:** The Mean Squared Error (MSE) for this model's predictions is **75**. During the optimization process, the model will adjust its parameters to try and make this MSE number as close to 0 as possible.

**Mean Absolute Error (MAE)**

$$MAE = \frac{1}{n} \sum_{i=1}^{n} \vert{}Y_i - \hat{Y}_i\vert{}$$

- $n$: Total number of data points
- $Y_i$: Actual (true) target value  
- $\hat{Y}_i$: Model's predicted value  
- $\vert{}\cdot\vert{}$: Absolute value function (converts negative differences to positive)

*  ***Worked Example Calculation***

    * **(Sum of Absolute Error):**  $10 + 10 + 5 = 25$
    * **(Mean):** $25 / 3 = 8.33$


**Root Mean Square Error** 

$$RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (Y_i - \hat{Y}_i)^2} = \sqrt{MSE}$$

- $n$: Total number of data points
- $Y_i$: Actual (true) target value
- $\hat{Y}_i$: Model's predicted value  
- $\sqrt{\cdot}$: Square root function (converts squared units back to original units)

*  ***Worked Example Calculation***
   * **Sum of Squared Errors:** $100 + 100 + 25 = 225$
   * **MSE (Mean):** $225 / 3 = 75$
   * **RMSE (Square Root):** $\sqrt{75} \approx \mathbf{8.66}$


** ***Loss function is a function / relationship. of errors.***

** ***Loss function is a function of betas (model parameters).***

The goal of model training is to find the parameter set $\boldsymbol{\beta}^*$ that minimizes the loss function.
When the model finishes training, save $\boldsymbol{\beta}^*$ (the optimal weights) to build the final trained model. 

```text 

Loss (Error)
    │
    │  \                           /
    │   \                         /
    │    \  Gradient Descent     /
    │     \   (Sliding down)    /
    │      \       --->        /
    │       \                 /
Loss_min ────► * <─────────────   <-- Bottom of Curve (Minimum Error Score)
    │          │
    └──────────┼─────────────────────── β (Parameter value)
               │
               ▼
              β* (Beta Star: The parameter value that gives Loss_min)

