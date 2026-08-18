# Week 01 - Introduction to AI/ML

## Key Topics

- What is intelligence
- What is artificial intelligence (AI)
- Deterministic vs Probabilistic Relationships 
- What is Machine Learning 
- ML Subdomains (base)
- Neural networks & Deep Learning 


## 1. What is intelligence

* Intelligence is ability to learn, reason and think abstractly so that can solve problems and adapt to new situations.
    - Linguistic : ability to explain (spoken or written words, reading) via languages.
    - Logical & Mathematical : ability understand logical concepts and think in mathematics and find solutions base on reason
    - Spatial : Ability to understand environment around you (visualizing space)

## 2. What is artificial intelligence AI

* AI: Building systems that mimic human intelligence to learn, reason, and adapt from data without manual programming.

    ### Mapping of human intelligence types to AI subfields
    Linguistic -> Natural Language Processing
    Logical & Mathematical -> Symbolic AI
    Spatial -> Computer Vision

## 3. Deterministic vs Probabilistic Relationships

### Deterministic Relationships
* A known, fixed relationship between variables that can be explicitly defined or programmed using exact rules. [known function]

 Key Property: *Fully known, explicit rules with zero randomness, given input $X$, output $Y$ is always guaranteed. (ex: Physics equations $F = ma$, or logic gates : IF age >= 18 THEN adult)*

### Probabilistic Relationships (non-deterministic)
* A relationship where inputs predict the likelihood of outcomes rather than a guaranteed result, accounting for uncertainty or noise. [unknown function but guest the function]

Key Property: *Involves randomness or incomplete data; input $X$ yields output $Y$ with a probability $P(Y\vert{}X)$. (ex: weather forecasting)*

 ** **Important: The process of discovering or approximating a function $f(X) \approx Y$ between variables in a non-deterministic environment, is called training ML model**


### 4. What is Machine Learning 
* An algorithm that has learned from experience - that is, from data.
    - ML: A branch of AI focused on building systems that learn and **adapt from experience** without direct manual programming.

    Traditional Programming :  **[Data] + [Program] ==> [Result]**

    Machine Learning Approach : **[Data] + [Result] ==> [Program]**

    ***Components***
    - a decision process
    - an error function
    - an optimization process

### 5. ML Subdomains
* Subdomains are divided into the types of data that are being used to train.


### 6. Neural networks & Deep Learning 
* Artificial Neural Network is a model that is trying to imitate the human brain mathematical process.

    - Input layer : input data
    - Output layer : output result
    - Hidden layer : processing and leaning (*hidden layers are not in every Neural Network*)

* If Neural Network contents 3 or more Hidden layers is called. **Deep Neural Network** (DNNs)
    
* Deep Learning is training a Deep Neural Network.

## Big Picture
```text
┌─────────────────────┐
│         AI          │
│  ┌────────────────┐ │
│  │ Machine        │ │
│  │ Learning       │ │
│  │  ┌───────────┐ │ │
│  │  │ Neural    │ │ │
│  │  │ Networks  │ │ │
│  │  │ ┌────────┐│ │ │
│  │  │ │  Deep  ││ │ │
│  │  │ │Learning││ │ │
│  │  │ └────────┘│ │ │
│  │  └───────────┘ │ │
│  └────────────────┘ │
└─────────────────────┘
```



