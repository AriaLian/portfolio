+++
title = "Player Retention Prediction"
summary = "Applying Random Forests and Support Vector Machines (SVM) for player retention prediction."
description = ""
featuredImage = ""
tags = ["SVM", "Random Forests"]
categories = ["AI"]
collections = [""]
weight = 3
draft = false
+++

## Abstract

The project intends to apply machine learning approaches to forecast player online gaming behavior, ascertain which aspects most importantly affect player retention, and find which classification method best fits in estimating engagement levels. Using a dataset comprising player demographics, in-game behavior measures, and engagement results, we will apply **Random Forests** and **Support Vector Machines (SVM)** to identify trends in the data and generate accurate projections for player retention.​​​​​​​

{{< button href="https://colab.research.google.com/drive/17WBf2OMGxyWT5ShJ4F2C8iJHr0aXvD3y" target="_blank" color="color-colab" >}}
{{< icon "colab" >}} View on Google Colab
{{< /button >}}

## Introduction

Player retention and engagement are crucial in the online gaming industry. Retaining existing players is often more cost effective than acquiring new ones. By exploring patterns in online gaming behavior and predicting player engagement levels, game developers can optimize the player experience and marketing strategies in order to increase the overall retention rates. 

### Objective

The objective of this project is to apply machine learning approaches to predict player engagement levels in online games. Specifically, the project aims to: 

- Identify and analyze key features that influence player retention and engagement.
- Develop and evaluate classification models, such as Support Vector Machines (SVM) and Random Forests to predict engagement levels (High, Medium, Low).
- Determine which classification model performs better through a comprehensive performance metrics evaluation.

### Scope

This work focuses on a synthetic dataset comprising demographic and game-specific information, in-game behavioral measures, and engagement statistics. This study, however, is limited to the current dataset and might not fully reflect real-world complexity including social interactions or outside variables impacting involvement. With possible improvements in future implementations, the results are thus scoped to offer fundamental insights for comparable real-world applications.

### Hypothesis

Higher engagement levels are more likely of players with more gameplay frequency, longer session length, and more in-game progress. Furthermore influencing involvement could include demographic and game-specific elements including age, game type, and challenge. 

### Research Questions

1. What are the most important features that influence player engagement in online games?
2. How accurately can machine learning models classify players into different engagement levels?
3. Which classification algorithm (Support Vector Machines (SVM) or Random Forest) is better at predicting player engagement?
4. What insights can be gained from the analysis to improve player retention strategies?

## Data Description

### Dataset Source

The dataset used in this project was obtained from Kaggle: [Predict Online Gaming Behavior Dataset](https://www.kaggle.com/datasets/rabieelkharoua/predict-online-gaming-behavior-dataset/data). It contains a variety of features describing player characteristics and behavior, along with a target variable indicating the engagement level. 

### Dataset Structure

To effectively predict player engagement levels, we begin by examining and extracting relevant features from the dataset. The features can be grouped into player demographics, game-related metrics, player usage patterns and interaction metrics. Below is an overview of the columns in the dataset: 

1. **PlayerID**: Unique identifier for each player (excluded from modeling).
2. **Age**: Age of the player.
3. **Gender**: Gender of the player (Male/Female).
4. **Location**: Region of the player (e.g., USA, Europe, Asia).
5. **GameGenre**: The genre of game the player is playing (e.g., Strategy, Sports, Action, RPG, Simulation).
6. **GameDifficulty**: Level of difficulty of the game (Easy, Medium, Hard).
7. **PlayTimeHours**: Total hours spent playing.
8. **SessionsPerWeek**: Average number of game sessions per week.
9. **AvgSessionDurationMinutes**: Average duration of a game session (in minutes).
10. **InGamePurchases**: Indicates if the player has made any in-game purchases (Yes/No).
11. **PlayerLevel**: Current progression level of the player.
12. **AchievementsUnlocked**: Number of achievements unlocked by the player.
13. **EngagementLevel**: Target variable indicating the engagement category (Low, Medium, High).

```py
# Load the dataset
df = pd.read_csv('online_gaming_behavior_dataset.csv')
df.head()
```

![](dataset1.png)
![](dataset2.png)

```py
df.info()
```
```
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 40034 entries, 0 to 40033
Data columns (total 13 columns):
 #   Column                     Non-Null Count  Dtype  
---  ------                     --------------  -----  
 0   PlayerID                   40034 non-null  int64  
 1   Age                        40034 non-null  int64  
 2   Gender                     40034 non-null  object 
 3   Location                   40034 non-null  object 
 4   GameGenre                  40034 non-null  object 
 5   PlayTimeHours              40034 non-null  float64
 6   InGamePurchases            40034 non-null  int64  
 7   GameDifficulty             40034 non-null  object 
 8   SessionsPerWeek            40034 non-null  int64  
 9   AvgSessionDurationMinutes  40034 non-null  int64  
 10  PlayerLevel                40034 non-null  int64  
 11  AchievementsUnlocked       40034 non-null  int64  
 12  EngagementLevel            40034 non-null  object 
dtypes: float64(1), int64(7), object(5)
memory usage: 4.0+ MB
```

### Data Cleaning and Preprocessing

```py
# Separate features and label
X = df.iloc[:, 1:12] # Drop the 'PlayerID' column
y = df.iloc[:, -1]

features = df.drop(columns=['EngagementLevel'])
target = df['EngagementLevel']

# Extract categorical and numerical features
categorical_features = X.select_dtypes(include=['object']).columns
numerical_features = X.select_dtypes(exclude=['object']).columns
```

#### Handling Missing Values

Columns with missing values were identified and addressed:
- Numerical features: Missing values were imputed using median values.
    ```py
    # Impute missing values in numerical features
    imputer = SimpleImputer(strategy='median')
    X_encoded[numerical_features] = imputer.fit_transform(X_encoded[numerical_features])
    ```
- Categorical features: Missing values were replaced using the most frequent value or a separate "unknown" category.

#### Feature Scaling and Normalization

Continuous variables (e.g., **`PlayTimeHours`**, **`SessionsPerWeek`**, **`AvgSessionDurationMinutes`**, **`PlayerLevel`**, **`AchievementsUnlocked`**) were standardized using z-score normalization `StandardScaler` to ensure consistent scaling for the models.

```py
# Standardize numerical features
scaler = StandardScaler()
X_encoded[numerical_features] = scaler.fit_transform(X_encoded[numerical_features])
```

#### Encoding Categorical Data

- **One-Hot Encoding**: Applied to features like **`Gender`**, **`Location`**, **`GameGenre`**, and **`GameDifficulty`** to convert them into numerical format.
    ```py
    # One-Hot Encode categorical features
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded_cats = encoder.fit_transform(X[categorical_features])

    # Convert encoded categories to DataFrame and concatenate
    encoded_cats_df = pd.DataFrame(encoded_cats, columns=encoder.get_feature_names_out(categorical_features))
    X_encoded = pd.concat([X[numerical_features], encoded_cats_df], axis=1)
    ```
- **Label Encoding**: Used for the target variable **`EngagementLevel`** (Low = 0, Medium = 1, High = 2).

### Exploratory Data Analysis (EDA)

#### Statistical Summaries

Descriptive statistics (mean, median, standard deviation) for numerical variables can provide insights into central tendency and variability:

![](EDA1.png)
![](EDA2.png)

#### Visualizations

1. **Histograms**: We use Seaborn **`sns.countplot()`** to create a bar plot showing the distribution of targets.
    ```py
    # Visualize target distribution
    sns.countplot(x=target)
    plt.title('Target Distribution')
    plt.show()
    ```
    ![](targets.png)

    This plot helps quickly identify the distribution of categories within the target variable, giving insight into class imbalance.

2. **Count Plots**: We use Pandas `pd.crosstab()` to generate contingency tables for each categorical feature against the target. Each table shows how each level of the categorical variable relates to the target variable.
    ```py
    # Make contingency tables for each categorical column against the target column
    categorical_features_graph = features.select_dtypes(include=['object'])
    fig, axes = plt.subplots(2, 2, figsize=(15, 15))
    for i, feature in enumerate(categorical_features_graph):
        pd.crosstab(categorical_features_graph[feature], target).plot(kind='bar', stacked=True, ax=axes[i//2, i%2])
        axes[i//2, i%2].set_title(f'{feature} vs EngagementLevel')
    plt.show()
    ```
    ![](tables.png)

    Stacked bar charts of categorical features help show how each feature category distributes across the target variable’s classes.

3. **Correlation Heatmap**: To visualize the correlation matrix as a heatmap, we use `sns.heatmap()` to reveal linear relationships between the numerical features with a color gradient. The heatmap shows both the magnitude and direction (positive/negative) of relationships.
    ```py
    # Calculate the correlation matrix
    corr_matrix = X_encoded[numerical_features].corr()

    # Create a heatmap of the correlation matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5) 
    plt.title('Correlation Heatmap')
    plt.show()
    ```
    ![](Heatmap.png)

4. **Pairplot**: Creates scatter plots for each pair of numerical features. The diagonal plots show the distributions and it allows for the coloring of the points by engagement levels.
    ```py
    # Create a pairplot of the dataset
    player = pd.concat([X, y], axis=1)
    sns.pairplot(player, hue='EngagementLevel')
    plt.show()
    ```
    ![](featured.png)

## Methodology

### Problem Definition

The dataset includes various player-specific behavioral and demographic features. The target variable, **EngagementLevel**, is a categorical variable with three classes: **High**, **Medium**, and **Low**. Our goal is to map the input feature set {{< katex >}} \(X=\{x_1,x_2,…,x_n\}\) to the output variable {{< katex >}} \(Y=\{y_1,y_2,y_3\}\), where each {{< katex >}} \(𝑥_𝑖\) represents a feature extracted from player behavior data, and {{< katex >}} \(y_1,y_2,y_3\) represent the three levels of engagement.

Given the input features {{< katex >}} \(X\), we need to find a hypothesis {{< katex >}} \(h(X)\) that minimizes the classification error between the predicted engagement level {{< katex >}} \(\hat y\) and the actual engagement level {{< katex >}} \(y\). This problem is therefore formulated as a multi-class classification task. 

### Support Vector Machines (SVM)

![](SVM.png)

Support Vector Machines (SVM) are margin-based classifiers that aim to find an optimal decision boundary separating different classes. Given a set of training samples {{< katex >}} \((x_i,y_i)\), SVM seeks to find a hyperplane defined as:
$$
f(x)=w^Tx+b
$$
that maximizes the margin between classes while minimizing classification errors.

For multi-class classification, SVM typically employs a one-vs-rest strategy, where multiple binary classifiers are trained. The optimization objective for a linear SVM can be written as:
$$
\min_{w, b} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^{N} \xi_i
$$
subject to:
​$$
y_i (w^T x_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0
$$
where:
- {{< katex >}} \(\xi_i\) are slack variables allowing misclassification
- {{< katex >}} \(C\) is a regularization parameter controlling the trade-off between margin maximization and classification error

In this project, a **linear kernel** was used because LDA projects the data into a lower-dimensional space where classes are approximately linearly separable. The linear kernel also offers better interpretability and lower computational cost compared to nonlinear kernels.

### Random Forest

Random Forest is an ensemble learning method that combines multiple decision trees to improve classification performance and robustness. Each decision tree is trained on a bootstrap sample of the data, and at each split, a random subset of features is considered. For classification tasks, the output of the random forest is the class chosen by most trees. 

![](RF.png)

The training algorithm for random forests applies the general technique of **bootstrap aggregating**, or bagging, to tree learners. Given a training set {{< katex >}} \(X = \{x_1, ..., x_n\}\) with responses {{< katex >}} \(Y = \{y_1, ..., y_n\}\), bagging repeatedly ({{< katex >}} \(B\) times) selects a random sample with replacement of the training set and fits trees to these samples:

For {{< katex >}} \(b = 1, ...,B\):

1. Sample, with replacement, {{< katex >}} \(n\) training examples from {{< katex >}} \(X\), {{< katex >}} \(Y\); call these {{< katex >}} \(X_b\), {{< katex >}} \(Y_b\).
2. Train a classification tree {{< katex >}} \(f_b\) on {{< katex >}} \(X_b\), {{< katex >}} \(Y_b\).

After training, predictions for unseen samples {{< katex >}} \(x'\) can be made by averaging the predictions from all the individual regression trees on {{< katex >}} \(x'\):

$$
\hat{f}=\frac 1B ∑_{b=1}^B f_b(x′)
$$

or by taking the plurality vote in the case of classification trees.

### Dimensionality Reduction

Dimensionality reduction is crucial for simplification of the dataset. We want to reduce noise and duplicated information by maintaining the basic variance in the data by dimension count reduction. 

We reduced the dataset dimensions using **Linear Discriminant Analysis (LDA)**. Unlike unsupervised methods such as Principal Component Analysis (PCA), LDA is a **supervised dimensionality reduction** method that explicitly incorporates class labels. It ensures that samples from different engagement levels are well separated in the reduced feature space. Since the target variable contains three classes, the maximum number of LDA components is {{< katex >}} \(C−1=2\). So we transformed the data into a two-dimensional space (by specifying **`n_components=2`**).

```py
# Perform LDA on the standardized data
n_components = 2
lda = LDA(n_components=n_components)
X_reduced = lda.fit_transform(X_encoded, y)

# Use KMeans to visualize the clusters
n_clusters = len(np.unique(y))

plt.figure(figsize=(5, 5))
kmeans = KMeans(n_clusters=n_clusters, n_init=n_clusters, random_state=42)
clusters = kmeans.fit_predict(X_reduced)
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=clusters, cmap='jet', alpha=0.5)
plt.title('KMeans Clustering')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.show()
```

To qualitatively assess class separability, K-Means clustering was applied to the LDA-transformed data, and the resulting clusters were visualized in two-dimensional space.

![](KMeans.png)

### Model Training

#### Train-Test Split Strategy

To assess the models' performance on unprocessed data, the dataset was divided in training and testing sets in an 80–20 ratio. This divide lowers the over-fitting risk and guarantees sufficient data for testing and training.

```py
# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X_reduced, y, test_size=0.2, random_state=42)

# One-hot encode the labels
encoder = OneHotEncoder(sparse_output=False)
y_train_encoded = encoder.fit_transform(y_train.values.reshape(-1, 1))
y_test_encoded = encoder.transform(y_test.values.reshape(-1, 1))
```

#### Support Vector Machines (SVM)

For Support Vector Machines (SVM), two key hyperparameters can be adjusted. These adjustments aimed to find a balance between model complexity and generalization ability.

- **C (Regularization Parameter)**: Controls the trade-off between minimizing training error and maximizing the margin's width. Here, we used the default value `C = 1.0`, providing a balanced trade-off between margin maximization and classification error.
- **Kernel**: We used a `linear` kernel since the dataset's reduced dimensions were linearly separable.

```py
# Classify the data using SVM
classifier = SVC(kernel="linear", probability=True)
classifier.fit(X_train, y_train.ravel())
y_pred_svm = classifier.predict(X_test)
y_score_svm = classifier.decision_function(X_test)
```

#### Random Forest

For the **Random Forest** model, several hyperparameters can be fine-tuned.

- **Number of Trees**: Determines the number of decision trees in the forest.
- **Max Depth**: Controls the maximum depth of the trees.
- **Min Samples Split**: Specifies the minimum number of samples required to split a node.

```py
# Train a Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred_rf = model.predict(X_test)
y_score_rf = model.predict_proba(X_test)
```
Here, default hyperparameters were used to establish a baseline ensemble model. This choice reduces model complexity and mitigates the risk of overfitting, while still benefiting from the robustness and variance reduction properties of ensemble learning.

### Evaluation Metrics

The performance of both models was evaluated using the following metrics:

1. **Accuracy**: Proportion of correctly predicted engagement levels out of total predictions.
2. **Precision**: Ability of the model to identify true positives for each engagement level.
3. **Recall**: Ability of the model to capture all relevant instances of each engagement level.
4. **F1-Score**: Harmonic mean of precision and recall, providing a balanced metric.
5. **Confusion Matrix**: A detailed breakdown of the model's predictions for each class.
6. **ROC Curve & AUC**: Evaluates model performance using true/false positive rates.

### Tools and Libraries

The project used Python and the following libraries:

- **Data Handling and Visualization**:
    - **NumPy** and **Pandas**: Handle and preprocess data.
    - **Matplotlib** and **Seaborn**: Create statistical plots for data visualization.
- **Preprocessing and Dimensionality Reduction**:
    - **StandardScaler**, **SimpleImputer**, and **OneHotEncoder**: Standardize, impute missing values, and encode categorical features.
    - **LDA**: Reduce feature dimensions while preserving class separability.
- **Modeling and Algorithms**:
    - **SVC**: Implements support vector classification for linear and non-linear data.
    - **RandomForestClassifier**: Uses decision tree ensembles for robust classification.
    - **KMeans**: Groups data into clusters based on feature similarities.
- **Model Evaluation and Metrics**:
    - **Accuracy Score**: Calculates the ratio of correct predictions.
    - **Classification Report**: Summarizes precision, recall, and F1-score.
    - **Confusion Matrix**: Shows a breakdown of correct and incorrect predictions.
    - **ROC Curve & AUC**: Evaluates model performance using true/false positive rates.
    - **Label Binarize**: Converts multi-class labels into binary format for evaluation metrics.

## Results and Discussion

### Model Performance

The two models, **Support Vector Machines (SVM)** and **Random Forest**, were trained on the dataset, and their performance was evaluated on the test set. Below are the results of the evaluation metrics for each model.

1. **Support Vector Machines (SVM)**
    - **Classification Accuracy**: **82.30%**
    - **Classification Report**:
        ```
        Classification Report for SVM:
                    precision    recall  f1-score   support

                High       0.88      0.84      0.86      2035
                 Low       0.80      0.71      0.75      2093
              Medium       0.81      0.88      0.84      3879

            accuracy                           0.82      8007
           macro avg       0.83      0.81      0.82      8007
        weighted avg       0.82      0.82      0.82      8007
        ```
    

2. **Random Forest**
    - **Classification Accuracy**: **81.29%**
    - **Classification Report**:
        ```
        Classification Report for Random Forest:
                    precision    recall  f1-score   support

                High       0.86      0.84      0.85      2035
                 Low       0.78      0.70      0.74      2093
              Medium       0.80      0.86      0.83      3879

            accuracy                           0.81      8007
           macro avg       0.82      0.80      0.81      8007
        weighted avg       0.81      0.81      0.81      8007
        ```

### Comparison of Models

The confusion matrices and classification metrics for **Support Vector Machines (SVM)** and **Random Forest** reveal how well each model classified engagement levels (High, Medium, and Low). These insights provide a deeper understanding of the strengths and weaknesses of each approach.

| Metric | SVM | Random Forest |
| --- | --- | --- |
| Accuracy | 82.30% | 81.29% |
| Precision (High) | 88% | 86% |
| Precision (Medium) | 81% | 80% |
| Precision (Low) | 80% | 78% |
| Recall (High) | 84% | 84% |
| Recall (Medium) | 88% | 86% |
| Recall (Low) | 71% | 70% |
| F1-Score (High) | 86% | 85% |
| F1-Score (Medium) | 84% | 83% |
| F1-Score (Low) | 75% | 74% |

### Visual Representation

#### Confusion Matrices

The confusion matrices reveal the classification accuracy for each engagement level.

1. **SVM Confusion Matrix**
    
    The SVM confusion matrix shows noticeable misclassifications, particularly between Low and Medium engagement levels. Medium engagement had relatively better classification with fewer misclassifications compared to other levels.
    
    ![](Matrix1.png)
    
    - High engagement was sometimes misclassified as Medium (263 instances).
    - Low engagement was more frequently confused with Medium (548 instances).
    - Medium engagement had relatively fewer misclassifications but some overlap with Low (310 instances) and High (174 instances).
2. **Random Forest Confusion Matrix**
    
    The Random Forest confusion matrix demonstrates better performance overall, particularly in distinguishing Medium engagement from other levels. Misclassifications are fewer than in SVM, especially for Medium and High engagement.
    
    ![](Matrix2.png)
    
    - High engagement was occasionally confused with Medium (250 instances).
    - Low engagement was frequently misclassified as Medium (563 instances).
    - Medium engagement showed better classification than SVM, with fewer overlaps with Low (344 instances) and High (206 instances).

#### ROC Curves

ROC curves visualize the model's ability to separate classes, with the area under the curve (AUC) indicating the model's effectiveness.

1. **SVM ROC Curve**
    
    ![](ROC1.png)
    
    - AUC for Low (Class 0): 0.94
    - AUC for Medium (Class 1): 0.91
    - AUC for High (Class 2): 0.85
    The SVM ROC curve highlights strong performance for Low engagement levels, but its performance slightly declines for High engagement.
2. **Random Forest ROC Curve**
    
    ![](ROC2.png)
    
    - AUC for Low (Class 0): 0.93
    - AUC for Medium (Class 1): 0.89
    - AUC for High (Class 2): 0.90
    The Random Forest ROC curve shows more balanced performance across all engagement levels, particularly for Medium and High engagement.

### Analysis of Results

1. **SVM**:
    - Performed well in distinguishing engagement levels but struggled slightly with overlapping classes.
    - High AUC for Class 0 and Class 2 demonstrates its effectiveness in separating those levels.
2. **Random Forest**:
    - Outperformed SVM across almost all metrics, achieving higher accuracy and F1-scores.
    - Strong ensemble learning approach allowed it to handle the complex relationships between features effectively.

#### Insights Gained

- Players with high **`PlayTimeHours`** and **`AchievementsUnlocked`** were more likely to be classified as highly engaged (Class 2).
- The distinction between Low (Class 0) and Medium (Class 1) engagement was the most challenging for both models.

#### Challenges Faced

1. **Class Overlap**: Medium engagement levels often overlapped with Low and High levels, leading to some misclassifications.
2. **Feature Interactions**: Correlated features required careful preprocessing to avoid redundancy.

The Random Forest model demonstrated better performance, making it the recommended model for predicting player engagement in this dataset.

## Conclusion and Future Work

### Summary of Findings

This project successfully applied machine learning techniques to predict player engagement levels in online gaming. Key findings from the analysis include:

- **Feature Importance**: Player engagement levels are significantly influenced by behavioral metrics like **`PlayTimeHours`** and **`AchievementsUnlocked`**.
- **Model Performance**: The Random Forest model outperformed Support Vector Machines (SVM) across key evaluation metrics, achieving higher accuracy, precision, recall, and F1-scores.
- **Insights for Engagement**: High engagement levels were associated with players exhibiting frequent and longer gameplay, advanced achievements, and a preference for challenging games.


### Limitations

Despite the success of the models, the project faced several limitations:

- **Synthetic Dataset**: The use of a synthetic dataset limits the real-world applicability of the results, as it may not capture the full complexity of actual player behaviors.
- **Class Overlap**: Medium engagement levels overlapped with Low and High levels, leading to some misclassifications in both models.
- **Feature Scope**: The dataset lacked some potentially impactful features, such as social interactions or psychological motivations, which could further enhance prediction accuracy.

### Future Work

To build upon the findings of this project, the following improvements and extensions are proposed:

1. **Integration of Real-World Data**: Applying the models to real-world player datasets to validate and refine their performance.
2. **Additional Features**: Incorporating features such as player communication patterns, game updates, and external factors (e.g., seasonal trends) to improve predictive capabilities.
3. **Advanced Models**: Exploring other machine learning approaches, such as Gradient Boosting Machines (e.g., XGBoost, LightGBM) or deep learning techniques for enhanced performance.
4. **Dynamic Prediction**: Developing a dynamic engagement prediction model that adapts to real-time player behavior changes.
5. **Player Segmentation**: Combining engagement prediction with clustering algorithms to segment players into actionable categories for targeted interventions.

In conclusion, the project successfully demonstrated the feasibility of predicting player engagement levels using machine learning, laying the groundwork for future advancements in player behavior analysis.

## References

1. Support vector machine. (2024, November 23). In *Wikipedia*. https://en.wikipedia.org/wiki/Support_vector_machine
2. Hofmann, T., Schölkopf, B., & Smola, A. J. (2007). Kernel methods in machine learning. *ArXiv*. https://doi.org/10.1214/009053607000000677
3. *Kernel method*. (n.d.). Engati. Retrieved November 25, 2024, from https://www.engati.com/glossary/kernel-method
4. *Support Vector Machines*. (n.d.). Retrieved November 25, 2024, from https://web.archive.org/web/20181012163919/http://svms.org/
5. Random forest. (2024, October 2). In *Wikipedia*. https://en.wikipedia.org/wiki/Random_forest
6. Breiman, L. Random Forests. *Machine Learning* **45**, 5–32 (2001). https://doi.org/10.1023/A:1010933404324
7. Dimensionality reduction. (2024, October 26). In *Wikipedia*. https://en.wikipedia.org/wiki/Dimensionality_reduction
8. Van Der Maaten, L., Postma, E. O., & Van Den Herik, H. J. (2009). Dimensionality reduction: A comparative review. *Journal of machine learning research*, *10*(66-71), 13.
9. Holtel, Frederik (2023-02-20). "Linear Discriminant Analysis (LDA) Can Be So Easy". Medium. Retrieved 2024-05-18.