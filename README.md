# Fake-News-Detection-in-R

"**Benchmarking Classical Models for Misinformation Detection Using Simple Text Features**" is a systematic study aimed at evaluating and comparing several classical machine learning models for classifying news articles as "FAKE" or "REAL". The core goal of the research was to assess how different feature representations, particularly **simple structural features** (like text length and title word count) versus **full text-based semantic features**, influence the predictive performance of classification models.

### Key Stages and Methodology

The project followed a structured five-stage architecture: data ingestion and cleaning, text preprocessing, feature extraction, model training, and evaluation.

1.  **Data Preparation and Cleaning:** The project used a dataset of 7,797 news articles. Data cleaning involved removing duplicate rows, corrupted or empty text fields, and standardizing the labels to the two valid classes: "FAKE" and "REAL". The dataset was confirmed to have a nearly equal distribution of FAKE and REAL articles, ensuring accuracy was an appropriate metric.
2.  **Feature Engineering:** The study investigated two main types of features:
    *   **Structural Features:** These included **article text length** (number of characters) and **title word count**.
    *   **Full Text-Based Semantic Features:** The `title` and `text` were concatenated into `full_text`. This text was processed using a full Natural Language Processing (NLP) pipeline, including lowercasing, removal of punctuation, numbers, and English stopwords, and **lemmatization**.
3.  **DTM Creation (TF_IDF):** The processed text was converted into a high-dimensional **Document-Term Matrix (DTM)**, serving as the semantic representation. Sparse terms were pruned (using a 0.95 sparsity threshold) to improve computational efficiency.
4.  **Model Training and Evaluation:** Four supervised machine learning models were trained on the DTM features: **Decision Tree**, **Naive Bayes**, **Random Forest**, and **Logistic Regression (GLMNET)**. A stratified 80/20 train/test split was used to maintain class balance during evaluation.

### Core Findings and Contribution

The project demonstrated a clear advantage of using semantic features over structural features alone:

*   **Structural Features Only:** Models trained solely on text length and title word count achieved only **moderate performance**, with accuracies ranging between **0.78 and 0.9045**.
*   **Full Text-Based Features:** Incorporating the high-dimensional DTM features derived from the NLP pipeline led to **substantial accuracy gains** (+20–30%) for all algorithms.
    *   **Random Forest** achieved the **highest test accuracy at 0.9045**.
    *   **Logistic Regression** (using elastic-net regularization) followed closely with an accuracy of **0.8981**.

The findings confirm that the **semantic structures** embedded in the textual content—not just simple length characteristics—are essential for effective fake news detection. The study contributes a systematic comparison of feature engineering strategies, providing evidence that classical machine learning models, when combined with robust text preprocessing, can achieve high accuracy in this domain.

##Pipline

The project's, fake news detection, framework follows a **five-stage architecture** designed to transform raw textual data into a high-dimensional semantic representation suitable for machine learning classification.

This overall workflow is illustrated in **Fig. 1. Complete Fake News Detection Architecture Using NLP and Machine Learning**.
<img width="730" height="276" alt="image" src="https://github.com/user-attachments/assets/58da66eb-9593-42b3-a829-29d0d236382f" />


### 1. Data Ingestion and Cleaning (A. Data Ingestion and Cleaning)

This stage involves preparing the raw dataset for analysis:

*   **Import and Selection:** The raw dataset, which contains article titles, full text, and FAKE/REAL labels, is imported. Only valid entries are retained.
*   **Cleaning:** Procedures are implemented to remove duplicate rows, corrupted or empty text fields, and inconsistent labels.
*   **Standardization:** All labels are standardized to uppercase and restricted to the two valid classes, "FAKE" and "REAL".
*   **Text Consolidation:** A new variable, `full_text`, is constructed by concatenating the article title and body to form a unified text input for subsequent processing.

### 2. Text Preprocessing (B. Text Preprocessing)

A comprehensive Natural Language Processing (NLP) pipeline is applied to the consolidated text to normalize and clean the corpus. This stage ensures the resulting vocabulary is clean, normalized, and linguistically meaningful:

*   **Lowercasing:** All tokens are converted to lowercase.
*   **Removal of Noise:** Punctuation, numbers, and extra whitespace are removed.
*   **Stopword Elimination:** English stop words are removed using the English stopword list.
*   **Lemmatization:** Tokens are reduced to their base or dictionary form.

### 3. Feature Extraction (C. Feature Extraction via DTM)

The cleaned text is converted into a numerical format for modeling:

*   **DTM Construction:** The preprocessed corpus is converted into a high-dimensional **Document-Term Matrix (DTM)**, where rows are articles and columns are unique lemmatized tokens.
*   **Term Constraints:** Terms appearing fewer than five times globally are removed.
*   **Sparsity Reduction:** Sparse terms are pruned using a **0.95 sparsity threshold** to improve computational efficiency.
*   **Final Input:** This resulting feature matrix captures vocabulary usage patterns characteristic of FAKE and REAL news and serves as the primary input for the models.

### 4. Model Training and Development (D. Model Development)

The project trains four supervised learning models on the DTM features:

*   **Data Split:** A **stratified 80/20 split** ensures both classes remain balanced across the training and testing sets.
*   **Model Training:** The four models are trained using specific control parameters:
    *   **Decision Tree:** Trained using Gini impurity and pruned based on the optimal complexity parameter (CP).
    *   **Naive Bayes:** Trained with Laplace smoothing to handle zero-frequency terms.
    *   **Random Forest:** Trained with optimized `mtry` and 100 decision trees.
    *   **Logistic Regression (GLMNET):** Trained with elastic-net regularization.
 
Decision Tree and Random Forest's plots are illustrated in fig. 6 and Fig. 8
<img width="574" height="334" alt="image" src="https://github.com/user-attachments/assets/aeda89ba-830d-4e1c-83e6-6464d0da1b19" />
<img width="574" height="334" alt="image" src="https://github.com/user-attachments/assets/0f741b13-4179-44e9-94ea-acbbc07b2ee4" />



### 5. Evaluation and Comparison (E. Evaluation and Comparison)

The models' performance is quantified and compared:

*   **Prediction:** Each model generates class predictions on the held-out test set.
*   **Metrics:** Performance is evaluated using confusion matrices and metrics including accuracy, precision, recall, specificity, and Cohen’s kappa.
*   **Comparison:** The accuracy of all four models is compared visually using charts (in Fig. 9 and Fig. 10).
  <img width="574" height="334" alt="image" src="https://github.com/user-attachments/assets/1b11e90e-f0d6-4906-a114-3eac9fba6a39" />


created and maintained by @aishaisunreal
