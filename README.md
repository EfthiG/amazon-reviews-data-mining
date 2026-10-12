# Amazon Reviews Data Mining & Machine Learning

**University project | Data Mining Techniques | Academic year 2024–2025**

An end-to-end exploration of customer reviews from the **Amazon Reviews 2023** dataset, combining exploratory data analysis, feature engineering, natural language processing, product clustering, recommendation algorithms, and sentiment classification. The work is organized into two Jupyter notebooks corresponding to the two parts of the university assignment.

## Project overview

Online product reviews contain both structured signals (star ratings, timestamps, product identifiers and prices) and unstructured feedback (review text). This project examines those signals to understand customer satisfaction, discover patterns across product categories, group related products, recommend items, and classify review sentiment.

The notebooks demonstrate the data-mining workflow, from loading and cleaning source data through feature extraction, visualization and machine-learning experiments.

## Part A — Data exploration and feature engineering

**Notebook:** `notebooks/part-a-data-exploration.ipynb`

The first part uses the Hugging Face `McAuley-Lab/Amazon-Reviews-2023` dataset and processes these five categories:

- All Beauty
- Books
- Electronics
- Toys and Games
- Home and Kitchen

The notebook is configured to request **up to 100,000 reviews per category**. It includes:

- Streaming review data from Hugging Face and exporting category-specific CSV files.
- Data cleaning, missing-value handling and text preparation.
- Rating-distribution visualizations and comparisons of average ratings across categories.
- Identification of frequently reviewed products with relatively low ratings.
- Top-five products by review count and extraction of common review keywords and phrases.
- Monthly rating trends, word clouds and exploratory plots.
- **VADER sentiment analysis**, combined with normalized star ratings to produce a weighted sentiment score (text weight **0.6**, rating weight **0.4** in the main workflow).

## Part B — Machine learning and recommendation systems

**Notebook:** `notebooks/part-b-machine-learning.ipynb`

The second part builds a more extensive pipeline using five categories:

- All Beauty
- Digital Music
- Gift Cards
- Health and Personal Care
- Magazine Subscriptions

This notebook is configured for **10,000 reviews per category**. It includes data preparation and exploratory analysis alongside the following machine-learning tasks:

### Product clustering

- Text preprocessing and **TF-IDF** vectorization.
- Combining textual and numerical product features.
- **K-Means** clustering, with elbow-method and silhouette-score analysis.
- Two-dimensional cluster visualizations using **PCA** and **t-SNE**.
- Summaries of product groups based on their attributes.

### Recommendation systems

- **User-based collaborative filtering:** finds users with similar rating patterns.
- **Item-based collaborative filtering:** finds products with similar interaction patterns.
- **Content-based filtering:** represents product descriptions with pretrained **Word2Vec** embeddings and compares products using cosine similarity.
- **Hybrid recommendations:** combines collaborative and content-based scores using configurable weights.
- Recommendation evaluation routines using prediction-error measures such as **MAE** and **RMSE**.

### Sentiment classification

- Review-text preprocessing and feature extraction with **TF-IDF** and word embeddings.
- Experiments with **Naive Bayes**, **K-Nearest Neighbors** and **Random Forest** classifiers.
- A **BERT-based** sentiment-analysis approach through Hugging Face Transformers.
- Model evaluation using cross-validation and metrics including **precision**, **recall**, **F1-score** and **accuracy**.

## Project structure

```text
amazon-reviews-data-mining/
├── README.md
├── notebooks/
│   ├── part-a-data-exploration.ipynb
│   └── part-b-machine-learning.ipynb
└── docs/
    └── Project2025PartA&B.pdf
```

The notebooks are standalone implementations with **different category selections and sample sizes**. Part B also includes its own data-loading and preprocessing steps; it does not simply import Part A's outputs.

## Technologies

- **Language / environment:** Python, Jupyter Notebook (also developed in Kaggle)
- **Data processing:** pandas, NumPy, Hugging Face `datasets`
- **Visualization:** Matplotlib, Seaborn, WordCloud
- **Natural language processing:** NLTK, VADER, Gensim Word2Vec, Hugging Face Transformers
- **Machine learning:** scikit-learn (clustering, dimensionality reduction, classification, similarity and evaluation)

## Running the notebooks

1. Clone the repository and open it in JupyterLab, Jupyter Notebook, or a compatible notebook environment such as Kaggle.
2. Install the Python dependencies used by the notebooks:

   ```bash
   pip install jupyter numpy pandas matplotlib seaborn datasets nltk vaderSentiment wordcloud scikit-learn gensim transformers torch scipy
   ```

3. Open either notebook from `notebooks/` and execute its cells. Each notebook defines its workflow and invokes `main()` in its final code cell.
4. Ensure internet access is available for Hugging Face dataset streaming and for downloading pretrained models. Some NLTK resources may also need to be downloaded when first used.
5. Run the notebook from a working directory where it can create and read its relative output folders, including `converted_reviews/` and, for Part B, `converted_metadata/`.

**Compute and storage:** These notebooks process substantial datasets and may download large pretrained embeddings/models. Execution time, memory requirements and availability of remote datasets can vary. For a quicker trial, reduce the `rows_per_category` value in the relevant notebook's `main()` function.

## Data source

**Amazon Reviews 2023**, McAuley Lab, accessed via Hugging Face:

https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023

The original review data and pretrained models are obtained from their respective providers; this repository focuses on the notebooks and assignment documentation rather than redistributing those large datasets.

## Authors

Group university assignment by:

- **Efthymia Gavgiotaki**
- **Anastasios Alevizakis**

## Academic context

Developed for the **Data Mining Techniques** course (2024–2025). The assignment is divided into **Part A: Pre-processing / Exploration and Feature Engineering** and **Part B: Learning Tasks**. See the assignment PDF in `docs/` for the original requirements.
