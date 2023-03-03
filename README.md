# ![](https://ga-dash.s3.amazonaws.com/production/assets/logo-9f88ae6c9c3871690e33280fcf557f33.png) NLP Classification Model: r/WorldNews, r/Geopolitics

### Description


The goal of this project is to create a classification model that can predict whether a post belongs to subreddit r/WorldNews or r/Geopolitics. The model takes as input post titles and comments, learning frequency of word usage and sentiment intensity in order to make classification decisions.

The challenges posed by this classification endeavor are due to the high degree of similarity of both content and tone of the two selected subreddits. r/WorldNews and r/Geopolitics. Both subreddits heavily feature content relating to geopolitical and cross-boundary issues of the political, military, and economic variety (e.g. Russia's invasion of Ukraine, subsequent trade sanctions). Due to the large volume of shared vocabulary and theme, the "average" post on either subreddit could easily and justifiably be cross-posted onto the other subreddit without issue. This challenge renders any human attempt at classification a coin flip. As such, we sought to understand whether machine-learning models could learn what humans may find imperceivable.

*Raw data was pulled using Reddt's PushShift API.*

---

### Notebook Overview

**[01_data_collection.ipynb](https://git.generalassemb.ly/elee190/project-3/blob/main/code/01_data_collection.ipynb)**

- A retriever function pulls data from Reddit's PushShift API, intaking subreddits and iterations as parameters. Raw data is saved to '../data/raw/' in .csv format.

**[02_EDA_preprocessing.ipynb](https://git.generalassemb.ly/elee190/project-3/blob/main/code/02_EDA_preprocessing.ipynb)**

- PushShift API provides a variety of metadata that is not useful for our classification model. Columns containing non-useful data types are dropped.
- There is a prevalence of non-English and emoji-laden posts that do not appear on the front-end of the subreddit website. These posts are generally AI-generated advertisements that are removed by moderators, but still appear when pulling data from the back-end. A non_ascii_dropper function identifies the ratio of non-ascii (non-English and emoji) characters that appear in posting, and drops the data if it does not meet a threshold ratio.
- Classification column is added, with our positive class indicating r/worldnews.
- A sentiment intensity score is added to each observation using NLTK Vader SentimentIntensityAnalyzer.

**[03_sentiments_model.ipynb](https://git.generalassemb.ly/elee190/project-3/blob/main/code/03_sentiments_model.ipynb)**

- Simple logistic regression models are run on sentiment intensity scores.
- Nothing notable is discovered here, sentiment intensity has little bearing on predictive power and performs similarly to a null model.

**[04_nlp_model.ipynb](https://git.generalassemb.ly/elee190/project-3/blob/main/code/04_nlp_model.ipynb)**

- Grid-search to identify optimal hyperparameters for various pipelines with a single transformer (CountVectorizer, TfidfVectorizer) and a single estimator (Logistic, Random Forest, Extra Trees, Gradient Boost). 

**[05_stacked_model.ipynb](https://git.generalassemb.ly/elee190/project-3/blob/main/code/05_stacked_model.ipynb)**

- Models with highest performance from 04_nlp_model.ipynb are combined into stacked and voting classifiers, using optimized hyperparameters identified in grid-search.

**[06_analysis.ipynb](https://git.generalassemb.ly/elee190/project-3/blob/main/code/06_analysis.ipynb)**

- Visuals construction for presentation purposes.
- Extraction of relevant unigrams, bigrams, and trigrams for presentation purposes.

**[_drafts.ipynb](https://git.generalassemb.ly/elee190/project-3/blob/main/code/_drafts.ipynb)**

- General "scratch" sheet.

---

### Methodology/Inferences/Assumptions

Due to the shared context between the two subreddits, we first sought to identify whether there were any non-language features that we could add to our model. Initial modeling was based on Sentiment Intensity (NLTK: Vader) scores. It was apparent that there was nothing a model could learned fromt sentiment scores alone that would allow it to outperform a baseline null model. As such, sentiment scores were not included in subsequent models.

Language processing began by comparing various pipelines. Each pipeline used either CountVectorizer or TfidfVectorizer as its initial transformer, before using either Logistic Regression, Random Forest classification, Extra Trees Random Forest classification, or Gradient Boost Classification as its estimator. Various hyperparameters for each pipeline (transformer and estimator) were tested via grid-search, and final scores were calculated on unseen data. The highest performing models and their associated hyperparameters were then combined into a stacked model, where the model achieved its highest performance to date.

Our stacked model's accuracy of 86.4% (baseline 61.1%) indicates that it has learned a signicant volume of patterns to assist in classification.

Unfortunately, our initial data sample size consists of sub ~2,000 observations. This was due to an issue with the PushShift API's server migration. However, our notebooks are organized in a programmatic fashion that will allow an easily repeatable analysis given the resolution of the API server issue.

As a result of the API issue, our observations have slightly imbalanced classes (~61/~39) split. While we did not seek to balance the classes via oversampling, undersampling, or synthetic data creation, these are all avenues that can be further explored.

---

### Conclusions/Recommendations

- Re-run models after creating synthetic data points to deal with imbalanced classes
- Increased observations would make our model more robust
- Further exploration of sentiment score variants
- Train and test model on comments data and aggregated comments-posts data

---
