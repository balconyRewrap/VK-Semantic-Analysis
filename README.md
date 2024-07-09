# Program for Sentiment Analysis of Posts and Comments on VKontakte Using a Neural Network Written in Python

## Neural Network for VKontakte Analysis

### ``Python Dependencies for Training the Neural Network:``
**Libraries:**
1) keras
2) numpy
3) sklearn
4) pickle
5) scipy
6) tensorflow
7) nltk
8) h5py

**Installation Command:**
```pip install keras numpy scikit-learn pickle scipy tensorflow nltk h5py```

### ``Datasets Used for Training the Neural Network``
**Primary Datasets:**
1) [RuTweetCorp (Rubtsova, 2013)](http://study.mokoron.com/)
2) [RuReviews (Smetanin and Komarov, 2019)](https://ieeexplore.ieee.org/document/8807792)
3) [Russian Language Toxic Comments (2ch.hk and Pikabu.ru)](https://www.kaggle.com/datasets/blackmoon/russian-language-toxic-comments)
4) [PolSentiLex 2015 and 2016](https://linis-crowd.org/)

**Supplementary Datasets:**
1) [Dictionary of Obscene Words](https://mat2.slovaronline.com/)
The dataset isn't provided; had to parse it manually.
2) [List of Russian Obscene Words for Banning](https://github.com/bars38/Russian_ban_words)

### ``Dependencies for Training and Using the Neural Network:``
**If training on CPU:**
1) [HDF5](https://www.hdfgroup.org/downloads/hdf5)

**If training on or including GPU:**
1) cuda
2) cudnn

### ``Project Goal:``

Develop a program that can analyze the sentiment of posts and comments on VKontakte.

### ``Project Objectives:``

1) Develop an algorithm (***web scraper***) to collect necessary information from VKontakte (communities, user pages, list of communities or groups, etc.);
2) Create a neural network-based algorithm to analyze posts and comments.
3) Create a dataset in a format suitable for neural network training;
4) Write an algorithm to structure input data appropriately for neural network processing;
5) Train the neural network on the created dataset to classify posts and comments by sentiment (e.g., positive or negative);
6) Develop an algorithm to present the results in a user-friendly format;
7) Develop an application programming interface (***API***) for integrating the program with other software.

### ``Theoretical Background:``

1) **Neural Network**: An algorithm for machine learning that models the functioning of the human brain, enabling a computer to learn from large amounts of data. It consists of interconnected elements called neurons, which process information and pass it to each other through connections called synapses. Neurons work together to solve information processing tasks, such as image classification, speech recognition, and time series prediction. Neural network training involves backpropagation, where the network adjusts its neuron parameters based on training data to achieve optimal results.
2) **Web Scraper**: A program that extracts data from web pages for subsequent processing or use. Web scrapers can gather information on products, prices, news, and other data from various internet sources.

### ``Existing Work on Similar Topics:``

1) [Twitter Semantic Analysis by Cercosa](https://github.com/Cercosa/Twitter_semantic_analysis)
2) [SemanticAnalysis by kokwai48699](https://github.com/kokwai4869/SemanticAnalysis)
3) [SemanticAnalysis by hudongyue1](https://github.com/hudongyue1/semanticAnalysis)
4) [SemanticAnalysis by Jared-Hall](https://github.com/Jared-Hall/SemanticAnalysis)
5) [SAoT by zackotterstrom](https://github.com/zackotterstrom/SAoT)

### ``Advantages of Our Project over Existing Solutions:``

1) Designed specifically for ***Russian language***;
2) Targeted for use with ***VKontakte*** social network;
3) Capable of automatically obtaining necessary information from sources without manual data input;
4) Easy to learn and use;
5) Includes an integration mechanism (***API***) with other software.

### ``End Product of Our Project:``

**Software** that provides users with a convenient and quick way to analyze the sentiment of their posts and comments on VKontakte. This can help users understand how their opinions will be perceived by others. It can also be useful for marketing research, public opinion analysis, and other applications related to text analysis in social media.

### ``Potential Issues During the Project:``

1) Large data requirements for training the neural network. To create an accurate model, the neural network must be trained on a large amount of text with known sentiment. Insufficient data can reduce the model's accuracy and effectiveness.
2) Effective data preprocessing algorithm. Before feeding data into the neural network, it must be preprocessed to make the texts cleaner and more structured. An ineffective preprocessing algorithm can reduce model quality.
3) Challenges in determining sentiment. Sentiment can be subjective and context-dependent, making it difficult to accurately determine sentiment, which can reduce model accuracy.

### ``Knowledge Areas Addressed in the Project:``

1) **Natural Language Processing (NLP)** for text data analysis.
2) **Machine Learning**, particularly **Deep Learning**, for creating a neural network that will determine text sentiment.
---
### ``Hardware Requirements for the Project:``
1) HP Pavilion 15:
    1. Processor — Intel Core I7-8550U 1.80GHz
    2. Graphics Card — Nvidia GeForce GTX 1050

2) Asus Vivobook Pro 15:
    1. Processor — Intel Core I5-11300H 3.10GHz
    2. Graphics Card — Nvidia GeForce GTX 1650

### ``Operating Systems Used for the Project: ``
[LINUX MINT 21.1](https://www.linuxmint.com/)

### ``Programming Language Used: ``
[Python](https://www.python.org/)
### ``Reason for Choosing Python:``
Python is used in our project because it has all the necessary tools for working with natural language and deep learning, and can also be used for creating web applications and other interfaces. Python has all the necessary libraries suitable for our project, such as requests and BeautifulSoup4 for web scraping.  
Python also has a simple and clear syntax, making it easily readable and understandable for all team members, regardless of their level.  
Python is also a cross-platform language, meaning applications written in Python can run on various operating systems without code changes, making it easy to port the code to other systems.

### ``Software Used: ``
1) **PyCharm** — an integrated development environment (***IDE***) for Python, developed by JetBrains. PyCharm has extensive development features, including code autocompletion, debugging, framework support, and testing.
2) **Pip** — a command-line tool for managing Python packages. With pip, you can install, update, and remove Python packages from the centralized PyPI (Python Package Index) repository. It also allows you to install packages from other sources, such as Git or SVN. Pip also handles dependency installation, making it essential for managing Python packages.

### ``Libraries Anticipated for Project Implementation: ``
1) **Requests** — a library for working with HTTP requests. It allows you to get data from web pages, send data to a server, and authenticate on websites.
2) **Beautiful Soup 4** — a library for parsing HTML and XML documents. It allows you to extract information from HTML pages, such as article texts, titles, and links.
3) **NumPy** — a library for working with multidimensional arrays and mathematical functions.
4) **Pandas** — a library for data manipulation, including reading and writing data in various formats, data manipulation, and data aggregation.
5) **Scikit-learn** — a machine learning library that includes classification, regression, clustering, model selection algorithms, and more.
6) **TensorFlow** — a library for developing and training neural networks.
7) **Keras** — a high-level API for working with neural networks, built on TensorFlow.
8) **NLTK (Natural Language Toolkit)** — a library for natural language processing, including tokenization, stemming, lemmatization algorithms, and more.
9) **Seaborn** — a data visualization library based on Matplotlib, but with a simpler and more beautiful interface.
10) **SpaCy** — a library for natural language processing in Python. It is used for tasks like tokenization, lemmatization, named entity recognition, and dependency parsing.
---
### ``Team Roles: ``
1) **Mikhail Rogachev** — Project Manager, Machine Learning Engineer, Python Developer
2) **Artem Kaplenko** — Data Analyst, Python Developer
3) **Viktor Khromov** — Tester
---
### ``Project Roadmap: ``
- [x] 1) Collecting test data — 3-4 weeks — **Kaplenko**;  
- [x] 2) Algorithm for structuring input data for neural network processing — 4-5 weeks — **Rogachev**;
- [x] 3) Development of the first version of the neural network — 5-6 weeks — **Rogachev**;
- [x] 4) Collecting and preprocessing data for training the neural network, including posts and comments from VKontakte — 6-8 weeks — **Rogachev**;
- [x] 5) Training the neural network on the initially collected data — 8-13 weeks — **Rogachev/Kaplenko**;
- [x] 6) Testing the neural network — 12-14 weeks — **Khromov**.
