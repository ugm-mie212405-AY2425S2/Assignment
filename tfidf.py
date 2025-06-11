import re
import nltk
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from nltk.corpus import stopwords
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Download stopwords
nltk.download('stopwords')

# English stopwords
stop_words = set(stopwords.words('english'))

# List of abstracts
abstracts = [
    """Report from the WHO that one of the highest causes of medication errors is Look Alike – Sound Alike (LASA) drugs, leading to errors in receiving information about the drugs, which of course will affect patient safety. Efforts to reduce medication errors have been widely implemented, such as conducting medication training, managing medications, and storing and labeling medications. However, all of that leads to human error, so the utilization of technology is needed to address this issue. The technology expected to help reduce medication errors is the utilization of artificial intelligence (AI). AI is designed for automation processes and systems that can learn independently, allowing the causes of medication errors such as LASA to be learned by the system and predicted automatically. Deep learning is a part of AI that works by providing solutions accurately and automatically. The Recurrent Neural Networks (RNN) algorithm is one of the deep learning methods that has been proven accurate in predictions based on previous research studies. In this study, LASA predictions were made using RNN with the aim of serving as an aid to reduce medication errors, thereby ensuring patient safety. The accuracy achieved is 99% for training and 81% for testing.""",

    """The proliferation of cyber security attacks necessitates advanced and efficient detection methods. This study explores the application of Convolutional Neural Networks (CNNs) for classifying cyber security attacks using a comprehensive dataset containing various attack types and network traffic features. Emphasizing the role of hyperparameter optimization (HPO) techniques, this research aims to enhance the CNN model's performance in accurately detecting and classifying cyber attacks. Traditional machine learning approaches often need to catch up in capturing the complex patterns in such data, whereas CNNs excel in automatically extracting hierarchical features. Using the provided dataset, which includes attributes such as packet length, source and destination ports, protocol, and traffic type, we implemented various (HPO) techniques, including Grid Search, Random Search, and Bayesian Optimization, to identify the optimal CNN configurations. Our optimized CNN model significantly improved classification result. to baseline models without hyperparameter tuning. The results underline the importance of HPO in developing robust CNN models for cybersecurity applications. This study provides a practical framework for leveraging deep learning and optimization techniques to enhance cyber defense mechanisms, paving the way for future advancements in the field.""",

    """Tree identification is a very important to support almost all activities in the forest sector. Unfortunately, the inavailability of data and computer programs that is user friendly have caused ineficiency in tree identification. This research tries to make an expert system to identify trees by using the leaf images. To store the data in the knowledge base one must choose one of the some leaf images that are in the data base available in the program according the characteristic of the leaf. Each leaf image has a code and the accumulation of all codes build a tree code then this code is saved in the knowledge base. The tree code is used to identify a tree by making the comparison between input chosen by user and the tree code in the knowledge base using forward chaining. User who has information about a tree can add to the knowledge base but this information must be validated by an expert before it is used in the system. Another task of an expert is to give a CF (certainty factor) for each tree. The result of this research shows that no more errors are found due to input mistakes and the program is more user friendly. Another advantage is that the knowledge base is more flexible, dynamic and well organized Validation of knowledge base by experts can increase the quality and accuracy of using the knowledge base system."""
]

# Function to clean and tokenize text
def preprocess(text):
  text = text.lower()
  text = re.sub(r'[^a-z\s]', '', text)
  words = text.split()
  filtered = [word for word in words if word not in stop_words and len(word) > 1]
  return filtered

# Function to get term frequencies per abstract
def term_frequencies(abstract):
  for idx, text in enumerate(abstract, 1):
    words = preprocess(text)
    freq = Counter(words)
    print(f'\nTerm frequencies for Abstract {idx}')
    for term, count in sorted(freq.items()):
      print(f'{term}: {count}')

# Function to create wordclouds
def generate_wordclouds(abstracts):
  for idx, text in enumerate(abstracts, 1):
    words = preprocess(text)
    freq = Counter(words)

    wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(freq)

    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(f'Wordcloud for Abstract {idx}')
    plt.show()

# Call the functions
term_frequencies(abstracts)
generate_wordclouds(abstracts)

abstracts = [
    """Report from the WHO that one of the highest causes of medication errors is Look Alike – Sound Alike (LASA) drugs, leading to errors in receiving information about the drugs, which of course will affect patient safety. Efforts to reduce medication errors have been widely implemented, such as conducting medication training, managing medications, and storing and labeling medications. However, all of that leads to human error, so the utilization of technology is needed to address this issue. The technology expected to help reduce medication errors is the utilization of artificial intelligence (AI). AI is designed for automation processes and systems that can learn independently, allowing the causes of medication errors such as LASA to be learned by the system and predicted automatically. Deep learning is a part of AI that works by providing solutions accurately and automatically. The Recurrent Neural Networks (RNN) algorithm is one of the deep learning methods that has been proven accurate in predictions based on previous research studies. In this study, LASA predictions were made using RNN with the aim of serving as an aid to reduce medication errors, thereby ensuring patient safety. The accuracy achieved is 99% for training and 81% for testing.""",

    """The proliferation of cyber security attacks necessitates advanced and efficient detection methods. This study explores the application of Convolutional Neural Networks (CNNs) for classifying cyber security attacks using a comprehensive dataset containing various attack types and network traffic features. Emphasizing the role of hyperparameter optimization (HPO) techniques, this research aims to enhance the CNN model's performance in accurately detecting and classifying cyber attacks. Traditional machine learning approaches often need to catch up in capturing the complex patterns in such data, whereas CNNs excel in automatically extracting hierarchical features. Using the provided dataset, which includes attributes such as packet length, source and destination ports, protocol, and traffic type, we implemented various (HPO) techniques, including Grid Search, Random Search, and Bayesian Optimization, to identify the optimal CNN configurations. Our optimized CNN model significantly improved classification result. to baseline models without hyperparameter tuning. The results underline the importance of HPO in developing robust CNN models for cybersecurity applications. This study provides a practical framework for leveraging deep learning and optimization techniques to enhance cyber defense mechanisms, paving the way for future advancements in the field.""",

    """Tree identification is a very important to support almost all activities in the forest sector. Unfortunately, the inavailability of data and computer programs that is user friendly have caused ineficiency in tree identification. This research tries to make an expert system to identify trees by using the leaf images. To store the data in the knowledge base one must choose one of the some leaf images that are in the data base available in the program according the characteristic of the leaf. Each leaf image has a code and the accumulation of all codes build a tree code then this code is saved in the knowledge base. The tree code is used to identify a tree by making the comparison between input chosen by user and the tree code in the knowledge base using forward chaining. User who has information about a tree can add to the knowledge base but this information must be validated by an expert before it is used in the system. Another task of an expert is to give a CF (certainty factor) for each tree. The result of this research shows that no more errors are found due to input mistakes and the program is more user friendly. Another advantage is that the knowledge base is more flexible, dynamic and well organized Validation of knowledge base by experts can increase the quality and accuracy of using the knowledge base system."""
]

# TF-IDF Vectorizer with stopwords
Vectorizer = TfidfVectorizer(stop_words=stopwords.words('english'))

# Compute TF-IDF matrix
tfidf_matrix = Vectorizer.fit_transform(abstracts)

# Compute cosine similarity between documents
cosine_sim = cosine_similarity(tfidf_matrix)

print('Cosine Similarity Matrix:')
print('\n', np.round(cosine_sim, 3))

plt.figure(figsize=(8, 6))
sns.heatmap(cosine_sim, annot=True, cmap='Blues', xticklabels=["Abstract 1", "Abstract 2", "Abstract 3"], yticklabels=["Doc 1", "Doc 2", "Doc 3"])
plt.title('Cosine Similarity Heatmap of Abstracts (TF-IDF)', fontsize=14)
plt.xlabel('Document')
plt.ylabel('Document')
plt.tight_layout()
plt.show()