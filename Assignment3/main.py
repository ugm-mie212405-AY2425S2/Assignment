import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
import string

# ============================
# 1. BACA STOPWORD DARI FILE .TXT
# ============================
def load_stopwords_txt(filepath):
    with open(filepath, "r", encoding="utf-8") as file:
        stopwords = set([line.strip() for line in file if line.strip()])
    return stopwords

stopwords_id = load_stopwords_txt("stopwords.txt")  # Ganti nama file jika perlu

# ============================
# 2. DEFINISI ABSTRAK PANJANG
# ============================
abstract1 = """
Leukemia is a type of cancer that is on white blood cell. This disease are characterized by abundance of abnormal 
white blood cell called lymphoblast in the bone marrow. Classification of blood cell types, calculation of the ratio 
of cell types and comparison with normal blood cells can be the subject of diagnosing this disease. The diagnostic 
process is carried out manually by hematologists through microscopic image. This method is likely to provide 
a subjective result and time-consuming.
The application of digital image processing techniques and machine learning in the process of classifying 
white blood cells can provide more objective results. This research used thresholding method as segmentation and  
multilayer method of back propagation perceptron with variations in the extraction of textural features, geometry, 
and colors. The results of segmentation testing in this study amounted to 68.70%. Whereas the classification test 
shows that the combination of feature extraction of GLCM features, geometry features, and color features gives the 
best results. This test produces an accuration value 91.43%, precision value of 50.63%, sensitivity 56.67%, F1Score 51.95%, 
and specitifity 94.16%.

"""

abstract2 = """
White blood cells are classified into five types (basophils, eosinophils, neutrophils, lymphocytes and monocytes) 
with additional classes lymphoblast cells from microscope images are processed. By applying image processing, 
image its white blood cells extracted using the Histogram Oriented Gradient. Feature extraction results obtained 
then classified using Support Vector Machine method by comparing the results of two different kernel parameters: 
kernel Linear and kernel Radial Basis Function (RBF). Classification evaluated with these parameters: Accuracy, 
specificity, and sensitivity.
Obtained an accuracy of 72.26% from the detection of white blood cells in the microscope image. The average value 
of microscope images of patients and different kernel every white blood cells (monocytes, basophils, neutrophils, 
eosinophils, lymphocytes and lymphoblast) were evaluated with these parameters. Results of the study show 
the classification system has an average value of 82.20% accuracy (RBF Patient 1), 81.63% (RBF Patient 2) and 
78.73% (Linear Patient 1), 79.55% (Linear Patient 2 ), then the value of specificity of 89.91% (RBF patient 1), 
92.18% (RBF patient 2) and 88.06% (Linear patient 1), 91.34% (Linear patient 2), and sensitivity values 15 , 
45% (RBF patient 1), 12.97% (RBF patient 2) and 13.33% (Linear patient 1), 12.50% (Linear patient 2).

"""

abstract3 = """
The traditional procedure of classification of blood cells using a microscope in the laboratory of hematology to 
obtain information types of blood cells. It has become a cornerstone in the laboratory of hematology to diagnose 
and monitor hematologic disorders. However, the manual procedure through a series of labory test can take a while. 
Thresfore, this research can be helpful in the early stages of the classification of white blood cells automatically 
in the medical field.
Efforts to overcome the length of time and for the purposes of early diagnose can use the image processing technique 
based on morphology of blood cells. This research aims to classify the white blood cells based on cell morphology with 
the k-nearest neighbor (knn). Image processing algorithms used hough circle, thresholding, feature extraction, then 
to the process of classification was used the method of k-nearest neighbor (knn).
In the process of testing used 100 images to be aware of its kind. The test results showed segmentation 
accuracy of 78% and testing the classification of 64%.
"""

abstracts = [abstract1, abstract2, abstract3]

# ============================
# 3. PREPROCESSING TEKS
# ============================
def preprocess(text, stopwords):
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    words = text.split()
    filtered_words = [word for word in words if word not in stopwords]
    return " ".join(filtered_words)

abstracts_cleaned = [preprocess(doc, stopwords_id) for doc in abstracts]

# ============================
# 4. TF-IDF DAN COSINE SIMILARITY
# ============================
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(abstracts_cleaned)
cos_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# ============================
# 5. OUTPUT HASIL
# ============================
print("=== Cosine Similarity Matrix ===")
for i in range(len(abstracts)):
    for j in range(len(abstracts)):
        if i != j:
            print(f"Similarity(doc{i+1}, doc{j+1}) = {cos_sim[i][j]:.4f}")
