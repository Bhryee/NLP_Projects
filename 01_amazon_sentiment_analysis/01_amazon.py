
# METİN ÖNİŞLEME

## Adım 1 : amazon.xlsx verisini oku
import numpy as np
import pandas as pd

df = pd.read_excel("/content/gdrive/MyDrive/ML DL Kaynakça/NLP/amazon.xlsx")
print(df.T.head())

## Adım 2: Review Değişkeni Üzerinde:
### Tüm harfleri küçük harfe çevirme


df['Review'] = df['Review'].str.lower()
df['Review'].head()

### Regular Expression. Noktalama işaretlerini silme

df['Review'] = df['Review'].str.replace('[^\w\s]', '', regex = True)
df['Review'].head()

### Sayısal ifadeleri silme

df['Review'] = df['Review'].str.replace('\d', '', regex=True)
df['Review'].head()

### Bilgi içermeyen kelimeleri (stopwords) veriden silme

import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

sw = stopwords.words('english')

df['Review'] = df['Review'].apply(lambda x: " ".join(x for x in str(x).split() if x not in sw))
df['Review'].head()

### 1000'den az geçen kelimeleri veriden silme

# Review sütununda float tipinde olan değerleri filtrele
float_degerler = df[df['Review'].apply(lambda x: isinstance(x, float))]
print(float_degerler['Review'])

# NaN değerleri 'Review' sütunundan sil
df = df.dropna(subset=['Review'])

w = pd.Series(' '.join(df['Review']).split()).value_counts()
w.head()

drop = w[w <= 1]

df['Review'] = df['Review'].apply(lambda x: " ".join(x for x in x.split() if x not in drop))
print(df['Review'])

### Tokenization

from textblob import TextBlob
nltk.download("punkt")

df['Review'].apply(lambda x: TextBlob(x).words).head()

### Lemmatization (kök) işlemi

from textblob import Word
nltk.download('wordnet')

df['Review'] = df['Review'].apply(lambda sentence: " ".join([Word(word).lemmatize() for word in sentence.split()]))
df['Review'].head()

# METİN GÖRSELLEŞTİRME

## Barplot Görselleştirme İşlemi


# kelimeleri sayısal hale getir
tf = df["Review"].apply(lambda sentence: pd.value_counts(sentence.split(" "))).sum(axis=0).reset_index()

# dataframe'in sütunlarını yeniden adlandır
tf.columns = ["words", "tf"]

tf_sorted = tf.sort_values("tf", ascending=False);

print("İşlem tamamlandı")

print(tf)

#Barplot
import matplotlib.pyplot as plt

tf[tf["tf"] > 500].plot.bar(x="words", y="tf")
plt.show()

## WorldCloud Görselleştirme İşlemi

text = " ".join(i for i in df.Review)
print(text)

from wordcloud import WordCloud

wordcloud = WordCloud().generate(text)
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

wordcloud = WordCloud(max_font_size=60,
                      max_words=100,
                      background_color="white").generate(text)
plt.figure()
plt.imshow(wordcloud, interpolation="bilinear")
plt.axis("off")
plt.show()

#şablonlara göre WordCloud
from PIL import Image

tr_mask = np.array(Image.open("/content/gdrive/MyDrive/ML DL Kaynakça/NLP/ayzek.png"))

wc = WordCloud(background_color="white",
               max_words=1000,
               mask=tr_mask,
               contour_width=3,
               contour_color="blue")

wc.generate(text)
plt.figure(figsize=[10, 10])
plt.imshow(wc, interpolation="bilinear")
plt.axis("off")
plt.show()

# Sentiment Analysis (Duygu Analizi)

# polarity_scores() hesaplanması
# compound'ta 0'dan büyük bir sayı varsa cümle pozitif, küçükse negatif
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

for sentence in df['Review'].head(10):
  print(sentence)
  print(sia.polarity_scores(sentence))
  print("\n")

x = df['Review'][0:10].apply(lambda x: "positive" if sia.polarity_scores(x)['compound']> 0 else "negative")
print(x.head(10))

df["sentiment_label"] = df["Review"].apply(lambda x: "positive" if sia.polarity_scores(x)["compound"]> 0 else "negative")
print(df['sentiment_label'].head(2))

df['sentiment_label'].value_counts()

df.groupby("sentiment_label")["Star"].mean()

df.head()

# Machine Learning (Makine Öğrenmesi)

from sklearn.preprocessing import LabelEncoder

df["sentiment_label"] = LabelEncoder().fit_transform(df["sentiment_label"])

df.head(2)
# pos = 1
# neg = 0

y = df["sentiment_label"] #output
X = df["Review"] #input

X.head(1)

# TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(X)

tfidf_ngram = TfidfVectorizer(ngram_range = (2,3))
X_tfidf_ngram = tfidf_ngram.fit_transform(X)

X_tfidf.shape

X_tfidf[0]

X_tfidf_ngram.shape

X_tfidf_ngram[0]

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

lr = LogisticRegression().fit(X_tfidf, y)

cross_val_score(lr, X_tfidf, y, cv=5).mean()

sentence = df["Review"][10]
sentence_transform = TfidfVectorizer().fit(X).transform([sentence])

predict = lr.predict(sentence_transform)

print(f"Predict Sentence: {sentence}\n Predict: {predict}")

sentence = pd.Series(df["Review"].sample(1).values)
sentence_transform = TfidfVectorizer().fit(X).transform(sentence)

predict = lr.predict(sentence_transform)

print(f"Predict Sentence: {sentence}\n Predict: {predict}")

from sklearn.metrics import classification_report

# Test veri seti üzerinde tahmin yapalım (verilen örnekte X, tüm veri seti olarak alınmış).
y_pred = lr.predict(X_tfidf)

# Classification report oluşturalım.
report = classification_report(y, y_pred)
print(report)

# Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_validate

rf = RandomForestClassifier(random_state = 42)

rf_params = {"max_depth": [8, None],  # max derinlik
             "max_features": [7, "auto"], #bölünmelerde göz önünde bulunacak olan max değişken
             "min_samples_split": [2, 5, 8], # bir yaprakta ne kadar örnek olucak
             "n_estimators": [100, 200]} # kaç tane ağaç eğitilicek

grid = GridSearchCV(rf,
                    rf_params,
                    cv=5,
                    n_jobs=-1,
                    verbose=1).fit(X_tfidf, y)

grid.best_params_

final = rf.set_params(**grid.best_params_, random_state = 42).fit(X_tfidf, y)

cross_val_score(final, X_tfidf, y, cv=5, n_jobs=-1).mean()

y_pred = rf.predict(X_tfidf)


report = classification_report(y, y_pred)
print(report)
