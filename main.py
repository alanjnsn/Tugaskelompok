# Import library
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
import re
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

data_train = {
  "komentar": [
    "film itu jelek sekali",
    "film yang saya tonton sangat bermanfaat dan berkualitas",
    "penawaran diskon sangat bermanfaat dan berkualitas",
    "Produk penawaran diskon tidak sesuai harapan",
    "website resmi sangat bermanfaat dan berkualitas",
    "Tempat kualitas barang nyaman dan menyenangkan",
    "website resmi membuat saya kesal",
    "Pelayanan sistem pembayaran sangat memuaskan",
    "Saya menikmati waktu saya di pengalaman belanja",
    "Saya menyesal telah menggunakan restoran tersebut",
    "Saya sangat kecewa dengan toko online ini",
    "Tempat promo spesial membuat saya tidak nyaman",
    "penawaran diskon membuat saya senang",
    "Produk website resmi sangat membantu",
    "Produk hotel tempat saya menginap tidak sesuai harapan",
    "produk ini membuat saya senang",
    "aplikasi ini membuat saya senang",
    "Saya merekomendasikan pengalaman belanja kepada teman saya",
    "platform digital sangat tidak bermanfaat dan mengecewakan",
    "Pengalaman dengan promo spesial sangat mengecewakan",
    "Saya suka toko online ini yang diberikan",
    "Tempat fasilitas umum nyaman dan menyenangkan",
    "Saya suka toko online ini yang diberikan",
    "pengiriman membuat saya senang",
    "pengalaman belanja sangat bermanfaat dan berkualitas",
    "Pelayanan sistem pembayaran sangat buruk",
    "Tempat website resmi membuat saya tidak nyaman",
    "produk ini sangat tidak bermanfaat dan mengecewakan",
    "Saya sangat puas dengan promo spesial",
    "Pelayanan pengiriman sangat memuaskan",
    "website resmi membuat saya kesal",
    "Saya tidak suka sistem pembayaran yang diberikan",
    "hotel tempat saya menginap sangat tidak bermanfaat dan mengecewakan",
    "Produk produk ini sangat membantu",
    "Tempat produk ini nyaman dan menyenangkan",
    "Tempat website resmi membuat saya tidak nyaman",
    "Pengalaman dengan sistem pembayaran sangat mengecewakan",
    "Saya menyesal telah menggunakan promo spesial",
    "Saya suka hotel tempat saya menginap yang diberikan",
    "Saya tidak suka produk ini yang diberikan",
    "Saya tidak suka toko online ini yang diberikan",
    "Saya suka produk ini yang diberikan",
    "promo spesial sangat bermanfaat dan berkualitas",
    "Produk restoran tersebut tidak sesuai harapan",
    "Saya sangat puas dengan promo spesial",
    "Saya sangat kecewa dengan promo spesial",
    "Saya merekomendasikan toko online ini kepada teman saya",
    "Saya suka penawaran diskon yang diberikan",
    "Pengalaman dengan kualitas barang sangat positif",
    "Saya menikmati waktu saya di fasilitas umum",
    "Pelayanan promo spesial sangat buruk",
    "Produk pengiriman sangat membantu",
    "Saya sangat puas dengan film yang saya tonton",
    "kualitas barang sangat bermanfaat dan berkualitas",
    "Saya tidak suka penawaran diskon yang diberikan",
    "Saya merekomendasikan fasilitas umum kepada teman saya",
    "film yang saya tonton sangat tidak bermanfaat dan mengecewakan",
    "Saya sangat puas dengan pengalaman belanja",
    "Saya sangat puas dengan produk ini",
    "Saya sangat kecewa dengan kualitas barang",
    "Pengalaman dengan promo spesial sangat mengecewakan",
    "Pelayanan penawaran diskon sangat memuaskan",
    "Pengalaman dengan pengalaman belanja sangat mengecewakan",
    "pengiriman sangat bermanfaat dan berkualitas",
    "Saya menikmati waktu saya di penawaran diskon",
    "Saya suka produk ini yang diberikan",
    "aplikasi ini sangat tidak bermanfaat dan mengecewakan",
    "Produk restoran tersebut tidak sesuai harapan",
    "Saya menikmati waktu saya di website resmi",
    "kualitas barang sangat tidak bermanfaat dan mengecewakan",
    "Pengalaman dengan website resmi sangat positif",
    "toko online ini membuat saya senang",
    "Pengalaman dengan hotel tempat saya menginap sangat mengecewakan",
    "aplikasi ini membuat saya senang",
    "restoran tersebut sangat tidak bermanfaat dan mengecewakan",
    "Saya sangat puas dengan toko online ini",
    "Tempat platform digital nyaman dan menyenangkan",
    "Pengalaman dengan toko online ini sangat mengecewakan",
    "film yang saya tonton sangat bermanfaat dan berkualitas",
    "pengiriman membuat saya senang",
    "Pelayanan platform digital sangat buruk",
    "Saya tidak suka penawaran diskon yang diberikan",
    "Saya suka toko online ini yang diberikan",
    "platform digital sangat bermanfaat dan berkualitas",
    "Pengalaman dengan aplikasi ini sangat positif",
    "Pelayanan pengiriman sangat memuaskan",
    "Saya suka toko online ini yang diberikan",
    "Pengalaman dengan platform digital sangat positif",
    "Saya suka pengiriman yang diberikan",
    "Tempat promo spesial nyaman dan menyenangkan",
    "pengalaman belanja sangat bermanfaat dan berkualitas",
    "Pelayanan produk ini sangat buruk",
    "toko online ini membuat saya kesal",
    "Saya menikmati waktu saya di restoran tersebut",
    "Saya sangat kecewa dengan fasilitas umum",
    "Saya tidak suka layanan pelanggan yang diberikan",
    "Pelayanan sistem pembayaran sangat memuaskan",
    "Pengalaman dengan platform digital sangat mengecewakan",
    "toko online ini membuat saya kesal",
    "Saya tidak akan merekomendasikan layanan pelanggan kepada siapa pun",
    "Saya menyesal telah menggunakan aplikasi ini",
    "Pengalaman dengan kualitas barang sangat positif",
    "Saya sangat kecewa dengan produk ini",
    "Saya merekomendasikan kualitas barang kepada teman saya",
    "Saya tidak suka kualitas barang yang diberikan",
    "fasilitas umum membuat saya senang",
    "Pelayanan sistem pembayaran sangat buruk",
    "Saya sangat puas dengan penawaran diskon",
    "aplikasi ini sangat bermanfaat dan berkualitas",
    "Tempat produk ini nyaman dan menyenangkan",
    "Pengalaman dengan layanan pelanggan sangat mengecewakan",
    "Pelayanan toko online ini sangat buruk",
    "kualitas barang sangat bermanfaat dan berkualitas",
    "Saya menikmati waktu saya di sistem pembayaran",
    "Pelayanan pengiriman sangat memuaskan",
    "Pengalaman dengan promo spesial sangat mengecewakan",
    "Saya merekomendasikan restoran tersebut kepada teman saya",
    "Produk website resmi tidak sesuai harapan",
    "Pelayanan sistem pembayaran sangat buruk",
    "Pengalaman dengan kualitas barang sangat positif",
    "Saya merekomendasikan restoran tersebut kepada teman saya",
    "Saya menyesal telah menggunakan film yang saya tonton",
    "Pelayanan film yang saya tonton sangat buruk",
    "Saya tidak suka layanan pelanggan yang diberikan",
    "Tempat penawaran diskon membuat saya tidak nyaman",
    "Produk platform digital sangat membantu",
    "Pengalaman dengan hotel tempat saya menginap sangat mengecewakan",
    "Saya tidak suka aplikasi ini yang diberikan",
    "Pelayanan pengalaman belanja sangat buruk",
    "Saya suka fasilitas umum yang diberikan",
    "platform digital sangat bermanfaat dan berkualitas",
    "Produk layanan pelanggan sangat membantu",
    "Pelayanan pengalaman belanja sangat memuaskan",
    "Saya menikmati waktu saya di aplikasi ini",
    "Saya sangat puas dengan toko online ini",
    "Saya suka penawaran diskon yang diberikan",
    "Saya suka restoran tersebut yang diberikan",
    "Saya merekomendasikan website resmi kepada teman saya",
    "sistem pembayaran sangat bermanfaat dan berkualitas",
    "Pengalaman dengan produk ini sangat mengecewakan",
    "Produk promo spesial tidak sesuai harapan",
    "Pelayanan platform digital sangat memuaskan",
    "Produk website resmi tidak sesuai harapan",
    "Tempat pengiriman membuat saya tidak nyaman",
    "Pengalaman dengan sistem pembayaran sangat mengecewakan",
    "Saya sangat puas dengan website resmi",
    "Pengalaman dengan promo spesial sangat mengecewakan",
    "promo spesial sangat tidak bermanfaat dan mengecewakan",
    "Saya menikmati waktu saya di fasilitas umum",
    "Pengalaman dengan sistem pembayaran sangat mengecewakan",
    "hotel tempat saya menginap sangat tidak bermanfaat dan mengecewakan",
    "Pelayanan toko online ini sangat memuaskan",
    "hotel tempat saya menginap sangat tidak bermanfaat dan mengecewakan",
    "Tempat layanan pelanggan membuat saya tidak nyaman",
    "Saya suka website resmi yang diberikan",
    "Tempat aplikasi ini membuat saya tidak nyaman",
    "Saya sangat kecewa dengan aplikasi ini",
    "Tempat restoran tersebut membuat saya tidak nyaman",
    "hotel tempat saya menginap sangat tidak bermanfaat dan mengecewakan",
    "Produk sistem pembayaran tidak sesuai harapan",
    "Saya menikmati waktu saya di sistem pembayaran",
    "website resmi sangat bermanfaat dan berkualitas",
    "layanan pelanggan sangat tidak bermanfaat dan mengecewakan",
    "website resmi sangat tidak bermanfaat dan mengecewakan",
    "Saya sangat puas dengan penawaran diskon",
    "website resmi sangat bermanfaat dan berkualitas",
    "Pelayanan hotel tempat saya menginap sangat memuaskan",
    "Produk pengalaman belanja tidak sesuai harapan",
    "Saya merekomendasikan promo spesial kepada teman saya",
    "Tempat sistem pembayaran nyaman dan menyenangkan",
    "Tempat platform digital nyaman dan menyenangkan",
    "Tempat toko online ini nyaman dan menyenangkan",
    "website resmi membuat saya senang",
    "Pelayanan pengalaman belanja sangat buruk",
    "website resmi sangat tidak bermanfaat dan mengecewakan",
    "Saya merekomendasikan promo spesial kepada teman saya",
    "Produk pengalaman belanja sangat membantu",
    "Saya tidak suka hotel tempat saya menginap yang diberikan",
    "Tempat pengalaman belanja membuat saya tidak nyaman",
    "Saya sangat kecewa dengan pengiriman",
    "kualitas barang membuat saya senang",
    "Saya sangat kecewa dengan film yang saya tonton",
    "Tempat platform digital nyaman dan menyenangkan",
    "Tempat kualitas barang nyaman dan menyenangkan",
    "Saya menikmati waktu saya di pengalaman belanja",
    "hotel tempat saya menginap sangat bermanfaat dan berkualitas",
    "Produk hotel tempat saya menginap sangat membantu",
    "Saya merekomendasikan aplikasi ini kepada teman saya",
    "sistem pembayaran membuat saya kesal",
    "Saya sangat puas dengan restoran tersebut",
    "platform digital membuat saya senang",
    "fasilitas umum membuat saya senang",
    "Pengalaman dengan sistem pembayaran sangat mengecewakan",
    "website resmi membuat saya senang",
    "Produk pengiriman sangat membantu",
    "film yang saya tonton membuat saya kesal",
    "Saya tidak akan merekomendasikan aplikasi ini kepada siapa pun",
    "Tempat platform digital membuat saya tidak nyaman",
    "Saya tidak akan merekomendasikan fasilitas umum kepada siapa pun",
    "Produk layanan pelanggan tidak sesuai harapan",
    "Saya sangat puas dengan website resmi",
    "Saya suka promo spesial yang diberikan",
    "Saya menikmati waktu saya di hotel tempat saya menginap",
    "Tempat toko online ini nyaman dan menyenangkan",
    "Saya tidak suka platform digital yang diberikan",
    "produk ini membuat saya kesal",
    "Saya tidak suka film yang saya tonton yang diberikan",
    "aplikasi ini sangat tidak bermanfaat dan mengecewakan",
    "Saya sangat puas dengan produk ini",
    "Saya tidak suka penawaran diskon yang diberikan",
    "Saya suka kualitas barang yang diberikan",
    "Produk promo spesial sangat membantu",
    "platform digital membuat saya kesal",
    "Saya menikmati waktu saya di pengiriman",
    "Saya menyesal telah menggunakan layanan pelanggan",
    "Saya merekomendasikan sistem pembayaran kepada teman saya",
    "Saya suka platform digital yang diberikan",
    "film yang saya tonton membuat saya kesal",
    "Saya merekomendasikan toko online ini kepada teman saya",
    "Saya tidak akan merekomendasikan platform digital kepada siapa pun",
    "Produk restoran tersebut sangat membantu",
    "sistem pembayaran sangat bermanfaat dan berkualitas",
    "Saya menyesal telah menggunakan layanan pelanggan",
    "Pelayanan hotel tempat saya menginap sangat memuaskan",
    "kualitas barang sangat tidak bermanfaat dan mengecewakan",
    "Saya tidak akan merekomendasikan restoran tersebut kepada siapa pun",
    "Pengalaman dengan film yang saya tonton sangat mengecewakan",
    "Tempat promo spesial membuat saya tidak nyaman",
    "Saya menyesal telah menggunakan restoran tersebut",
    "Pelayanan pengalaman belanja sangat buruk",
    "Saya suka pengiriman yang diberikan",
    "Pengalaman dengan produk ini sangat positif",
    "Tempat fasilitas umum membuat saya tidak nyaman",
    "Produk platform digital sangat membantu",
    "Produk layanan pelanggan tidak sesuai harapan",
    "Pengalaman dengan sistem pembayaran sangat positif",
    "Saya merekomendasikan fasilitas umum kepada teman saya",
    "Saya merekomendasikan promo spesial kepada teman saya",
    "Tempat penawaran diskon nyaman dan menyenangkan",
    "Pelayanan film yang saya tonton sangat buruk",
    "kualitas barang membuat saya senang",
    "Produk toko online ini tidak sesuai harapan",
    "Pengalaman dengan layanan pelanggan sangat positif",
    "Pengalaman dengan pengiriman sangat positif",
    "Tempat website resmi nyaman dan menyenangkan",
    "produk ini sangat bermanfaat dan berkualitas",
    "Saya sangat kecewa dengan penawaran diskon",
    "Saya sangat puas dengan promo spesial",
    "Pelayanan aplikasi ini sangat memuaskan",
    "Saya merekomendasikan pengalaman belanja kepada teman saya",
    "film yang saya tonton membuat saya senang",
    "Saya tidak akan merekomendasikan film yang saya tonton kepada siapa pun",
    "Saya tidak akan merekomendasikan promo spesial kepada siapa pun",
    "toko online ini sangat bermanfaat dan berkualitas",
    "fasilitas umum membuat saya kesal",
    "Saya tidak suka pengalaman belanja yang diberikan",
    "Saya sangat puas dengan aplikasi ini",
    "Saya sangat puas dengan fasilitas umum",
    "Tempat film yang saya tonton nyaman dan menyenangkan",
    "Produk platform digital tidak sesuai harapan",
    "Saya sangat kecewa dengan hotel tempat saya menginap",
    "Saya tidak akan merekomendasikan film yang saya tonton kepada siapa pun",
    "promo spesial membuat saya senang",
    "Pelayanan promo spesial sangat buruk",
    "Produk fasilitas umum tidak sesuai harapan",
    "Pengalaman dengan hotel tempat saya menginap sangat positif",
    "layanan pelanggan membuat saya senang",
    "Produk hotel tempat saya menginap tidak sesuai harapan",
    "pengalaman belanja membuat saya kesal",
    "Saya tidak suka pengiriman yang diberikan",
    "Tempat promo spesial nyaman dan menyenangkan",
    "Produk produk ini tidak sesuai harapan",
    "Saya sangat kecewa dengan penawaran diskon",
    "Saya tidak suka layanan pelanggan yang diberikan",
    "Pengalaman dengan pengiriman sangat mengecewakan",
    "website resmi membuat saya kesal",
    "Tempat toko online ini membuat saya tidak nyaman",
    "promo spesial membuat saya kesal",
    "Pengalaman dengan kualitas barang sangat positif",
    "Saya menikmati waktu saya di kualitas barang",
    "Pengalaman dengan film yang saya tonton sangat positif",
    "produk ini membuat saya senang",
    "Saya tidak akan merekomendasikan layanan pelanggan kepada siapa pun",
    "pengalaman belanja sangat bermanfaat dan berkualitas",
    "Pelayanan hotel tempat saya menginap sangat buruk",
    "pengiriman sangat bermanfaat dan berkualitas",
    "Saya menikmati waktu saya di layanan pelanggan",
    "film yang saya tonton membuat saya senang",
    "Produk website resmi tidak sesuai harapan",
    "Pelayanan restoran tersebut sangat memuaskan",
    "toko online ini membuat saya kesal",
    "Produk produk ini sangat membantu",
    "Pengalaman dengan layanan pelanggan sangat positif",
    "Pengalaman dengan promo spesial sangat positif",
    "Pelayanan promo spesial sangat memuaskan",
    "Tempat produk ini membuat saya tidak nyaman",
    "Saya suka film yang saya tonton yang diberikan",
    "Pelayanan fasilitas umum sangat memuaskan",
    "pengalaman belanja sangat tidak bermanfaat dan mengecewakan",
    "Tempat fasilitas umum nyaman dan menyenangkan"
  ],
  "label": [0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1]
}

kamus_normalisasi = {
    "gk": "tidak",
    "ga": "tidak",
    "nggak": "tidak",
    "tdk": "tidak",
    "bgt": "banget",
    "bgtu": "begitu",
    "bgus": "bagus",
    "bgz": "bagus",
    "skrg": "sekarang",
    "udh": "sudah",
    "sdh": "sudah",
    "dgn": "dengan",
    "sm": "sama",
    "sy": "saya",
    "gw": "saya",
    "gua": "saya",
    "lu": "kamu",
    "loe": "kamu",
    "km": "kamu",
    "trs": "terus",
    "bkn": "bukan",
    "aj": "saja",
    "aja": "saja",
    "dr": "dari",
    "tp": "tapi",
    "mksh": "terima kasih",
    "tq": "terima kasih",
    "plis": "tolong",
    "pls": "tolong",
    "mntp": "mantap",
    "mantul": "mantap betul",
    "wkwkwk": "haha",
    "wkwk": "haha",
    "wk": "haha",
    "lol": "haha",
    "mbg":"makan bergizi gratis"
}

stopwords = set([
    "yang", "dan", "di", "ke", "dari", "itu", "ini", "untuk", "dengan", 
    "pada", "adalah", "karena", "bahwa", "sudah", "agar", "jika", "seperti"
])

# Buat stemmer
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Function normalisasi kata
def normalisasi_kata(kalimat, kamus):
    kata_kata = kalimat.split()
    hasil = [kamus.get(k.lower(), k) for k in kata_kata]
    return ' '.join(hasil)

# Pipeline preprocessing
def preprocess(kalimat):
    # Lowercase
    kalimat = kalimat.lower()
    
    # Hapus URL, mention, hashtag, angka, dan tanda baca
    kalimat = re.sub(r"http\S+|www\S+|https\S+", "", kalimat)
    kalimat = re.sub(r"@\w+", "", kalimat)
    kalimat = re.sub(r"#\w+", "", kalimat)
    kalimat = re.sub(r"[^a-zA-Z\s]", " ", kalimat)
    
    # Normalisasi
    kalimat = normalisasi_kata(kalimat, kamus_normalisasi)

    # Hapus stopwords
    kata_kata = kalimat.split()
    kata_kata = [k for k in kata_kata if k not in stopwords]

    # Stemming
    kalimat = ' '.join([stemmer.stem(k) for k in kata_kata])
    
    return kalimat

# Contoh data komentar sosial media
data = pd.DataFrame(data_train)
# comment_file = 'scrap_x1.csv'
# df= pd.read_csv('scrap_x1.csv')
# with open(comment_file,'r') as f:
#     lines = pd.read_csv(f, delimiter=';')

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    data["komentar"], data["label"], test_size=0.7, random_state=42
)

# TF-IDF Vectorizer (di dalam pipeline nanti)
tfidf = TfidfVectorizer()

xgb_pipeline = Pipeline([
    ('tfidf', tfidf),
    ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss'))
])

param_grid = {
    'xgb__max_depth': [3, 4, 5],
    'xgb__n_estimators': [50, 100, 150],
    'xgb__learning_rate': [0.01, 0.1, 0.2]
}

grid_search = GridSearchCV(xgb_pipeline, param_grid, cv=3, scoring='accuracy', verbose=1)

print("Melatih model XGBoost dengan GridSearch...")
grid_search.fit(X_train, y_train)

print("Parameter terbaik untuk XGBoost:", grid_search.best_params_)



# Pipelines untuk model
nb = make_pipeline(tfidf, MultinomialNB())
lr = make_pipeline(tfidf, LogisticRegression(max_iter=1000))
svm = make_pipeline(tfidf, SVC(probability=True))  # SVC harus pakai probability=True untuk soft voting
xgb = grid_search.best_estimator_

# Voting Classifier
voting_clf = VotingClassifier(
    estimators=[
        ('nb', nb),
        ('lr', lr),
        ('svm', svm),
        ('xgb', xgb)
    ],
    voting='soft'
)

# Training model ensemble
voting_clf.fit(X_train, y_train)

# Prediksi dan evaluasi
y_pred = voting_clf.predict(X_test)
print("\n=== Classification Report ===")
print(classification_report(y_test, y_pred, target_names=["negatif", "positif"]))

# print(voting_clf.predict(lines['Komentar']))
print(voting_clf.predict(['Program Makan Bergizi Gratis tidak berhasil memenuhi kebutuhan gizi']))

df = pd.read_csv('data.csv',sep=';')

komentar_mentah = "@makan Program makan gratis ini mntp bgt buat masyarakat indonesia skrg! 👍👍"

normalize = []

for i in df['Komentar']:
    hasil = preprocess(i)
    normalize.append(hasil)

# print(normalize)
finish= voting_clf.predict(normalize)
# print(finish)

mendukung = np.sum(finish == 1)
menolak = np.sum(finish == 0)

print(f"mendukung: {mendukung}, menolak: {menolak}")
