# Laporan Proyek Machine Learning 
#
#### Nama   : Ramdani Tarjianto
#### Email  : ramdani.tarjianto83@gmail.com
#
#
#
#
#


# Judul
##### Mendeteksi Penyakit Jantung dengan Algortima Neural Network dan Membandingkan Beberapa Algoritma Klasifikasi (Machine Learning)
#
#
#
## 1.DOMAIN PROYEK

###### 1.1 LATAR BELAKANG
Heart Disease, juga dikenal sebagai penyakit cardiovascular diseases (CVD) adalah penyebab kematian nomor 1 di dunia. Penyakit cardiovascular diseases (CVD) adalah gangguan pada jantung dan pembuluh darah termasuk penyakit jantung koroner, penyakit serebrovaskular, gagal jantung, dan jenis patologi lainnya. Secara keseluruhan, penyakit cardiovascular menyebabkan kematian sekitar 17 juta orang di seluruh dunia setiap tahun, dengan angka kematian meningkat untuk pertama kalinya dalam 50 tahun di Inggris (Chicco and Jurman, 2020). Karena masih kurangnya kesadaran akan gaya hidup sehat dan kurangnya informasi tentang penyakit jantung, yang mungkin membuat gejala awal tidak dapat dikenali. Proses deteksi penyakit jantung dapat dilakukan secara manual yaitu konsultasi langsung dengan ahli jantung dan beberapa pemeriksaan laboratorium, kemudian ahli jantung harus berkonsultasi kembali. Tentunya hal ini membutuhkan biaya yang relatif besar. Karena tingginya risiko kematian, maka diperlukan suatu sistem yang dapat mendeteksi penyakit jantung pada pasien secara akurat dengan harga yang murah (Wibisono and Fahrurozi, 2019). 

###### 1.2 MASALAH DALAM DOMAIN
Cara menerapkan algoritma machine learning untuk memprediksi pasien dengan penyakit jantung agar dapat memberikan perhatian yang lebih kepada pasien tersebut dan dapat meningkatkan pelayanan kesehatan yang lebih baik.

###### 1.3 HASIL RISET TERKAIT
Terdapat penelitian yang bertujuan untuk memprediksi pasien dengan penyakit jantung menggunakan algoritma C4.5  (http://research.pps.dinus.ac.id/index.php/Cyberku/article/view/4/4).
#
#
#
#
#
## 2.BUSINESS UNDERSTANDING
###### 2.1 PROBLEM STATEMENTS
Mendeteksi Penyakit Jantung dengan Algortima Neural Network dan Membandingkan Beberapa Algoritma Klasifikasi Machine Learning
###### 2.2 TUJUAN
- Melakukan evaluasi model prediksi penyakit jantung.
- Mengetahui faktor-faktor penyakit atau variabel bebas seperti gula darah, denyut jantung, kolesterol, dan lain-lain yang mempengaruhi terjadinya penyakit jantung.
- Membuat model penyakit jantung menggunakan algortima machine learning.

###### 2.3 MANFAAT 
- Memprediksi seseorang yang berpotensi penyakit jantung secara akurat yang berguna untuk meningkatkan pelayanan terhadap pasien tersebut.
- Membantu staf medis memprediksi apakah seorang pasien terkena penyakit jantung dengan menggunakan algoritma machine learing
- Memberitahu masyarakat luas tentang bagaimana cara bekerja dalam memprediksi penyakit jantung menggunakan algoritma machine learning

###### 2.4 SOLUTION STATEMENTS
Saya mengajukan algoritma machine learning sebagai solusi permasalahan, yaitu SVM dan membadingkannya dengan Logistic Regression, Neural Network, dan Naive Bayes.

Metode Neural Network (NN) (Derisma, 2020), Naive Bayes (Klasifikasi, Putra and Rini, 2019) dan Logistic Regression (Kumar and Devi Gandhi, 2018) diusulkan oleh banyak peneliti untuk prediksi penyakit jantung.

Neural Network memiliki kelebihan dalam mengambil keputusan dengan memprediksi data. Karena masukan yang digunakan dalam memprediksi penyakit lebih banyak dan diagnosis harus dilakukan pada tahap yang berbeda, Neural Network memperluas kemampuan prediktifnya pada tingkat hierarki yang berbeda dalam struktur jaringan berlapis-lapis. Struktur berlapis-lapis ini membantu dalam memilih fitur dari kumpulan data pada skala yang berbeda untuk menyempurnakannya menjadi fitur yang lebih spesifik (Boddu and Subhadra, 2019). Neural Network mempunyai kelemahan yaitu permasalahan dalam penentuan parameter sehingga perlu dilakukan eksperimen dalam menentukan tiap parameternya (Somantri and Cilacap, 2018). 

Naive Bayes akurasi yang dihasilkan cukup baik, hal ini karena keunggulan dari metode naive bayes sendiri yaitu mampu melakukan klasifikasi meskipun memiliki data training yang sedikit untuk estimasi parameternya (Devita, Wahyu Herwanto and Wibawa, 2018). Namun Naive Bayes memiliki kelemahan yaitu atribut atau fitur independen sering salah dan hasil estimasi probabilitas tidak dapat berjalan optimal (Prabowo and Muljono, 2018).

Logistic Regression sangat berguna untuk memprediksi ada atau tidaknya karakteristik atau hasil berdasarkan nilai dari set variabel prediktor (Putra and Rini, 2020) yang mana ini tepat untuk menentukan pasien dalam hal terkena penyakit jantung atau tidak. Tetapi Logistic Regression memiliki kelemahan yaitu rentan terhadap underfitting pada dataset yang kelasnya tidak seimbang, sehingga akan menghasilkan akurasi yang rendah (Rianto and Wahono, 2015).
#
#
#
#
#
## 3. DATA UNDERSTANDING
 Pada dataset yang digunakan pada proyek akhir ini adalah heart.csv. Dataset ini adalah untuk memprediksi apakah seseorang akan mengidap penyakit jantung atau tidak menggunakan teknik pembelajaran mesin. Dataset ini merupakan data nyata termasuk fitur penting pasien. Ada beberapa fitur yang harus dipahami dengan baik untuk memastikan atau mendapatkan akurasi dengan baik, Terdapat 1025 sampel data dan 14 column yang digunakan pada dataset ini, column tersebut diantaranya terdiri dari:
 - age = merupakan umur pasien
 - sex = merupakan jenis kelamin pasien
 - chest pain type  = merupakan jenis nyeri dada pada pasien
 - resting blood = merupakan jumlah resting blood pada pasien
 - serum cholesterol = merupakan jumlah kolesterol pada pasien
 - fasting blood sugar = merupakan jumlah gula darah pada pasien
 - resting electrocardiographic = merupakan jumlah elektrokardiografi pada pasein
 - maximum heart rate achieved = merupakan jumlah detak jantung maksimal pada pasien
 - exercise induced angina = merupakan jumlah agina pada pasein
 - oldpeak = merupakan jumlah oldpeak pada pasien
 - slope = merupakan jumlah slope pada pasien
 - cha = merupakan jumlah cha pada pasien
 - thal = merupakan jumlah thal pada pasien
 - target = merupakan nilai yang akan di prediksi 


Nilai yang akan diprediksi yaitu kolom target yang berisi nilai positif dan negatif, ini akan diprediksi menggunakan algoritma klasifikasi Neural Network yang akan mengeluarkan output 0 atau 1. 0 artinya pasien tersebut terkena penyakit jantung dan 1 tidak terkena penyakit jantung berdasarkan  dengan input dari 14 column tersebut. Dataset ini diambil dari kaggle
(https://www.kaggle.com/johnsmith88/heart-disease-dataset)
#
#
#
#
#
## 4. DATA PREPARATION
melakukan Data Preprocessing pada dataset heart disease (‘hear.csv’): dengan Memeriksa Missing Values, Outliers.kemudian missing values tersebut di isi dengan bagian tersebut dengan nilai rata-rata atau modus. Jika ada sampel yang merupakan outliers, sample tersebut harus di hapus.
![Data Preparations](https://raw.githubusercontent.com/RamdaniTarjianto/Dicoding/main/Screenshot%20(182).png)
tidak ada missing value pada dataset tersebut jadi data sudah bisa langsung di bagi menjadi train dan test menggunakan train_test_split
#
#
#
#
#
## 5. MODELING
![](https://raw.githubusercontent.com/RamdaniTarjianto/Dicoding/main/Screenshot%20(183).png)
untuk mendeteksi pasein dengan penyakit jantung saya menggunakn algortima Neural Network karena Neural Network memiliki kelebihan dalam mengambil keputusan dengan memprediksi data. Karena masukan yang digunakan dalam memprediksi penyakit lebih banyak dan diagnosis harus dilakukan pada tahap yang berbeda, Neural Network memperluas kemampuan prediktifnya pada tingkat hierarki yang berbeda dalam struktur jaringan berlapis-lapis. Struktur berlapis-lapis ini membantu dalam memilih fitur dari kumpulan data pada skala yang berbeda untuk menyempurnakannya menjadi fitur yang lebih spesifik (Boddu and Subhadra, 2019) dan menghasilkan akurasi 81%. kemudian juga saya membandingkan dengan beberapa algoritma machine learning lainnya agar mendapatkan model terbaik untuk permasalahan ini, model machine learning seperti Logistic Regression , Naive Bayes, SGD dan menghasilakn nilai akurasi sebagai berikut :
| No | Algoritma | Nilai Akurasi |
| --- | --- | --- |
| 1 | Neural Network | 0.81 |
| 2 | Stochastic Gradient Descent | 0.59 |
| 3 | Logistic Regression | 0.8 |
| 4 | Naive Bayes | 0.8 |

#
#
#
#
#
## 6. EVALUATION
![](https://raw.githubusercontent.com/RamdaniTarjianto/Dicoding/main/Screenshot%20(184).png)
Pada gambar tersebut menjelaskan, dengan menggunakan confusion matrix kita akan mendapatkan 4 sebagai representasi hasil proses klasifikasi yakni True Positive (TP) =  91, True Negative (TN) = 75, False Positive (FP) = 24, False Negative (FN) = 15. Nilai True Positif (TP) merupakan data positif yang terdeteksi dengan benar, sedangkan nilai True Negative (TN) merupakan data negatif yang terdeteksi dengan benar. False Positive (FP) merupakan data positif yang terdeteksi dengan salah, sedangkan False Negative (FN) merupakan data negatif yang terdeteksi dengan salah.



