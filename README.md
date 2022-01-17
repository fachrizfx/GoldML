# Laporan Proyek Machine Learning 
This project is a part of Dicoding's Applied Machine Learning Class Project. Project resources that have been used to help build this model are listed below in the reference area. Please cite this GitHub Repository page if you used our model as a guidance reference. Images that have been used in this markdown may not be rendered due to one or another reason, please try refreshing the page to see the image rendered.

## Domain Proyek

Emas adalah salah satu logam berharga yang digunakan sebagai sebuah bentuk mata uang di beberapa negara. Dikutip dari [01] emas bisa di jadikan investasi yang aman karena memiliki nilai yang stabil. Oleh karena itu, tujuan yang diharapkan adalah kita bisa memprediksi harga emas agar bisa memperoleh beberapa kegunaan seperti memprediksi kapan untuk membeli dan menjual emas.

## Business Understanding

### Problem Statements

Setelah mengetahui tujuan model, kita bisa membuat model yang menjawab permasalahan-permasalahan seperti berikut:

-   Bagaimana persentase kenaikan harga emas dari periode January 2019 hingga 2022?
-   Bagaimana harga emas untuk dikemudian hari?
-   Bagaimana rata-rata kenaikan harga emas setiap hari?

### Goals

Untuk menjawab semua permasalahan diatas, kita bisa membuat goal sebagai berikut:

-   Mengetahui pola kenaikan harga emas setiap hari.
-   Membuat model machine learning seakurat mungkin dengan tujuan memprediksi harga emas di kemudian hari.
-   Mengetahui kenaikan harga emas setiap hari berdasarkan pola data historisnya.

    ### Solution statements

    -   Menggunakan algoritma K-Nearest Neighbor, Random Forest, Boosting Algorithm, dan Neural Network untuk menyelesaikan masalah regresi.
    -   Melakukan hyperparameter tuning agar memiliki performa lebih baik
    -   Mengevaluasi model dengan metrik MSE, RMSE, atau MAE
    -   Menggunakan fungsi pct_change() untuk mengetahui kenaikan pada data

## Data Understanding

Dataset yang akan kita gunakan ini di dapatkan dari Yahoo Finance. Dataset ini berisi data historis harian harga emas dari tanggal Jan 02, 2019 sampai hari model ini dibuat yaitu Jan 10, 2022. Dataset ini memiliki 767 rows dan 7 columns. Dataset dapat diunduh pada link berikut: [YahooFinance].

Pada dataset ini saya tidak melakukan reduksi dimensi dengan PCA dikarenakan saya ingin menjadi 'Open' saja yang sebagai data input untuk memprediksi harga emas. Alasanya adalah karena jika kita ingin memprediksi harga emas kita belum tahu nilai 'High' dan 'Low'nya.

Untuk mendeteksi outliers kita bisa menggunakan beberapa teknik, antara lain:

-   Hypothesis Testing
-   Z-score method
-   IQR Method

Sebelumnya, untuk mengetahui apakah ada outliers pada data kita, kita bisa melakukan teknik visualisasi Boxplot. Maka dari itu kita akan melakukan visualisasi terlebih dahulu.

![Gambar3](https://camo.githubusercontent.com/6db15b998f6de510b4dbf243ff2d5a338a94fd9abd64c355dd1b6a8dbd78152d/68747470733a2f2f64726976652e676f6f676c652e636f6d2f75633f6578706f72743d766965772669643d315f4e396e4e594b596b5570547172344f343951333853754b667347364276726b)

![Gambar4](https://camo.githubusercontent.com/5f0fbbfe7db212812996cdba784dcace8b02726020b15960ec6c2ae23b249d38/68747470733a2f2f64726976652e676f6f676c652e636f6d2f75633f6578706f72743d766965772669643d315f4e66314c6f62532d4b6e77796e70596c39784f5870492d77536653326d5754)

Dari visualasi boxplot diatas kita bisa lihat bahwa dataset kita memiliki outliers pada column 'Volume'. Kita bisa saja menghapusnya tetapi karena ini merupakan dataset yang berpengaruh terhadap urutan data, maka jika kita hilangkan akan menghasilkan data yang hilang. Oleh karena itu kita akan menggantinya dengan nilai Median.

Untuk lebih memastikan adanya outliers pada data kita, kita bisa melihat histogram pada data. Berikut adalah output histogram 'Volume' pada data:

![Gambar5](https://camo.githubusercontent.com/3a5859ce5912c9f4027e0396cb2b9d1f37f88a1ee9f6df2804fe027931a16d97/68747470733a2f2f64726976652e676f6f676c652e636f6d2f75633f6578706f72743d766965772669643d315f536967703132647a7537794261474f54666f66394658706831645537414347)

Jika kita lihat pada [03] bagian histogram ia **menyatakan jika data pada histogram terdistribusi ke arah kiri itu menunjukan adanya outliers pada data.** Pada histogram volume kita, kita bisa lihat bahwa data terdistribusi ke arah kiri yang mengindikasikan adanya outliers pada data.

Dikutip dari [03] tahap untuk memastikan adanya outliers adalah dengan cara melihat skewness value. Skewness dari rentan -1 sampai 1 terbilang normal distribution, dan nilai yang perubahannya sangat besar mengindikasikan adanya outliers. Untuk melihat skewness value kita bisa menuliskan code berikut:

```
print('skewness value of Adj Close: ',df['Adj Close'].skew())
print('skewness value of Volume: ',df['Volume'].skew())
print('skewness value of Open: ',df['Open'].skew())
print('skewness value of High: ',df['High'].skew())
print('skewness value of Low: ',df['Low'].skew())
```

Output:

```
skewness value of Adj Close:  -0.4961562768585715
skewness value of Volume:  4.457710908666603
skewness value of Open:  -0.49109950423367865
skewness value of High:  -0.4754118412774641
skewness value of Low:  -0.500601583042607
```

Dari output diatas kita bisa lihat bahwa data pada column 'Volume' memiliki outliers, hal ini dikarena ia memiliki nilai 4.8 yang berarti rightly skewed yang mengindikasikan adanya outliers.

### Variabel-variabel pada Gold Historical Price dataset adalah sebagai berikut:

-   Open : harga emas saat pembukaan hari
-   High : harga maksimal emas pada hari itu
-   Low : harga terendah emas pada hari itu
-   Close : harga emas saat penutupan hari
-   Adj Close : harga emas saat penutupan hari yang disesuaikan oleh beberapa faktor. Untuk lebih lengkapnya bisa dilihat pada link berikut: [Kaggle]
-   Volume : volume perdagangan. Untuk lebih lengkapnya bisa dilihat pada link berikut: [02]

### Data Loading

Pada tahap pertama seperti biasa kita akan import semua library yang dibutuhkan dan melakukan data loading. Pada project ini saya akan melakukan data loading menggunakan url yang didapatkan pada link berikut [YahooFinance]. Setelah melakukan loading kita akan melihat output sebagai berikut:

![Gambar1](https://camo.githubusercontent.com/7ceb46cba92cb4b1630c68ff4aed47587609d795c8df890eae786c323a2e0bfd/68747470733a2f2f64726976652e676f6f676c652e636f6d2f75633f6578706f72743d766965772669643d315f4c63433961557141614c615555435175485734307157544e494c38782d7479)

Dari gambar diatas kita bisa lihat bahwa terdapat tanggal-tanggal yang hilang seperti tanggal 5 dan 6 January 2019. Jika kita lihat pada kalender, tanggal-tanggal yang hilang adalah tanggal mereka tidak buka (open) yaitu pada hari Sabtu dan Minggu. Selanjutnya kita bisa lihat informasi mengenai data pada dataset ini menggunakan code berikut:

```
df.info()
```

Output:

![Gambar2](https://camo.githubusercontent.com/1ca87a6380679a316085772e4ee1558b2dd4e37f217b84e034a9aace7030723b/68747470733a2f2f64726976652e676f6f676c652e636f6d2f75633f6578706f72743d766965772669643d315f4d52394e384c583268514633554b2d7235677055644d4d57724272536e376d)

Kita bisa lihat bahwa dataset kita tidak memiliki null data. Data type untuk dataset ada tiga yaitu:

-   float64
-   int64
-   object

### Multivariate Analysis

Setelah kita menangani outliers kita dapat melanjutkan ke tahap Data Analysis menggunakan Multivariate Analysis. Berikut adalah gambar Pairplot yang menunjukan relasi pasangan antar data dalam dataset.

![Gambar6](https://camo.githubusercontent.com/23ce454b84a285a4b9dbc55774f8401750df6704783ef34f53f64f7a1cce833a/68747470733a2f2f64726976652e676f6f676c652e636f6d2f75633f6578706f72743d766965772669643d315f75716f61516f63696344774e736a795f43633173776562416e304754345a6a)

Dari grafik diatas kita bisa lihat bahwa column 'Open', 'High', dan 'Low' memiliki korelasi yang tinggi dengan output variabel kita, yaitu 'Adj Close'. Maka kita bisa drop column 'Volume', dan 'Close'. Alasannya adalah column Volume memiliki korelasi yang rendah sekali, dan column close tidak berguna karena outcome variabel kita adalah Adj Close . Perbedaan keduanya ada pada bagian variabel-variabel di atas.

## Data Preparation

Karena dataset ini tidak memiliki fitur kategori maka kita tidak perlu melakukan Data Transform.

### Menangani Outliers

Selanjutnya kita akan menggantikan nilai outlier dengan nilai Median pada data. Dikutip dari [03] tidak di rekomendasikan untuk menggantikannya dengan nilai Mean karena sangat rentan terhadap outlier. Ada beberapa teknik untuk menangani outliers, antara lain:

-   Hypothesis Testing
-   Z-score method
-   IQR Method

Disini saya memilih untuk menggunakan IQR. Alasannya adalah saya lebih sering menggunakan metode ini dan juga dari metode yang digunakan pada [03] adalah metode IQR.

## Data Spliting

Sebelum masuk ke tahap selanjutnya yaitu Data Transformation kita harus melakukan teknik train_test_split dari Library Scikit Learn agar tidak terjadi kebocoran data. Pada cell pertama saya akan melakukan split. Jumlah test_size yang saya gunakan adalah 15% alasan saya menggunakan 15% adalah jika kita menggunakan 20% data test akan ada sebanyak 154 yang berarti terlalu banyak untuk dataset yang kecil seperti ini. Paramter shuffle saya jadikan False agar data tetap dalam urutan waktu yang merupakan hal sangat penting. Untuk melakukannya dapat menulis code berikut:

## Data Transformation

Pada tahap ini saya akan melakukan Standarisasi. Untuk standarisai kita memiliki bebrapa pilihan, antara lain:

-   MinMaxScaler
-   StandardScaler
-   dll

Disini saya akan menggunakan StandardScaler. Dikutip dari [06] "MinMaxScaler adalah jenis scaler yang menskalakan nilai minimum dan maksimum masing-masing menjadi 0 dan 1. Sementara StandardScaler menskalakan semua nilai antara min dan max sehingga berada dalam kisaran dari min ke max". Itu lah alasan saya menggunakan StandardScaler.

## Modeling

Seperti yang disebutkan pada bagian Solution Statement Model ini akan menyelesaikan masalah regresi menggunakan beberapa algoritma antara lain, K-Nearest Neighbor, Random Forest, Boosting Algorithm, dan Neural Network. Model yang memiliki nilai error yang paling sedikit adalah model yang akan dipilih.

### Development

Seperti yang telah sebutkan, saya akan menggunakan algoritma K-Nearest Neighbor, Random Forest, Boosting Algorithm, dan Neural Network. Pada cell pertama saya akan membuat DataFrame untuk tahap evaluation nanti. Pada tahap development saya akan menggunakan teknik GridSearchCV dari Library Scikit Learn pada model KNN, Random Forest, dan Boosting Algorithm, untuk menemukan hyperparamter yang tepat.

#### KNN

Algoritma KNN bekerja dengan cara menentukan jumlah tetangga yang dinotasikan dengan K, kemudian algoritma akan menkalkulasi jarak antara data baru dengan data poin K. Selanjutnya adalah algoritma akan mengambil sejumlah nilai K terdekat, lalu menentukan kelas dari data baru tersebut.

Dilihat dari [04], algoritma KNN secara default menggunakan metrik Minskowski, tetapi terdapat juga metrik yang lain yaitu Euclidean, dan Manhattan.

Metrik Euclidean menghitung jaraknya sebagai akar kuadrat dari jumlah selisih kuadrat antara titik a dan b. Sedangkan Metrik Euclidean merupakan generalisasi dari Euclidean dan Manhattan distance. Lalu metrik manhattan dihitung dengan menkalkulasikan jumlah dari nilai absolute dua vektor. Semua itu dapat dituliskan sebagai berikut:

![Gambar7](https://www.saedsayad.com/images/KNN_similarity.png)

Sumber: https://saedsayad.com/k_nearest_neighbors_reg.htm

Jika kita lihat output pada fungsi '.best*params*', kita bisa lihat bahwa hyperparamter yang tepat untuk model KNN ini adalah 'brute' untuk algoritma, 'minkowski' untuk metrik, 10 untuk n_neighbors. Maka dari itu model KNN akan saya gunakan parameter tersebut. Paramter 'n_neighbors' menentukan banyaknya nilai K pada model kita. Paramter selanjutnya adalah 'algorithm', pada model ini saya menggunakan algoritma 'Brute'. Algoritma ini mengandalkan kekuatan komputasi, cara kerjanya adalah dengan mencoba setiap kemungkinan sehingga dapat meminimalisir jumlah error.

#### Random Forest

Algoritma random forenst adalah salah satu algoritma supervised learning. Ia termasuk ke dalam kategori ensemble learning. Teknik untuk membuat model ensemble ada dua, yaitu bagging, dan boosting. Cara kerjanya cukup simple yaitu pertama-tama akan dilakukan pemecahan data (bagging) secara acak, setelah itu akan di masukan ke dalam algoritma decision tree. Untuk prediksi akhir dari model akan dilakukan kalkulasi rata-rata prediksi dari seluruh pohon dalam model ensemble, prediksi akhir dengan cara ini hanya berlaku pada kasus regresi. Pada kasus klasifikasi prediksi akhir akan diambil dari prediksi terbanyak pada seluruh pohon.

Berdasarkan output GridSearch parameter yang tepat adalah None untuk 'max_depth', dan 100 untuk 'n_estimators'. Paramter 'n_estimators' adalah jumlah pohon yang ada dalam forest, semakin banyak jumlah pohon akan memberikan kinerja model yang lebih baik. Salah satu kekurangan jumlah 'n_estimators' yang tinggi adalah membuat kode lebih lambat.

#### Boosting Algorithm

Boosting algorithm sama seperti random forest, yaitu sama-sama termasuk dalam kategori ensemble, bedanya adalah algoritma ini membuat model ensemble dengan cara boosting dibanding bagging. Pada model ensemble ini model akan dilatih secara berurutan dibanding secara pararel. Cara kerjanya juga cukup simple yaitu dengan membangun model dari data, lalu akan membuat model kedua yang bertujuan untuk memperbaiki kesalahan model pertama. Model akan terus ditambahkan sampai pada tahap data latih sudah terprediksi dengan baik atau telah mencapai batas maksimum model.

Output dari GridSearch menunjukan bahwa hyperparamter yang tepat adalah 0.5 untuk 'learning_rate', dan untuk 76 'n_estimators'. Paramter 'n_estimators' seperti yang dijelaskan pada bagian Random Forest adalah jumlah pohon yang ada dalam forest. Semakin banyak jumlahnya maka akan memberikan kinerja mode yang lebih baik, kekurangannya adalah semakin tinggi jumlah pohon akan membuat kode lebih lambat. Parameter selanjutnya adalah 'learning_rate', parameter ini mengontrol loss function yang akan menkalkulasi weight dari base models yang ada, sehingga jumlah learning_rate yang pas akan memberikan performa yang lebih.

#### Neural Network

Neural Network adalah salah satu model yang populer digunakan. Model ini bekerja dengan adanya input layer, dan output layer, namun ada juga hidden layer. Pembahasan lebih lengkapnya tidak akan dibahas disini, tetapi dapat dilihat pada link berikut: [07]. Layer-layer tersebut dapat memiliki ratusan ribu parameter bahkan jutaan paramter. Namun agar model ini bekerja, layer-layer tersebut akan mencari pola-pola pada data.

Pada model Neural Network saya tidak memakai teknik GridSearch. Sehingga paramater yang saya rubah pada model yang sudah di fine tuning hanyalah paramter 'learning_rate' pada optimizer, dan jumlah 'epoch' pada bagian fungsi .fit(). Hal ini bertujuan agar saya bisa melihat perbedaan performa saat training, apakah jarak antara data asli dekat atau lumayan jauh. Epoch adalah melatih model Neural Network dengan data latih yang sama selama cycle yang ditentukan. Nah paramter 'epoch' pada fungsi .fit() ini menentukan jumlah cycle untuk melatih model.

## Evaluation

Pada bagian metrik saya akan gunakan adalah MSE, pada kasus ini saya memilih metrik MSE. Alasan saya memilih metrik ini adalah karena MAE adalah skor linear, yang memiliki arti perbedaan individu antara data akan diberi bobot yang sama dalam rata-rata. Walaupun menurut beberapa sumber lebih baik memilih metrik RMSE, saya tetap memilih metrik MSE. Berdasarkan [05] MSE bekerja dengan melakukan pengurangan data aktual dengan data prediksi dan hasilnya dikuadratkan (squared) lalu dijumlahkan secara keseluruhan dan membaginya dengan banyaknya data.

![Gambar9](https://1.bp.blogspot.com/-BhCZ4B8uQqI/X-HjGU2kcsI/AAAAAAAACkQ/EdNE0ynOwDIR9RYD_uxRMhps2DFFs5jgQCNcBGAsYHQ/s364/rumus%2BMSE.jpg)

Sumber: https://www.khoiri.com/2020/12/pengertian-dan-cara-menghitung-mean-squared-error-mse.html

Hasil evaluasi adalah sebagai berikut:

![Gambar10](https://camo.githubusercontent.com/0e7eb05d3ce601b45218310ac7eb793c8298254805eb6a74b3d23896bf99b8ce/68747470733a2f2f64726976652e676f6f676c652e636f6d2f75633f6578706f72743d766965772669643d316e6b79506f6d6f3062485669487a6f4c51635363596a67383539584877586e6a)

![Gambar11](https://camo.githubusercontent.com/a39d7f137c186dad1ed118e657dfe9e4f789d68f52b1d7d3ee2200d509e610c7/68747470733a2f2f64726976652e676f6f676c652e636f6d2f75633f6578706f72743d766965772669643d3161477a4d696a70624e695234686768676776756c2d4c733150475f6b534d526a)

Dari grafik diatas kita bisa simpulkan bahwa model dengan algoritma KNN dan KNN yang sudah di fine tuning adalah model yang memiliki kinerja yang paling bagus, keduanya memiliki performa yang sama. Oleh karena itu kita bisa mengambil model KNNTune1 dan KNN, tetapi saya akan mengambil KNNTune1.

Setelah selesai melakukan semua proses, sekarang kita bisa menjawab permasalahan pada problem statement.

-   Bagaimana persentase kenaikan harga emas dari periode January 2019 hingga 2022?
    Kenaikan harga dari January hingga 2022 mencapai 34.6%! Atau sebesar 59.7 AUD.

-   Bagaimana harga emas untuk dikemudian hari?
    Saya akan melakukan prediksi dengan Test set, hasil dari model cukup memuaskan dengan perbedaan harga sebesar 0.2 AUD! Dari data aslinya yaitu 229 AUD yang berarti model memprediksi harga emas sebesar 228.8 AUD.

-   Bagaimana rata-rata kenaikan harga emas setiap hari?
    Rata-rata kenaikan harga emas per hari pada data kita mencapai sebesar 0.043%. Persentase kenaikan tersebut sangat sedikit hal ini di karenakan terdapat beberapa nilai yang mengalami penurunan. Penurunan pada harga emas adalah hal yang wajar/normal.

**---Ini adalah bagian akhir laporan---**

## Referensi

<br />[[yahoofinance]]ETFS Physical Gold (GOLD.AX) Stock Historical Prices & Data. (n.d.). Yahoo Finance. Retrieved January 10, 2022, from https://finance.yahoo.com/quote/GOLD.AX/history?period1=1546300800&period2=1641772800&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true
<br />[[01]]Cara Aman dan Waktu yang Tepat untuk Investasi Emas. (2021, October 2). CNN Indonesia. Retrieved January 10, 2022, from https://www.cnnindonesia.com/ekonomi/20210928113408-97-700375/cara-aman-dan-waktu-yang-tepat-untuk-investasi-emas
<br />[[02]]Rodo, F. (2019, December 18). Harga Emas Terkait Volume Perdagangan. Indo Gold. Retrieved January 10, 2022, from https://blog.indogold.id/harga-emas-terkait-volume-perdagangan/
<br />[[03]]Magaga, A. (2021, April 5). Identifying, Cleaning and replacing outliers | Titanic Dataset. Medium. Retrieved January 10, 2022, from https://medium.com/analytics-vidhya/identifying-cleaning-and-replacing-outliers-titanic-dataset-20182a062893
<br />[[04]]Scikit Learn. (n.d.). sklearn.neighbors.KNeighborsRegres. Retrieved January 10, 2022, from https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html
<br />[[05]]K. (2020, December 22). Pengertian dan Cara Menghitung Mean Squared Error MSE. Khoiri. Retrieved January 10, 2022, from https://www.khoiri.com/2020/12/pengertian-dan-cara-menghitung-mean-squared-error-mse.html
<br />[[06]]Kumar, A. (2020, July 27). MinMaxScaler vs StandardScaler – Python Examples. Vitalflux. Retrieved January 10, 2022, from https://vitalflux.com/minmaxscaler-standardscaler-python-examples/
<br />[[07]]Hardesty, L. (2017, April 17). Explained: Neural networks. MIT News. Retrieved January 10, 2022, from https://news.mit.edu/2017/explained-neural-networks-deep-learning-0414
<br />[[kaggle]]Siddhartha, M. (2021, July 20). Gold Prices Prediction Dataset (Version 1) [Data for this study is collected from November 18th 2011 to January 1st 2019 from various sources. The data has 1718 rows in total and 80 columns in total. Data for attributes, such as Oil Price, Standard and Poor’s (S&P) 500 index, Dow Jones Index US Bond rates (10 years), Euro USD exchange rates, prices of precious metals Silver and Platinum and other metals such as Palladium and Rhodium, prices of US Dollar Index, Eldorado Gold Corporation and Gold Miners ETF were gathered. The dataset has 1718 rows in total and 80 columns in total. Data for attributes, such as Oil Price, Standard and Poor’s (S&P) 500 index, Dow Jones Index US Bond rates (10 years), Euro USD exchange rates, prices of precious metals Silver and Platinum and other metals such as Palladium and Rhodium, prices of US Dollar Index, Eldorado Gold Corporation and Gold Miners ETF were gathered. The historical data of Gold ETF fetched from Yahoo finance has 7 columns, Date, Open, High, Low, Close, Adjusted Close, and Volume, the difference between Adjusted Close and Close is that the closing price of a stock is the price of that stock at the close of the trading day. Whereas the adjusted closing price takes into account factors such as dividends, stock splits, and new stock offerings to determine a value. So, Adjusted Close is the outcome variable which is the value you have to predict.]. Kaggle. https://www.kaggle.com/sid321axn/gold-price-prediction-dataset

<br />
<br />

[yahoofinance]: https://finance.yahoo.com/quote/GOLD.AX/history?period1=1546300800&period2=1641772800&interval=1d&filter=history&frequency=1d&includeAdjustedClose=true
[01]: https://www.cnnindonesia.com/ekonomi/20210928113408-97-700375/cara-aman-dan-waktu-yang-tepat-untuk-investasi-emas#:~:text=Emas%20dianggap%20salah%20satu%20instrumen,namun%20penurunannya%20tidak%20terlalu%20tajam.
[02]: https://blog.indogold.id/harga-emas-terkait-volume-perdagangan/
[03]: https://medium.com/analytics-vidhya/identifying-cleaning-and-replacing-outliers-titanic-dataset-20182a062893
[04]: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html
[05]: https://www.khoiri.com/2020/12/pengertian-dan-cara-menghitung-mean-squared-error-mse.html#:~:text=Cara%20menghitung%20Mean%20Squared%20Error%20(MSE)%20adalah%20melakukan%20pengurangan%20nilai,dengan%20banyaknya%20data%20yang%20ada.
[06]: https://vitalflux.com/minmaxscaler-standardscaler-python-examples/#:~:text=The%20MinMaxscaler%20is%20a%20type,range%20from%20min%20to%20max.
[07]: https://news.mit.edu/2017/explained-neural-networks-deep-learning-0414
[kaggle]: https://www.kaggle.com/sid321axn/gold-price-prediction-dataset
