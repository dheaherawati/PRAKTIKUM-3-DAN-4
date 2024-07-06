
laporan 3 dan 4

laporan 3

## Deteksi Garis dan tepi
import cv2 // membaca dan menampilkan gambar 
import numpy as np// pustaka fundamental untuk komputasi ilmiah dalam Python
import matplotlib.pyplot as plt//membuat berbagai jenis plot dan visualisasi data.

img = cv2.imread("gambar 2.jpg")// untuk membaca gambar

cv2.imshow("Gambar asli parkiran",img)// menampilkan Gambar
cv2.waitKey(0)
cv2.destroyAllWindows()

## Menampilkan tepi pada gambar
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)// untuk Mengonversi gambar berwarna menjadi gambar grayscale
edges = cv2.Canny(img, 100, 150)//  untuk mendeteksi tepi dalam gambar menggunakan metode Canny Edge Detection

cv2.imshow("Gambar asli parkirann",edges)// menampilkan gambar
cv2.waitKey(0)
cv2.destroyAllWindows()

## tampilan menggunakan figure, axis (fungsi matplotlib)
fix, axs = plt.subplots(1, 2, figsize = (10,10))// Membuat gambar subplot dengan satu baris (1), dua kolom (2), dan ukuran gambar figsize=(10, 10).
ax = axs.ravel()
ax[0].imshow(gray, cmap = 'gray')//Menampilkan gambar grayscale di subplot pertama dengan judul "Gambar Grayscale".
ax[0].set_title("gambar grayscale")

ax[1].imshow(edges, cmap = 'gray')//Menampilkan hasil deteksi tepi menggunakan metode Canny di subplot kedua dengan judul "Deteksi Tepi"
ax[1].set_title("gambar grayscale")

### Menampilkan garis pada 
lines = cv2.HoughLinesP(edges, 1, np.pi/100, 30, maxLineGap=5)
img_line = img.copy()

for line in lines:
        x1, y1, x2, y2, = line[0]
        cv2.line(img_line, (x1,y1), (x2,y2), (0,0,225),1)

### tampilan menggunakan figure, axis (fungsi matplotlib)
gray1 = cv2.cvtColor(img_line, cv2.COLOR_BGR2RGB)
fix, axs = plt.subplots(1, 2, figsize = (10,10))
ax = axs.ravel()
ax[0].imshow(gray, cmap = 'gray')
ax[0].set_title("gambar grayscale")

ax[1].imshow(gray1, cmap = 'gray')
ax[1].set_title("gambar edges")


laporan 4


import cv2 // membaca dan menampilkan gambar 
import matplotlib.pyplot as plt//membuat berbagai jenis plot dan visualisasi data.
import numpy as np// pustaka fundamental untuk komputasi ilmiah dalam Python
import skimage
from skimage.feature import graycomatrix,graycoprops

img = skimage.data.astronaut()
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

fig, axs = plt.subplots(1,2, figsize = (10,10))

ax = axs.ravel()
ax[0].imshow(img)
ax[0].set_title("RGB")

ax[1].imshow(img_gray, cmap="gray")
ax[1].set_title("GRAY")

mean = np.mean(img_gray.ravel())
std = np.std(img_gray.ravel())

print(mean, std)

glcm= graycomatrix(img_gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True) //Menghitung GLCM dari gambar grayscale (gray) dengan jarak 1 piksel dan sudut 0 derajat. Parameter levels=256 menunjukkan bahwa ada 256 level intensitas grayscale. symmetric=True menghasilkan GLCM simetris, dan normed=True menghasilkan matriks GLCM yang dinormalisasi.

contrast = graycoprops(glcm, 'contrast')[0,0]//Menghitung beberapa properti statistik dari GLCM yang dihasilkan seperti kontras, dissimilaritas, homogenitas, energi, dan korelasi.
dissimilarity = graycoprops(glcm, 'dissimilarity')[0,0]
homogeneity = graycoprops(glcm, 'homogeneity')[0,0]
energy = graycoprops(glcm, 'energy')[0,0]
correlation = graycoprops(glcm, 'correlation')[0,0]

print(f'contrast :{contrast}')//Untuk mencetak nilai properti GLCM seperti kontras, dissimilaritas, homogenitas, energi, dan korelasi yang telah dihitung,
print(f'dissimilarity :{dissimilarity}')
print(f'homogeneity :{homogeneity}')
print(f'energy :{energy}')
print(f'correlation :{correlation}')