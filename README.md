# ML_Fake_Image_Detection

**Feature extraction** of '[real](https://github.com/fardinabbasi/ML_Fake_Image_Detection/tree/main/real)' and '[fake](https://github.com/fardinabbasi/ML_Fake_Image_Detection/tree/main/fake)' images and implementation of the best classification method (using various machine learning models such as **Random Forest**, **SVM**, and **Logistic Regression**) to identify fake images.

The dataset comprises approximately **3400 images**, including both real and fake images of seas, mountains, and jungles, distributed evenly. AI generative models, including [Stable Diffusion](https://stablediffusionweb.com/#demo), [DALL.E](https://openai.com/dall-e-2), [Dreamstudio](https://beta.dreamstudio.ai/dream), [Crayion](https://www.craiyon.com/), and [Midjourney](https://www.midjourney.com/home/?callbackUrl=%2Fapp%2F), are used to create fake images.

A few sample images are provided below:

| Category | Real Image | Fake Image |
| --- | --- | --- |
| Sea | <img src="/real/810199456_real_none_sea_4.jpg" width="400" height="400"> | <img src="/fake/810199456_fake_stable_sea_2.jpg" width="400" height="400"> |
| Jungle | <img src="/real/810199456_real_none_jungle_8.jpg" width="400" height="400"> | <img src="/fake/810199456_fake_dalle_jungle_2.jpg" width="400" height="400"> |
| Mountain | <img src="/real/810199456_real_none_mountain_2.jpg" width="400" height="400"> | <img src="/fake/810199456_fake_stable_mountain_5.jpg" width="400" height="400"> |

Furthermore, in addition to the required feature extraction process, the deep features are already available as "[features.csv](https://github.com/fardinabbasi/ML_Fake_Image_Detection/blob/main/features.csv)" along with their corresponding labels in "[labels.csv](https://github.com/fardinabbasi/ML_Fake_Image_Detection/blob/main/labels.csv)".

## Data Preparation
### Feature Extraction
In this project, in addition to the provided **deep features**, **handcrafted features** were also extracted. The approach involved utilizing two commonly employed techniques: **Local Binary Patterns (LBP)** and **Fast Fourier Transform (FFT)**.

**LBP** is a texture descriptor technique that characterizes the local structure of an image
by comparing the intensity of a central pixel with its surrounding neighbors. By applying LBP, the project aimed to capture relevant textural details that could contribute
to the understanding and classification of the images or data.The LBP is implemented as below:
```ruby
  def lbp(self, path):
    try:
      image_path = self.image_dir + path
      image = io.imread(image_path, as_gray=True)
      image = resize(image, self.image_size)
      lbp = feature.local_binary_pattern(image, 8, 1, method='uniform')
      histogram, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 59))
      return histogram
    except Exception as e:
      print(e)
      print(image_path)
      print("Something happened in LBP")
```
**FFT** transforms a signal from the time domain to the frequency domain, enabling
the identification of different frequency components within the signal. By using FFT,
we extracted frequency-based features that could provide insights into the underlying
patterns or characteristics of the data.
```ruby
  def fft(self, path):
    try:
      image_path = self.image_dir + path
      image = io.imread(image_path, as_gray=True)

      resized_image = resize(image, self.image_size)

      fft_image = np.fft.fft2(resized_image)
      fft_shifted = np.fft.fftshift(fft_image)

      magnitude_spectrum = np.log(1 + np.abs(fft_shifted))
      return magnitude_spectrum.flatten()
    except Exception as e:
      print(e)
      print(image_path)
      print("Something happened in FFT")
```
### Preprocessing
In this project, the dataset has been preprocessed as below:

1. **Handling Null Values**: In any real-world dataset, there are always a few null values.

2. **Data Cleansing**: Data cleansing is the process of identifying and correcting corrupt or inaccurate records in a dataset. It involves detecting incomplete, incorrect, inaccurate, or irrelevant parts of the data and then taking actions such as replacing, modifying, or deleting the problematic data. For instance:
    - There are a few incorrect labels, such as "forest" or "Jungle" instead of "jungle," "DALL.E," and other derivatives instead of "DALL-E."
    - Removing irrelevant parts of labels, including image formats and student IDs.

3. **Standardization**: In Standardization, we transform our values such that the mean of the values is 0 and the standard deviation is 1.

4. **Test & Train Split**: Train data is used for training the model, validation data is used for tuning hyperparameters and choosing the best model, and test data is used for evaluating the chosen model.

Lables Summary:
<div style="display: flex;">
    <img src="/readme_images/preprocessing1.jpg">
    <img src="/readme_images/preprocessing2.jpg">
    <img src="/readme_images/preprocessing3.png">
</div>

### Dimension Reduction
In this project the **PCA** and **LOL** techniques are used to reduce dimension.

**Linear Optimal Low-Rank Projection (LOL)**:
The key intuition behind LOL is that we can jointly use the means and variances from
each class (like LDA and CCA), but without requiring more dimensions than samples
(like PCA), or restrictive sparsity assumptions. Using random matrix theory, we are
able to prove that when the data are sampled from a Gaussian, LOL finds a better
low-dimensional representation than PCA, LDA, CCA, and other linear methods

**Principal Component Analysis (PCA)**:
PCA is a widely used technique for dimension reduction. It identifies a new set of variables, called principal components, that are linear combinations of the original features.
These components are ordered in terms of the amount of variance they explain in the
data. 

PCA result for reducing **deep features** into 3 dimensions are shown below:

<img src="/readme_images/pca.jpg">

## Classification
For classification, 3 classification models are implemented including **Logistic Regression**, **SVM**, and **Random Forest**.

### Logistic Regression
For training the model, the "**Newton-Cholesky**" solver is used, which is recommended when the number of samples is much larger than the number of features.
The classification report and the confusion matrix are shown below, demonstrating the performance of the model:

| Result | Deep Features | Handcrafted Features |
| --- | --- | --- |
| Classification Report | <img src="/readme_images/LR1.jpg"> | <img src="/readme_images/LR3.png"> |
| Confusion Matrix | <img src="/readme_images/LR2.png"> | <img src="/readme_images/LR4.png"> |

### SVM
The optimization function for **Soft SVM** is written as follows:

$$
\min_{w, b, \xi} \frac{1}{2}\|w\|^2 + C\sum_{i=1}^{n}\xi_i
$$

subject to:

$$
\begin{align*}
& y_i(w^T x_i + b) \geq 1 - \xi_i, \quad i = 1, 2, \ldots, n \\
& \xi_i \geq 0, \quad i = 1, 2, \ldots, n
\end{align*}
$$

**C** is a **hyperparameter** which determines the **trade-off** between lower error or higher margin. In order to choose this hyperparameter, we used **grid search** technique and the best C for deep feature equals to 0.1 and for handcrafted features equals to 1.

| Result | Deep Features | Handcrafted Features |
| --- | --- | --- |
| Classification Report | <img src="/readme_images/svm1.jpg"> | <img src="/readme_images/SVM3.jpg"> |
| Confusion Matrix | <img src="/readme_images/svm2.jpg"> | <img src="/readme_images/SVM4.jpg"> |

### Random Forest
[Random forests](https://www.mygreatlearning.com/blog/random-forest-algorithm/) or random decision forests are an **ensemble learning** method for classification, regression, and other tasks that operates by constructing a multitude of decision trees at training time.

Two importent **hyperparameters** to find in random forest method, are the **number of estimators** and the **maximum depth**. The Best Hyperparameters for **deep features** are found by **Randomized Search CV**:

**Best Hyperparameters: {'n_estimators': 85, 'max_depth': 100}**

The classification report and the confusion matrix are shown as below which demonstrate how well the model works with deep features:
| Classification Report | Confusion Matrix |
| --- | --- |
| <img src="/readme_images/RF1.png"> | <img src="/readme_images/RF2.png"> |

Also the first tree is shown as below:

<img src="/readme_images/RF3.png">

## Clustering
For clustering, 2 models are implemented including **Mini Batch K-Means**, and **Gaussian Mixture Model**.

### Mini Batch K-Means
The **Mini-Batch K-means** algorithm is utilized as a solution to the increasing computation time of the traditional **K-means** algorithm when analyzing **large datasets**.

The clustering results for different number of clusters are shown as below:

| Number of Clusters | Deep Features | Handcrafted Features |
| --- | --- | --- |
| 2 | <img src="/readme_images/Clustering1.png"> | <img src="/readme_images/c1.png"> |
| 3 | <img src="/readme_images/Clustering2.jpg"> | <img src="/readme_images/C2.png"> |
| 6 | <img src="/readme_images/Clustering3.png"> | <img src="/readme_images/C3.png"> |
| 9 | <img src="/readme_images/Clustering4.png"> | <img src="/readme_images/C4.png"> |
| 50 | <img src="/readme_images/Clustering5.png"> | <img src="/readme_images/C5.png"> |

The best number of clusters, is **elbow point** in the plot of **inertia** with respect to number of clusters:
| Deep Features | Handcrafted Features |
| --- | --- |
| <img src="/readme_images/C6.png"> | <img src="/readme_images/Clustering6.png"> |

### Gaussian Mixture Model
**Gaussian Mixture Models (GMMs)** are powerful probabilistic models used for clustering and density estimation. By combining multiple Gaussian components, GMMs
can represent various data patterns and capture the underlying structure of the data.

The **Expectation-Maximization (EM)** algorithm is commonly employed to estimate the parameters of Gaussian Mixture Models (GMMs), including the mean, covariance, and cluster weights.

To find the optimal number of components in a cluster, the **Akaike Information Criterion (AIC)** and **Bayesian Information Criterion (BIC)** are commonly used measures.

| Deep Features | Handcrafted Features |
| --- | --- |
| <img src="/readme_images/m1.png"> | <img src="/readme_images/m2.png"> |
