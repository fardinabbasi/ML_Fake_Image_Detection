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
In this project, besides the given deep features, handcrafted features were needed to be extracted. The approach involved using two common techniques, namely Local Binary Patterns
(LBP) and Fast Fourier Transform (FFT).

LBP is a texture descriptor technique that characterizes the local structure of an image
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
FFT transforms a signal from the time domain to the frequency domain, enabling
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
