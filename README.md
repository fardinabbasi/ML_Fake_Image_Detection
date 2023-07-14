# ML_Fake_Image_Detection

**Feature extraction** of '[real](https://github.com/fardinabbasi/ML_Fake_Image_Detection/tree/main/real)' and '[fake](https://github.com/fardinabbasi/ML_Fake_Image_Detection/tree/main/fake)' images and implementation of the best classification method (using various machine learning models such as **Random Forest**, **SVM**, and **Logistic Regression**) to identify fake images. The dataset comprises approximately **3400 images**, including both real and fake images of seas, mountains, and jungles, distributed evenly. AI generative models, including [Stable Diffusion](https://stablediffusionweb.com/#demo), [DALL.E](https://openai.com/dall-e-2), [Dreamstudio](https://beta.dreamstudio.ai/dream), [Crayion](https://www.craiyon.com/), and [Midjourney](https://www.midjourney.com/home/?callbackUrl=%2Fapp%2F), are used to create fake images.



A few sample images are provided below:

| Category | Real Image | Fake Image |
| --- | --- | --- |
| Sea | <img src="/real/810199456_real_none_sea_4.jpg" width="400" height="400"> | <img src="/fake/810199456_fake_stable_sea_2.jpg" width="400" height="400"> |
| Jungle | <img src="/real/810199456_real_none_jungle_8.jpg" width="400" height="400"> | <img src="/fake/810199456_fake_dalle_jungle_2.jpg" width="400" height="400"> |
| Mountain | <img src="/real/810199456_real_none_mountain_2.jpg" width="400" height="400"> | <img src="/fake/810199456_fake_stable_mountain_5.jpg" width="400" height="400"> |

Furthermore, in addition to the required feature extraction process, the deep features are already available as "[features.csv](https://github.com/fardinabbasi/ML_Fake_Image_Detection/blob/main/features.csv)" along with their corresponding labels in "[labels.csv](https://github.com/fardinabbasi/ML_Fake_Image_Detection/blob/main/labels.csv)".

## Data Preparation
In this project, besides the given deep features that are likely extracted from a CNN,
we need to extract handcrafted features
