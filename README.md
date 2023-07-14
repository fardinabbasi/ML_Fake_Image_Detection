# ML_Fake_Image_Detection

Feature extraction of '[real](https://github.com/fardinabbasi/ML_Fake_Image_Detection/tree/main/real)' and '[fake](https://github.com/fardinabbasi/ML_Fake_Image_Detection/tree/main/fake)' images and implementation of the best classification method (using various machine learning models such as Random Forest, SVM, and Logistic Regression) to identify fake images. The dataset comprises approximately 3400 images, including both real and fake images of seas, mountains, and jungles, distributed evenly. A few sample images are provided below



| Real Image | Fake Image |
| --- | --- |
| <img src="/real/810199456_real_none_sea_4.jpg" width="400" height="400"> | <img src="/fake/810199456_fake_stable_sea_2.jpg" width="400" height="400"> |
| <img src="/real/810199456_real_none_jungle_8.jpg" width="400" height="400"> | <img src="/fake/810199456_fake_dalle_jungle_2.jpg" width="400" height="400"> |
| <img src="/real/810199456_real_none_mountain_2.jpg" width="400" height="400"> | <img src="/fake/810199456_fake_stable_mountain_5.jpg" width="400" height="400"> |

Besides the feature extraction, that needs to be done, the deep features are already provided as "features.csv" and their corresponding label as "labels.csv".
## Feature Extraction
