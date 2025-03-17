# Aerial-Semantic-Segmentation-Drone-using-Pytorch
Aerial Semantic Segmentation of Drone  Captured Images, using Pytorch.

### 🔧 Development Details
- **👨‍💻 Developer:** [Ravin Kumar](https://mr-ravin.github.io)
- **📂 GitHub Repository:** [https://github.com/mr-ravin/Aerial-Semantic-Segmentation-Drone-using-Pytorch](https://github.com/mr-ravin/Aerial-Semantic-Segmentation-Drone-using-Pytorch)
  
----
### Dataset Related Information

- Aerial Image Dataset: https://www.kaggle.com/datasets/santurini/semantic-segmentation-drone-dataset

- Number of Raw Classes: 24, indexed from 0 to 23. After processing this data, these classes are grouped into 5 classes, indexed from 0 to 4.

|  index | Class         |
|---------|--------------|
|    0    |  unlabeled   |
|    1    |  paved-area  |
|    2    |  dirt        |
|    3    |  grass       |
|    4    |  gravel      |
|    5    |  water       |
|    6    |  rocks       |
|    7    |  pool        |
|    8    |  vegetation  |
|    9    |  roof        |
|    10   |  wall        |
|    11   |  window      |
|    12   |  door        |
|    13   |  fence       |
|    14   |  fence-pole  |
|    15   |  person      |
|    16   |  dog         |
|    17   |  car         |
|    18   |  bicycle     |
|    19   |  tree        |
|    20   |  bald-tree   |
|    21   |  ar-marker   |
|    22   |  obstacle    |
|    23   |  conflicting |


### Final Number of Grouped Classes: 5

| grouped class id | individual classes                   | group color in rgb  | grouped class name  |
|------------------|--------------------------------------|---------------------|---------------------|
|        0         | 0, 6, 10, 11, 12, 13, 14, 21, 22, 23 |    [155,38,182]     | obstacles           |
|        1         | 5, 7                                 |    [14,135,204]     | water               |
|        2         | 2, 3, 8, 19, 20                      |    [124,252,0]      | nature              |
|        3         | 15, 16, 17, 18                       |    [255,20,147]     | moving              |
|        4         | 1, 4, 9                              |    [169,169,169]    | landable            |

### Visualise a sample from the dataset

![image](https://github.com/mr-ravin/Aerial-Semantic-Segmentation-Drone-using-Pytorch/blob/main/results/view_dataset.jpg?raw=true)

## Directory Structure
```
|-- dataset/
|      |-- images/ # contains all rgb images as .jpg
|      |-- masks/  # contains all ground truth masks as .png
|      |-- train/
|      |     |-- images/
|      |-- val/
|           |-- images/
|-- utils
|    |-- preprocess.py
|    |-- compute.py
|    |-- grephics.py
|
|-- train_files.txt
|-- valid_files.txt
|-- dataloader.py
|-- run.ipynb
|-- weights/       # it will contain the weight file after model training.
|-- results/
       |-- view_dataset.jpg
       |-- inference_results.jpg
       |-- overall_analysis.jpg

```

Information regarding the images present in our train set is present inside `train_files.txt`and image information for validation set is present inside `valid_files.txt`. It's provided so that one can reproduce the results.

### Steps to Train the model
- Download the dataset and place its images and masks inside `dataset/images/` and `dataset/masks/` respectively.
- Run the jupyter notebook: `run.ipynb`

### Model Performance
We have trained the model upto 25 epochs and have shared the performance matrices and some sample results below.

![image](https://github.com/mr-ravin/Aerial-Semantic-Segmentation-Drone-using-Pytorch/blob/main/results/overall_analysis.png?raw=true)


![image](https://github.com/mr-ravin/Aerial-Semantic-Segmentation-Drone-using-Pytorch/blob/main/results/inference_results.jpg?raw=true)
