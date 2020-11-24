<h1> Environmental Sound Classification Using Deep Learning </h1>

The purpose of this project is to construct a machine learning model that can classify multiple different environmental sound classes, specifically focusing on the identification of particular urban sounds. 

In addition, the following can provide a simple introduction to sound classification, with the main goal of exploring the PyTorch machine learning library and its functionalities. 

The focus of this project is on the PyTorch library, but the Keras implementation is given for the comparison of obtained results. Therefore, although the Keras model performs slightly better, everything below will be focused solely around development of a deep learning model with the PyTorch library.

## Methods and Algorithms Used
* Data Preprocessing
* Digital Signal Processing (Log-Mel Spectrograms)
* Data Augmentation
* Machine Learning
* Convolutional Neural Networks
* Sound Classification
* K-Fold Cross Validation
* etc.

## Technologies
* Python
* Librosa
* PyTorch
* Keras
* NumPy, Pandas, Matplotlib, Seaborn
* Jupyter Notebook
* etc. 

## Dataset
* UrbanSound8K
  
Analysis of the results obtained from the cross-validation method is given in the [Results](#results) section. In short, the cross-validation examined with the CNN model for sound classification gives **72%** accuracy and the loss of **0.934** measured on the UrbanSound8K dataset with the PyTorch implementation. The Keras model gives slightly better results, **75%** accuracy and **0.924** for the loss. More detailed results for the Keras model can be found in the Jupyter Notebook file.

<h1> Table Of Contents </h1>

- [Project Description](#project-description)
- [UrbanSound8K](#urbansound8k)
- [Data Augmentation](#data-augmentation)
- [CNN Model](#cnn-model)
- [In a Nutshell](#in-a-nutshell)
  - [**1. data_preprocessing.ipynb**](#1-data_preprocessingipynb)
  - [**2. evaluate_classifier.ipynb**](#2-evaluate_classifieripynb)
- [Results](#results)
- [References](#references)

# Project Description

Deep learning techniques applied to the classification of environmental sounds are specifically focused on the identification of particular urban sounds from the [UrbanSound8K](https://urbansounddataset.weebly.com/urbansound8k.html) dataset used in this project.

The project consists of three Jupyter Notebook files. The first contains code for data preparation and the other two contain the implementation of sound classifier model with PyTorch and Keras. 

In the data preprocessing step, examples from the dataset are converted to [log-mel spectrograms](https://towardsdatascience.com/getting-to-know-the-mel-spectrogram-31bca3e2d9d0). Spectrograms provide temporal and spectral characteristics of sound signals and therefore can be used as image inputs in the training step. 

To increase the amount of the data two data augmentation techniques are used. The both techniques are designed specifically for augmentation of audio data, but applied as image augmentation techniques.

The model has a simple CNN architecture, composed of three convolutional layers and two dense layers. 

The evaluation of the model is performed with the 10-fold cross validation method on pre-defined folds.

# UrbanSound8K
"This dataset contains 8732 labeled sound excerpts (<=4s) of urban sounds from **10 classes**: 
1. air_conditioner,
2. car_horn, 
3. children_playing, 
4. dog_bark, 
5. drilling, 
6. enginge_idling, 
7. gun_shot, 
8. jackhammer, 
9. siren and 
10. street_music

All excerpts are taken from field recordings uploaded to [Freesound](ww.freesound.org). The files are pre-sorted into ten folds (folders named fold1-fold10) to help in the reproduction of and comparison with the automatic classification results reported in [this](http://www.justinsalamon.com/uploads/4/3/9/4/4394963/salamon_urbansound_acmmm14.pdf) article.

In addition to the sound excerpts, a CSV file containing metadata about each excerpt is also provided.

Audio files are in the WAV format and the sampling rate, bit depth, and number of channels are the same as those of the original file uploaded to Freesound (and hence may vary from file to file)." \[[1](#references)\]

<br>

![image](images/ran-spec-3x3.png)
<p align="center">
  <b>Figure 1.</b> <i> examples of log-mel spectorgrams from the dataset</i>
</p>

<br>

# Data Augmentation
The size of the dataset is relatively small, so utilization of data augmentation techniques is recommended. In this project, an online method of data augmentation is applied. Therefore, the model is basically never trained on the exact same examples. Although this approach is much more time consuming, effects of overfitting are significantly reduced.

The two main techniques are used for the purposes of this project:
* **Background Gaussian Noise**
  * mixing the sample with background white noise
* **Time Shifting**
  * shifting the image to the right (in time), a part of the image which ends out of the fixed length of a frame is cut off 

Each technique is applied with a given probability, so multiple different combinations are possible when a new example is generated.

<br>

![image](images/data-aug.png)
<p align="center">
  <b>Figure 2.</b> <i> an example of different possible combinations of data augmentation</i>
</p>
<br>


# CNN Model
The proposed CNN architecture is parameterized as follows:
* **1:** 24 filters with a receptive field of (5,5). This is followed by (3,3) strided max-pooling over the last two dimensions (time and frequency respectively) and a ReLU activation function.
* **2:** 36 filters with a receptive field of (4,4). Like layer-1, this is followed by (2,2) strided max-pooling and a ReLU activation function.
* **3**: 48 filters with a receptive field of (3,3). This is followed by a  ReLU activation function (no pooling).
* **4**: 60 hidden units, followed by a ReLU activation function and dropout layer.
* **5**: 10 output units, followed by a softmax activation function.

<br>

![image](images/model-1-arh.png)
<p align="center">
  <b>Figure 3.</b> <i>the architecture of CNN model </i> [3]
</p>

<br>

# In a Nutshell   
The code is organized into two Jupyter Notebook files:
1. [`data_preprocessing.ipynb`](notebooks/data_preprocessing.ipynb)
2. [`evaluate_classifier.ipynb`](notebooks/evaluate_classifier.ipynb)

## **1. data_preprocessing.ipynb**
The first notebook includes two sections:
1. Download and extract the dataset file
2. Feature extraction

* The most important parts of the first notebook are shown in the following two code blocks. The first code block contains a function for computing a log-mel spectrogram from an audio .WAV file. The computation of the audio spectrogram is done with the [Librosa](https://librosa.org/doc/latestt/index.html) Python module  and its functions for audio and music processing. The last part of the function includes padding or cutting of a spectrogram so that it results in exactly 128 samples afterwards. Finally, each example is processed in a way that its output spectrogram has a shape (128,128), where the first dimension represents a number of Mel bands.

```python
def compute_melspectrogram_with_fixed_length(audio, sampling_rate, num_of_samples=128):
    try:
        # compute a mel-scaled spectrogram
        melspectrogram = librosa.feature.melspectrogram(y=audio, 
                                                        sr=sampling_rate, 
                                                        hop_length=HOP_LENGTH,
                                                        win_length=WINDOW_LENGTH, 
                                                        n_mels=N_MEL)

        # convert a power spectrogram to decibel units (log-mel spectrogram)
        melspectrogram_db = librosa.power_to_db(melspectrogram, ref=np.max)
        
        melspectrogram_length = melspectrogram_db.shape[1]
        
        # pad or fix the length of spectrogram 
        if melspectrogram_length != num_of_samples:
            melspectrogram_db = librosa.util.fix_length(melspectrogram_db, 
                                                        size=num_of_samples, 
                                                        axis=1, 
                                                        constant_values=(0, -80.0))
    except Exception as e:
        return None 
    
    return melspectrogram_db
```
* In the next block, iteration through the dataset and the process of spectrogram feature extraction is shown.

```python
SOUND_DURATION = 2.95   # fixed duration of an audio excerpt in seconds

features = []

# iterate through all dataset examples and compute log-mel spectrograms
for index, row in tqdm(us8k_metadata_df.iterrows(), total=len(us8k_metadata_df)):
    file_path = f'{US8K_AUDIO_PATH}/fold{row["fold"]}/{row["slice_file_name"]}'
    audio, sample_rate = librosa.load(file_path, duration=SOUND_DURATION, res_type='kaiser_fast')
    
    melspectrogram = compute_melspectrogram_with_fixed_length(audio, sample_rate)
    label = row["classID"]
    fold = row["fold"]
    
    features.append([melspectrogram, label, fold])

# Convert into a Pandas DataFrame 
us8k_df = pd.DataFrame(features, columns=["melspectrogram", "class", "fold"])
```
## **2. evaluate_classifier.ipynb**
The second notebook includes six sections:
1. Create a custom Dataset class
2. Data augmentation
3. CNN model
4. Helper functions
5. 10-fold cross validation
6. Results

* A PyTorch's `torch.utils.data.Dataset` class is implemented as a data loading utility support for the purposes of this project.
<br></br>
* The following two code blocks show implementation of the `__call__` method of custom augmentation classes.

```python
class MyRightShift(object):
    ...
    
    def __call__(self, image):
        if np.random.random() > self.shift_prob:
          return image

        # create a new array filled with the value of the min pixel
        shifted_image= np.full(self.input_size, np.min(image), dtype='float32')

        # randomly choose a start postion
        rand_position = np.random.randint(1, self.width_shift_range)

        # shift the image
        shifted_image[:,rand_position:] = copy.deepcopy(image[:,:-rand_position])

        return shifted_image
```
```python
class MyAddGaussNoise(object):
    ...
    
    def __call__(self, spectrogram):
      if np.random.random() > self.add_noise_prob:
          return spectrogram

      # set some std value 
      min_pixel_value = np.min(spectrogram)
      if self.std is None:
        std_factor = 0.03     # factor number 
        std = np.abs(min_pixel_value*std_factor)

      # generate a white noise spectrogram
      gauss_mask = np.random.normal(self.mean, 
                                    std, 
                                    size=self.input_size).astype('float32')
      
      # add white noise to the sound spectrogram
      noisy_spectrogram = spectrogram + gauss_mask

      return noisy_spectrogram
```


* The next three blocks contain an implementation of PyTorch's `torch.nn.Module` class. The first two blocks show only the most important parts of implementation, like initialization of the network's architecture and its forward function. The last block is an excerpt from the model's `fit` function.
  
``` python
def __init__(self, device):
    super(Net, self).__init__()
    self.device = device

    self.conv1 = nn.Conv2d(in_channels=1, out_channels=24, kernel_size=5, padding=0)
    self.conv2 = nn.Conv2d(in_channels=24, out_channels=36, kernel_size=4, padding=0)
    self.conv3 = nn.Conv2d(in_channels=36, out_channels=48, kernel_size=3, padding=0)

    self.fc1 = nn.Linear(in_features=48, out_features=60)
    self.fc2 = nn.Linear(in_features=60, out_features=10)
```


```python
def forward(self, x):
    # cnn layer-1
    x = self.conv1(x)
    x = F.max_pool2d(x, kernel_size=(3,3), stride=3)
    x = F.relu(x)

    # cnn layer-2
    x = self.conv2(x)
    x = F.max_pool2d(x, kernel_size=(2,2), stride=2)
    x = F.relu(x)

    # cnn layer-3
    x = self.conv3(x)
    x = F.relu(x)

    # global average pooling 2D
    x = F.avg_pool2d(x, kernel_size=x.size()[2:])
    x = x.view(-1, 48)

    # dense layer-1
    x = self.fc1(x)
    x = F.relu(x)
    x = F.dropout(x, p=0.5)

    # dense output layer
    x = self.fc2(x)

    return x
```
* This code excerpt shows an implementation of the mini-batch training approach in this project.
```python
for epoch in range(epochs):
    self.train()

    for step, batch in enumerate(train_loader):
        X_batch = batch['spectrogram'].to(self.device)
        y_batch = batch['label'].to(self.device)
        
        # zero the parameter gradients
        self.optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            # forward + backward 
            outputs = self.forward(X_batch)
            loss = self.criterion(outputs, y_batch)
            loss.backward()

            # update the parameters
            self.optimizer.step() 
``` 
* This is a helper function for processing a fold in the K-fold cross validation method designed specifically for the purposes of this project. 

```python
def process_fold(fold_k, dataset_df, epochs=100, batch_size=32, num_of_workers=0):
    #split the data
    train_df = dataset_df[dataset_df['fold'] != fold_k]
    test_df = dataset_df[dataset_df['fold'] == fold_k]

    # normalize the data
    train_df, test_df = normalize_data(train_df, test_df)

    # init train data loader
    train_ds = UrbanSound8kDataset(train_df, transform=train_transforms)
    train_loader = DataLoader(train_ds, 
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True,
                              num_workers=num_of_workers)
    
    # init test data loader
    test_ds = UrbanSound8kDataset(test_df, transform=test_transforms)
    test_loader = DataLoader(test_ds, 
                            batch_size=batch_size,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=num_of_workers)

    # init model
    model = init_model()

    # pre-training accuracy
    score = model.evaluate(test_loader)
    print("Pre-training accuracy: %.4f%%" % (100 * score[1]))
      
    # train the model
    start_time = datetime.now()
    history = model.fit(train_loader, epochs=epochs, val_loader=test_loader)
    end_time = datetime.now() - start_time
    print("\nTraining completed in time: {}".format(end_time))

    return history
```

* Other two helper functions are `normalize_data` and `init_model`, which normalize the data in respect to the train data and initialize the model to its random initial state, respectively.
* The 10-fold cross validation procedure is performed on pre-defined data folds. More in-depth motivation explaining why it's important to do exactly the 10-fold validation is given in the next section, and on [this](https://urbansounddataset.weebly.com/urbansound8k.html) page.
  
<br>

# Results
The model evaluation is done with the 10-fold cross validation method. In most cases, examples from the same class in the dataset are derived from the same audio source file, hence it is necessary to use the pre-defined folds. Otherwise, obtained results can be significantly inflated and can cause inaccurate perception of the classifier's performance.

Loss and accuracy are metrics used for the evaluation. For the loss function cross-entropy is selected and the accuracy is defined as the percentage of correctly classified instances.

For each fold, the loss is calculated by taking the minimum loss score of all epochs on the validation set. Likewise, the accuracy of each fold is selected as the best validation accuracy over the epochs.

Here are the results of 10-fold cross validation after repeating the procedure three times and calculating the mean. 

<center> 

|fold|accuracy| loss
|---|:-:|:-:|
|fold-1| 0.75|0.738|
|fold-2|0.71|0.925|
|fold-3|0.70|1.061|
|fold-4|0.69|1.030|
|fold-5|0.76|0.685|
|fold-6|0.71|1.177|
|fold-7|0.68|1.016|
|fold-8|0.66|1.190|
|fold-9|0.76|0.761|
|fold-10|0.74|0.760|
|**Total**|**0.72**|**0.934**| 
</center>
<b>Table 1.</b> <i> shows average loss and accuracy scores for each fold over 3 cross-validation runs</i>

<br></br>

![image](images/CV-accuracy.png)
<p align="center">
  <b>Figure 4.</b> <i>the accuracy measured by 10-fold cross validation score</i>
</p>

<br></br>

![image](images/CV-loss.png)
<p align="center">
  <b>Figure 5.</b> <i>the loss measured by 10-fold cross validation score </i>
</p>

<br></br>

* **What's next?**
  * more in-depth analysis of the results
  * keep tuning the hyperparameters
  * test the transfer learning approach
  * examine different data augmentation techniques, try to modify techniques specifically to particular sound classes
  * implement a real-time sound classification system
  
<br>

# References

1. J. Salamon, C. Jacoby, and J.P.Bello, "A Dataset and Taxonomy for Urban Sound Research," in 22nd ACM International Conference on Multimedia (ACM-MM'14), Orlando, FL, USA, Nov. 2014, pp. 1041–1044. \[Online\]. Available: http://www.justinsalamon.com/uploads/4/3/9/4/4394963/salamon_urbansound_acmmm14.pdf 
2. J. Salamon and J. P. Bello, "Deep Convolutional Neural Networks and Data Augmentation for Environmental Sound Classification", submitted, 2016.\[Online\]. Available: https://arxiv.org/pdf/1608.04363.pdf
3. Zohaib Mushtaq, Shun-Feng Su, "Environmental sound classification using a regularized deep convolutional neural network with data augmentation", Applied Acoustics, Volume 167, 2020, 107389, ISSN 0003-682X, https://doi.org/10.1016/j.apacoust.2020.107389. (http://www.sciencedirect.com/science/article/pii/S0003682X2030493X)
