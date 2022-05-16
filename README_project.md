# Facebook Marketplace's Recommendation Ranking System
Facebook Marketplace is a platform for buying and selling products on Facebook. This is an implementation of the system behind the system which uses AI to recommend the most relevant listings based on a personalised search query.

## Milestone 1 - Cleaning tabular and image data
### Tabular
Pandas was the main library use to clean the tbaular data after downloading the data from the RDS database.

To begin cleaning the data I first summarised the data profile of all the fields. For this I used the pandas_profiling library

```
pip install pandas_profiling
from pandas_profiling import ProfileReport
```
This was useful and saved time compared to creating code to complete field by field.

I implemented a number of methods to identify numerical outliers, in this case for the price column. These were:

- Iqr - using the interquartile range to any rows that feel outside of this range. In this case it was not useful as there is high spread in prices for the dataset due to the range of products.

- zscore - This identifys fields that sat outside 3 standard deviations from the mean. This is what i used in this case. It helped identify the clear outliers that were clearly dummy values. For this reason as they were data quality issues and the products were not actual products i decided to remove the rows. Although there remained high variation in the data the remaining data points were reflective of the range of products.

### Images
I made use of the following libraries to clean the image data:

- numpy - inorder to manipulate the image arrays
- PIL - to manipluate the image data
- skimage - another tool used to manipulate the image data
- matplotlib - in order to visualise the data when working in a notebook

The main reason i used both PIL and skimage is that i felt that skimage prodvided improve functionality to manipulate the images. However, once identifying the ImageFilter options in PIL I ended up using this more so.

I am yet to implement data augmentation to support the training process and therefore, undecided on which library to use for this.

Here is a summary of some of the methods i included in my clean_images.py:

- Resize to square - in the dataset there were varying image sizes. I have reduced them all to the same size and shaped(256x256). the reason for this is that this is generally considered best practice.

- Convert to grayscale - I have converted the images to grayscale. Again this is generally best practice as it reduces the complexity of the image for which should improve the image classification outcome.

- Normalise - By normalising the image we are then able to create a threshold to create a biniary array. This can be used in image classification to create a siply white or black outline of the main item in the image, further simplifying for the image classification. This was generally effective in identifying the main shape of each image. there are different types of thresholding, local vs gloabal and manual vs via an algorithm.

- Erosion/Dilation - This process supports further simplification of the binary image. It does this by either dimisnhing features from an image or accentuating them.


#### Useful resources for this milestone:
- resizing images - https://stackoverflow.com/questions/43512615/reshaping-rectangular-image-to-square
- Erode/Dilation - https://realpython.com/image-processing-with-the-python-pillow-library/#image-segmentation-and-superimposition-an-example
- PIL ImageFilter - https://coderzcolumn.com/tutorials/python/image-filtering-in-python-using-pillow
- Overall guides to image prepartion for ML:
- https://machinelearningmastery.com/start-here/#dlfcv
- https://machinelearningmastery.com/deep-learning-for-computer-vision/
- https://machinelearningmastery.com/best-practices-for-preparing-and-augmenting-image-data-for-convolutional-neural-networks/
- https://towardsdatascience.com/massive-tutorial-on-image-processing-and-preparation-for-deep-learning-in-python-1-e534ee42f122
- https://blog.roboflow.com/you-might-be-resizing-your-images-incorrectly/

## Milestone 2 - Simple regression and image classification models
### Regression
I attempted using the following fields for my price prediction:
- Longitude and Latitude of the location data - I also attempted clustering this data but found iminimal improvements
- Product catgeory data - I attempted both category hierarchies as well as trying them seperately. I found the the prediction was best/simpliest with just Cat_0

To begin the process of modelling development I first intialised a Linear regression model without hyperparameter tuning. This was a useful way to test onward development (i.e. training a model and then scoring) as well as, creating a baseline model which could be used for to assess other model choices.

I introduced scaling with the StandardScaler class from sklearn, fitting this with the training data and then applying this to the training and test data. In this way i was able to avoid data leakage.

I used the GridSearchCV class in sklearn to perform hyperparameter tuning across numerous models. The models tuned were:

- KNeighborsRegressor
- DecisionTreeRegressor
- Lasso
- GradientBoostingRegressor

By storing each model into a dictionary this enabled me to train and test each model with ease by iterating through each model.

I attmepted this first without the cross validation but finishing my development by introducing cross validation delivered via a pipeline.

The best performing model was the KNeighbours model with a root mean square error result of 154.99

The model could be improved by:
- Further hyperparamter tuning and testing with additional models
- More data
- Test other scaling methods

### Image classficiation
For this part of the project i had to do the following steps:
1. Import the image detail data from an RDS. This data held the image_id and the product_id each image related to
2. Clean the imported image detail data
3. Import cleaned images i had previously transformed
4. Further transform the images into arrays in order to use for modelling
5. Combine image_detail data and the image arrays to create one dataset with all required data for the images
6. Import the cleaned product data
7. Complete any transformations required for the products data to fit the requirement for image classification
8. Combine image data and product data to create one dataset for modelling
9. Create train and test model

One challenge i was faced with during this process was the fact I had used dummy encoding and therefore, one of the actegories had already been dropped. Time permitting i would re-run the Tabular cleanse to not dummy encode and use this data. However, the geocoding process was time consuming. For this reason i used a process to reintroduce the dropped category by asessing if all other category fields were set to 0. This helped me learn some new functions in ptyhon:

- np.where - used to sum values from multiple dataframe colunms (e.g. total sum = 0)
- .pop - used to insert a dataframe colunm t a specific index

To convert string categories to ordinal numerical values i used the LabelEncoder from sklearn.

For the model train and test i used a LogisticRegression with a multinominal agrument to allow for multi-class classification of the images.

The overall classficiation performance was 18% with the best classfication by category for Home_Garden of 29%. I used a confusion matrix to assess this using both seaborn and matplotlib.

This model could be improved with:

- Using a CNN instead of a logistic regression
- More data


#### Useful resources for this milestone:
- https://medium.com/swlh/linear-regression-machine-learning-in-python-227f062bd07c
- https://machinelearningmastery.com/optimize-regression-models/#:~:text=Optimize%20Regression%20Models,-Regression%20models%2C%20like&text=These%20regression%20models%20involve%20the,optimization%20algorithms%20can%20be%20used.
- https://machinelearningmastery.com/modeling-pipeline-optimization-with-scikit-learn/
- https://machinelearningmastery.com/elastic-net-regression-in-python/
- https://machinelearningmastery.com/difference-between-a-batch-and-an-epoch/
- https://machinelearningmastery.com/data-preparation-without-data-leakage/
- https://machinelearningmastery.com/types-of-classification-in-machine-learning/
- https://machinelearningmastery.com/discrete-probability-distributions-for-machine-learning/
- https://machinelearningmastery.com/multinomial-logistic-regression-with-python/
- https://towardsdatascience.com/hyperparameter-tuning-in-lasso-and-ridge-regressions-70a4b158ae6d