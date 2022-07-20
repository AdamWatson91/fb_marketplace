# Facebook Marketplace Search Ranking
Facebook marketplace uses a Search Ranking system to help users better find what they are looking for. This is based on complex unstrucutred data like images and text. The marketplace leverages computer vision and natural language processing (NLP) techniques to deliver the service. For more information [see here](https://engineering.fb.com/2018/10/02/ml-applications/under-the-hood-facebook-marketplace-powered-by-artificial-intelligence/).

This project aims to reproduce some of these AI powered features. In particular, training a multimodal deep neural network model that predicts the category of a product based on its image and text description.

Once the model is trained, it is deployed by creating an API with FastAPI. This API so accepts images and text and applies the trained model to produce a prediction. 

This programme is then containerised using Docker, and uploaded to an EC2 instance.

# Source data
There were three sources of data which were then cleansed, transformed, and combined. This data was sourced from gumtree to get product images and their relevant product descriptions among other features. The below explains the data source:

- Products.csv - Tabular dataset holding data relevant to the product such as the uid, description, price and location. This data includes the label (category) that will be the target variable for the mulit-class classification prediction.
- Images.csv - Tabular dataset holding data relevant to the images, including the uid for the product the image relates to.
    - NOTE: Multiple images exist for some products
- image_folder - This holds the actual image file in jpg format.
    - NOTE: Not included here due to size 

# Data preparation
In order to prepare the data for training purposes it was required to both cleanse Tabular data and merge. In addition, a holdout sample of the images was created for downstream testing of the API.

The following programs were created to prodvide the nescessary fucntionality for cleansing the data:

- clean_tabular.py - This program creates a class (TabularCleanse) which includes various methods for common functions for data cleansing
- clean_images.py - This program creates a class (ImageCleanse) which includes various methods for common functions for image cleansing.

The above were then used in the following programs to perform the data cleanse:

- cleanse_tabular.py - used to transform the products.csv data into the pickle file 'products_cleansed.pkl'. The key functions applied are
    - Removing null values and outliers from price
    - Converting price from string to float
    - Splitting columns to increase features
    - Dropping colunms not required for further development
- cleanse_images.py - used to transform the images and create a holdout sample of images. Creates pickle file 'cleansed_img_tabular' to hold data for those images not in the holdout sample. The following transformations are applied:
    - Images are resized to 256x256 using the same ratio as input with black padding applied.
- cleanse_product_images.py - used to merge and transform pickle files 'products_cleansed.pkl' & 'cleansed_img_tabular' to create one file 'cleansed_product_images.pkl'. Products without images are dropped.
    - NOTE: 'cleansed_product_images.npy' also output in order to build and train model using google colab for gpu access.

# Model training
Three models were built, trained and validated using pytorch and transfer learning. The following programs were used:

- model_train_bert.ipynb - trains text multi-class classification using transfer learning from bert embeddings. The following files are output:
    - bert_decoder.pkl - dictionary with the class names and indexes applied during training.
    - bert_model.pt - the trained model weights
- model_train_resnet50.ipynb - trains image multi-class classfication using a CNN. Feature vector was created using transfer leavering (RESNET-50) with a classfication module added. The following files are output:
    - image_decoder.pkl - dictionary with the class names and indexes applied during training.
    - image_model.pt - the trained model weights
- product_classification_mixed_modal.ipynb - This trains a mixed modal multi-class classification using both bert and RESNET-50. The following files are output:
    - combined_decoder.pkl - dictionary with the class names and indexes applied during training.
    - combined_model.pt - the trained model weights

# Model deployment
The trained models have been deployed using FastAPI and containerised using docker.

The api is created in api_deployment.py. This provides the functionality to try all free models.

In order to allow users to upload, text and images that can be trained using the models a processing step is required. processor_image.py and processor_text.py apply the required transformations to uploaded text or images for the model.

# Other
There are some other files included with details below. Please note these files were developed prior to final refactoring of the data preperation and may no longer work without further development

- simple_img_classficiation.ipynb - This file was created to test image classficiation quality using machine learning rather than deep learning. Overall validation was around 18%
- simple_regression.ipynb - This file uses traditional machine learning to perform regression for price prediction.
- transfer_learning_vision_model.ipynb - This file was for intial development of a CNN for image classfication without transfer learning. Validation accuracy around 45%