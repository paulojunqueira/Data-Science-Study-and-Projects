# Data Science Study and Projects
This repo contains a list and summary of projects that I have done in the context of Data Science. This repository is constantly being built and updated.


# Summary

- [Large Language Models - LLM](#Large-Langue-Models---LLM)
- [NLP](#NLP)
- [Computer Vision](#Computer-Vision)
- [Classification](#Classification)
- [Python and PySpark](#Python-and-PySpark)
- [Exploratory Data Analysis](#EDA)


---------------------------------------------------------------------------------------------------------------------------------------------------------
# Large Language Models - LLM
- [Story Generation and QA from PDF with LLMs - Notebook](https://www.kaggle.com/code/paulojunqueira/story-generation-and-qa-from-pdf-with-llm)
  - Kaggle Notebook that uses llama-2 LLM model to create a story, save as pdf and then uses this story in a QA chain with LLM model to allow the user to ask questions about the document
  - Keywords: LangChain | Hugging Face | Llama-2 | LLM Model | Generative AI | RAG

# NLP

- [MLO Toxic Lang Train with DistilBert - Notebook TRAIN](https://www.kaggle.com/code/paulojunqueira/mlo-toxic-lang-distilbert-train)
- [MLO Toxic Lang Train with DistilBert - Notebook INFER](https://www.kaggle.com/code/paulojunqueira/mlo-toxic-lang-distillbert-infer)
  - Kaggle Notebook for training a baseline distilBert Model to classify tweeter text into toxic and non-toxic in PTBR language
  - Keywords: Classification | Hugging Face | BERT | NLP | Train | Competition
    
    
# Computer Vision

- [Vehicle Tracking Time in Parking Lot](https://www.kaggle.com/code/paulojunqueira/yolo-v8-vehicle-time-tracking-in-parking-lot)
  - In this Kaggle Notebook, it is used the YOLO V8 tracking to track vehicles and time it in a parking lot. Also, a simple color mapping is create to infer the vehicle color.
  - Keywords: YOLO V8 | Tracking | Python | Image Detection

- [Autolabelling with Autodistill and GroundedSAM](https://www.kaggle.com/code/paulojunqueira/autolabeling-with-autodistill)
  - In this Kaggle Notebook, to teste the library autodistill, a video was breaked into frames that were used in the zero-shot object detection GroundedSAM to label automatically label images and create a dataset
  - Keywords: Autodistill | GroundedSAM | Python | Image Detection

- [Zero-shot Object Detection with GroundingDINO](https://www.kaggle.com/code/paulojunqueira/zero-shot-object-detection-with-groundingdino)
  - In this Kaggle Notebook, it is tested the Zero-shot object detection model called GroundingDINO. Transform text description input into object detection in a imagem without training
  - Keywords: GroundingDINO | Python | Image Detection

- [Detecting and Counting Vehicles with Yolo V8 - Notebook](https://www.kaggle.com/code/paulojunqueira/yolo-v8-vehicles-detecting-counting)
  - In this Kaggle Notebook, a Yolo V8 model was used to detect vehicles in a video. After dectecion, the number of vehicles that passes in each direction (up\down) are counted.
  - Keywords: Yolo V8 | Python | Image Detection

- [Detecting and Tracking People in a ROI - Notebook](https://www.kaggle.com/code/paulojunqueira/yolo-v8-people-detection-and-tracking-in-roi)
  - In this Kaggle Notebook, a Yolo V8 model was used to detect and track peoaple passing in a region of interest (ROI).
  - Keywords: Yolo V8 | Python | Image Detection | Tracking | Detection | ROI

- [Autoencoder with MNIST](https://www.kaggle.com/code/paulojunqueira/autoencoder-implementation-with-mnist)
  - In this notebook, an Autoencoder is implemented in pytorch. Then, it was used to reconstruct the learned features from the inputed MNIST dataset imagens. From random noise to similar MNIST imanges. The objective is to learn how the structure works as it could be used for dimensionalty reduction, anommaly detection and more.
  - Keywords: Autoencoder | Dimensionalty Reduction | Pytorch | Neural Network

- [RBM Boltzmann Experiment](https://www.kaggle.com/code/paulojunqueira/rbm-boltzmann-experiment)
  - In this notebook, a Restricted Boltzman Machines (RBM) network is used to learn how to reconstruct the input, imagens from the datasets Dogs Vs Cats and MNIST. RBM is an algorithm used for many purposes such as dimensionalty reduction, regression and as generative outputs.
  - Keywords: Image | RBM | Dimensionalty Reduction | Neural Network
  
- [Audio to Image Pipeline - BirdCLEF 2022](https://www.kaggle.com/code/paulojunqueira/birdclef2022-audio-to-image) 
  - This notebook implements one of the pipelines for audio transformation to Spectograms developed for the BirdCLEF 2022 competition where our team achieve the Bronze medal in 68th position.
  - Keywords: Sound | Image | Transformation | Pipeline | Competition

- [CNN for MNIST with Pytorch and Transfer Learning (timm)](https://www.kaggle.com/code/paulojunqueira/mnist-with-pytorch-and-transfer-learning-timm)
  - This notebooks apply the transfer learning with pytorch and timm libraries in a classification task for the MNIST Dataset
  - Keywords: Classification | Image | Pytorch | CNN | timm | Transfer Learning

- [CNN for MNIST with Pytorch](https://www.kaggle.com/code/paulojunqueira/cnn-for-mnist-with-pytorch)
  - This notebooks explores the pytorch library to develop a CNN model for the Classification task of the MNIST Dataset
  - Keywords: Classification | Image | Pytorch | CNN
        

    
# Classification
- [Training Pipiline for AMEX Kaggle Competition](https://www.kaggle.com/code/paulojunqueira/training-pipeline-for-amex)
  - In this Kaggle Notebook, a complete pipeline was created for the AMEX Competition. The competition was to predict predict deafault. Different models and variables can be used, with feature engineering, tunning and oof analysis. The challenge of the competion was cope with the huge size of the data.
  - Keywords: Classification | GPU Processing | CATBOOST | XGB | TABNET | LOGIT | LGBM | Feature Engineering | Tuning

- [Catboost and SpaceShip Dataset](https://www.kaggle.com/code/paulojunqueira/spaceship-catboost-data-aug-shap)
  - Introductory notebook on the SpaceShip dataset. EDA, feature engineering, tabular data augmentation, and Catboost model
  - Keywords: Classification | Data Augmentation | EDA
    
- [Simple Models Comparation for Titanic DataSet](https://www.kaggle.com/code/paulojunqueira/titanic-simple-models-comparison)
  - Introductory notebook exploring the classic titanic datset. EDA, feature engineering and Classification Models comparison
  - Keywords: Classification | Ensemble | EDA  

# Python and PySpark
- [Ensemble Optimization with GA](https://www.kaggle.com/code/paulojunqueira/ensemble-optimization-with-ga)
  - In this notebook, a Genetic Algorithm (GA) is implemented to optmize the OOF files from predictions for a better performance in a ensemble
  - Keywords: Genetic Algorithm | Optimization | Ensemble | OOF   

- [Hands on Pyspark Overview](https://www.kaggle.com/code/paulojunqueira/hands-on-pyspark-introduction-101)
  - Notebook that compiles pllenty pyspark functions and transforms
  - Keywords: PySpark | Python | Pandas | Functions | Aggregations | Distributed Computation
    
- [Custom MLP](https://www.kaggle.com/code/paulojunqueira/custom-mlp-0-78)
  - Notebook that implements a custom multi layer perceptron (MLP)
  - Keywords: Python | Neural Network | Numpy
    
- [Gradient Descent Experiments](https://www.kaggle.com/code/paulojunqueira/gradient-descent-experiments-and-visualizations)
  - Notebook with experiments and visualizations of Gradient Descent
  - Keywords: Python | Visualizations | Gradient Descent | Numpy

- [A Pathfinder Algorithm](https://github.com/paulojunqueira/Pathfinder-Project)
  - A repository containing the implementation of a algorithm called Pathfinder. The pathfinder finds the least cost path from A -> B.
  - Keywords: Python | Numpy | Optimization
     
- [N Queen Problem and EA](https://github.com/paulojunqueira/N-Queen-Problem-Evolutionary-Algorithm)
  - In this repo is implemented a Evolutionary Algorithm to solve an optimization problem called N-Queens. In this problem N chees queens have to be place in a board NxN and none of them can be kill by each other.
  - Keywords: Optimization | Evolutionary Algorithms | Python | N-Queen
    
- [Repo with Many other DS Developments](https://github.com/paulojunqueira/Python-Projects-and-Studies) and [ML implementations](https://github.com/paulojunqueira/Machine-Learning-Implementations/tree/master)
  - Github repository with many other small projects applications. For instance, K-means implementation, Regression analysis, SVM, Decision Trees etc.
  - Keywords: Machine Learning | Data Science | Python 

# EDA
- [EDA ML Olympiad - Toxic Language 2024](https://www.kaggle.com/code/paulojunqueira/eda-ml-olympiad-toxic-language)
  - Exploratory Data Analysis  for the ML Olympiacs - Toxic Language (PTBR). Data from Tweeter (X) with PTBR Tweets and classification of toxic or non-toxic language;  
  - Keywords:  Classification | NLP | Competition | EDA | Visualization
    
- [EDA and Web Scrapping for BirdCLEF 2023 Competition](https://www.kaggle.com/code/paulojunqueira/pew-pew-overview-birdclef-2023)
  - Exploratory Data Analysis for the BirdCLEF 2023. Analysis of most\least common bird sounds, locations and other informations. Sound to spectogram transformation and analysis. Also, was created a Web scraping to look for additional information from the birds in the wikipedia page
  - Keywords: Web Scraping | Sound Classification | Spectogram | Competition | EDA | Visualization

- [Brief Analysis for AMEX Competition](https://www.kaggle.com/code/paulojunqueira/brief-analysis-of-dataset-variables-amex)
  - Exploratory Data Analysis for the AMEX 2022 Competition.
  - Keywords: EDA | Classification | Competition
    
- [EDA for BirdCLEF 2022 Competition](https://www.kaggle.com/code/paulojunqueira/little-bird-what-sound-is-that-eda)
  - Exploratory Data Analysis for the BirdCLEF 2022. Analysis of most\least common bird sounds, locations and other informations. Sound to spectogram transformation and analysis.
  - Keywords: Sound Classification | Spectogram | Competition | EDA | Visualization

- [EDA for ENEM Data (in PT-BR)](https://github.com/paulojunqueira/Python-Projects-and-Studies/blob/master/Data%20Analysis/Estudo_Analise_Dados_ENEM_2019ipynb.ipynb)
  - This study presents a statistical analysis of the data from the ENEM. This was done in PT-BR as one of the first studies that I have done.
  - Keywords: Statistical Analysis | Academic Data | EDA

- [Used Car Sales Analysis (in PT-BR)](https://github.com/paulojunqueira/Python-Projects-and-Studies/blob/master/Data%20Analysis/Analise_Venda_Carros_Usados.ipynb)
  - This study presents an analysis and some basic model development for an used car sales data. This was done in PT-BR as one of the first studies that I have done.
  - Keywords: Python | Analysis | Visualizations | EDA | Models




