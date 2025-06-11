This is a repository to show my machine learning and deep learning projects.
The projects:

1] Credit card Dimensionality Reduction using Autoencoders

This project implements an autoencoder for dimensionality reduction on the "Default of Credit Card Clients" dataset from the UCI Machine Learning Repository. The dataset consists of 30,000 samples with 24 features related to credit card usage and default prediction. The autoencoder architecture compresses the input features from 24 to 16 dimensions while preserving key information, achieving a Mean Squared Error (MSE) of 0.0011 and an R² score of approximately 0.66 on both training and test datasets. The use of MinMaxScaler for feature scaling, along with techniques like LeakyReLU activation and dropout for regularization, ensures effective training and generalization. Overall, the model demonstrates a successful reduction in dimensionality while retaining significant variance from the original dataset.

2] Customer segmentation

This project focuses on customer segmentation using K-means clustering to analyze spending patterns based on income levels. The Jupyter notebook contains code for loading and preprocessing customer data from a CSV file, performing exploratory data analysis, and implementing K-means clustering to segment customers into distinct groups. The analysis utilizes Python libraries such as pandas, numpy, matplotlib, and scikit-learn. The methodology involves determining the optimal number of clusters using the elbow method and visualizing the results through scatter plots. The final output segments customers into four distinct groups based on their income and spending patterns, which can be leveraged for targeted marketing strategies or personalized customer experiences

3] EDA

This project explores the UCI Adult dataset through comprehensive Exploratory Data Analysis (EDA), revealing significant demographic and income insights. The analysis uncovers that approximately 24% of individuals earn more than $50K, with males substantially overrepresented in the high-income category (30% vs. 10% for females). Key findings highlight age trends showing high-income individuals tend to be older, with males averaging 44.63 years and females 42.13 years. Additionally, the study demonstrates gender disparities in work patterns, with males working an average of 42.43 hours per week compared to females' 36.41 hours, while both genders maintain a median of 40 work hours weekly. The research methodology involved rigorous data preprocessing, statistical analysis, and visualization techniques to extract meaningful patterns across gender, age, and income dimensions, with future goals including machine learning model development and deeper exploration of education and occupation correlations.

4] EmotionDetectionusingCNN

This project implements an emotion recognition system using a Convolutional Neural Network (CNN) to classify images into seven emotion categories: anger, disgust, fear, happiness, neutral, sadness, and surprise. The system utilizes a dataset of 35,887 labeled grayscale images and employs data preprocessing techniques like normalization, resizing, and augmentation. A custom CNN architecture is designed with convolutional, max-pooling, and fully connected layers, along with dropout layers to mitigate overfitting. The model is trained using categorical cross-entropy loss and optimized with early stopping and learning rate schedulers, achieving an accuracy of around 60%.

5] GAN for digit Generation

This project utilizes a Generative Adversarial Network (GAN) to generate realistic images of the handwritten digit 4, leveraging the MNIST dataset. The GAN consists of a generator, which creates synthetic images from random noise, and a discriminator, which distinguishes between real and generated images. The generator employs dense, batch normalization, and Conv2DTranspose layers to output a 28x28 grayscale image, while the discriminator uses a convolutional neural network for binary classification. Trained with the Adam optimizer and binary cross-entropy loss over 100 epochs, the GAN successfully learns to produce 100 synthetic images that closely resemble the original MNIST digit 4 samples, demonstrating a consistent reduction in generator loss and stable training.
