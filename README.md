# Phishing Classifier: Detecting Fraudulent Websites Using AI

In today's tech-driven world, cyberattacks are an ever-present threat. One of the most common and successful forms of
attack is **phishing**. Phishing is a type of fraud where an attacker impersonates a legitimate entity—usually through
emails, text messages, or social media—aiming to steal sensitive information like personal details or credit card
information. Typically, these attacks involve redirecting victims to fake websites designed to look authentic, tricking
them into providing personal data or paying for services they’ll never receive. Phishing can also serve as a vehicle for
more malicious attacks, such as ransomware.

The high success rate of phishing is often due to the simplicity of its execution, the availability of tools that
facilitate website cloning, and the lack of cybersecurity awareness, particularly among vulnerable groups like the
elderly and young children. For these reasons, the development of automatic detection systems for fraudulent websites
has become crucial to help protect users.

This project focuses on applying and improving artificial intelligence techniques to detect phishing websites, aiming to
enhance the accuracy and performance of current models. Various machine learning (ML) algorithms, such as **Logistic
Regression** and **Random Forest**, along with deep learning techniques like **Convolutional Neural Networks (CNNs)**,
will be explored.

## Repository Overview

This repository contains the code for six experiments exploring both traditional ML and deep learning models for
phishing website classification:

1. **Execution of Classical Machine Learning Models:**
    - A baseline experiment using classical models (Decision Tree, Random Forest, Logistic Regression, Naïve Bayes, and
      Support Vector Machine).
    - Script: `classical_ml_models.py`

2. **Dataset Scaling and Repetition of Experiment 1:**
    - Standardizing the dataset and re-evaluating the models.
    - Script: `scaled_data_models.py`

3. **Voting Ensemble of Random Forest, Logistic Regression, and Support Vector Machine:**
    - Combining the outputs of these models to enhance accuracy.
    - Script: `voting_ensemble.py`

4. **Neural Network with a Single Hidden Layer:**
    - Implementing a simple feedforward neural network.
    - Script: `simple_nn.py`

5. **Convolutional Neural Network (CNN):**
    - Implementing a CNN model for image-based feature extraction and classification.
    - Script: `cnn_model.py`

## Getting Started

### Prerequisites

To run the experiments, ensure you have the following installed:

- Python 3.1x
- Required libraries (see `requirements.txt` for details)

### Running the Experiments

Each experiment is contained in its own script. To run an experiment, execute the relevant script. For example, to run
the **CNN** experiment:

```bash
python3 cnn_model.py
```

## Additional Content

### Image Preparation

The images used in the CNN model are generated by the `generate_images.py` script and stored in the `figures/` folder.
You can regenerate these images by running the `generate_images.py` script.

```bash
python3 generate_images.py
```

Pre-generated images from the `generate_images.py` script are stored as follows:

- **Training set**:
    - Legitimate: `figures/train/legitimate/` (4585 images)
    - Phishing: `figures/train/phishing/` (4559 images)

- **Test set**:
    - Legitimate: `figures/test/legitimate/` (1130 images)
    - Phishing: `figures/test/phishing/` (1156 images)

These images are used as input to the CNN model in `cnn_model.py` for training and testing purposes.

### Explainable AI (XAI)

To enhance transparency and trust in the model predictions, we use **XAI** methods to identify the most significant
features contributing to the classification of websites. This analysis provides insights into how the model
differentiates between legitimate and phishing websites. You can run the XAI analysis with the following command:

```bash
python3 xai_analysis.py
```

## Model Summary

|          Experiment          |                                                    Model Type                                                    |            Key Techniques            |          Script          | 
|:----------------------------:|:----------------------------------------------------------------------------------------------------------------:|:------------------------------------:|:------------------------:|
|         Classical ML         | Decision Tree (DT), Random Forest (RF), Logistic Regression (LR), Naïve Bayes (NB), Support Vector Machine (SVM) |         Baseline performance         | `classical_ml_models.py` |
|         Scaled Data          |                                               DT, RF, LR, NB, SVM                                                |         Data standardization         | `scaled_data_models.py`  |
|       Voting Ensemble        |                                                   RF, LR, SVM                                                    | Model ensemble for enhanced accuracy |   `voting_ensemble.py`   |
|    Simple Neural Network     |                                               Single Hidden Layer                                                |    Basic neural network structure    |      `simple_nn.py`      |
| Convolutional Neural Network |                                        Convolutional Neural Network (CNN)                                        |    Feature extraction for images     |      `cnn_model.py`      |
