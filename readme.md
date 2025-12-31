## Inference for ICH

- ICH means Intangible Cultural Heritage
- This project aims to develop a machine learning approach to infer the cultural heritage value of a given object based on its textual and visual materials.
- We will extract four ontology-based features from a given textual and visual corpus of ICH objects, and then use machine learning algorithms to train a model to predict the cultural heritage value of the object.

## Methodology

1. Data Collection: Collect a dataset of ICH objects with their textual and visual mateirals.
2. Building Dictionary on four principles: 'Historical / Aesthetic / Semiotic / Sociological' as the backgound of the ICH.
3. Deep Feature Extraction: Extract four ontology-based features from the textual and visual corpus of ICH objects.
4. Projecting the extracted features into the dictionary space.
5. Comput the cultural congnition on the projected space. Employ AHP: Analytic Hierarchy Process to determine the relative importance of the four features for the cultural cognition hidden behind the core of intangible cultural heritage.
6. Make decision based on computed cultural cognition. Employ Machine Learning: Train a machine learning model to predict the cultural heritage value of the object based on the extracted features.
7. Evaluation: Evaluate the performance of the model using various metrics such as accuracy, precision, recall, F1-score, and AUC-ROC.

## Usage

- Python 3.9 or higher is required.
- Codes in directory 'conditionalProjection' is used for Chinese embedding and projection calculation (Please ensure that you have configured the correct directory paths for your datasets and output vectors).
- Perform projection and similarity calculation using the command below:

```bash
python projection2.py
```

- You can perform classification tasks by executing the command below:

```bash
python classifying.py
```

- Or, execute the classification task using the command below, while also displaying the classification boundaries:

```bash
python classifying2.py
```

- Get simple assessment values by:

```bash
paython assessment.py
```

### Notes

- The project is still in the development phase, and we will continuously report on our latest research findings.
