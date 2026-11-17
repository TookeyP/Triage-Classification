# Triage-Classification
Using Machine Learning to predict accident &amp; emergency patient acuity scores.

This project was submitted as my final project of my MSc  in Data Science &amp; Artificial Intelligence.
As it's basis the project uses a dataset of 560,000 patient visits to three hospital emergency departments in the USA.
The dataset along with the original research paper can be found  [HERE](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0201016).

The purpose of this project was to combine ML classification with explainable outputs to produce the foundations of a clinical decision support system.
That could be used in A&amp;E departments to appropriately triage patients.

A series of experiments were undertaken to evaluate ML models for this task.
Additionally, an LLM-generated minority class oversampling technique was implemented to improve model performance for the minority class.

The final proof-of-concept system developed utilises an XGBoost Model with SHAP explanations to produce a patient's predicted triage score and the top 5 contributing factors. This is intended to serve as the backend for staff decision support system.

## Data Files
**original.rdata** - This is the original raw dataset used. (not included, available from the link above).

**processed_triage_dataset.parquet** - The processed dataset. Produced as output from Data Preprocessing.ipynb.

**esi_1_sample.csv** - The sample of 200 ESI 1 patient records provided to the LLM for production of similar synthetic records.

**chief complaint values.docx** - A list of all possible values for chief complaint. Provided to the LLM.

**LLM Output Master.csv** - 500 Synthetic patient records produced by the LLM. Used in both Model Development.ipynb and Final Model Testing &amp; PoC.ipynb.

## SRC Files
**Data Preprocessing.ipynb** - Loads the original dataset, initial data exploration and cleaning is performed. Feature distribution plots produced. Null values imputed using KNN imputation. Correlation of features to the target feature examined. Final dataframe created, formatted and outputted to file.

**Model Development.ipynb** - Loads the cleaned dataset and performs a series of experiments. File is split into sections 0-6: sections 0-2 must be executed, sections 3-6 can be executed independently of one another.

**Final Model Testing &amp; PoC.ipynb** - Loads the cleaned dataset and synthetic samples, combines them and then applies the final model. Model tested on test set for final evaluation.
The Proof-of-Concept system included in this file. File is split into 4 sections: Sections 0-1 are required prerequisites. Section 2 is testing model. Section 3 is Proof-of-Concept.
