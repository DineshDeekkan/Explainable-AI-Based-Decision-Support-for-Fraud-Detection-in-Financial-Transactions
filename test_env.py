import sys
print("Python version:", sys.version)

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
from kafka import KafkaProducer
from pyspark.sql import SparkSession
import shap
from faker import Faker

print("All core libraries installed successfully!")

df = pd.read_csv("transactions_dataset.csv")
print(df["true_label"].value_counts())