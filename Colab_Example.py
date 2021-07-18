# Imports for accessing Datathon data using Google BigQuery.
from google.colab import auth
from google.cloud import bigquery

import os
# Import some useful libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Below imports are used to print out pretty pandas dataframes
from IPython.display import display, HTML

# We need to authenticate our Google account
auth.authenticate_user()

# Set up the environment for using BigQuery

project_id = 'physionet-data' # You can create a dummy project in your google cloud console, and then put its project_id here
os.environ['GOOGLE_CLOUD_PROJECT'] = project_id

# Wrapper function to help you execute a query and store it in a pandas dataframe directly

def run_query(query):
  return pd.io.gbq.read_gbq(
      query,
      project_id=project_id,
      verbose=False,
      configuration={'query': {
          'useLegacySql': False
      }})

# Same query from the GCP example

df = run_query("""
SELECT p.subject_id, p.dob, a.hadm_id,
    a.admittime, p.expire_flag,
    MIN (a.admittime) OVER (PARTITION BY p.subject_id) AS first_admittime
FROM `physionet-data.mimiciii_clinical.admissions` a
INNER JOIN `physionet-data.mimiciii_clinical.patients` p
  ON p.subject_id = a.subject_id
ORDER BY a.hadm_id, p.subject_id;
""")

df.head(20)