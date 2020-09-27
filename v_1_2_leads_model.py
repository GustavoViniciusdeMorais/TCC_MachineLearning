# -*- coding: utf-8 -*-
"""v_1_2_leads_model.ipynb

Created by: Gustavo Vinicius de Morais

https://www.zend-zce.com/en/yellow-pages/ZEND031130

"""

#imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
import seaborn as sn
import matplotlib.pyplot as plt

# get data frame
from google.colab import files
uploaded = files.upload()

# import data frame
df = pd.read_csv('SampleData.csv', sep=',', encoding='cp1252')

# treat column names
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

# create a column class based on a column value
df['Class'] = pd.np.where(df.lead_stage.str.contains("Not Interested"), "LOW",
              pd.np.where(df.lead_stage.str.contains("Lost"), "LOW",
              pd.np.where(df.lead_stage.str.contains("Unreachable"), "LOW",
              pd.np.where(df.lead_stage.str.contains("Junk Lead"), "LOW",
              pd.np.where(df.lead_stage.str.contains("Not Eligible"), "LOW",
              pd.np.where(df.lead_stage.str.contains("Qualified"), "MEDIUM",
              pd.np.where(df.lead_stage.str.contains("Not Called"), "MEDIUM",
              pd.np.where(df.lead_stage.str.contains("Interested"), "HIGH",
              pd.np.where(df.lead_stage.str.contains("Closed"), "HIGH", "null"
)))))))))

# drop ID's columns
df = df.drop('prospect_id' , axis='columns')
df = df.drop('lead_number' , axis='columns')

# convert type of column's values to string to apply label encoder
df = df.astype(str)

# label encoder all at once
new_df = df.apply(LabelEncoder().fit_transform)

# Adjust the scale of the values in the columns
#StandardScaler
df_scaled = pd.DataFrame(StandardScaler().fit_transform(new_df), columns=new_df.columns)

# get independent and dependent variables
previsores = df_scaled.iloc[:,0:120].values
classe = new_df.iloc[:,120].values

df.if_you_are_a_working_professional_please_mention.unique()
#df.lead_profile.unique()

# choose best attributes

# Create an SelectKBest object to select features with two best ANOVA F-Values
selector = SelectKBest(f_classif, k=10)

# Choose the best attributes to the model
selector.fit(previsores, classe)

# Show the name of the columns in the data set that are the best attributes to the model
cols = selector.get_support(indices=True)
features_df_new = df.iloc[:,cols]

# show the columns that best contribute to the model
features_df_new.head()

# build dataframe with the features that best contributes to the model
df_selcted_features_1 = df_scaled.filter(['lead_stage', 'engagement_score', 'lead_quality', 'tags','lead_profile', 'if_you_are_a_working_professional_please_mention'])
df_selcted_features_1.head()

# get best independent variables
best_features = df_selcted_features_1.values

# get train test data
X_treinamento2, X_teste2, y_treinamento2, y_teste2 = train_test_split(best_features,
                                                                  classe,
                                                                  test_size=0.3,
                                                                  random_state=0)

# train model
# I am going to use the best attributes to build the model
svm = SVC(kernel='linear')
svm.fit(X_treinamento2, y_treinamento2)

# show model results
p_y = svm.predict(X_teste2)

taxa_acerto = accuracy_score(y_teste2, p_y)
taxa_acerto # 0.8892496392496393

# shows when the model predicted the right class

cf_matrix = confusion_matrix(y_teste2, p_y)
#print(cf_matrix)
df_cm = pd.DataFrame(cf_matrix, range(3), range(3))
#print(df_cm.head())
sn.heatmap(df_cm, annot=True, fmt="d")
plt.show()