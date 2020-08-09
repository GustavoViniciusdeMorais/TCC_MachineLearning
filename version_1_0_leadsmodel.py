# -*- coding: utf-8 -*-
"""Version_1_0_LeadsModel.ipynb

Created by: Gustavo Vinicius

"""

#imports
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.svm import SVC 
from sklearn.ensemble import ExtraTreesClassifier

from google.colab import files
uploaded = files.upload()

df = pd.read_csv('SampleData.csv', sep=',', encoding='cp1252')

df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(', '').str.replace(')', '')

df['Class'] = pd.np.where(df.lead_stage.str.contains("Not Interested"), "LOW",
              pd.np.where(df.lead_stage.str.contains("Lost"), "LOW",
              pd.np.where(df.lead_stage.str.contains("Unreachable"), "LOW",
              pd.np.where(df.lead_stage.str.contains("Junk Lead"), "LOW",
              pd.np.where(df.lead_stage.str.contains("Not Eligible"), "LOW",
              pd.np.where(df.lead_stage.str.contains("Qualified"), "MEDIUM",
              pd.np.where(df.lead_stage.str.contains("Not Called"), "MEDIUM",
              pd.np.where(df.lead_stage.str.contains("Interested"), "HIGH",
              pd.np.where(df.lead_stage.str.contains("Closed"), "HIGH", "null"
))))))))) # create a column class based on a column value

# drop ID's columns
df = df.drop('prospect_id' , axis='columns')
df = df.drop('lead_number' , axis='columns')

print(len(df.columns)) # show number of columns

df = df.astype(str)

previsores = df.iloc[:,0:120].values
classe = df.iloc[:,120].values

classe

labelencoder = LabelEncoder()
previsores[:,0] = labelencoder.fit_transform(previsores[:,0])
previsores[:,1] = labelencoder.fit_transform(previsores[:,1])
previsores[:,2] = labelencoder.fit_transform(previsores[:,2])
previsores[:,3] = labelencoder.fit_transform(previsores[:,3])
previsores[:,4] = labelencoder.fit_transform(previsores[:,4])
previsores[:,5] = labelencoder.fit_transform(previsores[:,5])
previsores[:,6] = labelencoder.fit_transform(previsores[:,6])
previsores[:,7] = labelencoder.fit_transform(previsores[:,7])
previsores[:,8] = labelencoder.fit_transform(previsores[:,8])
previsores[:,9] = labelencoder.fit_transform(previsores[:,9])
previsores[:,10] = labelencoder.fit_transform(previsores[:,10])
previsores[:,11] = labelencoder.fit_transform(previsores[:,11])
previsores[:,12] = labelencoder.fit_transform(previsores[:,12])
previsores[:,13] = labelencoder.fit_transform(previsores[:,13])
previsores[:,14] = labelencoder.fit_transform(previsores[:,14])
previsores[:,15] = labelencoder.fit_transform(previsores[:,15])
previsores[:,16] = labelencoder.fit_transform(previsores[:,16])
previsores[:,17] = labelencoder.fit_transform(previsores[:,17])
previsores[:,18] = labelencoder.fit_transform(previsores[:,18])
previsores[:,19] = labelencoder.fit_transform(previsores[:,19])
previsores[:,20] = labelencoder.fit_transform(previsores[:,20])
previsores[:,21] = labelencoder.fit_transform(previsores[:,21])
previsores[:,22] = labelencoder.fit_transform(previsores[:,22])
previsores[:,23] = labelencoder.fit_transform(previsores[:,23])
previsores[:,24] = labelencoder.fit_transform(previsores[:,24])
previsores[:,25] = labelencoder.fit_transform(previsores[:,25])
previsores[:,26] = labelencoder.fit_transform(previsores[:,26])
previsores[:,27] = labelencoder.fit_transform(previsores[:,27])
previsores[:,28] = labelencoder.fit_transform(previsores[:,28])
previsores[:,29] = labelencoder.fit_transform(previsores[:,29])
previsores[:,30] = labelencoder.fit_transform(previsores[:,30])
previsores[:,31] = labelencoder.fit_transform(previsores[:,31])
previsores[:,32] = labelencoder.fit_transform(previsores[:,32])
previsores[:,33] = labelencoder.fit_transform(previsores[:,33])
previsores[:,34] = labelencoder.fit_transform(previsores[:,34])
previsores[:,35] = labelencoder.fit_transform(previsores[:,35])
previsores[:,36] = labelencoder.fit_transform(previsores[:,36])
previsores[:,37] = labelencoder.fit_transform(previsores[:,37])
previsores[:,38] = labelencoder.fit_transform(previsores[:,38])
previsores[:,39] = labelencoder.fit_transform(previsores[:,39])
previsores[:,40] = labelencoder.fit_transform(previsores[:,40])
previsores[:,41] = labelencoder.fit_transform(previsores[:,41])
previsores[:,42] = labelencoder.fit_transform(previsores[:,42])
previsores[:,43] = labelencoder.fit_transform(previsores[:,43])
previsores[:,44] = labelencoder.fit_transform(previsores[:,44])
previsores[:,45] = labelencoder.fit_transform(previsores[:,45])
previsores[:,46] = labelencoder.fit_transform(previsores[:,46])
previsores[:,47] = labelencoder.fit_transform(previsores[:,47])
previsores[:,48] = labelencoder.fit_transform(previsores[:,48])
previsores[:,49] = labelencoder.fit_transform(previsores[:,49])
previsores[:,50] = labelencoder.fit_transform(previsores[:,50])
previsores[:,51] = labelencoder.fit_transform(previsores[:,51])
previsores[:,52] = labelencoder.fit_transform(previsores[:,52])
previsores[:,53] = labelencoder.fit_transform(previsores[:,53])
previsores[:,54] = labelencoder.fit_transform(previsores[:,54])
previsores[:,55] = labelencoder.fit_transform(previsores[:,55])
previsores[:,56] = labelencoder.fit_transform(previsores[:,56])
previsores[:,57] = labelencoder.fit_transform(previsores[:,57])
previsores[:,58] = labelencoder.fit_transform(previsores[:,58])
previsores[:,59] = labelencoder.fit_transform(previsores[:,59])
previsores[:,60] = labelencoder.fit_transform(previsores[:,60])
previsores[:,61] = labelencoder.fit_transform(previsores[:,61])
previsores[:,62] = labelencoder.fit_transform(previsores[:,62])
previsores[:,63] = labelencoder.fit_transform(previsores[:,63])
previsores[:,64] = labelencoder.fit_transform(previsores[:,64])
previsores[:,65] = labelencoder.fit_transform(previsores[:,65])
previsores[:,66] = labelencoder.fit_transform(previsores[:,66])
previsores[:,67] = labelencoder.fit_transform(previsores[:,67])
previsores[:,68] = labelencoder.fit_transform(previsores[:,68])
previsores[:,69] = labelencoder.fit_transform(previsores[:,69])
previsores[:,70] = labelencoder.fit_transform(previsores[:,70])
previsores[:,71] = labelencoder.fit_transform(previsores[:,71])
previsores[:,72] = labelencoder.fit_transform(previsores[:,72])
previsores[:,73] = labelencoder.fit_transform(previsores[:,73])
previsores[:,74] = labelencoder.fit_transform(previsores[:,74])
previsores[:,75] = labelencoder.fit_transform(previsores[:,75])
previsores[:,76] = labelencoder.fit_transform(previsores[:,76])
previsores[:,77] = labelencoder.fit_transform(previsores[:,77])
previsores[:,78] = labelencoder.fit_transform(previsores[:,78])
previsores[:,79] = labelencoder.fit_transform(previsores[:,79])
previsores[:,80] = labelencoder.fit_transform(previsores[:,80])
previsores[:,81] = labelencoder.fit_transform(previsores[:,81])
previsores[:,82] = labelencoder.fit_transform(previsores[:,82])
previsores[:,83] = labelencoder.fit_transform(previsores[:,83])
previsores[:,84] = labelencoder.fit_transform(previsores[:,84])
previsores[:,85] = labelencoder.fit_transform(previsores[:,85])
previsores[:,86] = labelencoder.fit_transform(previsores[:,86])
previsores[:,87] = labelencoder.fit_transform(previsores[:,87])
previsores[:,88] = labelencoder.fit_transform(previsores[:,88])
previsores[:,89] = labelencoder.fit_transform(previsores[:,89])
previsores[:,90] = labelencoder.fit_transform(previsores[:,90])
previsores[:,91] = labelencoder.fit_transform(previsores[:,91])
previsores[:,92] = labelencoder.fit_transform(previsores[:,92])
previsores[:,93] = labelencoder.fit_transform(previsores[:,93])
previsores[:,94] = labelencoder.fit_transform(previsores[:,94])
previsores[:,95] = labelencoder.fit_transform(previsores[:,95])
previsores[:,96] = labelencoder.fit_transform(previsores[:,96])
previsores[:,97] = labelencoder.fit_transform(previsores[:,97])
previsores[:,98] = labelencoder.fit_transform(previsores[:,98])
previsores[:,99] = labelencoder.fit_transform(previsores[:,99])
previsores[:,100] = labelencoder.fit_transform(previsores[:,100])
previsores[:,101] = labelencoder.fit_transform(previsores[:,101])
previsores[:,102] = labelencoder.fit_transform(previsores[:,102])
previsores[:,103] = labelencoder.fit_transform(previsores[:,103])
previsores[:,104] = labelencoder.fit_transform(previsores[:,104])
previsores[:,105] = labelencoder.fit_transform(previsores[:,105])
previsores[:,106] = labelencoder.fit_transform(previsores[:,106])
previsores[:,107] = labelencoder.fit_transform(previsores[:,107])
previsores[:,108] = labelencoder.fit_transform(previsores[:,108])
previsores[:,109] = labelencoder.fit_transform(previsores[:,109])
previsores[:,110] = labelencoder.fit_transform(previsores[:,110])
previsores[:,111] = labelencoder.fit_transform(previsores[:,111])
previsores[:,112] = labelencoder.fit_transform(previsores[:,112])
previsores[:,113] = labelencoder.fit_transform(previsores[:,113])
previsores[:,114] = labelencoder.fit_transform(previsores[:,114])
previsores[:,115] = labelencoder.fit_transform(previsores[:,115])
previsores[:,116] = labelencoder.fit_transform(previsores[:,116])
previsores[:,117] = labelencoder.fit_transform(previsores[:,117])
previsores[:,118] = labelencoder.fit_transform(previsores[:,118])
previsores[:,119] = labelencoder.fit_transform(previsores[:,119])

df.iloc[:,120]

X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(previsores,
                                                                  classe,
                                                                  test_size=0.3,
                                                                  random_state=0)

forest = ExtraTreesClassifier()
forest.fit(X_treinamento, y_treinamento)

importancias = forest.feature_importances_
importancias

importances = list(zip(forest.feature_importances_, df.columns))
importances.sort()
importances # show importance of the attributes to the model

# I am going to use the best attributes to build the model

#ID's list of important columns to the model:
#78 eventname, 77 amt, 60 any_other_please_specify, 58 digital_advertisement, 96 what_attracted_you_to_consider_someschool, 6 job_title, 59 through_recommendations, 31 address_2, 35 zip, 100 lead_type, 55 newspaper_article, 13 do_not_call, 52 next_follow_up, 14 lead_stage,
# [78, 77, 60, 58, 96, 6, 59, 31, 35, 100, 55, 13, 52, 14,]

X_treinamento2 = X_treinamento[:,[78, 77, 60, 58, 96, 6, 59, 31, 35, 100, 55, 13, 52, 14,]]
X_teste2  = X_teste[:,[78, 77, 60, 58, 96, 6, 59, 31, 35, 100, 55, 13, 52, 14,]]

svm = SVC(kernel='linear')
svm.fit(X_treinamento2, y_treinamento)

previsoes2 = svm.predict(X_teste2)

taxa_acerto = accuracy_score(y_teste, previsoes2)
taxa_acerto # 0.7463924963924964