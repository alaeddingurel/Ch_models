
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import nltk

from sklearn.model_selection import cross_val_score

from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from sklearn.svm import SVR
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV

def gs_sgd(train, y_train, scoring="recall", f1_param="macro"):
    model_dict = dict()
    sgd_grid = GridSearchCV(
            estimator=SGDClassifier(),
            param_grid={
                'alpha': [0.00001, 0.0001, 0.001, 0.01, 0.1],
                'l1_ratio': [0.15, 0.20, 0.4, 0.6, 0.8]
            },
            cv=5, scoring=scoring, verbose=1, n_jobs=-1)
    
    grid_result = sgd_grid.fit(train_tfidf, y_train)
    
    best_params = grid_result.best_params_
    
    best_sgd = SGDClassifier(alpha=best_params["alpha"])
    best_sgd.fit(train, y_train)
    
    sgd_predicted = best_sgd.predict(test_tfidf)
    
    model_dict['model'] = best_sgd
    model_dict['prediction'] = sgd_predicted

    print("SGD F1 Score : ")
    print(f1_score(y_test, sgd_predicted, average=f1_param))
    print("SGD Accuracy Score : ")
    print(accuracy_score(y_test, sgd_predicted))
    print("SGD Recall : ")
    print(recall_score(y_test, sgd_predicted))
    print("SGD Precision : ")
    print(precision_score(y_test, sgd_predicted))

    return model_dict
    
def gs_svm(train, y_train, scoring="recall", f1_param="macro"):
    model_dict = dict()
    gsc = GridSearchCV(
        estimator=svm.LinearSVC(),
        param_grid={
            'C': [0.1, 1, 10, 100, 1000]
        },
        cv=5, scoring=scoring, verbose=1, n_jobs=-1)
    grid_result = gsc.fit(train, y_train)

    best_params = grid_result.best_params_
    best_svr = svm.LinearSVC(C=best_params["C"])
    best_svr.fit(train, y_train)
    svm_predicted = best_svr.predict(test_tfidf)
    
    model_dict['model'] = best_svr
    model_dict['prediction'] = svm_predicted
    
    
    print("SVM F1 Score : ")
    print(f1_score(y_test, svm_predicted, average=f1_param))
    print("SVM Accuracy Score : ")
    print(accuracy_score(y_test, svm_predicted))
    print("SVM Recall : ")
    print(recall_score(y_test, svm_predicted))
    print("SVM Precision : ")
    print(precision_score(y_test, svm_predicted))
    
def gs_rf(train, y_train, scoring="recall", f1_param="macro"):
    model_dict = dict()
    rf_grid = GridSearchCV(
        estimator=RandomForestClassifier(),
        param_grid={
            'n_estimators': [10, 20, 50, 100]
            
        },
        cv=5, scoring=scoring, verbose=1, n_jobs=-1)

    grid_result = rf_grid.fit(train, y_train)
    
    best_params = grid_result.best_params_
    
    best_rf = RandomForestClassifier(n_estimators=best_params['n_estimators'])
    best_rf.fit(train_tfidf, y_train)
    
    rf_predicted = best_rf.predict(test_tfidf)
    
    model_dict['model'] = best_rf
    model_dict['prediction'] = rf_predicted
    
    print("RF F1 Score : ")
    print(f1_score(y_test, rf_predicted, average=f1_param))
    print("RF Accuracy Score : ")
    print(accuracy_score(y_test, rf_predicted))
    print("RF Recall : ")
    print(recall_score(y_test, rf_predicted))
    print("RF Precision : ")
    print(precision_score(y_test, rf_predicted))
        
    

# In[2]:


peoplesdaily = "/home/user/Desktop/china_document_level/corpus/Document/Peoples_Daily/20190128-Peoples_Daily_800_agreed.json"
rcv = "/home/user/Desktop/china_document_level/corpus/Document/RCV1_Code_China/20190221_Reuters_codeChina_SVM_RF_Bert_preds_Adjudication_base.json"
scmp = "/home/user/Desktop/china_document_level/corpus/Document/Scmp/20190321_scmp_AgreedByAll.xlsx"

peoplesdaily_df = pd.read_json(peoplesdaily, lines=True)
rcv_df = pd.read_json(rcv, lines=True)
scmp_df = pd.read_excel(scmp)

# df = pd.read_json(file, lines=True)


# In[3]:


peoplesdaily_df = peoplesdaily_df[['text','label']]
rcv_df = rcv_df[['text','label']]
scmp_df = scmp_df[['text','label']]


# In[4]:


rcv_df.loc[rcv_df.label == 2, 'label'] = 0 


# In[5]:


scmp_df = scmp_df.loc[~np.isnan(scmp_df['label'])]


# In[6]:


df = peoplesdaily_df.append(rcv_df, ignore_index=True)


# In[7]:


df = df.append(scmp_df, ignore_index=True)


# In[8]:


len(df)


# In[9]:


df = df.rename(columns={'text': 'sentence'})


# In[10]:


df.sentence = df.sentence.str.replace('\d+', '')
df_sent = (df['sentence'].str.len() > 10)
ne_df = df.loc[df_sent]
df = ne_df
df = df[~df.sentence.str.startswith('https')]
df = df[~df.sentence.str.startswith('url : ')]


# In[11]:


import nltk

w_tokenizer = nltk.tokenize.WhitespaceTokenizer()
lemmatizer = nltk.stem.WordNetLemmatizer()


# In[12]:


def lemmatize_text(text):
    return [lemmatizer.lemmatize(w) for w in w_tokenizer.tokenize(text)]

df.sentence = df.sentence.apply(lemmatize_text)


# In[13]:


df_X = df['sentence']
df_Y = df['label']

df_fina = pd.concat([df_X, df_Y], axis=1, sort=False)


# In[14]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df_X, df_Y, test_size=0.2, random_state=80)


# In[15]:


def dummy_fun(doc):
    return doc


# In[16]:


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vect = TfidfVectorizer(stop_words='english', ngram_range=(1,2), tokenizer=dummy_fun, preprocessor=dummy_fun, analyzer='word', token_pattern=r'\w{1,}', max_features=20000)
train_tfidf = tfidf_vect.fit_transform(X_train)
test_tfidf = tfidf_vect.transform(X_test)



svr = gs_svm(train_tfidf, y_train)
rf = gs_rf(train_tfidf, y_train)
sgd = gs_sgd(train_tfidf, y_train)

# In[125]:


prediction_df = pd.DataFrame(columns=["svm", "sgd", "rf", "result"])

prediction_df['svm'] = svm_predicted
prediction_df['sgd'] = sgd_predicted
prediction_df['rf'] = rf_predicted


for idx, elem in prediction_df.iterrows():
    count = 0
    if elem['svm'] == 1:
        count = count + 1
    if elem['sgd'] == 1:
        count = count + 1
    if elem['rf'] == 1:
        count = count + 1
    if count >= 2:
        prediction_df['result'].values[idx] = 1
    else:
        prediction_df['result'].values[idx] = 0
        
print("Result F1 Score : ")
print(f1_score(y_test, prediction_df['result'].tolist(), average="macro"))
from sklearn.metrics import accuracy_score
print("Result Accuracy Score : ")
print(accuracy_score(y_test, prediction_df['result'].tolist()))
print("Result Recall : ")
print(recall_score(y_test, prediction_df['result'].tolist()))
print("Result Precision : ")
print(precision_score(y_test, prediction_df['result'].tolist()))

print(classification_report(y_test, prediction_df['result'].tolist(), target_names=['class 0', 'class 1']))


# In[126]:


prediction_df


# In[127]:


prediction_df_one = pd.DataFrame(columns=["svm", "sgd", "rf", "result"])
prediction_df_one['svm'] = svm_predicted
prediction_df_one['sgd'] = sgd_predicted
prediction_df_one['rf'] = rf_predicted

for idx, elem in prediction_df_one.iterrows():
    count = 0
    if elem['svm'] == 1:
        count = count + 1
    if elem['sgd'] == 1:
        count = count + 1
    if elem['rf'] == 1:
        count = count + 1
    if count >= 1:
        prediction_df_one['result'].values[idx] = 1
    else:
        prediction_df_one['result'].values[idx] = 0
        
print("Result F1 Score : ")
print(f1_score(y_test, prediction_df_one['result'].tolist(), average="macro"))
from sklearn.metrics import accuracy_score
print("Result Accuracy Score : ")
print(accuracy_score(y_test, prediction_df_one['result'].tolist()))
print("Result Recall : ")
print(recall_score(y_test, prediction_df_one['result'].tolist()))
print("Result Precision : ")
print(precision_score(y_test, prediction_df_one['result'].tolist()))

print(classification_report(y_test, prediction_df_one['result'].tolist(), target_names=['class 0', 'class 1']))


# In[130]:


prediction_df_three = pd.DataFrame(columns=["svm", "sgd", "rf", "result"])
prediction_df_three['svm'] = svm_predicted
prediction_df_three['sgd'] = sgd_predicted
prediction_df_three['rf'] = rf_predicted

for idx, elem in prediction_df_three.iterrows():
    count = 0
    if elem['svm'] == 1:
        count = count + 1
    if elem['sgd'] == 1:
        count = count + 1
    if elem['rf'] == 1:
        count = count + 1
    if count >= 3:
        prediction_df_three['result'].values[idx] = 1
    else:
        prediction_df_three['result'].values[idx] = 0
        
print("Result F1 Score : ")
print(f1_score(y_test, prediction_df_three['result'].tolist(), average="macro"))
from sklearn.metrics import accuracy_score
print("Result Accuracy Score : ")
print(accuracy_score(y_test, prediction_df_three['result'].tolist()))
print("Result Recall : ")
print(recall_score(y_test, prediction_df_three['result'].tolist()))
print("Result Precision : ")
print(precision_score(y_test, prediction_df_three['result'].tolist()))

print(classification_report(y_test, prediction_df_three['result'].tolist(), target_names=['class 0', 'class 1']))


# In[149]:


prediction_df_three['y_test'] = y_test.values
prediction_df_three['X_test'] = X_test.values


# In[150]:


prediction_df_three


# In[154]:


prediction_df_three.iloc[23].X_test


# In[33]:


df_all = pd.DataFrame()


# In[34]:


df_all['sentence'] = X_test
df_all['label'] = y_test


# In[35]:


df_all = df_all.loc[df_all.label == 1]


# In[36]:


X_test = df_all['sentence']
y_test = df_all['label']


# In[37]:


from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vect = TfidfVectorizer(stop_words='english', ngram_range=(1,2), tokenizer=dummy_fun, preprocessor=dummy_fun, analyzer='word', token_pattern=r'\w{1,}', max_features=20000)
train_tfidf = tfidf_vect.fit_transform(X_train)
test_tfidf = tfidf_vect.transform(X_test)


# In[40]:


ones_predict_sgd = sgd.predict(test_tfidf)
ones_predict_svm = best_svr.predict(test_tfidf)
ones_predict_lr = lr.predict(test_tfidf)


# In[41]:


print(accuracy_score(y_test, ones_predict_sgd))
print(accuracy_score(y_test, ones_predict_svm))
print(accuracy_score(y_test, ones_predict_lr))


# In[42]:


len(df_all)


# In[43]:


y_test


# In[44]:


from sklearn.metrics import classification_report


# In[45]:


print(classification_report(y_test, ones_predict_sgd, target_names=["class 0", "class 1"]))


# In[46]:


print(classification_report(y_test, ones_predict_svm, target_names=["class 0", "class 1"]))


# In[47]:


print(classification_report(y_test, ones_predict_lr, target_names=["class 0", "class 1"]))


# In[48]:


len(df)


# In[49]:


df.to_excel("all_china.xlsx")


# In[139]:


X_test.tolist()

