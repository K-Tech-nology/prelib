import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler, MinMaxScaler, Normalizer

class Transformers :
    def __init__(self, data, features, target) :
        self.data = data
        self.features = features
        self.target = target

    def col_split(self) :
        
        X = self.data.drop(self.target, axis = 1)

        num_cols = []
        obj_cols = []
        time_cols = [] 
        
        num_dtypes = ['float','int']
        obj_dtypes = ['object']
        time_dtypes = ['datetime']
        
        #Identify Data Types per Column
        for column in X.columns :
            
            if X[column].dtype in obj_dtypes :
                obj_cols.append(column)
                
            elif X[column].dtype in num_dtypes :
                num_cols.append(column)
                    
            elif X[column].dtype in time_dtypes :
                time_cols.append(column)
                
                
        return num_cols, obj_cols, time_cols
    
    def num_transformer(self, transformers, num_cols) :
        
        X_numeric = self.data[num_cols]
        
        scaling = transformers
        
        X_scaled = scaling.fit_transform(X_numeric)
        
        return X_scaled
    
    def cat_transformer(self, transformers, cat_features) :
        cat_cols = cat_features
        
        X_cat = self.data[cat_cols]
        
        encoding = transformers
        
        for column in X_cat.columns :
            X_cat[f'{column}_label'] = encoding.fit_transform(X_cat[column])
        
        X_encoded = X_cat
        
        return X_encoded
    
    def text_transformer(self, text_features, number_words, maxlen) :
        
        import re
        import string
        from tensorflow.keras.preprocessing.text import Tokenizer
        from tensorflow.keras.preprocessing.sequence import pad_sequences
        from nltk.corpus import stopwords
        
        punct_ = string.punctuation 
        stop_words = stopwords.words('english')
        
        text = self.data[text_features]
        
        text = re.sub(r'[0-9]',' ',text) # remove numbers
        text = re.sub(r'#\w+',' ',text) # remove hashtag
        text = re.sub(r'@\w+',' ',text) # remove @
        text = re.sub(r'http:\S+',' ',text) # remove HTTP
        
        text = text.translate(str.maketrans(' ',' ',punct_)) # remove any punctuation
        
        text = text.lower().strip() # remove whitespace and lowercase letter
        
        text = " ".join([word for word in text.split() if word not in stop_words]) # remove stop words
        
        tokenizer = Tokenizer(num_words = number_words)
        tokenizer.fit_on_texts(text)
        
        sequence = tokenizer.texts_to_sequences(text)
        
        pad_seq = pad_sequences(sequence, maxlen = maxlen)
        
        X_text = pd.DataFrame(pad_seq)
        
        return X_text