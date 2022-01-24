import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

class RecommendationSystem:

    def __init__(self):
        # load the vectorizer
        self.loaded_vectorizer = pickle.load(open('./models/vectorizer.pickle', 'rb'))

        # load model from file
        self.loaded_model = pickle.load(open("./models/model.pickle", "rb"))

        # Loadrecommendation file
        self.recommendation_df = pd.read_pickle("./models/itembased-df.pickle")

        # Reading Actual Data file
        self.ratings = pd.read_csv('./data/sample30.csv' , encoding='latin-1')

    def recommendation_products(self, username):

        top_5_recommended_product = pd.DataFrame()
        error = None

         ## check the username is present in the given dataset or not
        if username.lower() not in self.ratings['reviews_username'].str.lower().values:
            return None
        
        try:
            # top 20 recommeded products for the username given in the input
            top_20_recommended_products = pd.DataFrame(self.recommendation_df.loc[username].sort_values(ascending=False)[0:20])
                
            
            def sentiment_predict(reviews_text):
                test_data = TfidfVectorizer(vocabulary=self.loaded_vectorizer.get_feature_names_out())
                X_test_value = test_data.fit_transform(reviews_text)
                #print(X_test_value)
                sentiment_output = pd.DataFrame(self.loaded_model.predict(X_test_value), columns=['sentiment'])
                reviews_text.reset_index(drop=True, inplace=True)
                sentiment_output.reset_index(drop=True, inplace=True) 
                sentiment_output = pd.concat([reviews_text, sentiment_output], axis=1)
                return sentiment_output
            
            sentimetment_result_df = pd.DataFrame()
            for index, row in top_20_recommended_products.iterrows():
                product_reviews = self.ratings[self.ratings['name'].isin([row.name])].reviews_text
                sentimetment_result = sentiment_predict(product_reviews)
                sentimetment_result['name'] = row.name
                sentimetment_result_df = sentimetment_result_df.append(sentimetment_result)

            gb = sentimetment_result_df.groupby(['name','sentiment']).size().reset_index(name='counts')
            gb_total = sentimetment_result_df.groupby(['name']).size().reset_index(name='total_counts')
            gb = pd.merge(gb,gb_total[['name','total_counts']],on='name', how='right')
            gb['percentage'] = (gb['counts']/gb['total_counts'])*100
            gb_final = gb[(gb['percentage'] >=50) & (gb['sentiment'] == 1)]
            top_5_recommended_product = gb_final.sort_values('percentage', ascending=False).head(5)
            show_flag = True
            print(top_5_recommended_product.name)
        
        except KeyError:
            print("The username input is not present in the dataset")
            error = 'The username input is not present in the dataset'
            return error
         
        return top_5_recommended_product