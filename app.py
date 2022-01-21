from flask import Flask, render_template
from flask import request
import pickle
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from xgboost import XGBClassifier


app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")
 
@app.route("/Submit",  methods=['POST'])
def Submit():
    error = None
    if request.method == 'POST':
        if request.form['username']:
            print(request.form['username'])
            #Moving forward code
            forward_message = "Moving Forward..."
           
            # load the vectorizer
            loaded_vectorizer = pickle.load(open('./models/vectorizer.pickle', 'rb'))

            # load model from file
            loaded_model = pickle.load(open("./models/model.pickle", "rb"))

            # Loadrecommendation file
            recommendation_df = pd.read_pickle("./models/itembased-df.pickle")
            username = request.form['username']
            top_recommended_products = pd.DataFrame(recommendation_df.loc[username].sort_values(ascending=False)[0:20])

            # Reading Actual Data file
            ratings = pd.read_csv('./data/sample30.csv' , encoding='latin-1')
            
            def sentiment_predict(reviews_text):
                #print(reviews_text)
                test_data = CountVectorizer(vocabulary=loaded_vectorizer.get_feature_names_out())
                X_test_value = test_data.fit_transform(reviews_text)
                sentiment_output = pd.DataFrame(loaded_model.predict(X_test_value), columns=['sentiment'])
                reviews_text.reset_index(drop=True, inplace=True)
                sentiment_output.reset_index(drop=True, inplace=True) 
                sentiment_output = pd.concat([reviews_text, sentiment_output], axis=1)
                return sentiment_output

                       
            sentimetment_result_df = pd.DataFrame()
            for index, row in top_recommended_products.iterrows():
                product_reviews = ratings[ratings['name'].isin([row.name])].reviews_text
                sentimetment_result = sentiment_predict(product_reviews)
                sentimetment_result['name'] = row.name
                sentimetment_result_df = sentimetment_result_df.append(sentimetment_result)
                
            
            gb = sentimetment_result_df.groupby(['name','sentiment']).size().reset_index(name='counts')
            gb_total = sentimetment_result_df.groupby(['name']).size().reset_index(name='total_counts')
            gb = pd.merge(gb,gb_total[['name','total_counts']],on='name', how='right')
            gb['percentage'] = (gb['counts']/gb['total_counts'])*100
            gb_final = gb[(gb['percentage'] >=50) & (gb['sentiment'] == 1)]
            show_flag = True
            print(gb_final.name)
        
        else:
            error = 'User Name Is empty. Please '    

    return render_template('index.html', row_data=list(gb_final.name), show_content = show_flag, error=error)

if __name__ == '__main__':
    app.run()
