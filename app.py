from flask import Flask, render_template
from flask import request
from model import RecommendationSystem
import warnings

warnings.filterwarnings("ignore")
app = Flask(__name__)

#Instance of the recommendation object
recommendation_system = RecommendationSystem()

@app.route("/")
def home():
    return render_template("index.html")
 
@app.route("/Submit",  methods=['POST'])
def Submit():
    error = None
    if request.method == 'POST':
        if request.form['username']:

            print(request.form['username'])
            show_flag =False
            error = None

            #Calling model file
            top_5_products = recommendation_system.recommendation_products(request.form['username'])

            if top_5_products is None:
                error = 'The username input is not present in the dataset'
                return render_template("index.html", column_names=None, 
                        row_data=None, zip=zip, show_content = show_flag, error=error)
            else:
                show_flag = True
                error = None
                return render_template("index.html", column_names=top_5_products.columns.values, 
                row_data=list(top_5_products.name), zip=zip, show_content = show_flag, error=error)

if __name__ == '__main__':
    app.run()
