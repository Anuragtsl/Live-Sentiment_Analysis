from flask import Flask, render_template, request
import pickle
import re
import nltk

classifier = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pk', 'rb'))
stop_words = set(nltk.corpus.stopwords.words('english'))

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        message = request.form['message']
        message = re.sub(r'http\S+','', message)
        message = ' '.join([word for word in message.split() if word not in (stop_words)])
        message = re.sub(r'[^\w\s]','', message)
        message = re.sub(r'\d+','', message)
        message = re.sub(r'\s+(.)\1+\b','', message).strip()
        if(len(message)>1):
            vect = vectorizer.transform([message])
            my_prediction = classifier.predict(vect)[0]
        
            return render_template('result.html', prediction=my_prediction)
        
        else:
            return render_template('result.html', prediction=-1)

if __name__=='__main__':
    app.run(debug=True)