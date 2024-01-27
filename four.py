
from flask import Flask
app = Flask(__name__)

@app.route('/')
def hello():
    return 'Hello Folla!'

# if __name__ == '__main__':
#     # run app in debug mode on port 5000
#     app.run(debug=True, port=5000, host='0.0.0.0')
