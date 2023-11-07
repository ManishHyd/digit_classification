from flask import Flask

app = Flask(__name__)

@app.route("/<name>")
def hello_world(name):
    return "<p>Hello, World!</p>" + name
@app.route("/<name>/<a>/<b>")
def sum_of_numbers(name,a,b):
    return name  + " your sum is " + str(int(a) + int(b))