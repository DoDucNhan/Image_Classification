from flask import render_template
from app import app


@app.route('/', methods=['GET'])
@app.route('/flowers', methods=['GET'])
def login():
    return render_template('flowers.html', title='Flowers', header='HOA', status1="active")


@app.route('/cats_dogs', methods=['GET'])
def cats_dogs():
    return render_template('cats_dogs.html', title='Cats vs dogs', header='CHÓ MÈO', status2="active")



