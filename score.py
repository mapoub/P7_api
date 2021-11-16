from flask import Flask, jsonify, request

app = Flask(__name__)

@app.route('/api', methods=['POST', 'DELETE', 'GET'])
def my_microservice():
    return jsonify({'Hello 2': 'World!'})

@app.route('/api/person/<person_id>')
def person(person_id):
    dico = {}
    if person_id =="client1":
        dico["score"] = "35"
        response = jsonify(dico)
    elif person_id =="client2":
        response = jsonify({'score': '50'})
    elif person_id =="client3":
        response = jsonify({'score': '65'})
    elif person_id =="client4":
        response = jsonify({'score': '40'})
    #response = jsonify({'Hello': person_id})
    return response



@app.route('/', methods=['POST', 'DELETE', 'GET'])
def my_microservice2():
    print(request)
    print(">>>>>>>>")
    print(request.environ)
    print(">>>>>>>>")
    response = jsonify({'Hello': 'World!'})
    print(">>>>>>>>")
    print(response)
    print(">>>>>>>>")
    print(response.data)
    return response

#if __name__ == '__main__':
#    app.run(host="51.158.147.66",port=7778)
