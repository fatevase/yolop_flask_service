from flask import Flask, jsonify, request


app = Flask(__name__)

@app.route("/", methods=['GET'])
def homePage():
    home = 200
    return jsonifyResult(home, "index")



@app.route("/get", methods=['GET'])
def getArticles():
    test = -1
    if request.args is not None:
        test = request.args.get('test')
    return jsonifyResult(test, "hello word")

@app.route("/post", methods=['POST'])
def postArticles():
    post_type = request.json['handle_type']
    post_input = request.json['raw_data']
    post_info = request.json['info']
    
    result = {"info":info}
    return jsonifyResult(data=result)

def jsonifyResult(code=0, data=[]):
    return jsonify({"code":code, "data":data})   

if __name__ == "__main__":
    app.run(debug=True)
    