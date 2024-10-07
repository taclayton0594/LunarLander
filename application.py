from flask import Flask,request,render_template
from src.components.model_trainer import RLModelTrainer

application = Flask(__name__)

app = application

## Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

#@app.route('/LunarRover',methods=['GET','POST'])
#def run_lunar_rover():


if __name__=="__main__":
    #app.run(host='0.0.0.0', port=8080)      

    LunarLanderMdl = RLModelTrainer()
    LunarLanderMdl.start_RL_training()
