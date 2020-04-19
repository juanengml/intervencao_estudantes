from flask import Flask
from flask_restful  import Resource, Api,reqparse
import pickle
import pandas as pd
from console_logging.console import Console

console = Console()

app = Flask(__name__)
api = Api(app)


def load_model():
 console.log("[*] LOADING MODEL")
 filename = 'student_intervention.sav'
 modelo = pickle.load(open(filename, 'rb'))
 console.log("[*] MODEL LOADED ")
 return modelo

modelo = load_model()

class evaluate(Resource):
   def post(self):
        console.log("[*] CARREGANDO ARGUMENTOS ")
        argumentos = reqparse.RequestParser()
        argumentos.add_argument('school_GP',type=int)
        argumentos.add_argument('school_MS',type=int)
        argumentos.add_argument('sex_F',type=int)
        argumentos.add_argument('sex_M',type=int)
        argumentos.add_argument('age',type=int)
        argumentos.add_argument('address_R',type=int)
        argumentos.add_argument('address_U',type=int)
        argumentos.add_argument('famsize_GT3',type=int)
        argumentos.add_argument('famsize_LE3',type=int)
        argumentos.add_argument('Pstatus_A',type=int)
        argumentos.add_argument('Pstatus_T',type=int)
        argumentos.add_argument('Medu',type=int)
        argumentos.add_argument('Fedu',type=int)
        argumentos.add_argument('Mjob_at_home',type=int)
        argumentos.add_argument('Mjob_health',type=int)
        argumentos.add_argument('Mjob_other',type=int)
        argumentos.add_argument('Mjob_services',type=int)
        argumentos.add_argument('Mjob_teacher',type=int)
        argumentos.add_argument('Fjob_at_home',type=int)
        argumentos.add_argument('Fjob_health',type=int)
        argumentos.add_argument('Fjob_other',type=int)
        argumentos.add_argument('Fjob_services',type=int)
        argumentos.add_argument('Fjob_teacher',type=int)
        argumentos.add_argument('reason_course',type=int)
        argumentos.add_argument('reason_home',type=int)
        argumentos.add_argument('reason_other',type=int)
        argumentos.add_argument('reason_reputation',type=int)
        argumentos.add_argument('guardian_father',type=int)
        argumentos.add_argument('guardian_mother',type=int)
        argumentos.add_argument('guardian_other',type=int)
        argumentos.add_argument('traveltime',type=int)
        argumentos.add_argument('studytime',type=int)
        argumentos.add_argument('failures',type=int)
        argumentos.add_argument('schoolsup',type=int)
        argumentos.add_argument('famsup',type=int)
        argumentos.add_argument('paid',type=int)
        argumentos.add_argument('activities',type=int)
        argumentos.add_argument('nursery',type=int)
        argumentos.add_argument('higher',type=int)
        argumentos.add_argument('internet',type=int)
        argumentos.add_argument('romantic',type=int)
        argumentos.add_argument('famrel',type=int)
        argumentos.add_argument('freetime',type=int)
        argumentos.add_argument('goout',type=int)
        argumentos.add_argument('Dalc',type=int)
        argumentos.add_argument('Walc',type=int)
        argumentos.add_argument('health',type=int)
        argumentos.add_argument('absences',type=int)
        dados  = argumentos.parse_args()
        result = [
               "aluno precisa de intervensão"
               if modelo.predict(pd.DataFrame([dados]))[0] == "yes"
               else "aluno não precisa de intervensao"
        ]
        console.log("[*] PREDICT MODELO LOADED  ")

        return{'mensage':result[0]},200

class About(Resource):
   def get(self):
    return {"mensage":"Api desenvolvida pela Angah 2020 - 12 - abril UPDATE "},200

api.add_resource(evaluate, '/angah/api/v1/')
api.add_resource(About, '/')

if __name__ == '__main__':
    console.log("[*] SERVING STARED API MODEL STUDENT INTERVENTION ")
    app.run(host='0.0.0.0',threaded=False,debug=True)
