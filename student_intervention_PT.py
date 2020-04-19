#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import f1_score
from console_logging.console import Console
import numpy as np

console = Console()

student_data = pd.read_csv("bases/student-data.csv")
console.log("[*] LOADING BASE STUDANTE_DATA_CSV")
print("Os dados dos estudantes foram lidos com êxito!")
n_students = student_data.shape[0]
n_features = student_data.shape[1] - 1
n_passed = student_data[student_data['passed']=='yes'].shape[0]
n_failed = student_data[student_data['passed']=='no'].shape[0]
passou = float(n_passed * 1.00)
allstudantes = float(n_students * 1.00)
grad_rate = ( passou / allstudantes ) * 100
print("Número total de estudantes: {}".format(n_students))
print("Número de atributos: {}".format(n_features))
print("Número de estudantes aprovados: {}".format(n_passed))
print("Número de estudantes reprovados: {}".format(n_failed))
print("Taxa de graduação: {:.2f}%".format(grad_rate))


feature_cols = list(student_data.columns[:-1])
console.log("[*] CARACTERISTICAS COLUNAS")
target_col = student_data.columns[-1]
console.log("[*] TARGETS COLUNAS")

print("Colunas de atributos:\n{}".format(feature_cols))
print("\nColuna-alvo: {}".format(target_col))

X_all = student_data[feature_cols]
console.log("[*] X TARGET ALL")
y_all = student_data[target_col]
console.log("[*] Y TARGET ALL")
print("\nFeature values:")
print(X_all.head())

def preprocess_features(X):
    ''' Pré-processa os dados dos estudantes e converte as variáveis binárias não numéricas em
        variáveis binárias (0/1). Converte variáveis categóricas em variáveis postiças. '''
    console.log("[*] X FUNCTION PREPROCESS - FEATURES")
    # Inicialize nova saída DataFrame
    output = pd.DataFrame(index = X.index)

    # Observe os dados em cada coluna de atributos
    for col, col_data in X.iteritems():

        # Se o tipo de dado for não numérico, substitua todos os valores yes/no por 1/0
        if col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0])

        # Se o tipo de dado for categórico, converta-o para uma variável dummy
        if col_data.dtype == object:
            # Example: 'school' => 'school_GP' and 'school_MS'
            col_data = pd.get_dummies(col_data, prefix = col)

        # Reúna as colunas revisadas
        output = output.join(col_data)

    return output

X_all = preprocess_features(X_all)
print("Processed feature columns ({} total features):\n{}".format(len(X_all.columns), list(X_all.columns)))
console.log("[*] Processed feature columns ({} total features):\n{}".format(len(X_all.columns), list(X_all.columns)))
from sklearn.model_selection import train_test_split
num_train = 300 # 75% dos dados

num_test = X_all.shape[0] - num_train

X_train, X_test, y_train, y_test = train_test_split(X_all,y_all,test_size=num_test,train_size = 0.75, random_state=42)
console.log("[*] split train and test")
print("O conjunto de treinamento tem {} amostras.".format(X_train.shape[0]))
print("O conjunto de teste tem {} amostras.".format(X_test.shape[0]))



def train_classifier(clf, X_train, y_train):
    ''' Ajusta um classificador para os dados de treinamento. '''
    console.log("[*] TREINANDO CLASSIFICADOR ")
    # Inicia o relógio, treina o classificador e, então, para o relógio
    start = time()
    clf.fit(X_train, y_train)
    end = time()

    # Imprime os resultados
    print("O modelo foi treinado em {:.4f} segundos".format(end - start))
    console.log("[*] O modelo foi treinado em {:.4f} segundos".format(end - start))


def predict_labels(clf, features, target):
    ''' Faz uma estimativa utilizando um classificador ajustado baseado na pontuação F1. '''

    # Inicia o relógio, faz estimativas e, então, o relógio para
    start = time()
    y_pred = clf.predict(features)
    end = time()

    # Imprime os resultados de retorno
    print("[*] As previsões foram feitas em {:.4f} segundos.".format(end - start))
    console.log("[*] As previsões foram feitas em {:.4f} segundos.".format(end - start))
    return f1_score(target.values, y_pred, pos_label="yes")


def train_predict(clf, X_train, y_train, X_test, y_test):
    ''' Treina e faz estimativas utilizando um classificador baseado na pontuação do F1. '''

    # Indica o tamanho do classificador e do conjunto de treinamento
    print("Treinando um {} com {} pontos de treinamento. . .".format(clf.__class__.__name__, len(X_train)))
    console.log("[*] Treinando um {} com {} pontos de treinamento. . .".format(clf.__class__.__name__, len(X_train)))
    # Treina o classificador
    train_classifier(clf, X_train, y_train)

    # Imprime os resultados das estimativas de ambos treinamento e teste
    print("Pontuação F1 para o conjunto de treino: {:.4f}.".format(predict_labels(clf, X_train, y_train)))
    print("Pontuação F1 para o conjunto de teste: {:.4f}.".format(predict_labels(clf, X_test, y_test)))

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

classificador_A = GaussianNB()
console.log("[X] CLASSIFICADOR GaussianNB ")
classificador_B = SVC(random_state=0)
console.log("[X] CLASSIFICADOR SVM ")
classificador_C = LogisticRegression(random_state=0,n_jobs=-1)
console.log("[X] CLASSIFICADOR LogisticRegression ")
for classificador in [classificador_A, classificador_B, classificador_C]:
    print("\tTreinando " + str(classificador) + "\n\n")
    console.log("[*] \tTRAINING " + str(classificador) + "\n\n")
    for n_train in [100, 200, 300]:
        train_predict(classificador, X_train[:n_train], y_train[:n_train], X_test, y_test)
        print ("\n")
    for n_train in [200]:
        train_predict(classificador, X_train[:n_train], y_train[:n_train], X_test, y_test)
        print ("\n")
    for n_train in [300]:
        train_predict(classificador, X_train[:n_train], y_train[:n_train], X_test, y_test)
        print ("\n")
    print ("\n\n")

from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.model_selection import ShuffleSplit
console.log("[*] LOAD LIBS MODEL SELECTION - GRID GridSearchCV")
console.log("[*] LOAD LIBS MODEL SELECTION - ShuffleSplit")
console.log("[*] LOAD LIBS METRICS F1 SCORE ")

parameters = [
    {'kernel': ['linear'], 'C': [1, 10, 100]}]

classificador =  SVC(random_state=0)
f1_scorer = make_scorer(f1_score, pos_label="yes")
console.log("[*] FI_score...")
grid_obj = GridSearchCV(classificador, parameters, scoring=f1_scorer)
console.log("[*] GRID SEARCH")
grid_obj =  grid_obj.fit(X_train, y_train)
classificador = grid_obj.best_estimator_

print("O modelo calibrado tem F1 de {:.4f} no conjunto de treinamento.".format(predict_labels(classificador, X_train, y_train)))
print("O modelo calibrado tem F1 de {:.4f} no conjunto de teste.".format(predict_labels(classificador, X_test, y_test)))

import pickle

predict_labels(classificador, X_test, y_test)
console.log("[*] SAVING MODEL ")

filename = 'student_intervention.sav'
pickle.dump(classificador, open(filename, 'wb'))
console.log("[*] SAVED MODEL ")
console.log("[*] DONE !!! TRAINING MODEL !!!")
