
# coding: utf-8

# In[1]:


# Importar bibliotecas
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import f1_score

# Ler os dados dos estudantes
student_data = pd.read_csv("student-data.csv")
print "Os dados dos estudantes foram lidos com êxito!"


# In[2]:


students_passed = student_data[student_data['passed'] == "yes"].shape[0]
students_not_passed = student_data[student_data['passed'] == "no"].shape[0]

total = students_passed + students_not_passed

print "Total calculado " + str(total) + " Total esperado " + str(student_data.shape[0])


# In[3]:


taxa_graduacao = float(students_passed) / total
print str(round(taxa_graduacao*100, 3)) + " %"


# In[4]:


# TODO: Calcule o número de estudante
n_students = student_data.shape[0]

# TODO: Calcule o número de atributos
# A últimoa coluna é a classe
n_features = student_data.shape[1] - 1 

# TODO: Calcule o número de alunos aprovados
n_passed = students_passed

# TODO: Calcule o número de alunos reprovados
n_failed = students_not_passed

# TODO: Calcule a taxa de graduação
grad_rate = taxa_graduacao * 100

# Imprima os resultados
print "Número total de estudantes: {}".format(n_students)
print "Número de atributos: {}".format(n_features)
print "Número de estudantes aprovados: {}".format(n_passed)
print "Número de estudantes reprovados: {}".format(n_failed)
print "Taxa de graduação: {:.2f}%".format(grad_rate)


# In[7]:


len(student_data.columns)
target_col = (student_data.columns[-1])
#student_data[target_col]
len(student_data.columns)
target_col = (student_data.columns[30:31])
#student_data[target_col]


# In[8]:


# Extraia as colunas dos atributos
feature_cols = list(student_data.columns[:-1])

# Extraia a coluna-alvo 'passed'
#Desta forma o nome da coluna é ignorado
#target_col = student_data.columns[-1] 
target_col = student_data.columns[30:31] 

# Mostre a lista de colunas
print "Colunas de atributos:\n{}".format(feature_cols)
print "\nColuna-alvo: {}".format(target_col)

# Separe os dados em atributos e variáveis-alvo (X_all e y_all, respectivamente)
X_all = student_data[feature_cols]
y_all = student_data[target_col]

# Mostre os atributos imprimindo as cinco primeiras linhas
print "\nFeature values:"
print X_all.head()


# In[9]:


def preprocess_features(X):
    ''' Pré-processa os dados dos estudantes e converte as variáveis binárias não numéricas em
        variáveis binárias (0/1). Converte variáveis categóricas em variáveis postiças. '''
    
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
print "Processed feature columns ({} total features):\n{}".format(len(X_all.columns), list(X_all.columns))

y_all = preprocess_features(y_all)
print "Processed feature columns ({} total features):\n{}".format(len(y_all.columns), list(y_all.columns))


# In[10]:


# TODO: Importe qualquer funcionalidade adicional de que você possa precisar aqui
from sklearn.model_selection import train_test_split

# TODO: Estabeleça o número de pontos de treinamento
num_train = 300

# Estabeleça o número de pontos de teste
num_test = X_all.shape[0] - num_train

# TODO: Emabaralhe e distribua o conjunto de dados de acordo com o número de pontos de treinamento e teste abaixo
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test, random_state=42, stratify=y_all)

# Mostre o resultado da distribuição
print "O conjunto de treinamento tem {} amostras.".format(X_train.shape[0])
print "O conjunto de teste tem {} amostras.".format(X_test.shape[0])


# In[11]:


def train_classifier(clf, X_train, y_train):
    ''' Ajusta um classificador para os dados de treinamento. '''
    
    # Inicia o relógio, treina o classificador e, então, para o relógio
    start = time()
    clf.fit(X_train, y_train)
    end = time()
    
    # Imprime os resultados
    print "O modelo foi treinado em {:.4f} segundos".format(end - start)

    
def predict_labels(clf, features, target):
    ''' Faz uma estimativa utilizando um classificador ajustado baseado na pontuação F1. '''
    
    # Inicia o relógio, faz estimativas e, então, o relógio para
    start = time()
    y_pred = clf.predict(features)
    end = time()
    
    # Imprime os resultados de retorno
    print "As previsões foram feitas em {:.4f} segundos.".format(end - start)
    return f1_score(target.values, y_pred, pos_label=1)

def predictions(clf, features):
    ''' Faz uma estimativa utilizando um classificador ajustado baseado na pontuação F1. '''
    
    # Inicia o relógio, faz estimativas e, então, o relógio para
    start = time()
    y_pred = clf.predict(features)
    end = time()
    
    # Imprime os resultados de retorno
    print "As previsões foram feitas em {:.4f} segundos.".format(end - start)
    return y_pred


def train_predict(clf, X_train, y_train, X_test, y_test):
    ''' Treina e faz estimativas utilizando um classificador baseado na pontuação do F1. '''
    
    # Indica o tamanho do classificador e do conjunto de treinamento
    print "Treinando um {} com {} pontos de treinamento. . .".format(clf.__class__.__name__, len(X_train))
    
    # Treina o classificador
    train_classifier(clf, X_train, y_train.values.ravel())
    
    # Imprime os resultados das estimativas de ambos treinamento e teste
    print "Pontuação F1 para o conjunto de treino: {:.4f}.".format(predict_labels(clf, X_train, y_train))
    print "Pontuação F1 para o conjunto de teste: {:.4f}.".format(predict_labels(clf, X_test, y_test))


# In[12]:


# TODO: Importe os três modelos de aprendizagem supervisionada do sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier

# TODO: Inicialize os três modelos
clf_A = GaussianNB()
clf_B = svm.SVC(random_state=0)
clf_C = RandomForestClassifier(random_state=0)

# TODO: Configure os tamanho dos conjuntos de treinamento
for clf in [clf_A, clf_B, clf_C]:
    print "Treinando " + str(clf) + "\n"
    for n_train in [100, 200, 300]:
        train_predict(clf, X_train[:n_train], y_train[:n_train], X_test, y_test)
        print "\n"
    print "\n\n"


# In[ ]:


# TODO: Importe 'GridSearchCV' e 'make_scorer'
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.model_selection import ShuffleSplit
from scipy.stats import expon 

# TODO: Crie a lista de parâmetros que você gostaria de calibrar
parameters = [
    {'kernel': ['linear'], 'C': [1, 10, 100]}]

# TODO: Inicialize o classificador
clf = svm.SVC(random_state=0)

# TODO: Faça uma função de pontuação f1 utilizando 'make_scorer' 
f1_scorer = make_scorer(f1_score, pos_label=1)

# TODO: Execute uma busca em matriz no classificador utilizando o f1_scorer como método de pontuação
cv_sets = ShuffleSplit(X_train.shape[0], test_size = 0.20, random_state = 40)
grid_obj = GridSearchCV(clf, parameters, scoring=f1_scorer, cv=cv_sets)

# TODO: Ajuste o objeto de busca em matriz para o treinamento de dados e encontre os parâmetros ótimos
grid_obj =  grid_obj.fit(X_train, y_train.values.ravel())

# Get the estimator
clf = grid_obj.best_estimator_

# Best parameters
print "Melhores parâmetros encontrados"
print grid_obj.best_params_

# Reporte a pontuação final F1 para treinamento e teste depois de calibrar os parâmetrosprint "Tuned model has a training F1 score of {:.4f}.".format(predict_labels(clf, X_train, y_train))
print "O modelo calibrado tem F1 de {:.4f} no conjunto de treinamento.".format(predict_labels(clf, X_train, y_train))
print "O modelo calibrado tem F1 de {:.4f} no conjunto de teste.".format(predict_labels(clf, X_test, y_test))

