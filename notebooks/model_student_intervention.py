import pandas as pd 
import pickle
from sklearn.metrics import f1_score
filename = '../model/student_intervention.sav'

def target_feature_split(df):
    # Extraia as colunas dos atributo
    feature_cols = list(df.columns[:-2])
    # Extraia a coluna-alvo 'passed'
    target_col = df.columns[-2:]
    # Separe os dados em atributos e variáveis-alvo (X_all e y_all, respectivamente)
    X_all = df[feature_cols]
    y_all = df[target_col]
    return X_all, y_all

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


def load_model(path_model=filename):
    model = pickle.load(open(path_model, 'rb'))
    return model

def load_dataset(path_dataset):
    df = pd.read_csv(path_dataset)
    features, target = target_feature_split(df)
    features = preprocess_features(features)
    return features, target

def inference(model, features, target):
    pred = model.predict(features)
    base_pred = target.join(pd.DataFrame({"predictions": pred}))
    return base_pred

def metrics(model, features, target):
    target.drop(columns=['cpf_student'], inplace=True)
    score = model.score(features, target)
    return score