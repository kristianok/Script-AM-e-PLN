from collections import Counter

import spacy
import pandas
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from sklearn.neighbors import KNeighborsClassifier


def treinar_testar(model, data, test, y_cols='Classificação'):
    correct = []
    predicted = []

    x = data.loc[:, data.columns != y_cols]
    y = data.loc[:, data.columns == y_cols]
    xTest = test.loc[:, data.columns != y_cols]
    yTest = test.loc[:, data.columns == y_cols]

    # print(x)
    # print('------')
    y = y.astype('int')
    yTest = yTest.astype('int')
    # print(y)
    # print(np.array(y).ravel())
    # print('---Test---')
    # print(xTest)
    model.fit(x, np.array(y).ravel())
    # print('------')
    predicted = model.predict(xTest)
    return np.array(yTest).ravel(), predicted


if __name__ == '__main__':
    df = pandas.read_excel(r"C:\Users\Kristiano\Downloads\DadosBotsInstagram.xlsx")
    binario = {'Sim': True, 'Não': False, 'Bot': True, 'Real': False}
    df = df.drop('Comentário', axis=1)
    with pandas.option_context("future.no_silent_downcasting", True):
        df = df.replace(binario)
    # with pandas.option_context('display.max_rows', None, 'display.max_columns', None):print(df)

    data, test = train_test_split(df, test_size=0.2)

    corretos, preditos = treinar_testar(RandomForestClassifier(), data, test)

    print(corretos)
    print(preditos)

    print("\nRandomForestClassifier")
    print("\nAcurácia: ")
    print(accuracy_score(corretos, preditos))
    print("\nRecall: ")
    print(recall_score(corretos, preditos))
    print("\nF1: ")
    print(f1_score(corretos, preditos))
    print("\nPrecisão: ")
    print(precision_score(corretos, preditos))

    corretos, preditos = treinar_testar(KNeighborsClassifier(), data, test)

    print(corretos)
    print(preditos)

    print("\nKNeighborsClassifier")
    print("\nAcurácia: ")
    print(accuracy_score(corretos, preditos))
    print("\nRecall: ")
    print(recall_score(corretos, preditos))
    print("\nF1: ")
    print(f1_score(corretos, preditos))
    print("\nPrecisão: ")
    print(precision_score(corretos, preditos))

    comentarios = pandas.read_excel(r"C:\Users\Kristiano\Downloads\ComentariosTratadosSpacy.xlsx")
    comentariosReais = comentarios[comentarios["Classificação"] == "Real"]
    comentariosBots = comentarios[comentarios["Classificação"] == "Bot"]
    comentarios = np.array(comentariosReais["Comentário"]).ravel()
    comentariosReais = np.array(comentariosReais["Comentário"]).ravel()
    comentariosBots = np.array(comentariosBots["Comentário"]).ravel()
    qtdReais = comentariosReais.shape[0]
    qtdBots = comentariosBots.shape[0]
    nlp = spacy.load("pt_core_news_sm")

    morfologiaBots = None

    for comentario in comentariosBots:
        doc = nlp(comentario)
        for sent in doc.sents:
            for token in sent:
                c = Counter(([token.pos_ for token in sent for sent in doc.sents]))
        if morfologiaBots is None:
            morfologiaBots = c
        morfologiaBots += c

    print(morfologiaBots)

    morfologiaUsuariosReais = None

    for comentario in comentariosReais:
        doc = nlp(comentario)
        for sent in doc.sents:
            for token in sent:
                c = Counter(([token.pos_ for token in sent for sent in doc.sents]))
        if morfologiaUsuariosReais is None:
            morfologiaUsuariosReais = c
        morfologiaUsuariosReais += c

    print(morfologiaUsuariosReais)
    totalMorfologia = sum(morfologiaUsuariosReais.values())
    for item, count in morfologiaUsuariosReais.items():
        morfologiaUsuariosReais[item] /= totalMorfologia
        morfologiaUsuariosReais[item] *= 100

    print(morfologiaUsuariosReais)

    totalMorfologia = sum(morfologiaBots.values())
    for item, count in morfologiaBots.items():
        morfologiaBots[item] /= totalMorfologia
        morfologiaBots[item] *= 100

    print(morfologiaBots)
    # for comentario in comentarios:
    # doc = nlp(comentario)
    # print(doc.text)
    # for token in doc:
    # print(token.text, token.pos_, token.dep_)
