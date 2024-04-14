from collections import Counter

import spacy
import pandas
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score, precision_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
from collections import OrderedDict
import nltk

nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('portuguese')


def aprendizadoMaquina(data, test, modelo):
    corretos, preditos = treinar_testar(modelo, data, test)

    print(corretos)
    print(preditos)

    print("\n")
    print(modelo)
    print("\nAcurácia: ")
    print(accuracy_score(corretos, preditos))
    print("\nRecall: ")
    print(recall_score(corretos, preditos))
    print("\nF1: ")
    print(f1_score(corretos, preditos))
    print("\nPrecisão: ")
    print(precision_score(corretos, preditos))


def plotPalavras(quantidade, palavras):
    names = list(palavras.keys())
    values = list(palavras.values())
    names = names[:quantidade]
    values = values[:quantidade]

    plt.barh(range(len(values)), values, tick_label=names)
    plt.gca().invert_yaxis()
    plt.show()


def contarPalavras(comentarios):
    dic = {}
    for comentario in comentarios:
        words = comentario.split()
        for raw_word in words:
            word = raw_word.lower()
            if word in stopwords or word == "?" or word == "!" or word == "." or word == ",":
                continue
            if word in dic:
                dic[word] += 1
            else:
                dic[word] = 1

    return dict(sorted(dic.items(), key=lambda item: item[1], reverse=True))


def traduzirTags(lista):
    i = 0
    for r in lista:
        if r[0] == "NOUN":
            lista[i] = ("SUBSTANTIVO", r[1])
        elif r[0] == "PUNCT":
            lista[i] = ("PONTUAÇÃO", r[1])
        elif r[0] == "ADJ":
            lista[i] = ("ADJETIVO", r[1])
        elif r[0] == "ADP":
            lista[i] = ("SUBSTANTIVO", r[1])
        elif r[0] == "VERB":
            lista[i] = ("VERBO", r[1])
        i += 1


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

    data, test = train_test_split(df, test_size=0.2, random_state=0)

    print("Modelo 1")

    aprendizadoMaquina(data, test, RandomForestClassifier())
    aprendizadoMaquina(data, test, KNeighborsClassifier())

    df = df.drop("Quantidade de caractéres na \"bio\"", axis=1)
    df = df.drop("Possui foto de perfil?", axis=1)
    df = df.drop("Quantidade de dígitos numéricos no nome de usuário", axis=1)
    df = df.drop("Quantidade de curtidas no comentário", axis=1)

    data, test = train_test_split(df, test_size=0.2, random_state=0)

    print("Modelo 2")

    aprendizadoMaquina(data, test, RandomForestClassifier())
    aprendizadoMaquina(data, test, KNeighborsClassifier())

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

    listaReais = list(morfologiaUsuariosReais.items())
    listaBots = list(morfologiaBots.items())
    listaReais = [r for r in listaReais if
                  r[0] != "DET" and r[0] != "ADV" and r[0] != "PRON" and r[0] != "PROPN" and r[0] != "ADP"]
    listaBots = [b for b in listaBots if
                 b[0] != "DET" and b[0] != "ADV" and b[0] != "PRON" and b[0] != "PROPN" and b[0] != "ADP"]

    traduzirTags(listaReais)
    traduzirTags(listaBots)

    percentualMorfologiaBots = np.array(listaBots)
    percentualMorfologiaReais = np.array(listaReais)

    for percentual in percentualMorfologiaBots:
        if float(percentual[1]) > 5:
            plt.scatter(percentual[0], float(percentual[1]), color="red", label="bot")

    for percentual in percentualMorfologiaReais:
        if float(percentual[1]) > 5:
            plt.scatter(percentual[0], float(percentual[1]), color="blue", label="real")

    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = OrderedDict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys())
    plt.xlabel("Classes")
    plt.ylabel("Percentual")
    plt.show()

    print('palavras bots')
    palavrasBots = contarPalavras(comentariosBots)
    print('palavras reais')
    palavrasReais = contarPalavras(comentariosReais)

    plt.rcParams['font.size'] = 7
    plotPalavras(15, palavrasBots)
    plotPalavras(15, palavrasReais)
