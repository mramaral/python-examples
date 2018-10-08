from sklearn.naive_bayes import MultinomialNB
# [gordinho, perna curta, au au]
porco1 = [1, 1, 0]
porco2 = [1, 1, 0]
porco3 = [1, 1, 0]
cachorro4 = [1, 1, 1]
cachorro5 = [0, 1, 1]
cachorro6 = [0, 1, 1]

dados = [porco1, porco2, porco3, cachorro4, cachorro5, cachorro6]
#1=porco -1=cachorro
marcacoes = [1, 1, 1, -1, -1, -1]

#modelo para cachorrinhos ou porquinhos
multCP = MultinomialNB()
#treinando o modelo com os dados acima
multCP.fit(dados, marcacoes)

#elementos desconhecidos
d1 = [1, 1, 1]
d2 = [0, 0, 0]
d3 = [0, 1, 0]

teste = [d1, d2, d3]

resultado = multCP.predict(teste)
print(resultado)
