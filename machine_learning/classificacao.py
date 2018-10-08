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
marcacoes_teste = [-1, -1, 1]
teste = [d1, d2, d3]
#tentando advinhar a classificacao dos elementos desconhecidos
resultado = multCP.predict(teste)
print(resultado)

#calculando a taxa de acerto
#calculando as diferencas
diferencas = resultado - marcacoes_teste
#contando os acertos
acertos = [d for d in diferencas if d == 0]
#calculando a taxa de acertos
taxa_de_acerto = 100.0*(len(acertos)/len(resultado))
                        
print(resultado)
print(diferencas)
#print(acertos)
print(taxa_de_acerto)
