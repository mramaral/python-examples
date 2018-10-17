import pandas as pd
df = pd.read_csv('busca.csv')
X_df = df[['home', 'busca', 'logado',]]
Y_df = df['comprou']
Xdummies_df = pd.get_dummies(X_df)
Ydummies_df = Y_df
X = Xdummies_df.values
Y = Ydummies_df.values

porcentagem_treino = 0.9
tamanho_de_treino = int(porcentagem_treino*len(Y))
tamanho_de_teste = int(len(Y)-tamanho_de_treino)

treino_dados = X[:tamanho_de_treino]
treino_marcacoes = Y[:tamanho_de_treino]

teste_dados = X[-tamanho_de_teste:]
teste_marcacoes = Y[-tamanho_de_teste:]

from sklearn.naive_bayes import MultinomialNB
modelo = MultinomialNB()
modelo.fit(treino_dados, treino_marcacoes)

resultado = modelo.predict(teste_dados)

diferencas = resultado - teste_marcacoes

acertos = [d for d in diferencas if d == 0]
total_de_acertos = len(acertos)
total_de_elementos = len(teste_dados)

taxa_de_acerto = 100.0 * total_de_acertos / total_de_elementos

print('Total de elementos: %f' % total_de_elementos)
print('Taxa de acerto: %f' % taxa_de_acerto)

acerto_de_um = sum(Y)
acerto_de_zero = len(Y)-acerto_de_um
taxa_de_acerto_base = (100.0*max(acerto_de_um, acerto_de_zero))/len(Y)
print('Taxa de acerto Base: %f' % taxa_de_acerto_base)
