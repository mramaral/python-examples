import csv
def carregar_acessos(arquivo):
    X = []
    Y = []
    
    arquivo = open(arquivo, "r")
    leitor = csv.reader(arquivo)
    next(leitor)

    for acessou_home, acessou_como_funciona, acessou_contato, comprou in leitor:
        X.append([int(acessou_home), int(acessou_como_funciona), int(acessou_contato)])
        Y.append(int(comprou))

    return X, Y
