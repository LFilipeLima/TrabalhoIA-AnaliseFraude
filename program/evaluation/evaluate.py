from itertools import groupby
from math import fsum, isfinite


class Evaluate:

    def __init__(self, y_true, probabilidades, valores, comissao=0.03):
        self.y_true = tuple(y_true)
        self.probabilidades = tuple(probabilidades)
        self.valores = tuple(valores)
        self.comissao = comissao
        self._validar_entradas()

    def _validar_entradas(self):
        if not (len(self.y_true) == len(self.probabilidades) == len(self.valores)):
            raise ValueError("As entradas devem ter o mesmo tamanho.")
        if not self.y_true:
            raise ValueError("A validação deve conter pelo menos uma transação.")
        if any(rotulo not in (0, 1) for rotulo in self.y_true):
            raise ValueError("Os rótulos devem ser 0 ou 1.")
        if any(not isfinite(p) or not 0 <= p <= 1 for p in self.probabilidades):
            raise ValueError("As probabilidades devem estar entre 0 e 1.")
        if any(not isfinite(valor) or valor < 0 for valor in self.valores):
            raise ValueError("Os valores devem ser finitos e não negativos.")
        if not isfinite(self.comissao) or not 0 <= self.comissao <= 1:
            raise ValueError("A comissão deve estar entre 0 e 1.")

    @staticmethod
    def _validar_corte(corte):
        # -inf representa recusar tudo, mesmo com probabilidades iguais a zero.
        if corte != float("-inf") and (not isfinite(corte) or not 0 <= corte <= 1):
            raise ValueError("O corte deve estar entre 0 e 1 ou ser -inf.")

    def calcular_lucro(self, corte):

        self._validar_corte(corte)

        lucro = 0.0

        for fraude, probabilidade, valor in zip(
            self.y_true, self.probabilidades, self.valores
        ):
            if probabilidade <= corte:
                if fraude == 1:
                    lucro -= valor
                else:
                    lucro += valor * self.comissao

        return lucro

    def calcular_metricas(self, corte):

        self._validar_corte(corte)

        fraudes_previstas = [
            probabilidade > corte
            for probabilidade in self.probabilidades
        ]

        TP = FP = FN = TN = 0

        for r_fraude, p_fraude in zip(self.y_true, fraudes_previstas):
            if r_fraude == 1 and p_fraude == True:
                TP += 1
            if r_fraude == 0 and p_fraude == True:
                FP += 1
            if r_fraude == 1 and p_fraude == False:
                FN += 1
            if r_fraude == 0 and p_fraude == False:
                TN += 1

        precisao = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * precisao * recall / (precisao + recall) if (precisao + recall) > 0 else 0

        return {"precisao": precisao, "recall": recall, "F1": f1, "matriz_confusao": [[TN, FP], [FN, TP]]}

    def buscar_corte(self):

        # Cada compra aprovada acrescenta uma comissão ou um prejuízo ao lucro.
        transacoes = sorted(
            zip(self.probabilidades, self.y_true, self.valores),
            key=lambda transacao: transacao[0],
        )

        melhor_corte = float("-inf")
        melhor_lucro = 0.0
        lucro_acumulado = 0.0

        for corte, grupo in groupby(transacoes, key=lambda transacao: transacao[0]):
            lucro_grupo = fsum(
                -valor if fraude == 1 else valor * self.comissao
                for _, fraude, valor in grupo
            )
            lucro_acumulado = fsum((lucro_acumulado, lucro_grupo))

            if lucro_acumulado > melhor_lucro:
                melhor_corte = corte
                melhor_lucro = lucro_acumulado

        return melhor_corte, melhor_lucro
