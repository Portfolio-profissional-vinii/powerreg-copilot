"""
tests/evaluation/eval_dataset_v2.py
Dataset de avaliação expandido para o Utilities Copilot.

Cobertura:
  - 25 REGULATÓRIO  (diretas, semânticas, paráfrases, informais, siglas, multi-chunk, ambíguas)
  - 20 OPERACIONAL  (AVG, COUNT, MAX, MIN, GROUP BY, ORDER BY, WHERE, DISTINCT, temporal, comparação)
  - 10 FRONTEIRA    (casos ambíguos entre REGULATORIO e OPERACIONAL)
  - 10 SEM_RESPOSTA (abstention — informação inexistente no corpus)
  - 5  PARÁFRASES   (grupos de perguntas semanticamente equivalentes)

IMPORTANTE: Este arquivo é SOMENTE LEITURA para a aplicação.
Coleção de ground truth baseada exclusivamente no corpus existente.

Módulos disponíveis:
  Módulo 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, REN 1000/2021

Indicadores operacionais confirmados na base:
  DEC, DECINC, DECIND, DECINE, DECINO, DECIP, DECIPC, DECXN, DECXNC, DECXP, DECXPC,
  FEC, FECINC, FECIND, FECINE, FECINO, FECIP, FECIPC, FECXN, FECXNC, FECXP, FECXPC, NumCon

Distribuidoras confirmadas: EAC, CEA, ETO (e dezenas de outras)
Anos confirmados: 2020–2026
"""

# ---------------------------------------------------------------------------
# ESTRUTURA DE CADA ENTRADA
# ---------------------------------------------------------------------------
# {
#   "id":                str,           # identificador único
#   "pergunta":          str,           # pergunta do usuário
#   "categoria_esperada": str,          # "REGULATORIO" | "OPERACIONAL" | "SEM_RESPOSTA"
#   "grupo_parafrases":  str | None,    # id do grupo de paráfrases, se aplicável
#   # Campos opcionais por categoria:
#   "modulo_esperado":   str | None,    # ex: "Módulo 8" — para REGULATORIO
#   "pagina_esperada":   int | None,    # página aproximada — opcional
#   "sql_esperado_contem": list[str] | None,  # tokens esperados no SQL — para OPERACIONAL
#   "resultado_esperado":  any | None,  # valor numérico esperado — para OPERACIONAL
#   "notas":             str | None,    # observações para o avaliador
# }
# ---------------------------------------------------------------------------

EVAL_SET_V2 = [

    # =========================================================================
    # REGULATÓRIO — 25 perguntas
    # =========================================================================

    # --- Módulo 8: Qualidade do Fornecimento / DIC / DEC ---
    {
        "id": "REG-01",
        "pergunta": "Como é calculado o indicador DIC?",
        "categoria_esperada": "REGULATORIO",
        "grupo_parafrases": "PARAF-DIC",
        "modulo_esperado": "Módulo 8",
        "pagina_esperada": None,
        "sql_esperado_contem": None,
        "resultado_esperado": None,
        "notas": "DIC = Duração de Interrupção Individual por Unidade Consumidora. Módulo 8, Seção 8.2.",
    },
    {
        "id": "REG-02",
        "pergunta": "Como se determina o DIC de uma unidade consumidora?",
        "categoria_esperada": "REGULATORIO",
        "grupo_parafrases": "PARAF-DIC",
        "modulo_esperado": "Módulo 8",
        "pagina_esperada": None,
        "sql_esperado_contem": None,
        "resultado_esperado": None,
        "notas": "Paráfrase de REG-01.",
    },
    {
        "id": "REG-03",
        "pergunta": "Qual é a fórmula de cálculo do DIC segundo o PRODIST?",
        "categoria_esperada": "REGULATORIO",
        "grupo_parafrases": "PARAF-DIC",
        "modulo_esperado": "Módulo 8",
        "pagina_esperada": None,
        "sql_esperado_contem": None,
        "resultado_esperado": None,
        "notas": "Paráfrase de REG-01.",
    },
    {
        "id": "REG-04",
        "pergunta": "me explica como chega no valor do dic",
        "categoria_esperada": "REGULATORIO",
        "grupo_parafrases": "PARAF-DIC",
        "modulo_esperado": "Módulo 8",
        "pagina_esperada": None,
        "sql_esperado_contem": None,
        "resultado_esperado": None,
        "notas": "Paráfrase informal de REG-01.",
    },
    {
        "id": "REG-05",
        "pergunta": "O indicador DEC tem alguma regra específica de cálculo definida no PRODIST?",
        "categoria_esperada": "REGULATORIO",
        "grupo_parafrases": None,
        "modulo_esperado": "Módulo 8",
        "pagina_esperada": None,
        "sql_esperado_contem": None,
        "resultado_esperado": None,
        "notas": "DEC = Duração Equivalente de Interrupção por Unidade Consumidora. Regra definida no Módulo 8.",
    },
    {
        "id": "REG-06",
        "pergunta": "Quais são os limites estabelecidos para o indicador FEC?",
        "categoria_esperada": "REGULATORIO",
        "grupo_parafrases": None,
        "modulo_esperado": "Módulo 8",
        "pagina_esperada": None,
        "sql_esperado_contem": None,
        "resultado_esperado": None,
        "notas": "FEC = Frequência Equivalente de Interrupção por UC. Limites no Módulo 8.",
    },
    {
        "id": "REG-07",
        "pergunta": "O que são variações de tensão de curta duração no sistema elétrico?",
        "categoria_esperada": "REGULATORIO",
        "grupo_parafrases": None,
        "modulo_esperado": "Módulo 8",
        "pagina_esperada": None,
        "sql_esperado_contem": None,
        "resultado_esperado": None,
        "notas": "VTCD definido na Seção 8.1 do Módulo 8.",
    },
    {
        "id": "REG-08",
        "pergunta": "Quais os prazos que a distribuidora tem para responder a uma solicitação de ressarcimento por dano elétrico?",
        "categoria_esperada": "REGULATORIO",
        "grupo_parafrases": "PARAF-RESSARC",
        "modulo_esperado": "Módulo 9",
        "pagina_esperada": None,
        "sql_esperado_contem": None,
        "resultado_esperado": None,
        "notas": "Módulo 9, Seção 9.3. Prazo de análise da solicitação.",
    },
    {
        "id": "REG-09",
        "pergunta": "quanto tempo a distribuidora tem pra responder pedido de ressarcimento de dano elétrico",
        "categoria_esperada": "REGULATORIO",
        "grupo_parafrases": "PARAF-RESSARC",
        "modulo_esperado": "Módulo 9",
        "pagina_esperada": None,
        "sql_esperado_contem": None,
        "resultado_esperado": None,
        "notas": "Paráfrase informal de REG-08.",
    },

    # --- Módulo 3: Acesso ao Sistema ---
    {
        "id": "REG-10",
        "pergunta": "Quais os requisitos de proteção para subestações de média e alta tensão?",
        "categoria_esperada": "REGULATORIO",
        "grupo_parafrases": "PARAF-SUBEST",
        "modulo_esperado": "Módulo 3",
        "pagina_esperada": None,
        "sql_esperado_contem": None,
        "resultado_esperado": None,
        "notas": "Módulo 3 tem 12 chunks com 'subestação'. Página original do EVAL_SET (15) não verificada.",
    },
    {
        "id": "REG-11",
        "pergunta": "me fala sobre proteção de subestação",
        "categoria_esperada": "REGULATORIO",
        "grupo_parafrases": "PARAF-SUBEST",
        "modulo_esperado": "Módulo 3",
        "pagina_esperada": None,
        "sql_esperado_contem": None,
        "resultado_esperado": None,
        "notas": "Paráfrase informal de REG-10.",
    },
    {
        "id": "REG-12",
        "pergunta": "Quais são as regras de aterramento para unidades consumidoras segundo o PRODIST?",
        "categoria_esperada": "REGULATORIO",
        "grupo_parafrases": None,
        "modulo_esperado": "Módulo 3",
        "pagina_esperada": None,
        "sql_esperado_contem": None,
        "resultado_esperado": None,
        "notas": "Módulo 3 tem 5 chunks com 'aterramento'.",
    },
    {
        "id": "REG-13",
        "pergunta": "Quais são as condições para conexão de geração distribuída à rede de distribuição?",
        "categoria_esperada": "REGULATORIO",
        "grupo_parafrases": None,
        "modulo_esperado": "Módulo 3",
        "pagina_esperada": None,
        "sql_esperado_contem": None,
        "resultado_esperado": None,
        "notas": "Módulo 3 trata de acesso ao sistema, incluindo geração distribuída.",
    },

    # --- Módulo 6: Tarifas e Faturamento ---
    {
        "id": "REG-14",
        "pergunta": "Como é definida a tarifa de distribuição de energia elétrica no PRODIST?",
        "categoria_esperada": "REGULATORIO",
        "grupo_parafrases": None,
        "modulo_esperado": "Módulo 6",
        "pagina_esperada": None,
        "sql_esperado_contem": None,
        "resultado_esperado": None,
        "notas": "Módulo 6 trata de tarifas. 160 chunks disponíveis.",
    },
    {
        "id": "REG-15",
        "pergunta": "Quais são as regras de faturamento para consumidores do Grupo A?",
        "categoria_esperada": "REGULATORIO",
        "grupo_parafrases": None,
        "modulo_esperado": "Módulo 6",
        "pagina_esperada": None,
        "sql_esperado_contem": None,
        "resultado_esperado": None,
        "notas": "Grupo A = alta e média tensão. Módulo 6.",
    },

    # --- Módulo 4: Medição ---
    {
        "id": "REG-16",
        "pergunta": "Quais são os requisitos técnicos para medidores de energia elétrica?",
        "categoria_esperada": "REGULATORIO",
        "grupo_parafrases": None,
        "modulo_esperado": "Módulo 4",
        "pagina_esperada": None,
        "sql_esperado_contem": None,
        "resultado_esperado": None,
        "notas": "Módulo 4 = Medição. 116 chunks.",
    },
    {
        "id": "REG-17",
        "pergunta": "Qual a periodicidade mínima para aferição de medidores conforme o PRODIST?",
        "categoria_esperada": "REGULATORIO",
        "grupo_parafrases": None,
        "modulo_esperado": "Módulo 4",
        "pagina_esperada": None,
        "sql_esperado_contem": None,
        "resultado_esperado": None,
        "notas": "Módulo 4, Seção de verificação periódica de medidores.",
    },

    # --- Módulo 1: Introdução / Glossário ---
    {
        "id": "REG-18",
        "pergunta": "O que é uma distribuidora de energia elétrica segundo o PRODIST?",
        "categoria_esperada": "REGULATORIO",
        "grupo_parafrases": None,
        "modulo_esperado": "Módulo 1",
        "pagina_esperada": None,
        "sql_esperado_contem": None,
        "resultado_esperado": None,
        "notas": "Módulo 1 = Introdução. 152 chunks com definições e glossário.",
    },

    # --- Módulo 2: Planejamento ---
    {
        "id": "REG-19",
        "pergunta": "Quais são as diretrizes de planejamento da expansão da rede de distribuição?",
        "categoria_esperada": "REGULATORIO",
        "grupo_parafrases": None,
        "modulo_esperado": "Módulo 2",
        "pagina_esperada": None,
        "sql_esperado_contem": None,
        "resultado_esperado": None,
        "notas": "Módulo 2 = Planejamento. 72 chunks.",
    },

    # --- Módulo 5: Sistema de Proteção ---
    {
        "id": "REG-20",
        "pergunta": "Quais são os critérios para coordenação de proteção em redes de distribuição?",
        "categoria_esperada": "REGULATORIO",
        "grupo_parafrases": None,
        "modulo_esperado": "Módulo 5",
        "pagina_esperada": None,
        "sql_esperado_contem": None,
        "resultado_esperado": None,
        "notas": "Módulo 5 = Proteção. 91 chunks.",
    },

    # --- Módulo 7: Operação ---
    {
        "id": "REG-21",
        "pergunta": "Quais os procedimentos operacionais obrigatórios para manobras em redes energizadas?",
        "categoria_esperada": "REGULATORIO",
        "grupo_parafrases": None,
        "modulo_esperado": "Módulo 7",
        "pagina_esperada": None,
        "sql_esperado_contem": None,
        "resultado_esperado": None,
        "notas": "Módulo 7 = Operação. 48 chunks.",
    },

    # --- REN 1000/2021 ---
    {
        "id": "REG-22",
        "pergunta": "Quais os direitos do consumidor de energia elétrica previstos na REN 1000/2021?",
        "categoria_esperada": "REGULATORIO",
        "grupo_parafrases": None,
        "modulo_esperado": "REN 1000/2021",
        "pagina_esperada": None,
        "sql_esperado_contem": None,
        "resultado_esperado": None,
        "notas": "REN 1000/2021 disponível com 20 chunks.",
    },

    # --- Módulo 11: Qualidade de Energia ---
    {
        "id": "REG-23",
        "pergunta": "Quais são os limites de harmônicos de tensão definidos pelo PRODIST?",
        "categoria_esperada": "REGULATORIO",
        "grupo_parafrases": None,
        "modulo_esperado": "Módulo 11",
        "pagina_esperada": None,
        "sql_esperado_contem": None,
        "resultado_esperado": None,
        "notas": "Módulo 11 = Qualidade da Energia. 82 chunks.",
    },

    # --- Módulo 10: Sistema de Informação ---
    {
        "id": "REG-24",
        "pergunta": "Quais informações a distribuidora é obrigada a reportar à ANEEL periodicamente?",
        "categoria_esperada": "REGULATORIO",
        "grupo_parafrases": None,
        "modulo_esperado": "Módulo 10",
        "pagina_esperada": None,
        "sql_esperado_contem": None,
        "resultado_esperado": None,
        "notas": "Módulo 10 = Sistema de Informação. 37 chunks.",
    },

    # --- Pergunta complexa (multi-chunk) ---
    {
        "id": "REG-25",
        "pergunta": "Explique a diferença entre DEC, FEC, DIC e FIC e como cada um é apurado.",
        "categoria_esperada": "REGULATORIO",
        "grupo_parafrases": None,
        "modulo_esperado": "Módulo 8",
        "pagina_esperada": None,
        "sql_esperado_contem": None,
        "resultado_esperado": None,
        "notas": "Requer múltiplos chunks. Testa capacidade multi-documento do RAG.",
    },

    # =========================================================================
    # OPERACIONAL — 20 perguntas
    # =========================================================================

    # --- AVG ---
    {
        "id": "OPE-01",
        "pergunta": "Qual foi o valor médio do indicador DEC em 2020?",
        "categoria_esperada": "OPERACIONAL",
        "grupo_parafrases": None,
        "modulo_esperado": None,
        "pagina_esperada": None,
        "sql_esperado_contem": ["AVG", "DEC", "2020"],
        "resultado_esperado": None,
        "notas": "AVG simples com filtro de ano e indicador.",
    },
    {
        "id": "OPE-02",
        "pergunta": "Qual a média de DECXNC por ano entre 2020 e 2023?",
        "categoria_esperada": "OPERACIONAL",
        "grupo_parafrases": None,
        "modulo_esperado": None,
        "pagina_esperada": None,
        "sql_esperado_contem": ["AVG", "DECXNC", "GROUP BY"],
        "resultado_esperado": None,
        "notas": "AVG com GROUP BY + filtro de período.",
    },
    {
        "id": "OPE-03",
        "pergunta": "Qual é a média do FEC para a distribuidora CEA em 2022?",
        "categoria_esperada": "OPERACIONAL",
        "grupo_parafrases": None,
        "modulo_esperado": None,
        "pagina_esperada": None,
        "sql_esperado_contem": ["AVG", "FEC", "CEA", "2022"],
        "resultado_esperado": None,
        "notas": "AVG com filtro de distribuidora + indicador + ano.",
    },

    # --- COUNT ---
    {
        "id": "OPE-04",
        "pergunta": "Quantos registros existem para a distribuidora CEA?",
        "categoria_esperada": "OPERACIONAL",
        "grupo_parafrases": None,
        "modulo_esperado": None,
        "pagina_esperada": None,
        "sql_esperado_contem": ["COUNT", "CEA"],
        "resultado_esperado": 19104,
        "notas": "COUNT com filtro de distribuidora. Resultado confirmado: 19104.",
    },
    {
        "id": "OPE-05",
        "pergunta": "Quantas distribuidoras distintas existem na base de dados?",
        "categoria_esperada": "OPERACIONAL",
        "grupo_parafrases": None,
        "modulo_esperado": None,
        "pagina_esperada": None,
        "sql_esperado_contem": ["COUNT", "DISTINCT", "SigAgente"],
        "resultado_esperado": None,
        "notas": "COUNT DISTINCT. Testa geração de DISTINCT.",
    },
    {
        "id": "OPE-06",
        "pergunta": "Quantos registros de DEC existem para o ano de 2021?",
        "categoria_esperada": "OPERACIONAL",
        "grupo_parafrases": None,
        "modulo_esperado": None,
        "pagina_esperada": None,
        "sql_esperado_contem": ["COUNT", "DEC", "2021"],
        "resultado_esperado": None,
        "notas": "COUNT com duplo filtro.",
    },

    # --- MAX / MIN ---
    {
        "id": "OPE-07",
        "pergunta": "Qual distribuidora teve o maior valor de FEC em 2021?",
        "categoria_esperada": "OPERACIONAL",
        "grupo_parafrases": None,
        "modulo_esperado": None,
        "pagina_esperada": None,
        "sql_esperado_contem": ["FEC", "2021"],
        "resultado_esperado": None,
        "notas": "MAX ou ORDER BY DESC LIMIT 1 — ambos são corretos. sql_esperado_contem não inclui MAX para não penalizar abordagem alternativa.",
    },
    {
        "id": "OPE-08",
        "pergunta": "Qual foi o menor valor de DEC registrado para a EAC em todo o período disponível?",
        "categoria_esperada": "OPERACIONAL",
        "grupo_parafrases": None,
        "modulo_esperado": None,
        "pagina_esperada": None,
        "sql_esperado_contem": ["MIN", "DEC", "EAC"],
        "resultado_esperado": None,
        "notas": "MIN com filtro de distribuidora e indicador.",
    },
    {
        "id": "OPE-09",
        "pergunta": "Qual o maior valor de FECIPC registrado em toda a base?",
        "categoria_esperada": "OPERACIONAL",
        "grupo_parafrases": None,
        "modulo_esperado": None,
        "pagina_esperada": None,
        "sql_esperado_contem": ["MAX", "FECIPC"],
        "resultado_esperado": None,
        "notas": "MAX simples.",
    },

    # --- GROUP BY ---
    {
        "id": "OPE-10",
        "pergunta": "Qual a média do DEC por distribuidora em 2023?",
        "categoria_esperada": "OPERACIONAL",
        "grupo_parafrases": None,
        "modulo_esperado": None,
        "pagina_esperada": None,
        "sql_esperado_contem": ["AVG", "DEC", "2023", "GROUP BY"],
        "resultado_esperado": None,
        "notas": "GROUP BY distribuidora.",
    },
    {
        "id": "OPE-11",
        "pergunta": "Qual a soma do DEC por mês para a distribuidora ETO em 2022?",
        "categoria_esperada": "OPERACIONAL",
        "grupo_parafrases": None,
        "modulo_esperado": None,
        "pagina_esperada": None,
        "sql_esperado_contem": ["ETO", "DEC", "2022", "GROUP BY"],
        "resultado_esperado": None,
        "notas": "GROUP BY mês (NumPeriodoIndice).",
    },

    # --- ORDER BY ---
    {
        "id": "OPE-12",
        "pergunta": "Liste as 5 distribuidoras com maior média de DEC em 2023, em ordem decrescente.",
        "categoria_esperada": "OPERACIONAL",
        "grupo_parafrases": None,
        "modulo_esperado": None,
        "pagina_esperada": None,
        "sql_esperado_contem": ["DEC", "2023", "ORDER BY", "LIMIT"],
        "resultado_esperado": None,
        "notas": "ORDER BY + LIMIT. Testa ranking.",
    },

    # --- WHERE (filtros compostos) ---
    {
        "id": "OPE-13",
        "pergunta": "Quais registros de DEC da EAC têm valor acima de 10 horas?",
        "categoria_esperada": "OPERACIONAL",
        "grupo_parafrases": None,
        "modulo_esperado": None,
        "pagina_esperada": None,
        "sql_esperado_contem": ["DEC", "EAC", "WHERE"],
        "resultado_esperado": None,
        "notas": "WHERE com filtro numérico.",
    },

    # --- Comparação entre anos ---
    {
        "id": "OPE-14",
        "pergunta": "Compare a média do DEC da distribuidora CEA nos anos de 2020 e 2023.",
        "categoria_esperada": "OPERACIONAL",
        "grupo_parafrases": None,
        "modulo_esperado": None,
        "pagina_esperada": None,
        "sql_esperado_contem": ["DEC", "CEA", "2020", "2023"],
        "resultado_esperado": None,
        "notas": "Comparação temporal. Pode gerar CTE ou subconsultas.",
    },

    # --- Comparação entre distribuidoras ---
    {
        "id": "OPE-15",
        "pergunta": "Qual distribuidora tem melhor DEC médio em 2022: CEA ou EAC?",
        "categoria_esperada": "OPERACIONAL",
        "grupo_parafrases": None,
        "modulo_esperado": None,
        "pagina_esperada": None,
        "sql_esperado_contem": ["DEC", "CEA", "EAC", "2022"],
        "resultado_esperado": None,
        "notas": "Comparação entre distribuidoras. Testa geração de subconsultas ou HAVING.",
    },

    # --- Filtro temporal (mês) ---
    {
        "id": "OPE-16",
        "pergunta": "Qual foi o valor médio de FEC em dezembro de 2022?",
        "categoria_esperada": "OPERACIONAL",
        "grupo_parafrases": None,
        "modulo_esperado": None,
        "pagina_esperada": None,
        "sql_esperado_contem": ["AVG", "FEC", "2022"],
        "resultado_esperado": None,
        "notas": "Filtro de mês (NumPeriodoIndice = 12).",
    },

    # --- Pergunta informal com sigla e sem contexto ---
    {
        "id": "OPE-17",
        "pergunta": "quanto foi o dec da eac",
        "categoria_esperada": "OPERACIONAL",
        "grupo_parafrases": None,
        "modulo_esperado": None,
        "pagina_esperada": None,
        "sql_esperado_contem": ["DEC", "EAC"],
        "resultado_esperado": None,
        "notas": "Minúsculo, sem ano, sem pontuação. Testa robustez do SQL agent.",
    },

    # --- DISTINCT ---
    {
        "id": "OPE-18",
        "pergunta": "Quais são os anos disponíveis na base de indicadores de continuidade?",
        "categoria_esperada": "OPERACIONAL",
        "grupo_parafrases": None,
        "modulo_esperado": None,
        "pagina_esperada": None,
        "sql_esperado_contem": ["DISTINCT", "AnoIndice"],
        "resultado_esperado": [2020, 2021, 2022, 2023, 2024, 2025, 2026],
        "notas": "DISTINCT + ORDER. Resultado esperado confirmado.",
    },

    # --- Múltiplos indicadores ---
    {
        "id": "OPE-19",
        "pergunta": "Qual a média de DEC e FEC para a distribuidora ETO em 2021?",
        "categoria_esperada": "OPERACIONAL",
        "grupo_parafrases": None,
        "modulo_esperado": None,
        "pagina_esperada": None,
        "sql_esperado_contem": ["ETO", "2021"],
        "resultado_esperado": None,
        "notas": "Testa múltiplos indicadores na mesma query.",
    },

    # --- Pergunta de interpretação (sem SQL complexo) ---
    {
        "id": "OPE-20",
        "pergunta": "Qual foi o total de registros coletados de indicadores de continuidade em 2024?",
        "categoria_esperada": "OPERACIONAL",
        "grupo_parafrases": None,
        "modulo_esperado": None,
        "pagina_esperada": None,
        "sql_esperado_contem": ["COUNT", "2024"],
        "resultado_esperado": None,
        "notas": "COUNT com filtro de ano.",
    },

    # =========================================================================
    # FRONTEIRA — 10 perguntas ambíguas REGULATORIO / OPERACIONAL
    # =========================================================================

    {
        "id": "FRONT-01",
        "pergunta": "O indicador DEC tem alguma regra de cálculo definida no PRODIST?",
        "categoria_esperada": "REGULATORIO",
        "grupo_parafrases": None,
        "modulo_esperado": "Módulo 8",
        "pagina_esperada": None,
        "sql_esperado_contem": None,
        "resultado_esperado": None,
        "notas": "FRONTEIRA: menciona DEC (operacional) mas pergunta sobre a REGRA (regulatório).",
    },
    {
        "id": "FRONT-02",
        "pergunta": "A ANEEL define metas de DEC para as distribuidoras?",
        "categoria_esperada": "REGULATORIO",
        "grupo_parafrases": None,
        "modulo_esperado": "Módulo 8",
        "pagina_esperada": None,
        "sql_esperado_contem": None,
        "resultado_esperado": None,
        "notas": "FRONTEIRA: pergunta sobre definição de metas, não sobre valor apurado.",
    },
    {
        "id": "FRONT-03",
        "pergunta": "Qual é o limite máximo de DEC tolerado para distribuidoras?",
        "categoria_esperada": "REGULATORIO",
        "grupo_parafrases": None,
        "modulo_esperado": "Módulo 8",
        "pagina_esperada": None,
        "sql_esperado_contem": None,
        "resultado_esperado": None,
        "notas": "FRONTEIRA: menciona 'máximo' (operacional) mas é sobre regra regulatória.",
    },
    {
        "id": "FRONT-04",
        "pergunta": "Como é feito o cálculo do ressarcimento em caso de dano elétrico?",
        "categoria_esperada": "REGULATORIO",
        "grupo_parafrases": None,
        "modulo_esperado": "Módulo 9",
        "pagina_esperada": None,
        "sql_esperado_contem": None,
        "resultado_esperado": None,
        "notas": "FRONTEIRA: 'cálculo' parece operacional mas é regra do Módulo 9.",
    },
    {
        "id": "FRONT-05",
        "pergunta": "A distribuidora CEA cumpre os limites regulatórios de DEC?",
        "categoria_esperada": "OPERACIONAL",
        "grupo_parafrases": None,
        "modulo_esperado": None,
        "pagina_esperada": None,
        "sql_esperado_contem": ["DEC", "CEA"],
        "resultado_esperado": None,
        "notas": "FRONTEIRA: menciona 'limites regulatórios' mas requer dados reais da CEA.",
    },
    {
        "id": "FRONT-06",
        "pergunta": "Quais distribuidoras ultrapassaram o FEC em 2022?",
        "categoria_esperada": "OPERACIONAL",
        "grupo_parafrases": None,
        "modulo_esperado": None,
        "pagina_esperada": None,
        "sql_esperado_contem": ["FEC", "2022"],
        "resultado_esperado": None,
        "notas": "FRONTEIRA: 'ultrapassaram' implica comparação com limite regulatório, mas requer dados.",
    },
    {
        "id": "FRONT-07",
        "pergunta": "Como a EAC se comportou em relação ao DEC em 2021 comparado à norma?",
        "categoria_esperada": "OPERACIONAL",
        "grupo_parafrases": None,
        "modulo_esperado": None,
        "pagina_esperada": None,
        "sql_esperado_contem": ["DEC", "EAC", "2021"],
        "resultado_esperado": None,
        "notas": "FRONTEIRA difícil: mistura dado operacional com referência à norma.",
    },
    {
        "id": "FRONT-08",
        "pergunta": "O que é DEC e qual foi o valor médio em 2023?",
        "categoria_esperada": "REGULATORIO",
        "grupo_parafrases": None,
        "modulo_esperado": "Módulo 8",
        "pagina_esperada": None,
        "sql_esperado_contem": None,
        "resultado_esperado": None,
        "notas": "FRONTEIRA: pergunta mista. Idealmente precisaria de ambos os agentes. Esperamos REGULATORIO pela primeira parte.",
    },
    {
        "id": "FRONT-09",
        "pergunta": "A interrupção de fornecimento gera obrigação de ressarcimento automático?",
        "categoria_esperada": "REGULATORIO",
        "grupo_parafrases": None,
        "modulo_esperado": "Módulo 9",
        "pagina_esperada": None,
        "sql_esperado_contem": None,
        "resultado_esperado": None,
        "notas": "FRONTEIRA: pode parecer pergunta de dado mas é questão normativa.",
    },
    {
        "id": "FRONT-10",
        "pergunta": "indicadores de qualidade",
        "categoria_esperada": "REGULATORIO",
        "grupo_parafrases": None,
        "modulo_esperado": "Módulo 8",
        "pagina_esperada": None,
        "sql_esperado_contem": None,
        "resultado_esperado": None,
        "notas": "FRONTEIRA: pergunta extremamente vaga. Pode ser roteada para qualquer lado.",
    },

    # =========================================================================
    # SEM_RESPOSTA — 10 perguntas (abstention)
    # =========================================================================
    # Informações que NÃO existem no corpus nem na base de dados

    {
        "id": "ABS-01",
        "pergunta": "Qual é a previsão de crescimento do setor elétrico brasileiro para 2030?",
        "categoria_esperada": "SEM_RESPOSTA",
        "grupo_parafrases": None,
        "modulo_esperado": None,
        "pagina_esperada": None,
        "sql_esperado_contem": None,
        "resultado_esperado": None,
        "notas": "Informação de projeção futura não presente no corpus.",
        "abstencao_esperada": True,
        "frase_abstencao": "Não encontrei essa informação",
    },
    {
        "id": "ABS-02",
        "pergunta": "Qual é o salário médio de um engenheiro elétrico no Brasil?",
        "categoria_esperada": "SEM_RESPOSTA",
        "grupo_parafrases": None,
        "modulo_esperado": None,
        "pagina_esperada": None,
        "sql_esperado_contem": None,
        "resultado_esperado": None,
        "notas": "Completamente fora do escopo do sistema.",
        "abstencao_esperada": True,
        "frase_abstencao": "Não encontrei essa informação",
    },
    {
        "id": "ABS-03",
        "pergunta": "Qual é o custo de instalação de um painel solar residencial?",
        "categoria_esperada": "SEM_RESPOSTA",
        "grupo_parafrases": None,
        "modulo_esperado": None,
        "pagina_esperada": None,
        "sql_esperado_contem": None,
        "resultado_esperado": None,
        "notas": "Fora do escopo. Não há dados de custo de instalação.",
        "abstencao_esperada": True,
        "frase_abstencao": "Não encontrei essa informação",
    },
    {
        "id": "ABS-04",
        "pergunta": "Qual distribuidora será privatizada no próximo ano?",
        "categoria_esperada": "SEM_RESPOSTA",
        "grupo_parafrases": None,
        "modulo_esperado": None,
        "pagina_esperada": None,
        "sql_esperado_contem": None,
        "resultado_esperado": None,
        "notas": "Informação de mercado futura. Não no corpus.",
        "abstencao_esperada": True,
        "frase_abstencao": "Não encontrei essa informação",
    },
    {
        "id": "ABS-05",
        "pergunta": "Qual o DEC do Módulo 8?",
        "categoria_esperada": "SEM_RESPOSTA",
        "grupo_parafrases": None,
        "modulo_esperado": None,
        "pagina_esperada": None,
        "sql_esperado_contem": None,
        "resultado_esperado": None,
        "notas": "Pergunta sem sentido: 'DEC do Módulo 8' mistura dado operacional com documento normativo. Testa alucinação.",
        "abstencao_esperada": True,
        "frase_abstencao": "Não encontrei essa informação",
    },
    {
        "id": "ABS-06",
        "pergunta": "Como faço para reclamar de energia cara na prefeitura?",
        "categoria_esperada": "SEM_RESPOSTA",
        "grupo_parafrases": None,
        "modulo_esperado": None,
        "pagina_esperada": None,
        "sql_esperado_contem": None,
        "resultado_esperado": None,
        "notas": "Procedimento administrativo não coberto pelo corpus.",
        "abstencao_esperada": True,
        "frase_abstencao": "Não encontrei essa informação",
    },
    {
        "id": "ABS-07",
        "pergunta": "Qual a meta de DEC da distribuidora Energisa para 2027?",
        "categoria_esperada": "SEM_RESPOSTA",
        "grupo_parafrases": None,
        "modulo_esperado": None,
        "pagina_esperada": None,
        "sql_esperado_contem": None,
        "resultado_esperado": None,
        "notas": "Ano 2027 não existe na base. Testa alucinação de dado futuro.",
        "abstencao_esperada": True,
        "frase_abstencao": "Não encontrei essa informação",
    },
    {
        "id": "ABS-08",
        "pergunta": "Quais são os indicadores ESG das distribuidoras brasileiras?",
        "categoria_esperada": "SEM_RESPOSTA",
        "grupo_parafrases": None,
        "modulo_esperado": None,
        "pagina_esperada": None,
        "sql_esperado_contem": None,
        "resultado_esperado": None,
        "notas": "Indicadores ESG não fazem parte do PRODIST nem da base operacional.",
        "abstencao_esperada": True,
        "frase_abstencao": "Não encontrei essa informação",
    },
    {
        "id": "ABS-09",
        "pergunta": "Qual é o artigo da Constituição Federal que regula a distribuição de energia?",
        "categoria_esperada": "SEM_RESPOSTA",
        "grupo_parafrases": None,
        "modulo_esperado": None,
        "pagina_esperada": None,
        "sql_esperado_contem": None,
        "resultado_esperado": None,
        "notas": "Constituição Federal não faz parte do corpus (apenas PRODIST e REN 1000).",
        "abstencao_esperada": True,
        "frase_abstencao": "Não encontrei essa informação",
    },
    {
        "id": "ABS-10",
        "pergunta": "Quanto custou o KWh médio no Brasil em 2021?",
        "categoria_esperada": "SEM_RESPOSTA",
        "grupo_parafrases": None,
        "modulo_esperado": None,
        "pagina_esperada": None,
        "sql_esperado_contem": None,
        "resultado_esperado": None,
        "notas": "Dado de preço de energia não está na base (que contém apenas indicadores de qualidade).",
        "abstencao_esperada": True,
        "frase_abstencao": "Não encontrei essa informação",
    },

    # =========================================================================
    # PARÁFRASES — 5 grupos já distribuídos acima
    # (REG-01..04: PARAF-DIC, REG-08..09: PARAF-RESSARC, REG-10..11: PARAF-SUBEST)
    # Adicionamos 2 grupos extras aqui para completar 5
    # =========================================================================

    {
        "id": "PARAF-01",
        "pergunta": "Quais são as penalidades por descumprimento dos indicadores de qualidade?",
        "categoria_esperada": "REGULATORIO",
        "grupo_parafrases": "PARAF-PENALID",
        "modulo_esperado": "Módulo 8",
        "pagina_esperada": None,
        "sql_esperado_contem": None,
        "resultado_esperado": None,
        "notas": "Paráfrase grupo PENALIDADES.",
    },
    {
        "id": "PARAF-02",
        "pergunta": "O que acontece se a distribuidora não cumprir o DEC?",
        "categoria_esperada": "REGULATORIO",
        "grupo_parafrases": "PARAF-PENALID",
        "modulo_esperado": "Módulo 8",
        "pagina_esperada": None,
        "sql_esperado_contem": None,
        "resultado_esperado": None,
        "notas": "Paráfrase informal de PARAF-01.",
    },
    {
        "id": "PARAF-03",
        "pergunta": "Qual é a média do DEC para todas as distribuidoras em 2022?",
        "categoria_esperada": "OPERACIONAL",
        "grupo_parafrases": "PARAF-AVG-DEC",
        "modulo_esperado": None,
        "pagina_esperada": None,
        "sql_esperado_contem": ["AVG", "DEC", "2022"],
        "resultado_esperado": None,
        "notas": "Paráfrase grupo AVG-DEC.",
    },
    {
        "id": "PARAF-04",
        "pergunta": "Qual o DEC médio das distribuidoras no ano de 2022?",
        "categoria_esperada": "OPERACIONAL",
        "grupo_parafrases": "PARAF-AVG-DEC",
        "modulo_esperado": None,
        "pagina_esperada": None,
        "sql_esperado_contem": ["AVG", "DEC", "2022"],
        "resultado_esperado": None,
        "notas": "Paráfrase de PARAF-03.",
    },
    {
        "id": "PARAF-05",
        "pergunta": "me diz a media do dec de 2022",
        "categoria_esperada": "OPERACIONAL",
        "grupo_parafrases": "PARAF-AVG-DEC",
        "modulo_esperado": None,
        "pagina_esperada": None,
        "sql_esperado_contem": ["AVG", "DEC", "2022"],
        "resultado_esperado": None,
        "notas": "Paráfrase informal de PARAF-03.",
    },
]

# ---------------------------------------------------------------------------
# Subsets para uso nos módulos de avaliação
# ---------------------------------------------------------------------------

REGULATORIO = [e for e in EVAL_SET_V2 if e["categoria_esperada"] == "REGULATORIO" and e.get("grupo_parafrases") is None or
               (e["categoria_esperada"] == "REGULATORIO" and e["id"].startswith("REG"))]

OPERACIONAL = [e for e in EVAL_SET_V2 if e["categoria_esperada"] == "OPERACIONAL" and e["id"].startswith("OPE")]

FRONTEIRA = [e for e in EVAL_SET_V2 if e["id"].startswith("FRONT")]

ABSTENCAO = [e for e in EVAL_SET_V2 if e["id"].startswith("ABS")]

PARAFRASES = [e for e in EVAL_SET_V2 if e.get("grupo_parafrases") is not None]

# Subset para Router (inclui todos — o router precisa ver casos de fronteira e abstention)
ROUTER_SET = EVAL_SET_V2

# Subset para Retrieval (apenas regulatórios com modulo_esperado definido)
RETRIEVAL_SET = [e for e in EVAL_SET_V2
                 if e["modulo_esperado"] is not None
                 and e["categoria_esperada"] == "REGULATORIO"]

# Subset para SQL Agent
SQL_SET = [e for e in EVAL_SET_V2
           if e["categoria_esperada"] == "OPERACIONAL"
           and e.get("sql_esperado_contem") is not None]


def resumo_dataset():
    """Imprime um resumo do dataset."""
    from collections import Counter
    cats = Counter(e["categoria_esperada"] for e in EVAL_SET_V2)
    grupos = Counter(e.get("grupo_parafrases") for e in EVAL_SET_V2 if e.get("grupo_parafrases"))
    print(f"Total: {len(EVAL_SET_V2)} perguntas")
    print(f"Por categoria: {dict(cats)}")
    print(f"Grupos de paráfrases: {len(grupos)} grupos, {sum(grupos.values())} perguntas")
    print(f"  Subsets — Router: {len(ROUTER_SET)} | Retrieval: {len(RETRIEVAL_SET)} | SQL: {len(SQL_SET)}")


if __name__ == "__main__":
    resumo_dataset()
