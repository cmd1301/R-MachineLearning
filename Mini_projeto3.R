# Mini-Projeto 3
# Prevendo a Inadimplência de Clientes com Machine Learning e Power BI

# Definindo a pasta de trabalho
setwd("C:/Users/Carlos Magno/Documents/PowerBI/Cap15")
getwd()

# Instalando os pacotes para o projeto
install.packages("Amelia") # funções para tratar valores ausentes
install.packages("caret") # construir modelos de ML e interpretar dados
install.packages("ggplot2")
install.packages("dplyr") # tratar dados
install.packages("reshape") # mudar a forma de alguns dados
install.packages("randomForest") # ML
install.packages("e1071") # ML

# Carregando os pacotes
library(Amelia)
library(ggplot2)
library(caret)
library(dplyr)
library(reshape)
library(randomForest)
library(e1071)

# Carregando o dataset
dados_clientes <- read.csv("dados/dataset.csv")

# Visualizando os dados e sua estrutura
View(dados_clientes)
str(dados_clientes)
summary(dados_clientes)

#### Análise Exploratória, Limpeza e Transformação ####

# Removendo a primeira coluna: ID
dados_clientes$ID <- NULL
dim(dados_clientes)
View(dados_clientes)

# Renomeando a coluna de classe
colnames(dados_clientes)
colnames(dados_clientes)[24] <- "inadimplente"
colnames(dados_clientes)
View(dados_clientes)

# Verificando valores ausentes e removendo do dataset
sapply(dados_clientes, function(x) sum(is.na(x)))
missmap(dados_clientes, main = "Valores Missing Observados")
dados_clientes <- na.omit(dados_clientes)

# Convertendo os atributos genero, escolaridade, estado civil e idade
# para fatores (categorias)

# Renomeando colunas categóricas
colnames(dados_clientes)
colnames(dados_clientes)[2] <- "genero"
colnames(dados_clientes)[3] <- "escolaridade"
colnames(dados_clientes)[4] <- "estado_civil"
colnames(dados_clientes)[5] <- "idade"
colnames(dados_clientes)
View(dados_clientes)

# Genero
View(dados_clientes$genero)
str(dados_clientes$genero)
summary(dados_clientes$genero)
dados_clientes$genero <- cut(dados_clientes$genero,
                             c(0,1,2),
                             labels = c("masculino", "feminino"))
View(dados_clientes$genero)

# Escolaridade
dados_clientes$escolaridade <- cut(dados_clientes$escolaridade,
                                   c(0,1,2,3,4),
                                   labels = c('posgrad', 'grad', 'ensino_medio', 'outros'))
View(dados_clientes$escolaridade)
summary(dados_clientes$estado_civil)

# Estado Civil
dados_clientes$estado_civil <- cut(dados_clientes$estado_civil,
                                   c(-1,0,1,2,3),
                                   labels = c('desconhecido', 'casado', 'solteiro', 'outro'))
View(dados_clientes$estado_civil)
summary(dados_clientes$estado_civil)

# Convertendo a variável para o tipo fator com faixa etária
str(dados_clientes$idade)
summary(dados_clientes$idade)
hist(dados_clientes$idade)
dados_clientes$idade <- cut(dados_clientes$idade,
                            c(0,30,50,100),
                            labels = c('jovem', 'adulto', 'idoso'))
View(dados_clientes$idade)
str(dados_clientes$idade)
summary(dados_clientes$idade)
View(dados_clientes)

# Convertendo a variável que indica pagamentos para o tipo fator
dados_clientes$PAY_0 <- as.factor(dados_clientes$PAY_0)
dados_clientes$PAY_2 <- as.factor(dados_clientes$PAY_2)
dados_clientes$PAY_3 <- as.factor(dados_clientes$PAY_3)
dados_clientes$PAY_4 <- as.factor(dados_clientes$PAY_4)
dados_clientes$PAY_5 <- as.factor(dados_clientes$PAY_5)
dados_clientes$PAY_6 <- as.factor(dados_clientes$PAY_6)

# Dataset após as conversões
str(dados_clientes)
sapply(dados_clientes, function(x) sum(is.na(x)))
missmap(dados_clientes, main = "Valores Missing Observados")
dados_clientes <- na.omit(dados_clientes)
missmap(dados_clientes, main = "Valores Missing Observados")
dim(dados_clientes)

# Alterando a variável dependente para o tipo fator
str(dados_clientes$inadimplente)
colnames(dados_clientes)
dados_clientes$inadimplente <- as.factor(dados_clientes$inadimplente)
str(dados_clientes$inadimplente)
View(dados_clientes)
dim(dados_clientes)

# Total de inadimplentes versus não-inadimplentes
# função table calcula proporções de uma variável
table(dados_clientes$inadimplente)
prop.table(table(dados_clientes$inadimplente))

# Plot da distribuição usando ggplot2
qplot(inadimplente, data = dados_clientes, geom = 'bar') +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))

# Set seed (random number generation)
set.seed(12345)

# Amostragem estratificada
# Seleciona as linhas de acordo com a variável inadimplente como strata
indice <- createDataPartition(dados_clientes$inadimplente, p = 0.75, list = FALSE)
dim(indice)

# Definimos os dados de treinamento como subconjunto do conjunto de dados original
# com números de índice de linha (conforme identificado acima) e todas as colunas
dados_treino <- dados_clientes[indice,] # fatiar o df
dim(dados_treino)
table(dados_treino$inadimplente)

# Porcentagens no dataset de treino
prop.table(table(dados_treino$inadimplente))

# Número de registros no dataset de treino
dim(dados_treino)

# Comparamos as porcentagens entre as classes de treinamento e dados originais
compara_dados <- cbind(prop.table(table(dados_treino$inadimplente)),
                       prop.table(table(dados_clientes$inadimplente)))
colnames(compara_dados) <- c('treinamento', 'original')
compara_dados

# Melt Data = converte colunas em linhas
reshape2::melt
melt_compara_dados <- melt(compara_dados)
melt_compara_dados

# Plot para ver distribuição do treinamento vs original
ggplot(melt_compara_dados, aes(x=x1, y=value)) +
  geom_bar(aes(fill = x2), stat = 'identity', position = 'dodge') +
  theme(axis.text.x = element_text(angle = 90, hjust = 1))
        
# Tudo que não está no dataset de treinamento está no de teste (sinal de -)
dados_teste <- dados_clientes[-indice, ]
dim(dados_treino)

#### Modelo de Machine Learning ####

# Construindo a primeira versão do modelo
modelo_v1 <- randomForest(inadimplente ~ ., data = dados_treino)
modelo_v1

# Avaliando o modelo
plot(modelo_v1)

# Previsões com dados de teste
previsoes_v1 <- predict(modelo_v1, dados_teste)

# Confusion Matrix
cm_v1 <- caret::confusionMatrix(previsoes_v1, dados_teste$inadimplente, positive = "1")
cm_v1

# Calculando Precision, Recall e F1-Score, métricas de avaliação do modelo preditivo
y <- dados_teste$inadimplente
y_pred_v1 <- previsoes_v1

precision <- posPredValue(y_pred_v1, y)
precision

recall <- sensitivity(y_pred_v1, y)
recall

F1 <- (2*precision*recall)/(precision+recall)
F1

# Balanceamento de classe
install.packages("performanceEstimation")
library(performanceEstimation)

# Aplicando o SMOTE: Synthethic Minority Over-sampling Technique
table(dados_treino$inadimplente)
prop.table(table(dados_treino$inadimplente))
set.seed(9560)
dados_treino_bal <- smote(inadimplente ~ ., data=dados_treino)
table(dados_treino_bal$inadimplente)
prop.table(table(dados_treino_bal$inadimplente))

# Construindo a segunda versão do modelo
modelo_v2 <- randomForest(inadimplente ~ ., data=dados_treino_bal)
modelo_v2

# Avaliando o modelo
plot(modelo_v2)

# Previsões com dados de teste
previsoes_v2 <- predict(modelo_v2, dados_teste)

# Confusion Matrix
cm_v2 <- caret::confusionMatrix(previsoes_v2, dados_teste$inadimplente, positive = "1")
cm_v2

# Calculando Precision, Recall e F1-Score, métricas de avaliação do modelo preditivo
y <- dados_teste$inadimplente
y_pred_v2 <- previsoes_v2

precision <- posPredValue(y_pred_v2, y)
precision

recall <- sensitivity(y_pred_v2, y)
recall

F1 <- (2*precision*recall)/(precision+recall)
F1

# Importânica das variáveis preditoras para as previsões
View(dados_treino_bal)
varImpPlot(modelo_v2)

# Obtendo as variáveis mais importantes
imp_var <- importance(modelo_v2)
varImportance <- data.frame(Variables = row.names(imp_var),
                            Importance = round(imp_var[ ,"MeanDecreaseGini"],2))

# Criando o rank de variáveis baseado na importância
rankImportance <- varImportance %>%
  mutate(Rank = paste0('#', dense_rank(desc(Importance))))

# Usando ggplot2 para visualizar a importância relativa das variáveis
ggplot(rankImportance,
       aes(x = reorder(Variables, Importance),
           y = Importance,
           fill = Importance)) +
  geom_bar(stat = 'identity') +
  geom_text(aes(x = Variables, y = 0.5, label = Rank),
            hjust = 0,
            vjust = 0.55,
            size = 4,
            colour = 'red') +
  labs(x = 'Variables') +
  coord_flip()

# Construindo a terceira visão do modelo apenas com as variáveis mais importantes
colnames(dados_treino_bal)
modelo_v3 <- randomForest(inadimplente ~ PAY_0 + PAY_2 + PAY_3 + PAY_AMT1 + PAY_AMT2 + PAY_5 + BILL_AMT1, data = dados_treino_bal)
modelo_v3

# Avaliando o modelo
plot(modelo_v3)

# Previsões com dados de teste
previsoes_v3 <- predict(modelo_v3, dados_teste)

# Confusion Matrix
cm_v3 <- caret::confusionMatrix(previsoes_v3, dados_teste$inadimplente, positive = "1")
cm_v3

# Calculando Precision, Recall e F1-Score, métricas de avaliação do modelo preditivo
y <- dados_teste$inadimplente
y_pred_v3 <- previsoes_v3

precision <- posPredValue(y_pred_v3, y)
precision

recall <- sensitivity(y_pred_v3, y)
recall

F1 <- (2*precision*recall)/(precision+recall)
F1

# Salvando o modelo em disco
saveRDS(modelo_v3, file = 'modelo_v3.rds')

# Carregando o modelo
modelo_final <- readRDS('modelo_v3.rds')

# Previsões com novos dados de 3 clientes

# Dados dos clientes
PAY_0 <- c(0, 0, 0)
PAY_2 <- c(0, 0, 0)
PAY_3 <- c(1, 0, 0)
PAY_AMT1 <- c(1100, 1000, 1200)
PAY_AMT2 <- c(1500, 1300, 1150)
PAY_5 <- c(0, 0, 0)
BILL_AMT1 <- c(350, 420, 280)

# Concatena em um dataframe
novos_clientes <- data.frame(PAY_0, PAY_2, PAY_3, PAY_AMT1, PAY_AMT2, PAY_5, BILL_AMT1)
View(novos_clientes)

# Checando os tipos de dados
str(dados_treino_bal)
str(novos_clientes)

# Convertendo os tipos de dados
novos_clientes$PAY_0 <- factor(novos_clientes$PAY_0, levels = levels(dados_treino_bal$PAY_0))
novos_clientes$PAY_2 <- factor(novos_clientes$PAY_2, levels = levels(dados_treino_bal$PAY_2))
novos_clientes$PAY_3 <- factor(novos_clientes$PAY_3, levels = levels(dados_treino_bal$PAY_3))
novos_clientes$PAY_5 <- factor(novos_clientes$PAY_5, levels = levels(dados_treino_bal$PAY_5))
str(novos_clientes)

# Previsões
previsoes_novo_cliente <- predict(modelo_final, novos_clientes)
View(previsoes_novo_cliente)

# Instalar o pacote tibble para converter o vetor para dataframe com a função enframe
install.packages("tibble")
library(tibble)
enframe(previsoes_novo_cliente)