# Analise_Default
---
Análise de Inadimplência com Redes Neurais e Modelos Ensemble
---

```{r setup, include=FALSE}
knitr::opts_chunk$set(echo = TRUE)
```

## 1. Carregando Pacotes e Dados

```{r}
library(ISLR)
library(dplyr)
library(ggplot2)
library(caret)
library(neuralnet)
library(pROC)
library(randomForest)
library(ipred)
library(xgboost)
library(tidyr)

set.seed(123)

data(Default)
df <- Default %>%
  mutate(
    default_bin = ifelse(default == "Yes", 1, 0),
    student_bin = ifelse(student == "Yes", 1, 0)
  ) %>%
  select(default_bin, student_bin, balance, income)
```

## 2. Análise Exploratória dos Dados

### 2.1 Balanceamento da Variável Target

```{r}
table(df$default_bin)
prop.table(table(df$default_bin))

ggplot(df, aes(x = factor(default_bin))) +
  geom_bar(fill = "#4682B4") +
  labs(x = "Inadimplência (0 = Não, 1 = Sim)", y = "Frequência", title = "Distribuição da variável default_bin")
```

> A base é fortemente desbalanceada: apenas cerca de 3% dos clientes são inadimplentes.

### 2.2 Correlação Entre Variáveis Numéricas

```{r}
cor(df[, c("balance", "income", "default_bin")])
pairs(df[, c("balance", "income", "default_bin")], main = "Relações entre variáveis")
```

> `balance` tem forte relação com inadimplência, enquanto `income` aparenta ter pouca influência.

### 2.3 Distribuição das Variáveis Numéricas

```{r}
df_long <- pivot_longer(df, cols = c(balance, income), names_to = "variavel", values_to = "valor")
ggplot(df_long, aes(x = valor)) +
  geom_histogram(fill = "#008080", bins = 30) +
  facet_wrap(~ variavel, scales = "free") +
  labs(title = "Distribuição de Balance e Income")
```

---

## 3. Pré-processamento

```{r}
normalize <- function(x) (x - min(x)) / (max(x) - min(x))
df$balance <- normalize(df$balance)
df$income <- normalize(df$income)

train_index <- sample(1:nrow(df), size = 0.8 * nrow(df))
train_data <- df[train_index, ]
test_data <- df[-train_index, ]
```

---

## 4. Validação Cruzada da Rede Neural

```{r}
k <- 5
folds <- sample(rep(1:k, length.out = nrow(train_data)))

compute_error <- function(data_train, data_test, hidden) {
  nn <- neuralnet(default_bin ~ student_bin + balance + income,
                  data = data_train,
                  hidden = hidden,
                  linear.output = FALSE)
  pred <- compute(nn, data_test[, c("student_bin", "balance", "income")])$net.result
  pred_class <- ifelse(pred > 0.5, 1, 0)
  mean(pred_class != data_test$default_bin)
}

cv_results <- data.frame()
for (h in 1:4) {
  for (i in 1:k) {
    train_fold <- train_data[folds != i, ]
    test_fold <- train_data[folds == i, ]
    error <- compute_error(train_fold, test_fold, h)
    cv_results <- rbind(cv_results, data.frame(type = "Cross-Validation", hidden = h, error = error))
  }
}

final_results <- data.frame()
for (h in 1:4) {
  test_error <- compute_error(train_data, test_data, h)
  final_results <- rbind(final_results, data.frame(type = "Test", hidden = h, error = test_error))
}

all_results <- bind_rows(cv_results, final_results) %>%
  group_by(type, hidden) %>%
  summarise(mean_error = mean(error), .groups = "drop")

ggplot(all_results, aes(x = hidden, y = mean_error, color = type)) +
  geom_line() + geom_point(size = 2) +
  labs(title = "Erro Médio por Número de Neurônios", x = "Neurônios na Camada Oculta", y = "Erro Médio")
```

---

## 5. Comparação com Modelos Ensemble

```{r}
avaliar_modelo <- function(y_true, y_prob, threshold = 0.5) {
  pred_class <- ifelse(y_prob > threshold, 1, 0)
  cm <- confusionMatrix(factor(pred_class), factor(y_true), positive = "1")
  erro <- mean(pred_class != y_true)
  data.frame(
    Recall = cm$byClass["Sensitivity"],
    Erro = erro
  )
}

# Random Forest
rf <- randomForest(factor(default_bin) ~ student_bin + balance + income, data = train_data)
rf_probs <- predict(rf, newdata = test_data, type = "prob")[, 2]
rf_metrics <- avaliar_modelo(test_data$default_bin, rf_probs)
rf_metrics$model <- "Random Forest"

# Bagging
bagging <- bagging(factor(default_bin) ~ ., data = train_data, nbagg = 25)
bagging_probs <- predict(bagging, newdata = test_data, type = "prob")[, 2]
bagging_metrics <- avaliar_modelo(test_data$default_bin, bagging_probs)
bagging_metrics$model <- "Bagging"

# Boosting
dtrain <- xgb.DMatrix(as.matrix(train_data[, c("student_bin", "balance", "income")]), label = train_data$default_bin)
dtest <- xgb.DMatrix(as.matrix(test_data[, c("student_bin", "balance", "income")]), label = test_data$default_bin)
boost <- xgboost(data = dtrain, nrounds = 100, objective = "binary:logistic", verbose = 0)
boost_probs <- predict(boost, dtest)
boost_metrics <- avaliar_modelo(test_data$default_bin, boost_probs)
boost_metrics$model <- "Boosting"

# Neural Net com 3 neurônios
nn3 <- neuralnet(default_bin ~ student_bin + balance + income,
                 data = train_data, hidden = 3, linear.output = FALSE)
nn3_probs <- compute(nn3, test_data[, c("student_bin", "balance", "income")])$net.result
nn3_metrics <- avaliar_modelo(test_data$default_bin, nn3_probs)
nn3_metrics$model <- "Neural Net"

# Consolidar
final_metrics <- bind_rows(rf_metrics, bagging_metrics, boost_metrics, nn3_metrics) %>%
  select(model, Erro, Recall)

print(final_metrics)
```

---

## 6. Conclusões

- A rede neural obteve a **menor taxa de erro (~2.55%)**, com **3 neurônios** na camada oculta.
- No entanto, o **Recall da rede foi apenas ~0.30**, o que significa que ela teve dificuldades em identificar inadimplentes.
- **Boosting** superou os demais em Recall (**~0.37**), o que pode ser mais relevante no contexto de crédito.
- O conjunto de dados é **fortemente desbalanceado**, sugerindo que abordagens como `SMOTE`, reweighting ou ajustes de threshold poderiam melhorar o Recall das redes neurais.

> **Recomendações**: Para aplicações práticas, sugere-se otimizar o balanceamento da base e priorizar Recall, dado o custo de não detectar um inadimplente.
