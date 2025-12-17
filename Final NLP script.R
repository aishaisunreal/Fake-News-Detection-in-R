#install.packages("dplyr") 
#install.packages("tidyverse")
#install.packages("rpart")
#install.packages("rpart.plot")
#install.packages("e1071")
#install.packages("caret")
#install.packages("randomForest")
#install.packages("glmnet")
#install.packages("text2vec")

library(NLP)
library(tm)
library(tokenizers)
library(dplyr)
library(textstem)
library(tidytext)
library(tidyverse)
library(caret)
library(rpart)
library(rpart.plot)
library(e1071)
library(randomForest)
library(glmnet)
library(text2vec)

news <- read.csv("news.csv", stringsAsFactors = FALSE, na.strings = "")

str(news)
head(news)
summary(news)

news <- news %>%
  select(title, text, label)

news <- news %>%
  distinct()

news <- news %>%
  filter(!is.na(title),
         !is.na(text),
         !is.na(label))

news$text_length_chars <- nchar(news$text)
hist(
  news$text_length_chars,
  main  = "Histogram of text length (characters)",
  xlab  = "Number of characters",
  col   = "#bc4749",
  border = "#6f1d1b"
)

news$title_word_count <- sapply(strsplit(news$title, "\\s+"), length)
hist(
  news$title_word_count,
  main = "Histogram of title word count",
  xlab = "Number of words in title",
  col  = "#8ecae6",
  border = "#023047"
)

news$label <- toupper(trimws(news$label))
news <- news %>%
  filter(label %in% c("FAKE", "REAL"))
news$label <- factor(news$label, levels = c("FAKE", "REAL"))

news <- news %>%
  mutate(
    text_length = nchar(text),
    title_word_count = stringr::str_count(title, "\\S+")
  )

tab <- table(news$label)
barplot(tab,
        main = "Fake vs Real News Distribution",
        xlab = "Label",
        ylab = "Count",
        col  = "steelblue")

hist(news$text_length,
     breaks = 30,
     main = "Distribution of Text Length (characters)",
     xlab  = "Number of characters",
     col   = "darkred",
     border = "white")

hist(news$title_word_count,
     breaks = 30,
     main = "Distribution of Title Word Count",
     xlab  = "Number of words in title",
     col   = "darkgreen",
     border = "white")

news$full_text <- paste(news$title, news$text)

corpus <- VCorpus(VectorSource(news$full_text))

corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removeWords, stopwords("english"))
corpus <- tm_map(corpus, stripWhitespace)
corpus <- tm_map(corpus, content_transformer(lemmatize_strings))

dtm <- DocumentTermMatrix(
  corpus,
  control = list(
    wordLengths = c(2, Inf),
    bounds = list(global = c(5, Inf))
  )
)

dtm <- removeSparseTerms(dtm, 0.95)


#####################################################

set.seed(42)
X <- as.data.frame(as.matrix(dtm))
y <- factor(news$label)

train_idx <- createDataPartition(y, p = 0.8, list = FALSE)

X_train <- X[train_idx, ]
X_test  <- X[-train_idx, ]
y_train <- y[train_idx]
y_test  <- y[-train_idx]

table(y_test)

train_df <- data.frame(label = y_train, X_train)
test_df <- data.frame(label = y_test, X_test)

train_df$label <- as.factor(train_df$label)
test_df$label  <- as.factor(test_df$label)


# Decision Tree
set.seed(42)

tree_control <- rpart.control(
  cp = 0.0005,
  minsplit = 10,
  maxdepth = 30
)

tree_model <- rpart(
  label ~ .,
  data = train_df,
  method = "class",
  parms = list(split = "gini"),
  control = tree_control
)

best_cp <- tree_model$cptable[which.min(tree_model$cptable[,"xerror"]), "CP"]
tree_model <- prune(tree_model, cp = best_cp)

summary(tree_model)

rpart.plot(tree_model, main = "Decision Tree")

tree_pred <- predict(tree_model, newdata = test_df, type = "class")
tree_pred_train <- predict(tree_model, newdata = train_df, type = "class")
tree_cm_train <- confusionMatrix(tree_pred_train, y_train)
tree_cm <- confusionMatrix(tree_pred, y_test)

tree_accuracy <- tree_cm$overall["Accuracy"]
tree_train_accuracy <- tree_cm_train$overall["Accuracy"]

print("Decision tree train accuracy: ")
tree_train_accuracy
print("test accuracy:")
tree_accuracy


#Random forest

set.seed(42)
rf_model <- randomForest(
  x = X_train,
  y = y_train,
  ntree = 100,       
  mtry = floor(sqrt(ncol(train_df) - 1)), 
  importance = TRUE
)

rf_model
rf_pred_train <- predict(rf_model, newdata = X_train)
rf_pred <- predict(rf_model, newdata = X_test)

rf_cm <- confusionMatrix(rf_pred, y_test)
rf_cm_train <- confusionMatrix(rf_pred_train, y_train)

rf_accuracy <- rf_cm$overall["Accuracy"]
rf_accuracy_train <- rf_cm_train$overall["Accuracy"]

print("random forest train accuracy:")
rf_accuracy_train
print("test accuracy: ")
rf_accuracy

varImpPlot(rf_model, n.var = 20)


# Naive Bayes
set.seed(42)

nb_model <- naiveBayes(
  x = X_train,
  y = y_train,
  laplace = 0.5,
  
)

nb_pred <- predict(nb_model, newdata = as.data.frame(X_test))
nb_pred_train <- predict(nb_model, newdata = as.data.frame(X_train))

nb_cm_train <- confusionMatrix(nb_pred_train, y_train)
nb_cm <- confusionMatrix(nb_pred, y_test)

nb_accuracy <- nb_cm$overall["Accuracy"]
nb_accuracy_train <- nb_cm_train$overall["Accuracy"]
print("Naive Bayes train accuracy: ")
nb_accuracy_train
print("test accuracy: ")
nb_accuracy


#Logistic regression
X_train_mat <- as.matrix(data.frame(lapply(X_train, as.numeric)))
X_test_mat  <- as.matrix(data.frame(lapply(X_test, as.numeric)))

set.seed(42)

y_train_num <- ifelse(y_train == "REAL", 1, 0)


log_model <- cv.glmnet(
  x = X_train_mat,
  y = y_train_num,
  family = "binomial",
  alpha = 0.5
)


log_prob_train <- predict(log_model, newx = X_train_mat, s = "lambda.min", type = "response")
log_prob       <- predict(log_model, newx = X_test_mat, s = "lambda.min", type = "response")

log_pred_train <- ifelse(log_prob_train > 0.5, "REAL", "FAKE")
log_pred_train <- factor(log_pred_train, levels = c("FAKE", "REAL"))

log_pred <- ifelse(log_prob > 0.5, "REAL", "FAKE")
log_pred <- factor(log_pred, levels = c("FAKE", "REAL"))

log_cm_train <- confusionMatrix(log_pred_train, y_train)
log_cm       <- confusionMatrix(log_pred,       y_test)

log_accuracy_train <- log_cm_train$overall["Accuracy"]
log_accuracy       <- log_cm$overall["Accuracy"]

print("logistic regression train accuracy:")
print(log_accuracy_train)
print("test accuracy: ")
print(log_accuracy)


model_results <- data.frame(
  Model    = c("Decision Tree", "Naive Bayes", "Random Forest", "Logistic Regression"),
  Accuracy = c(tree_cm$overall["Accuracy"],
               nb_cm$overall["Accuracy"],
               rf_cm$overall["Accuracy"],
               log_cm$overall["Accuracy"])
)

model_results


ggplot(model_results, aes(x = Model, y = Accuracy, fill = Model)) +
  geom_bar(stat = "identity", width = 0.6) +
  theme_minimal() +
  labs(title = "Model Accuracy Comparison",
       y = "Accuracy",
       x = "Model") +
  theme(axis.text.x = element_text(angle = 15, hjust = 1)) +
  scale_fill_brewer(palette = "Set2")

