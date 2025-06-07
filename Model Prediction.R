
# Load necessary libraries

install.packages("caret")
library(tidyverse)
library(caret)
library(ggplot2)
library(dplyr)
library(lattice)


# Read the CSV (replace "yourfile.csv" with the actual path or use textConnection for direct string input)
df <- read.csv("C:/BI/Statistical Learning/Assignment 1 Due date May 12 on 0900 Oclock/Data_HomeEx1/Question 2/cleaned_data.csv", stringsAsFactors = FALSE)


# Distribution of behaviors
table(df$Modifiers)

# Boxplot example
ggplot(df, aes(x = Modifiers, y = sdnorm)) +
  geom_boxplot() +
  labs(
    title = "Distribution the standard deviation norm of acceleration by Modifier",
    x = "Modifiers",
    y = "Standardized Value (sdnorm)"
  ) +theme(
    axis.text.x = element_text(angle = 45, hjust = 1),
    plot.title = element_text(hjust = 0.5)
  )


# Convert target to factor (if not already)
df$Modifiers <- as.factor(df$Modifiers)

# Split into train/test
set.seed(123)
train_index <- createDataPartition(df$Modifiers, p = 0.7, list = FALSE)
train_data <- df[train_index, ]
test_data <- df[-train_index, ]

#  k-Nearest Neighbors (kNN)
# Train using kNN
ctrl <- trainControl(method = "cv", number = 5)
set.seed(123)
knn_model <- train(Modifiers ~ ., data = train_data, method = "knn", trControl = ctrl, tuneLength = 10)

# View results of tuning
print(knn_model)

# Best value of k
cat("Best k:", knn_model$bestTune$k, "\n")

# Predict on test data
knn_pred <- predict(knn_model, newdata = test_data)

# Confusion matrix
cm <- confusionMatrix(knn_pred, test_data$Modifiers)
print(cm)

# Accuracy
cat("Accuracy on test set:", cm$overall['Accuracy'], "\n")


# Create and store the plot
p <- ggplot(knn_model)

# Plot model performance (Accuracy vs. k)
# Add title and labels
ggplot(knn_model) +
  ggtitle("kNN Model Accuracy vs. Number of Neighbors (k)") +
  xlab("Number of Neighbors (k)") +
  ylab("Accuracy") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))



# LDA Analysis

df <- na.omit(df)

# Step 4: Ensure target is a factor
df$Modifiers <- as.factor(df$Modifiers)

# Step 5: Split into training and testing sets
set.seed(123)
train_index <- createDataPartition(df$Modifiers, p = 0.8, list = FALSE)
train_data <- df[train_index, ]
test_data <- df[-train_index, ]

# Step 6: Remove near-zero variance predictors
nzv <- nearZeroVar(train_data, saveMetrics = TRUE)
train_data_nzv <- train_data[, !nzv$nzv]
test_data_nzv <- test_data[, colnames(train_data_nzv)]

# Step 7: Remove known constant columns that caused LDA failure
# Columns 9, 10, 20, 26, 29, 38 (based on previous error)
problem_cols <- colnames(train_data_nzv)[c(9, 10, 20, 26, 29, 38)]
train_data_clean <- train_data_nzv %>% select(-all_of(problem_cols))
test_data_clean <- test_data_nzv %>% select(-all_of(problem_cols))

# Step 8: Remove highly correlated predictors (to avoid collinearity)
cor_matrix <- cor(train_data_clean %>% select(-Modifiers))
high_corr <- findCorrelation(cor_matrix, cutoff = 0.95)
train_data_final <- train_data_clean %>% select(-high_corr)
test_data_final <- test_data_clean %>% select(colnames(train_data_final))

# Step 9: Make sure Modifiers is still a factor
train_data_final$Modifiers <- as.factor(train_data_final$Modifiers)
test_data_final$Modifiers <- as.factor(test_data_final$Modifiers)

# Step 10: Train LDA model
ctrl <- trainControl(method = "cv", number = 5)
lda_model <- train(Modifiers ~ ., data = train_data_final, method = "lda", trControl = ctrl)

# Step 11: Make predictions and evaluate
lda_pred <- predict(lda_model, test_data_final)
conf_mat <- confusionMatrix(lda_pred, test_data_final$Modifiers)

# Output the results
print(lda_model)
print(conf_mat)



# Nice visualization for the repot - test
cat("Overall Accuracy:", round(conf_mat$overall["Accuracy"], 4), "\n\n")

# Print the confusion matrix as a table
conf_df <- as.data.frame(conf_mat$table)
conf_df <- conf_df %>%
  rename(
    Actual = Reference,
    Predicted = Prediction,
    Count = Freq
  ) %>%
  arrange(desc(Actual), desc(Predicted))

print(conf_df)


# Confusion matrix heatmap
ggplot(conf_df, aes(x = Actual, y = Predicted, fill = Count)) +
  geom_tile(color = "white", linewidth = 0.6) +
  geom_text(aes(label = Count), size = 4.5, color = "black") +
  scale_fill_gradientn(
    colours = c("#f7f7f7", "#c7d6e0", "#91b3c7", "#6699cc"),
    name = "Count"
  ) +
  labs(
    title = "Confusion Matrix: LDA Classification Model",
    subtitle = paste("Overall Accuracy:", round(conf_mat$overall["Accuracy"], 4)),
    x = "Actual Behaviour",
    y = "Predicted Behaviour"
  ) +
  theme_minimal(base_size = 14) +
  theme(
    axis.text.x = element_text(angle = 45, hjust = 1, vjust = 1),
    axis.text.y = element_text(angle = 0, hjust = 1),
    plot.title = element_text(hjust = 0.5, size = 18, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5, size = 13),
    legend.position = "right",
    panel.grid.major = element_blank()
  )

