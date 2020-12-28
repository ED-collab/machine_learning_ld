###############################################
#          SOURCE CODE FOR CHAPTER 4          #
###############################################

# LOAD PACKLAGES ----
library(mlr)

library(tidyverse)

# LOAD DATA ----
install.packages("titanic")

data(titanic_train, package = "titanic")

titanicTib <- as_tibble(titanic_train)

titanicTib

# CLEAN DATA ----
fctrs <- c("Survived", "Sex", "Pclass")

titanicClean <- titanicTib %>%
  mutate_at(.vars = fctrs, .funs = factor) %>%
  mutate(FamSize = SibSp + Parch) %>%
  select(Survived, Pclass, Sex, Age, Fare, FamSize)

titanicClean

# PLOT DATA ----
titanicUntidy <- gather(titanicClean, key = "Variable", value = "Value", 
                        -Survived)
titanicUntidy 

titanicUntidy %>%
  filter(Variable != "Pclass" & Variable != "Sex") %>%
  ggplot(aes(Survived, as.numeric(Value))) +
  facet_wrap(~ Variable, scales = "free_y") +
  geom_violin(draw_quantiles = c(0.25, 0.5, 0.75)) +
  theme_bw()

titanicUntidy %>%
  filter(Variable == "Pclass" | Variable == "Sex") %>%
  ggplot(aes(Value, fill = Survived)) +
  facet_wrap(~ Variable, scales = "free_x") +
  geom_bar(position = "fill") +
  theme_bw()

# CREATE TASK AND LEARNER, AND ATTEMPT TO TRAIN MODEL ----
titanicTask <- makeClassifTask(data = titanicClean, target = "Survived")

logReg <- makeLearner("classif.logreg", predict.type = "prob")
#By setting the argument predict.type = "prob", the trained model will output the estimated 
#probabilities of each class when making predictions on new data, rather than just the 
#predicted class membership.
logRegModel <- train(logReg, titanicTask)
#Ah shit... NAs in Age variable are causing an error

# COUNT MISSING VALUES IN Age VARIABLE ----
titanicClean$Age #Take a look

sum(is.na(titanicClean$Age))

# IMPUTE MISSING VALUES ----
imp <- impute(titanicClean, cols = list(Age = imputeMean()))
#Mean imputation: take mean of variable & replace missing values with that

sum(is.na(titanicClean$Age))

sum(is.na(imp$data$Age))

# CREATE TASK WITH IMPUTED DATA AND TRAIN MODEL ----
titanicTask <- makeClassifTask(data = imp$data, target = "Survived")

logRegModel <- train(logReg, titanicTask)
#This time no error messages

# WRAP LEARNER ----
logRegWrapper <- makeImputeWrapper("classif.logreg",
                                   cols = list(Age = imputeMean()))

#makeImputeWrapper is going to wrap together both the learner and the imputation method, which
#is important because we want to cross-validate both

logRegWrapper

# CROSS-VALIDATE ----
kFold <- makeResampleDesc(method = "RepCV", folds = 10, reps = 50, 
                          stratify = TRUE)
#apply stratified, 10-fold cross-validation, repeated 50 times, to our wrapped learner.
#This means that goes for the imputation method as well (in this case probably doesn't matter since only mean)

logRegwithImpute <- resample(logRegWrapper, titanicTask, resampling = kFold, 
                             measures = list(acc, fpr, fnr))
logRegwithImpute

# EXTRACT ODDS RATIOS
logRegModelData <- getLearnerModel(logRegModel)
# turn our mlr model object, logRegModel, into an R model object using the getLearnerModel()

coef(logRegModelData)
#pass this R model object as the argument to the coef() function, which stands for coefficients 
#(another term for parameters), so this function returns the model parameters.
#The intercept is the log odds of surviving the Titanic disaster when all continuous variables 
#are 0 and the factors are at their reference levels.

exp(cbind(Odds_Ratio = coef(logRegModelData), confint(logRegModelData)))
# By taking their exponent
#can also calculate 95% confidence intervals using confint()

# USING THE MODEL TO MAKE PREDICTIONS ----
data(titanic_test, package = "titanic")

titanicNew <- as_tibble(titanic_test)

titanicNewClean <- titanicNew %>%
  mutate_at(.vars = c("Sex", "Pclass"), .funs = factor) %>%
  mutate(FamSize = SibSp + Parch) %>%
  select(Pclass, Sex, Age, Fare, FamSize)

predict(logRegModel, newdata = titanicNewClean)
#Here is the prob of survival/death for each passenger in the new data

# EXERCISES ----
# 1
titanicUntidy %>%
  filter(Variable != "Pclass" & Variable != "Sex") %>%
  ggplot(aes(Survived, as.numeric(Value))) +
  facet_wrap(~ Variable, scales = "free_y") +
  geom_violin(draw_quantiles = c(0.25, 0.5, 0.75)) +
  geom_point(alpha = 0.05, size = 3) +
  theme_bw()

# 2
titanicUntidy %>%
  filter(Variable == "Pclass" | Variable == "Sex") %>%
  ggplot(aes(Value, fill = Survived)) +
  facet_wrap(~ Variable, scales = "free_x") +
  geom_bar(position = "dodge") +
  theme_bw()

titanicUntidy %>%
  filter(Variable == "Pclass" | Variable == "Sex") %>%
  ggplot(aes(Value, fill = Survived)) +
  facet_wrap(~ Variable, scales = "free_x") +
  geom_bar(position = "stack") +
  theme_bw()

# 3
titanicNoFare <- select(titanicClean, -Fare)
titanicNoFareTask <- makeClassifTask(data = titanicNoFare, 
                                     target = "Survived")
logRegNoFare <- resample(logRegWrapper, titanicNoFareTask, 
                         resampling = kFold, 
                         measures = list(acc, fpr, fnr))
logRegNoFare

# 4
surnames <- map_chr(str_split(titanicTib$Name, "\\."), 1)

salutations <- map_chr(str_split(surnames, ", "), 2)

salutations[!(salutations %in% c("Mr", "Dr", "Master", 
                                 "Miss", "Mrs", "Rev"))] <- "Other"
# 5
fctrsInclSals <- c("Survived", "Sex", "Pclass", "Salutation")

titanicWithSals <- titanicTib %>%
  mutate(FamSize = SibSp + Parch, Salutation = salutations) %>%
  mutate_at(.vars = fctrsInclSals, .funs = factor) %>%
  select(Survived, Pclass, Sex, Age, Fare, FamSize, Salutation)

titanicTaskWithSals <- makeClassifTask(data = titanicWithSals, 
                                       target = "Survived")

logRegWrapper <- makeImputeWrapper("classif.logreg",
                                   cols = list(Age = imputeMean()))

kFold <- makeResampleDesc(method = "RepCV", folds = 10, reps = 50, 
                          stratify = TRUE)

logRegWithSals <- resample(logRegWrapper, titanicTaskWithSals, 
                           resampling = kFold, 
                           measures = list(acc, fpr, fnr))
logRegWithSals

