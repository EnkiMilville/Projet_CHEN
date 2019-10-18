#set-up
set.seed(299792458)


#### Import ####
################


#data.table
if (!require(data.table)) install.packages("data.table") 
require(data.table)

#rpart
if (!require(rpart)) install.packages("rpart") 
require(rpart)


# loading data
df <- fread("iris.csv")

Y_NAME <- "variety"
X_NAMES<- names(df)[-which(names(df)==Y_NAME)]
TEST_SIZE= 0.1
index_Y <- which(names(df)==Y_NAME)

#### Donnees ####
#################


#Split : Train and Test
train_id <- sample(1:nrow(df),floor(nrow(df)*(1-TEST_SIZE)))
train <- df[train_id,]
test <- df[-train_id]


#set algorithm to use
formula <- paste(Y_NAME ,paste(X_NAMES,collapse = "+"),sep="~")
fit <-  rpart(formula,
              method="class", data=train)



# plot tree
plot(fit, uniform=TRUE,
     main="Classification Tree for Species")
text(fit, use.n=TRUE, all=TRUE, cex=.5)


# prediction
pred <- predict(fit,newdata = test )
pred <-apply(pred,1,which.max)
pred[pred==1] <- "setosa"
pred[pred==2] <- "versicolor"
pred[pred==3] <- "virginica"

test_res <- unlist(test[,..index_Y])


mat.confusion <-table(pred,test_res)
accuracy <- sum(diag(mat.confusion))/nrow(test)

#display 
print(mat.confusion)
print(accuracy)


