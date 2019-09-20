rm(list=ls(all=T))

#set current working directory
setwd("D:/edvisor data science path/Project_2_R")
getwd()

#Load Libraries
x = c("ggplot2", "corrgram", "DMwR", "caret", "randomForest", "unbalanced", "C50", "dummies", "e1071", "Information",
      "MASS", "rpart", "gbm", "ROSE", 'sampling', 'DataCombine', 'inTrees')

#install.packages(x)
lapply(x, require, character.only = TRUE)
rm(x)


#Read the training data 

training_data = read.csv("train_cab.csv", header = T, na.strings = c(" ", "", "NA"))

#Explore the data

str(training_data)
training_data$fare_amount=as.numeric(training_data$fare_amount)

str(training_data)
training_data$pickup_datetime=as.numeric(training_data$pickup_datetime)
str(training_data)

##missing value analysis
#create data frame with missing percentage

missing_val = data.frame(apply(training_data,2,function(x){sum(is.na(x))}))


#convert rownames into columns
missing_val$Columns = row.names(missing_val)
                                
#rename the variablel name 
names(missing_val)[1] =  "Missing_percentage"

#calculate missing percentage

missing_val$Missing_percentage = (missing_val$Missing_percentage/nrow(training_data)) * 100

#arrange in decending order

missing_val = missing_val[order(-missing_val$Missing_percentage),]

#Rearrange the columns
row.names(missing_val) = NULL
missing_val = missing_val[,c(2,1)]

#Write output result back into disk

write.csv(missing_val, "Miising_percentage.csv", row.names = F)



#plot bar graph for missing value

ggplot(data = missing_val[1:7,], aes(x=reorder(Columns, -Missing_percentage),y = Missing_percentage))+
  geom_bar(stat = "identity",fill = "grey")+xlab("Parameter")+
  ggtitle("Missing data percentage (Train)") + theme_bw()

#from the above analysis we can conclude that two variables have missing values so we will impute them



#impute with mean in fare_amount
training_data$fare_amount[is.na(training_data$fare_amount)]=mean(training_data$fare_amount,na.rm = T)

#impute with median in fare_amount
training_data$fare_amount[is.na(training_data$fare_amount)]=median(training_data$fare_amount,na.rm = T)

#the value of median method is closest to so we will go with the median method 

#impute with mean passenger_count
training_data$passenger_count[is.na(training_data$passenger_count)]=mean(training_data$passenger_count,na.rm = T)

#impute with median passenger_count
training_data$passenger_count[is.na(training_data$passenger_count)]=median(training_data$passenger_count,na.rm = T)

#the value of the median method is closeset so wi will go with the median 

##Outlier Analysis

# BoxPlots - Distribution and Outlier Check

numeric_index = sapply(training_data,is.numeric) #selecting only numeric

#take the data witch contains only numeric values
numeric_data = training_data[,numeric_index]

#save the numeric columns name
cnames = colnames(numeric_data)


for (i in 1:length(cnames))
{
  assign(paste0("gn",i), ggplot(aes_string(y = (cnames[i]), x = "fare_amount"), data = subset(training_data))+ 
           stat_boxplot(geom = "errorbar", width = 0.5) +
           geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18,outlier.size=1, notch=FALSE) +
           theme(legend.position="bottom")+
           labs(y=cnames[i],x="fare_amount")+
           ggtitle(paste("Box plot of target for",cnames[i])))
}





# ## Plotting plots together
gridExtra::grid.arrange(gn1,gn5,gn2,gn3,gn4,gn5,gn6,gn7,ncol=9)


# # #Remove outliers using boxplot method
df = training_data

##detect and delete the outlayers 

#loop to remove from all variables
for(i in cnames){
  print(i)
  val = training_data[,i][training_data[,i] %in% boxplot.stats(training_data[,i])$out]
  print(length(val))
  training_data = training_data[which(!training_data[,i] %in% val),]
}


###Feature selection

rm(missing_val)
write.csv(numeric_data, "numeric_data.csv", row.names = F)
rm(numeric_data)


corrgram(training_data[,numeric_index],order = F,upper.panel = panel.pie,text.panel = panel.text,main="Correlation Plot" )

#we are going to use all the variables

#feature scalling 

#normality check
qqnorm(training_data$fare_amount)
hist(training_data$fare_amount)
hist(training_data$pickup_datetime)
hist(training_data$passenger_count)

new_data=subset(training_data,select=c("fare_amount","passenger_count"))

cnames1=colnames(new_data)

#apply normalization 

for (i in cnames1){
  print(i)
  training_data[,i]=(training_data[,i]-min(training_data[,i]))/(max(training_data[,i]-min(training_data[,i])))
}

#the datetime object needs standerization 
data_new=subset(training_data,select=c("pickup_datetime"))
cnames2=colnames(data_new)

for (i in cnames2){
  print(i)
  training_data[,i]=(training_data[,i]-mean(training_data[,i]))/sd(training_data[,i])
}


##start the ML concept*************Building the Machine leatniong model*********

#lets devide the data into train and test 
set.seed(123)
train_index=sample(1:nrow(training_data),0.8*nrow(training_data))
train = training_data[train_index,]
test = training_data[-train_index,]

#check for multi coleniarity
library(usdm)
vif(training_data[,-1])
vifcor(training_data[,-1],th=0.9)


#####Error Matrix**********

#MAPE
#calculate MAPE
MAPE = function(y, yhat){
  mean(abs((y - yhat)/y))
}

##*******************Linear Regression***********

#Linear_resgression
lm_model = lm(fare_amount ~., data = training_data)

#Summary of the model
summary(lm_model)


#Predict
predictions_LR = predict(lm_model, test[,2:7])

#Calculate MAPE
MAPE(test[,1], predictions_LR)


# ##***************************##DT_regressor***************************


fit = rpart(fare_amount ~ ., data = training_data, method = "anova")

#Predict for new test cases
predictions = predict(fit, test[,-1])


#calculate error matrix 

MAPE(test[,1], predictions)




############Randome Forest REgression***********************

RF_model = randomForest(fare_amount ~ ., train, importance = TRUE, ntree = 500)
predict = predict(RF_model, test[,-1])

###calculate the MAPE

MAPE(test[,1], predict)

#############now load the test data****************************
test_data = read.csv("test.csv", header = T, na.strings = c(" ", "", "NA"))


#Explore test data

str(test_data)
test_data$pickup_datetime=as.numeric(test_data$pickup_datetime)
str(training_data)

##prepare the test data for predictions

subset_for_c1=subset(test_data,select=c("passenger_count"))

c1=colnames(subset_for_c1)

for (i in c1){
  print(i)
 test_data[,i]=(test_data[,i]-min(test_data[,i]))/(max(test_data[,i]-min(test_data[,i])))
}

subset_for_d2=subset(test_data,select=c("pickup_datetime"))

d2=colnames(subset_for_d2)

for (i in d2){
  print(i)
  test_data[,i]=(test_data[,i]-mean(test_data[,i]))/sd(test_data[,i])
}
#prediction on test data
fare_predictions = predict(RF_model, test_data)

fare_predictions=as.data.frame(fare_predictions)

#save the output file

write.csv(fare_predictions, "fare_predictions.csv", row.names = F)


