rm(list=ls(all=T))

#set current working directory
setwd("D:/edvisor data science path/Project_1_R")
getwd()


#Load Libraries
x = c("ggplot2", "corrgram", "DMwR", "caret", "randomForest", "unbalanced", "C50", "dummies", "e1071", "Information",
      "MASS", "rpart", "gbm", "ROSE", 'sampling', 'DataCombine', 'inTrees')

#install.packages(x)
lapply(x, require, character.only = TRUE)
rm(x)

#Read the data 

training_data = read.csv("train.csv", header = T, na.strings = c(" ", "", "NA"))

#Explore the data

str(training_data)
training_data$target=as.factor(training_data$target)


#missing value analysis
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

ggplot(data = missing_val[1:10,], aes(x=reorder(Columns, -Missing_percentage),y = Missing_percentage))+
geom_bar(stat = "identity",fill = "grey")+xlab("Parameter")+
ggtitle("Missing data percentage (Train)") + theme_bw()


#from the above ananlysis we can conclude the data does not have any missing value so we will proceed further for the next step


##Outlier Analysis

# BoxPlots - Distribution and Outlier Check

numeric_index = sapply(training_data,is.numeric) #selecting only numeric

#take the data witch contains only numeric values
numeric_data = training_data[,numeric_index]

#save the numeric columns name
cnames = colnames(numeric_data)





for (i in 1:length(cnames))
   {
     assign(paste0("gn",i), ggplot(aes_string(y = (cnames[i]), x = "target"), data = subset(training_data))+ 
              stat_boxplot(geom = "errorbar", width = 0.5) +
              geom_boxplot(outlier.colour="red", fill = "grey" ,outlier.shape=18,
                           outlier.size=1, notch=FALSE) +
              theme(legend.position="bottom")+
              labs(y=cnames[i],x="target")+
              ggtitle(paste("Box plot of target for",cnames[i])))
   }



# ## Plotting plots together
gridExtra::grid.arrange(gn1,gn5,gn2,ncol=3)
gridExtra::grid.arrange(gn6,gn7,ncol=2)


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
qqnorm(training_data$var_0)
hist(training_data$var_0)

#apply normalization 
for (i in cnames){
  print(i)
  training_data[,i]=(training_data[,i]-min(training_data[,i]))/(max(training_data[,i]-min(training_data[,i])))
}

#Data Sampling beacause R runs on RAM so we have shrink the dataset afterall the computer which i am using have only 3.5 GB of total RAM available

#We are going to use stratified sampling with the reference variable target
table(training_data$target)
stratas = strata(training_data, c("target"), size = c(20000,3000), method = "srswor")
stratified_data = getdata(training_data, stratas)

###Machine Learning algorithms *****Model Development*********


library(DataCombine)
rm("numeric_data")

#Divide data into train and test using stratified sampling method
set.seed(1234)
train.index = createDataPartition(stratified_data$target, p = .80, list = FALSE)
train = stratified_data[ train.index,]
test  = stratified_data[-train.index,]


#Develop model** We have already tested the naive bayes algorithm is working best for this data set wo we are going to use that same algorithm just for testing ***
##NOTE-the computer which i am using have only 3.5 GB of available ram so that's why i am unable to use all the algorithms because it shows an errot memory exausted
#so i am using only Naive Bayes here just to show that the model is working same as in the python

library(e1071)

NB_model = naiveBayes(target ~ ., data = train)

#predict on test cases #raw
NB_Predictions = predict(NB_model, test[,1:201], type = 'class')


Conf_matrix = table(observed = test[,202], predicted = NB_Predictions)
confusionMatrix(Conf_matrix)


#False negative rate
FNR = 360/(360+240)
#FNR=60%

#Accuracy=90.22%

#now predicting which customer is going to make the specific transacion using the Test data provided with the assignement
rm(stratas)
test_data = read.csv("test.csv", header = T)

NB_Test = predict(NB_model,test_data[,1:200], type = 'class')


