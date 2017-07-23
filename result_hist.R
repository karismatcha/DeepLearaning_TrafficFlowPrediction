# Install packages if we don't have
if (!require("corrplot")) install.packages("corrplot")
if (!require("ggplot2"))  install.packages("ggplot2")
if (!require("caret"))    install.packages("caret")
if (!require("e1071"))    install.packages("e1071")
if (!require("car"))      install.packages("car")
if (!require("stats"))    install.packages("stats")
if (!require("gridExtra"))    install.packages("gridExtra")
if (!require("ROCR"))    install.packages("ROCR")
if (!require("fBasics"))    install.packages("fBasics")
if (!require("randomForest"))    install.packages("randomForest")
if (!require("FSelector"))    install.packages("FSelector")

library(reshape2)
library(party)
library(rpart)
library(rattle)
library(rpart.plot)
library(RColorBrewer)


library(dplyr)
library(caret)
library(ggplot2)
library(corrplot)
library(e1071)
library(car)
library(stats)
library(gridExtra)
library(ROCR)
library(fBasics)
library(randomForest)
library(magrittr)
library(tidyr)
library(C50)
library(rpart)
library(randomForest)
library(neuralnet)
library(rpart)
library(caret) 
library(foreach)
library(devtools)
library(lattice)
library(devtools)

speed <- read.csv("speed.csv")
result <- read.csv("result.csv")


#hist of CarsSpeed
hist(speed[[1]], # histogram
     col = "peachpuff", # column color
     border = "black", 
     prob = TRUE, # show densities instead of frequencies
     xlab = "Speed (km/h)",
     main = "Real speed")
lines(density(speed[[1]]), # density plot
      lwd = 2, # thickness of line
      col = "chocolate3")


abline(v = mean(speed[[1]]),
       col = "royalblue",
       lwd = 2)

abline(v = median(speed[[1]]),
       col = "red",
       lwd = 2)
legend(x = "topright", # location of legend within plot area
       c("Density plot", "Mean", "Median"),
       col = c("chocolate3", "royalblue", "red"),
       lwd = c(2, 2, 2))





speed <- data.frame(speed)
result <- data.frame(result)

speed_oneday <- speed[90:126,1]
result_oneday <- result[90:126,1]

dataset <- read.csv("new_data.csv")
time <- dataset[630:666,3]




speed_graph <- data.frame(
  data_date = time,
  speed = speed_oneday
)

result_graph <- data.frame(
  data_date = time,
  speed = result_oneday
)

ggplot() + 
  geom_point(data = speed_graph, aes(x = data_date, y = speed, group = 1), color = "red",stat='summary', fun.y=sum) +
  stat_summary(fun.y=sum, geom="line")+
  #geom_point(data = result_graph, aes(x = data_date, y = speed, group = 2), color = "blue") +
  xlab('time') +
  ylab('speed(km/h)')



