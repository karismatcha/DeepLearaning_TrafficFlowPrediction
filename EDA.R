library(dplyr)
library(ggplot2)
dataset <- read.csv("new_data.csv")
dataset <- dataset %>%  select(Recordtime,CarsTotal,SmoSpeed003,CarsSVo,CarsTVo,CarsLVo,Week,Holiday) 
colnames(dataset)[3] <- "CarsSpeed"
dataset <- dataset %>% filter(CarsSpeed != -99,CarsTotal != -99, CarsSVo != -99, CarsTVo != -99,CarsLVo != -99)
dataset <- data.frame(dataset)

#####################################################################################
#hist of CarsTotal
hist(dataset$CarsTotal, # histogram
     col = "peachpuff", # column color
     border = "black", 
     prob = TRUE, # show densities instead of frequencies
     xlab = "CarsTotal",
     main = "Number of Cars")
lines(density(dataset$CarsTotal), # density plot
      lwd = 2, # thickness of line
      col = "chocolate3")


abline(v = mean(dataset$CarsTotal),
       col = "royalblue",
       lwd = 2)

abline(v = median(dataset$CarsTotal),
       col = "red",
       lwd = 2)
legend(x = "topright", # location of legend within plot area
       c("Density plot", "Mean", "Median"),
       col = c("chocolate3", "royalblue", "red"),
       lwd = c(2, 2, 2))


#hist of CarsSpeed
hist(dataset$CarsSpeed, # histogram
     col = "peachpuff", # column color
     border = "black", 
     prob = TRUE, # show densities instead of frequencies
     xlab = "CarsSpeed",
     main = "Speed (km/h)")
lines(density(dataset$CarsSpeed), # density plot
      lwd = 2, # thickness of line
      col = "chocolate3")


abline(v = mean(dataset$CarsSpeed),
       col = "royalblue",
       lwd = 2)

abline(v = median(dataset$CarsSpeed),
       col = "red",
       lwd = 2)
legend(x = "topright", # location of legend within plot area
       c("Density plot", "Mean", "Median"),
       col = c("chocolate3", "royalblue", "red"),
       lwd = c(2, 2, 2))


#hist of CarsSVo
hist(dataset$CarsSVo, # histogram
     col = "peachpuff", # column color
     border = "black", 
     prob = TRUE, # show densities instead of frequencies
     xlab = "CarssSVo",
     main = "Number of Small Cars")
lines(density(dataset$CarsSVo), # density plot
      lwd = 2, # thickness of line
      col = "chocolate3")


abline(v = mean(dataset$CarsSVo),
       col = "royalblue",
       lwd = 2)

abline(v = median(dataset$CarsSVo),
       col = "red",
       lwd = 2)
legend(x = "topright", # location of legend within plot area
       c("Density plot", "Mean", "Median"),
       col = c("chocolate3", "royalblue", "red"),
       lwd = c(2, 2, 2))

#hist of CarsTVo
hist(dataset$CarsTVo, # histogram
     col = "peachpuff", # column color
     border = "black", 
     prob = TRUE, # show densities instead of frequencies
     xlab = "CarsTVo",
     main = "Number of Medium Cars")
lines(density(dataset$CarsTVo), # density plot
      lwd = 2, # thickness of line
      col = "chocolate3")


abline(v = mean(dataset$CarsTVo),
       col = "royalblue",
       lwd = 2)

abline(v = median(dataset$CarsTotal),
       col = "red",
       lwd = 2)
legend(x = "topright", # location of legend within plot area
       c("Density plot", "Mean", "Median"),
       col = c("chocolate3", "royalblue", "red"),
       lwd = c(2, 2, 2))

#hist of CarsLVo
hist(dataset$CarsLVo, # histogram
     col = "peachpuff", # column color
     border = "black", 
     prob = TRUE, # show densities instead of frequencies
     xlab = "CarsLVo",
     main = "Number of Large Cars")
lines(density(dataset$CarsLVo), # density plot
      lwd = 2, # thickness of line
      col = "chocolate3")


abline(v = mean(dataset$CarsLVo),
       col = "royalblue",
       lwd = 2)

abline(v = median(dataset$CarsLVo),
       col = "red",
       lwd = 2)
legend(x = "topright", # location of legend within plot area
       c("Density plot", "Mean", "Median"),
       col = c("chocolate3", "royalblue", "red"),
       lwd = c(2, 2, 2))

#####################################################################################
#Recortime and CarsSpeed
scatter_plot <- ggplot(dataset,
       aes(x=Recordtime,
           y=CarsSpeed,
           color = -CarsSpeed))+
  geom_point()
scatter_plot + geom_point() + labs(x = "Time (a.m.) ", y = "Speed (km/h)")
  
#Recortime and CarsTotal
scatter_plot <- ggplot(dataset,
       aes(x=Recordtime,
           y=CarsTotal,
           color = -CarsTotal))+
  geom_point()
scatter_plot + geom_point() + labs(x = "Time (a.m.) ", y = "Number of Cars")

#CarsSpeed and CarsTotal
scatter_plot <- ggplot(dataset,
       aes(x=CarsTotal,
           y=CarsSpeed,
           color = -CarsSpeed))+
  geom_point()
scatter_plot + geom_point() + labs(x = "Number of Cars", y = "Speed (km/h)") + 
  geom_smooth(method='lm',formula=y~x)

#Week and CarsSpeed
scatter_plot <- ggplot(dataset,
       aes(x=Week,
           y=CarsSpeed,
           color = -CarsSpeed))+
  geom_point() 
scatter_plot + geom_point() + labs(x = "Days ", y = "Speed (km/h)")

#Week and CarsTotal
scatter_plot <- ggplot(dataset,
       aes(x=Week,
           y=CarsTotal,
           color = -CarsTotal))+
  geom_point()
scatter_plot + geom_point() + labs(x = "Days ", y = "Number of Cars")

#Holiday and CarsSpeed
scatter_plot <- ggplot(dataset,
       aes(x=Holiday,
           y=CarsSpeed,
           color = -CarsSpeed))+
  geom_point()
scatter_plot + geom_point() + labs(x = "Holiday", y = "Speed (km/h)")

#Holiday and CarsTotal
scatter_plot <- ggplot(dataset,
       aes(x=Holiday,
           y=CarsTotal,
           color = -CarsTotal))+
  geom_point()
scatter_plot + geom_point() + labs(x = "Holiday", y = "Number of Cars")
#####################################################################################
hol_1 <- dataset %>% filter(Holiday == 1)
hol_0 <- dataset %>% filter(Holiday == 0)

boxplot(hol_0$CarsTotal)
title(main="Boxplot of CarsTotal in weekdays",ylab="Number of Cars")
boxplot(hol_0$CarsSpeed)
title(main="Boxplot of CarsSpeed in weekdays",ylab="Speed")
boxplot(hol_1$CarsTotal)
title(main="Boxplot of CarsTotal in holidays",ylab="Number of Cars")
boxplot(hol_1$CarsSpeed)
title(main="Boxplot of CarsSpeed in holidays",ylab="Speeds")

#delete RecordTime
new_dataset <- dataset %>% within(rm(Recordtime))


cormat <- round(cor(new_dataset),2)
# Get lower triangle of the correlation matrix
get_lower_tri<-function(cormat){
  cormat[upper.tri(cormat)] <- NA
  return(cormat)
}
# Get upper triangle of the correlation matrix
get_upper_tri <- function(cormat){
  cormat[lower.tri(cormat)]<- NA
  return(cormat)
}
upper_tri <- get_upper_tri(cormat)
# Melt the correlation matrix
library(reshape2)
melted_cormat <- melt(upper_tri, na.rm = TRUE)

#normal ggplot
ggplot(data = melted_cormat, aes(x=Var2, y=Var1, fill=value)) + 
  geom_tile()

#Heatmap
reorder_cormat <- function(cormat){
  # Use correlation between variables as distance
  dd <- as.dist((1-cormat)/2)
  hc <- hclust(dd)
  cormat <-cormat[hc$order, hc$order]
}
# Reorder the correlation matrix
cormat <- reorder_cormat(cormat)
upper_tri <- get_upper_tri(cormat)
# Melt the correlation matrix
melted_cormat <- melt(upper_tri, na.rm = TRUE)
# Create a ggheatmap
ggheatmap <- ggplot(melted_cormat, aes(Var2, Var1, fill = value))+
  geom_tile(color = "white")+
  scale_fill_gradient2(low = "blue", high = "red", mid = "white", 
                       midpoint = 0, limit = c(-1,1), space = "Lab", 
                       name="Pearson\nCorrelation") +
  theme_minimal()+ # minimal theme
  theme(axis.text.x = element_text(angle = 45, vjust = 1, 
                                   size = 12, hjust = 1))+
  coord_fixed()
ggheatmap + 
  geom_text(aes(Var2, Var1, label = value), color = "black", size = 4) +
  theme(
    axis.title.x = element_blank(),
    axis.title.y = element_blank(),
    panel.grid.major = element_blank(),
    panel.border = element_blank(),
    panel.background = element_blank(),
    axis.ticks = element_blank(),
    legend.justification = c(1, 0),
    legend.position = c(0.6, 0.7),
    legend.direction = "horizontal")+
  guides(fill = guide_colorbar(barwidth = 7, barheight = 1,
                               title.position = "top", title.hjust = 0.5))

