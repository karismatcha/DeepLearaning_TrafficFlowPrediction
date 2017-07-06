library(dplyr)
dataset <- read.csv("clustering.csv")
dataset <- dataset %>%  select(Recordtime,CarsTotal,CarsSpeed,CarsSVo,CarsTVo,CarsLVo,Week,Holiday) 
dataset <- data.frame(dataset)
i <- 0 


#filter(dataset,Recordtime == Recordtime[73])




#while(i<36) { filter(dataset,Recordtime == Recordtime[i+73]);i <- i+1}
#dataset <- dataset %>% filter(Recordtime == Recordtime[i+73])
dataset

