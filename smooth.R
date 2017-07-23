library(ggplot2)

read_data <- read.csv("for_smooth.csv")
read_speed <- read_data[6]
read_time <- read_data[20]


ggplot(read_speed, read_time, main="lowess(speed)")


scatter_plot <- ggplot(read_data,
                       aes(x=run,
                           y=CarsSpeed,
                           color = -CarsSpeed))+
  geom_point()
scatter_plot + geom_point() + labs(x = "run ", y = "Speed (km/h)")


lines(lowess(read_speed, read_time), col=2)
lines(lowess(read_speed, read_time, f=.2), col=3)
legend(5, 120, c(paste("f=", c("2/3", ".2"))), lty=1, col=2:3)