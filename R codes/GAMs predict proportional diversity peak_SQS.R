

# Load the necessary packages
library(mgcv)

######################################################################################################
# Ammonoidea
# Read data
file_path <- "F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Data for GAMs predict proportional diversity peak/Ammonoidea_ratio_summary.csv"
df <- read.csv(file_path, stringsAsFactors = FALSE)

# View the data structure
str(df)
head(df)
# Fit the GAM model
gam_model <- gam(ratio_mean ~ s(age), data = df)

# Make predictions on the original data points to obtain the smoothed ratio_mean
df$ratio_gam_fit <- predict(gam_model, newdata = df)

# Export the results including the original data and the GAM fitted values
write.csv(df, file = "F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Data for GAMs predict proportional diversity peak/Ammonoidea_ratio_with_gamfit.csv", row.names = FALSE)
# Create an evenly spaced age sequence
new_age <- seq(min(df$age), max(df$age), length.out = 200)
new_data <- data.frame(age = new_age)

# Predicted fitted value
new_data$ratio_gam_fit <- predict(gam_model, newdata = new_data)

# Plot
par(mfrow=c(3,4), mar=c(4,4,2,1)) # 12 clades

plot(df$age, df$ratio_mean, pch=16, col="grey", xlab="Age (Ma)", ylab="Proportional diversity", main="Ammonoidea",
     xlim = rev(range(df$age)), ylim = c(0.2, 1))
arrows(
  x0 = df$age, 
  y0 = df$ratio_mean - df$ratio_ci, 
  x1 = df$age, 
  y1 = df$ratio_mean + df$ratio_ci,
  code = 3, angle = 90, length = 0.03, col = "grey"
)

lines(new_data$age, new_data$ratio_gam_fit, col="blue", lwd=2)

######################################################################################################
# Camerata
# Read data
file_path <- "F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Data for GAMs predict proportional diversity peak/Camerata_ratio_summary.csv"
df <- read.csv(file_path, stringsAsFactors = FALSE)

# View the data structure
str(df)
head(df)
# Fit the GAM model
gam_model <- gam(ratio_mean ~ s(age), data = df)

# Make predictions on the original data points to obtain the smoothed ratio_mean
df$ratio_gam_fit <- predict(gam_model, newdata = df)

# Export the results including the original data and the GAM fitted values
write.csv(df, file = "F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Data for GAMs predict proportional diversity peak/Camerata_ratio_with_gamfit.csv", row.names = FALSE)
# Create an evenly spaced age sequence
new_age <- seq(min(df$age), max(df$age), length.out = 200)
new_data <- data.frame(age = new_age)

# Predicted fitted value
new_data$ratio_gam_fit <- predict(gam_model, newdata = new_data)

# Plot
plot(df$age, df$ratio_mean, pch=16, col="grey", xlab="Age (Ma)", ylab="Proportional diversity", main="Camerata",
     xlim = rev(range(df$age)), ylim = c(0, 0.6))
arrows(
  x0 = df$age, 
  y0 = df$ratio_mean - df$ratio_ci, 
  x1 = df$age, 
  y1 = df$ratio_mean + df$ratio_ci,
  code = 3, angle = 90, length = 0.03, col = "grey"
)
lines(new_data$age, new_data$ratio_gam_fit, col="blue", lwd=2)

######################################################################################################

# Conodonta

# Read data
file_path <- "F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Data for GAMs predict proportional diversity peak/Conodonta_ratio_summary.csv"
df <- read.csv(file_path, stringsAsFactors = FALSE)

# View the data structure
str(df)
head(df)
# Fit the GAM model
gam_model <- gam(ratio_mean ~ s(age), data = df)

# Make predictions on the original data points to obtain the smoothed ratio_mean
df$ratio_gam_fit <- predict(gam_model, newdata = df)

# Export the results including the original data and the GAM fitted values
write.csv(df, file = "F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Data for GAMs predict proportional diversity peak/Conodonta_ratio_with_gamfit.csv", row.names = FALSE)
# Create an evenly spaced age sequence
new_age <- seq(min(df$age), max(df$age), length.out = 200)
new_data <- data.frame(age = new_age)

# Predicted fitted value
new_data$ratio_gam_fit <- predict(gam_model, newdata = new_data)

# Plot
plot(df$age, df$ratio_mean, pch=16, col="grey", xlab="Age (Ma)", ylab="Proportional diversity", main="Conodonta",
     xlim = rev(range(df$age)), ylim = c(0, 1.4))
arrows(
  x0 = df$age, 
  y0 = df$ratio_mean - df$ratio_ci, 
  x1 = df$age, 
  y1 = df$ratio_mean + df$ratio_ci,
  code = 3, angle = 90, length = 0.03, col = "grey"
)
lines(new_data$age, new_data$ratio_gam_fit, col="blue", lwd=2)

######################################################################################################
# Fenestrida
# Read data
file_path <- "F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Data for GAMs predict proportional diversity peak/Fenestrida_ratio_summary.csv"
df <- read.csv(file_path, stringsAsFactors = FALSE)

# View the data structure
str(df)
head(df)
# Fit the GAM model
gam_model <- gam(ratio_mean ~ s(age), data = df)

# Make predictions on the original data points to obtain the smoothed ratio_mean
df$ratio_gam_fit <- predict(gam_model, newdata = df)

# Export the results including the original data and the GAM fitted values
write.csv(df, file = "F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Data for GAMs predict proportional diversity peak/Fenestrida_ratio_with_gamfit.csv", row.names = FALSE)
# Create an evenly spaced age sequence
new_age <- seq(min(df$age), max(df$age), length.out = 200)
new_data <- data.frame(age = new_age)

# Predicted fitted value
new_data$ratio_gam_fit <- predict(gam_model, newdata = new_data)

# Plot
plot(df$age, df$ratio_mean, pch=16, col="grey", xlab="Age (Ma)", ylab="Proportional diversity", main="Fenestrida",
     xlim = rev(range(df$age)), ylim = c(0, 0.5))
arrows(
  x0 = df$age, 
  y0 = df$ratio_mean - df$ratio_ci, 
  x1 = df$age, 
  y1 = df$ratio_mean + df$ratio_ci,
  code = 3, angle = 90, length = 0.03, col = "grey"
)
lines(new_data$age, new_data$ratio_gam_fit, col="blue", lwd=2)



######################################################################################################
# Fusulinoidea
# Read data
file_path <- "F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Data for GAMs predict proportional diversity peak/Fusulinoidea_ratio_summary.csv"
df <- read.csv(file_path, stringsAsFactors = FALSE)

# View the data structure
str(df)
head(df)
# Fit the GAM model
gam_model <- gam(ratio_mean ~ s(age), data = df)

# Make predictions on the original data points to obtain the smoothed ratio_mean
df$ratio_gam_fit <- predict(gam_model, newdata = df)

# Export the results including the original data and the GAM fitted values
write.csv(df, file = "F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Data for GAMs predict proportional diversity peak/Fusulinoidea_ratio_with_gamfit.csv", row.names = FALSE)
# Create an evenly spaced age sequence
new_age <- seq(min(df$age), max(df$age), length.out = 200)
new_data <- data.frame(age = new_age)

# Predicted fitted value
new_data$ratio_gam_fit <- predict(gam_model, newdata = new_data)

# Plot
plot(df$age, df$ratio_mean, pch=16, col="grey", xlab="Age (Ma)", ylab="Proportional diversity", main="Fusulinoidea",
     xlim = rev(range(df$age)), ylim = c(0, 1))
arrows(
  x0 = df$age, 
  y0 = df$ratio_mean - df$ratio_ci, 
  x1 = df$age, 
  y1 = df$ratio_mean + df$ratio_ci,
  code = 3, angle = 90, length = 0.03, col = "grey"
)
lines(new_data$age, new_data$ratio_gam_fit, col="blue", lwd=2)


######################################################################################################

# Graptoloidea
# Read data
file_path <- "F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Data for GAMs predict proportional diversity peak/Graptoloidea_ratio_summary.csv"
df <- read.csv(file_path, stringsAsFactors = FALSE)

# View the data structure
str(df)
head(df)
# Fit the GAM model
gam_model <- gam(ratio_mean ~ s(age), data = df)

# Make predictions on the original data points to obtain the smoothed ratio_mean
df$ratio_gam_fit <- predict(gam_model, newdata = df)

# Export the results including the original data and the GAM fitted values
write.csv(df, file = "F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Data for GAMs predict proportional diversity peak/Graptoloidea_ratio_with_gamfit.csv", row.names = FALSE)
# Create an evenly spaced age sequence
new_age <- seq(min(df$age), max(df$age), length.out = 200)
new_data <- data.frame(age = new_age)

# Predicted fitted value
new_data$ratio_gam_fit <- predict(gam_model, newdata = new_data)

# Plot
plot(df$age, df$ratio_mean, pch=16, col="grey", xlab="Age (Ma)", ylab="Proportional diversity", main="Graptoloidea",
     xlim = rev(range(df$age)), ylim = c(0, 1.2))
arrows(
  x0 = df$age, 
  y0 = df$ratio_mean - df$ratio_ci, 
  x1 = df$age, 
  y1 = df$ratio_mean + df$ratio_ci,
  code = 3, angle = 90, length = 0.03, col = "grey"
)
lines(new_data$age, new_data$ratio_gam_fit, col="blue", lwd=2)


######################################################################################################

# Orthoceratoidea
# Read data
file_path <- "F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Data for GAMs predict proportional diversity peak/Orthoceratoidea_ratio_summary.csv"
df <- read.csv(file_path, stringsAsFactors = FALSE)

# View the data structure
str(df)
head(df)
# Fit the GAM model
gam_model <- gam(ratio_mean ~ s(age), data = df)

# Make predictions on the original data points to obtain the smoothed ratio_mean
df$ratio_gam_fit <- predict(gam_model, newdata = df)

# Export the results including the original data and the GAM fitted values
write.csv(df, file = "F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Data for GAMs predict proportional diversity peak/Orthoceratoidea_ratio_with_gamfit.csv", row.names = FALSE)
# Create an evenly spaced age sequence
new_age <- seq(min(df$age), max(df$age), length.out = 200)
new_data <- data.frame(age = new_age)

# Predicted fitted value
new_data$ratio_gam_fit <- predict(gam_model, newdata = new_data)

# Plot
plot(df$age, df$ratio_mean, pch=16, col="grey", xlab="Age (Ma)", ylab="Proportional diversity", main="Orthoceratoidea",
     xlim = rev(range(df$age)), ylim = c(0, 1.6))
arrows(
  x0 = df$age, 
  y0 = df$ratio_mean - df$ratio_ci, 
  x1 = df$age, 
  y1 = df$ratio_mean + df$ratio_ci,
  code = 3, angle = 90, length = 0.03, col = "grey"
)
lines(new_data$age, new_data$ratio_gam_fit, col="blue", lwd=2)

######################################################################################################

# Palaeocopida
# Read data
file_path <- "F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Data for GAMs predict proportional diversity peak/Palaeocopida_ratio_summary.csv"
df <- read.csv(file_path, stringsAsFactors = FALSE)

# View the data structure
str(df)
head(df)
# Fit the GAM model
gam_model <- gam(ratio_mean ~ s(age), data = df)

# Make predictions on the original data points to obtain the smoothed ratio_mean
df$ratio_gam_fit <- predict(gam_model, newdata = df)

# Export the results including the original data and the GAM fitted values
write.csv(df, file = "F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Data for GAMs predict proportional diversity peak/Palaeocopida_ratio_with_gamfit.csv", row.names = FALSE)
# Create an evenly spaced age sequence
new_age <- seq(min(df$age), max(df$age), length.out = 200)
new_data <- data.frame(age = new_age)

# Predicted fitted value
new_data$ratio_gam_fit <- predict(gam_model, newdata = new_data)

# Plot
plot(df$age, df$ratio_mean, pch=16, col="grey", xlab="Age (Ma)", ylab="Proportional diversity", main="Palaeocopida",
     xlim = rev(range(df$age)), ylim = c(0, 1.2))
arrows(
  x0 = df$age, 
  y0 = df$ratio_mean - df$ratio_ci, 
  x1 = df$age, 
  y1 = df$ratio_mean + df$ratio_ci,
  code = 3, angle = 90, length = 0.03, col = "grey"
)
lines(new_data$age, new_data$ratio_gam_fit, col="blue", lwd=2)

######################################################################################################

# Rugosa
# Read data
file_path <- "F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Data for GAMs predict proportional diversity peak/Rugosa_ratio_summary.csv"
df <- read.csv(file_path, stringsAsFactors = FALSE)

# View the data structure
str(df)
head(df)
# Fit the GAM model
gam_model <- gam(ratio_mean ~ s(age), data = df)

# Make predictions on the original data points to obtain the smoothed ratio_mean
df$ratio_gam_fit <- predict(gam_model, newdata = df)

# Export the results including the original data and the GAM fitted values
write.csv(df, file = "F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Data for GAMs predict proportional diversity peak/Rugosa_ratio_with_gamfit.csv", row.names = FALSE)
# Create an evenly spaced age sequence
new_age <- seq(min(df$age), max(df$age), length.out = 200)
new_data <- data.frame(age = new_age)

# Predicted fitted value
new_data$ratio_gam_fit <- predict(gam_model, newdata = new_data)

# Plot
plot(df$age, df$ratio_mean, pch=16, col="grey", xlab="Age (Ma)", ylab="Proportional diversity", main="Rugosa",
     xlim = rev(range(df$age)), ylim = c(0, 1))
arrows(
  x0 = df$age, 
  y0 = df$ratio_mean - df$ratio_ci, 
  x1 = df$age, 
  y1 = df$ratio_mean + df$ratio_ci,
  code = 3, angle = 90, length = 0.03, col = "grey"
)
lines(new_data$age, new_data$ratio_gam_fit, col="blue", lwd=2)

######################################################################################################

# Spiriferinida
# Read data
file_path <- "F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Data for GAMs predict proportional diversity peak/Spiriferinida_ratio_summary.csv"
df <- read.csv(file_path, stringsAsFactors = FALSE)

# View the data structure
str(df)
head(df)
# Fit the GAM model
gam_model <- gam(ratio_mean ~ s(age), data = df)

# Make predictions on the original data points to obtain the smoothed ratio_mean
df$ratio_gam_fit <- predict(gam_model, newdata = df)

# Export the results including the original data and the GAM fitted values
write.csv(df, file = "F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Data for GAMs predict proportional diversity peak/Spiriferinida_ratio_with_gamfit.csv", row.names = FALSE)
# Create an evenly spaced age sequence
new_age <- seq(min(df$age), max(df$age), length.out = 200)
new_data <- data.frame(age = new_age)

# Predicted fitted value
new_data$ratio_gam_fit <- predict(gam_model, newdata = new_data)

# Plot
plot(df$age, df$ratio_mean, pch=16, col="grey", xlab="Age (Ma)", ylab="Proportional diversity", main="Spiriferinida",
     xlim = rev(range(df$age)), ylim = c(0, 0.5))
arrows(
  x0 = df$age, 
  y0 = df$ratio_mean - df$ratio_ci, 
  x1 = df$age, 
  y1 = df$ratio_mean + df$ratio_ci,
  code = 3, angle = 90, length = 0.03, col = "grey"
)
lines(new_data$age, new_data$ratio_gam_fit, col="blue", lwd=2)

######################################################################################################

# Tabulata
# Read data
file_path <- "F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Data for GAMs predict proportional diversity peak/Tabulata_ratio_summary.csv"
df <- read.csv(file_path, stringsAsFactors = FALSE)

# View the data structure
str(df)
head(df)
# Fit the GAM model
gam_model <- gam(ratio_mean ~ s(age), data = df)

# Make predictions on the original data points to obtain the smoothed ratio_mean
df$ratio_gam_fit <- predict(gam_model, newdata = df)

# Export the results including the original data and the GAM fitted values
write.csv(df, file = "F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Data for GAMs predict proportional diversity peak/Tabulata_ratio_with_gamfit.csv", row.names = FALSE)
# Create an evenly spaced age sequence
new_age <- seq(min(df$age), max(df$age), length.out = 200)
new_data <- data.frame(age = new_age)

# Predicted fitted value
new_data$ratio_gam_fit <- predict(gam_model, newdata = new_data)

# Plot
plot(df$age, df$ratio_mean, pch=16, col="grey", xlab="Age (Ma)", ylab="Proportional diversity", main="Tabulata",
     xlim = rev(range(df$age)), ylim = c(0, 0.5))
arrows(
  x0 = df$age, 
  y0 = df$ratio_mean - df$ratio_ci, 
  x1 = df$age, 
  y1 = df$ratio_mean + df$ratio_ci,
  code = 3, angle = 90, length = 0.03, col = "grey"
)
lines(new_data$age, new_data$ratio_gam_fit, col="blue", lwd=2)

######################################################################################################

# Trilobita
# Read data
file_path <- "F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Data for GAMs predict proportional diversity peak/Trilobita_ratio_summary.csv"
df <- read.csv(file_path, stringsAsFactors = FALSE)

# View the data structure
str(df)
head(df)
# Fit the GAM model
gam_model <- gam(ratio_mean ~ s(age), data = df)

# Make predictions on the original data points to obtain the smoothed ratio_mean
df$ratio_gam_fit <- predict(gam_model, newdata = df)

# Export the results including the original data and the GAM fitted values
write.csv(df, file = "F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Data for GAMs predict proportional diversity peak/Trilobita_ratio_with_gamfit.csv", row.names = FALSE)
# Create an evenly spaced age sequence
new_age <- seq(min(df$age), max(df$age), length.out = 200)
new_data <- data.frame(age = new_age)

# Predicted fitted value
new_data$ratio_gam_fit <- predict(gam_model, newdata = new_data)

# Plot
plot(df$age, df$ratio_mean, pch=16, col="grey", xlab="Age (Ma)", ylab="Proportional diversity", main="Trilobita",
     xlim = rev(range(df$age)), ylim = c(0, 1))
arrows(
  x0 = df$age, 
  y0 = df$ratio_mean - df$ratio_ci, 
  x1 = df$age, 
  y1 = df$ratio_mean + df$ratio_ci,
  code = 3, angle = 90, length = 0.03, col = "grey"
)
lines(new_data$age, new_data$ratio_gam_fit, col="blue", lwd=2)

