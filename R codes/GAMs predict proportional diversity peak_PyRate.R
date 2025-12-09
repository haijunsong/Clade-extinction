

# Load the necessary packages
library(mgcv)

######################################################################################################
# Ammonoidea
# Read data (no header)
file_path <- "F:/1-投稿/2022-clades灭绝/Pyrate/12 clades 2025-11/output1myr/Ammonoidea vs. Nautiloidea,Coleoidea,Agnatha,placodermi,chondrichthyes,Actinopterygii,Coelacanthimorpha,Dipnoi,Dipnomorphadiversity_ratio_2025-11-13.csv"
raw_data <- read.csv(file_path, header = FALSE, stringsAsFactors = FALSE)

# Extract the first column as variable names
var_names <- raw_data[,1]
# Extract the rest as data
datamat <- as.matrix(raw_data[,-1])
# Convert to numeric
storage.mode(datamat) <- "numeric"

# Now, each row is a variable, columns are samples
# Turn it "long" so each variable是一列，每行是一个时间点
data_wide <- as.data.frame(t(datamat))
colnames(data_wide) <- var_names

# Now data_wide should have columns "age1", "mean_ratio", etc.
str(data_wide)
head(data_wide)

# Now do GAM
gam_model <- gam(mean_ratio ~ s(age1), data = data_wide)

# Predict fitted values at the original data points
data_wide$ratio_gam_fit <- predict(gam_model, newdata = data_wide)

# Export
write.csv(data_wide, file = "F:/1-投稿/2022-clades灭绝/Pyrate/12 clades 2025-11/output1myr/Ammonoidea_ratio_with_gamfit_PyRate.csv", row.names = FALSE)

# Sequence for smooth plot
new_age <- seq(min(data_wide$age1, na.rm = TRUE), max(data_wide$age1, na.rm = TRUE), length.out = 200)
new_data <- data.frame(age1 = new_age)
new_data$ratio_gam_fit <- predict(gam_model, newdata = new_data)

# Plot
par(mfrow=c(3,4), mar=c(4,4,2,1)) # 12 clades

plot(data_wide$age1, data_wide$mean_ratio, pch=16, col="grey", xlim=c(540,0),  xlab="Age (Ma)", ylab="Proportional diversity", main="Ammonoidea", ylim=c(0,1), type="n")
arrows(
  x0 = data_wide$age1,
  y0 = data_wide$hpd95lower,
  x1 = data_wide$age1,
  y1 = data_wide$hpd95higher,
  angle = 90,
  code = 3,
  length = 0.03,
  col = "lightgrey"
)
points(data_wide$age1, data_wide$mean_ratio, pch=16, col="dimgrey")

lines(new_data$age1, new_data$ratio_gam_fit, col="blue", lwd=2)

######################################################################################################
# Camerata
# Read data (no header)
file_path <- "F:/1-投稿/2022-clades灭绝/Pyrate/12 clades 2025-11/output1myr/Camerata vs. Pentacrinoideadiversity_ratio_2025-11-13.csv"
raw_data <- read.csv(file_path, header = FALSE, stringsAsFactors = FALSE)

# Extract the first column as variable names
var_names <- raw_data[,1]
# Extract the rest as data
datamat <- as.matrix(raw_data[,-1])
# Convert to numeric
storage.mode(datamat) <- "numeric"

# Now, each row is a variable, columns are samples
# Turn it "long" so each variable是一列，每行是一个时间点
data_wide <- as.data.frame(t(datamat))
colnames(data_wide) <- var_names

# Now data_wide should have columns "age1", "mean_ratio", etc.
str(data_wide)
head(data_wide)

# Now do GAM
gam_model <- gam(mean_ratio ~ s(age1), data = data_wide)

# Predict fitted values at the original data points
data_wide$ratio_gam_fit <- predict(gam_model, newdata = data_wide)

# Export
write.csv(data_wide, file = "F:/1-投稿/2022-clades灭绝/Pyrate/12 clades 2025-11/output1myr/Camerata_ratio_with_gamfit_PyRate.csv", row.names = FALSE)

# Sequence for smooth plot
new_age <- seq(min(data_wide$age1, na.rm = TRUE), max(data_wide$age1, na.rm = TRUE), length.out = 200)
new_data <- data.frame(age1 = new_age)
new_data$ratio_gam_fit <- predict(gam_model, newdata = new_data)

# Plot
plot(data_wide$age1, data_wide$mean_ratio, pch=16, col="grey", xlim=c(540,0), xlab="Age (Ma)", ylab="Proportional diversity", main="Camerata", ylim=c(0,1), type="n")
arrows(
  x0 = data_wide$age1,
  y0 = data_wide$hpd95lower,
  x1 = data_wide$age1,
  y1 = data_wide$hpd95higher,
  angle = 90,
  code = 3,
  length = 0.03,
  col = "lightgrey"
)
points(data_wide$age1, data_wide$mean_ratio, pch=16, col="dimgrey")
lines(new_data$age1, new_data$ratio_gam_fit, col="blue", lwd=2)

######################################################################################################

# Conodonta

# Read data (no header)
file_path <- "F:/1-投稿/2022-clades灭绝/Pyrate/12 clades 2025-11/output1myr/Conodonta vs. Agnatha,placodermi,chondrichthyes,Actinopterygii,Coelacanthimorpha,Dipnoi,Dipnomorphadiversity_ratio_2025-11-13.csv"
raw_data <- read.csv(file_path, header = FALSE, stringsAsFactors = FALSE)

# Extract the first column as variable names
var_names <- raw_data[,1]
# Extract the rest as data
datamat <- as.matrix(raw_data[,-1])
# Convert to numeric
storage.mode(datamat) <- "numeric"

# Now, each row is a variable, columns are samples
# Turn it "long" so each variable是一列，每行是一个时间点
data_wide <- as.data.frame(t(datamat))
colnames(data_wide) <- var_names

# Now data_wide should have columns "age1", "mean_ratio", etc.
str(data_wide)
head(data_wide)

# Now do GAM
gam_model <- gam(mean_ratio ~ s(age1), data = data_wide)

# Predict fitted values at the original data points
data_wide$ratio_gam_fit <- predict(gam_model, newdata = data_wide)

# Export
write.csv(data_wide, file = "F:/1-投稿/2022-clades灭绝/Pyrate/12 clades 2025-11/output1myr/Conodonta_ratio_with_gamfit_PyRate.csv", row.names = FALSE)

# Sequence for smooth plot
new_age <- seq(min(data_wide$age1, na.rm = TRUE), max(data_wide$age1, na.rm = TRUE), length.out = 200)
new_data <- data.frame(age1 = new_age)
new_data$ratio_gam_fit <- predict(gam_model, newdata = new_data)

# Plot
plot(data_wide$age1, data_wide$mean_ratio, pch=16, col="grey", xlim=c(540,0),, xlab="Age (Ma)", ylab="Proportional diversity", main="Conodonta", ylim=c(0,1), type="n")
arrows(
  x0 = data_wide$age1,
  y0 = data_wide$hpd95lower,
  x1 = data_wide$age1,
  y1 = data_wide$hpd95higher,
  angle = 90,
  code = 3,
  length = 0.03,
  col = "lightgrey"
)
points(data_wide$age1, data_wide$mean_ratio, pch=16, col="dimgrey")
lines(new_data$age1, new_data$ratio_gam_fit, col="blue", lwd=2)

######################################################################################################
# Fenestrida
# Read data (no header)
file_path <- "F:/1-投稿/2022-clades灭绝/Pyrate/12 clades 2025-11/output1myr/Fenestrida vs. Rugosa,Tabulata,Porifera,Cryptostomata,Trepostomata,Cystoporata,Esthonioporata,cyclostomatadiversity_ratio_2025-11-13.csv"
raw_data <- read.csv(file_path, header = FALSE, stringsAsFactors = FALSE)

# Extract the first column as variable names
var_names <- raw_data[,1]
# Extract the rest as data
datamat <- as.matrix(raw_data[,-1])
# Convert to numeric
storage.mode(datamat) <- "numeric"

# Now, each row is a variable, columns are samples
# Turn it "long" so each variable是一列，每行是一个时间点
data_wide <- as.data.frame(t(datamat))
colnames(data_wide) <- var_names

# Now data_wide should have columns "age1", "mean_ratio", etc.
str(data_wide)
head(data_wide)

# Now do GAM
gam_model <- gam(mean_ratio ~ s(age1), data = data_wide)

# Predict fitted values at the original data points
data_wide$ratio_gam_fit <- predict(gam_model, newdata = data_wide)

# Export
write.csv(data_wide, file = "F:/1-投稿/2022-clades灭绝/Pyrate/12 clades 2025-11/output1myr/Fenestrida_ratio_with_gamfit_PyRate.csv", row.names = FALSE)

# Sequence for smooth plot
new_age <- seq(min(data_wide$age1, na.rm = TRUE), max(data_wide$age1, na.rm = TRUE), length.out = 200)
new_data <- data.frame(age1 = new_age)
new_data$ratio_gam_fit <- predict(gam_model, newdata = new_data)

# Plot
plot(data_wide$age1, data_wide$mean_ratio, pch=16, col="grey", xlim=c(540,0),, xlab="Age (Ma)", ylab="Proportional diversity", main="Fenestrida", ylim=c(0,0.6), type="n")
arrows(
  x0 = data_wide$age1,
  y0 = data_wide$hpd95lower,
  x1 = data_wide$age1,
  y1 = data_wide$hpd95higher,
  angle = 90,
  code = 3,
  length = 0.03,
  col = "lightgrey"
)
points(data_wide$age1, data_wide$mean_ratio, pch=16, col="dimgrey")
lines(new_data$age1, new_data$ratio_gam_fit, col="blue", lwd=2)



######################################################################################################
# Fusulinoidea
# Read data (no header)
file_path <- "F:/1-投稿/2022-clades灭绝/Pyrate/12 clades 2025-11/output1myr/Fusulinoidea vs. othersdiversity_ratio_2025-11-13.csv"
raw_data <- read.csv(file_path, header = FALSE, stringsAsFactors = FALSE)

# Extract the first column as variable names
var_names <- raw_data[,1]
# Extract the rest as data
datamat <- as.matrix(raw_data[,-1])
# Convert to numeric
storage.mode(datamat) <- "numeric"

# Now, each row is a variable, columns are samples
# Turn it "long" so each variable是一列，每行是一个时间点
data_wide <- as.data.frame(t(datamat))
colnames(data_wide) <- var_names

# Now data_wide should have columns "age1", "mean_ratio", etc.
str(data_wide)
head(data_wide)

# Now do GAM
gam_model <- gam(mean_ratio ~ s(age1), data = data_wide)

# Predict fitted values at the original data points
data_wide$ratio_gam_fit <- predict(gam_model, newdata = data_wide)

# Export
write.csv(data_wide, file = "F:/1-投稿/2022-clades灭绝/Pyrate/12 clades 2025-11/output1myr/Fusulinoidea_ratio_with_gamfit_PyRate.csv", row.names = FALSE)

# Sequence for smooth plot
new_age <- seq(min(data_wide$age1, na.rm = TRUE), max(data_wide$age1, na.rm = TRUE), length.out = 200)
new_data <- data.frame(age1 = new_age)
new_data$ratio_gam_fit <- predict(gam_model, newdata = new_data)

# Plot
plot(data_wide$age1, data_wide$mean_ratio, pch=16, col="grey", xlim=c(540,0), xlab="Age (Ma)", ylab="Proportional diversity", main="Fusulinoidea", ylim=c(0,0.8), type="n")
arrows(
  x0 = data_wide$age1,
  y0 = data_wide$hpd95lower,
  x1 = data_wide$age1,
  y1 = data_wide$hpd95higher,
  angle = 90,
  code = 3,
  length = 0.03,
  col = "lightgrey"
)
points(data_wide$age1, data_wide$mean_ratio, pch=16, col="dimgrey")
lines(new_data$age1, new_data$ratio_gam_fit, col="blue", lwd=2)


######################################################################################################

# Graptoloidea
# Read data (no header)
file_path <- "F:/1-投稿/2022-clades灭绝/Pyrate/12 clades 2025-11/output1myr/Graptoloidea vs. Tentaculitadiversity_ratio_2025-11-13.csv"
raw_data <- read.csv(file_path, header = FALSE, stringsAsFactors = FALSE)

# Extract the first column as variable names
var_names <- raw_data[,1]
# Extract the rest as data
datamat <- as.matrix(raw_data[,-1])
# Convert to numeric
storage.mode(datamat) <- "numeric"

# Now, each row is a variable, columns are samples
# Turn it "long" so each variable是一列，每行是一个时间点
data_wide <- as.data.frame(t(datamat))
colnames(data_wide) <- var_names

# Now data_wide should have columns "age1", "mean_ratio", etc.
str(data_wide)
head(data_wide)

# Now do GAM
gam_model <- gam(mean_ratio ~ s(age1), data = data_wide)

# Predict fitted values at the original data points
data_wide$ratio_gam_fit <- predict(gam_model, newdata = data_wide)

# Export
write.csv(data_wide, file = "F:/1-投稿/2022-clades灭绝/Pyrate/12 clades 2025-11/output1myr/Graptoloidea_ratio_with_gamfit_PyRate.csv", row.names = FALSE)

# Sequence for smooth plot
new_age <- seq(min(data_wide$age1, na.rm = TRUE), max(data_wide$age1, na.rm = TRUE), length.out = 200)
new_data <- data.frame(age1 = new_age)
new_data$ratio_gam_fit <- predict(gam_model, newdata = new_data)

# Plot
plot(data_wide$age1, data_wide$mean_ratio, pch=16, col="grey", xlim=c(540,0),, xlab="Age (Ma)", ylab="Proportional diversity", main="Graptoloidea", ylim=c(0,1), type="n")
arrows(
  x0 = data_wide$age1,
  y0 = data_wide$hpd95lower,
  x1 = data_wide$age1,
  y1 = data_wide$hpd95higher,
  angle = 90,
  code = 3,
  length = 0.03,
  col = "lightgrey"
)
points(data_wide$age1, data_wide$mean_ratio, pch=16, col="dimgrey")
lines(new_data$age1, new_data$ratio_gam_fit, col="blue", lwd=2)


######################################################################################################
# Orthoceratoidea
# Read data (no header)
file_path <- "F:/1-投稿/2022-clades灭绝/Pyrate/12 clades 2025-11/output1myr/Orthoceratoidea vs. Nautiloideadiversity_ratio_2025-11-13.csv"
raw_data <- read.csv(file_path, header = FALSE, stringsAsFactors = FALSE)

# Extract the first column as variable names
var_names <- raw_data[,1]
# Extract the rest as data
datamat <- as.matrix(raw_data[,-1])
# Convert to numeric
storage.mode(datamat) <- "numeric"

# Now, each row is a variable, columns are samples
# Turn it "long" so each variable是一列，每行是一个时间点
data_wide <- as.data.frame(t(datamat))
colnames(data_wide) <- var_names

# Now data_wide should have columns "age1", "mean_ratio", etc.
str(data_wide)
head(data_wide)

# Now do GAM
gam_model <- gam(mean_ratio ~ s(age1), data = data_wide)

# Predict fitted values at the original data points
data_wide$ratio_gam_fit <- predict(gam_model, newdata = data_wide)

# Export
write.csv(data_wide, file = "F:/1-投稿/2022-clades灭绝/Pyrate/12 clades 2025-11/output1myr/Orthoceratoidea_ratio_with_gamfit_PyRate.csv", row.names = FALSE)

# Sequence for smooth plot
new_age <- seq(min(data_wide$age1, na.rm = TRUE), max(data_wide$age1, na.rm = TRUE), length.out = 200)
new_data <- data.frame(age1 = new_age)
new_data$ratio_gam_fit <- predict(gam_model, newdata = new_data)

# Plot
plot(data_wide$age1, data_wide$mean_ratio, pch=16, col="grey", xlim=c(540,0),, xlab="Age (Ma)", ylab="Proportional diversity", main="Orthoceratoidea", ylim=c(0,1), type="n")
arrows(
  x0 = data_wide$age1,
  y0 = data_wide$hpd95lower,
  x1 = data_wide$age1,
  y1 = data_wide$hpd95higher,
  angle = 90,
  code = 3,
  length = 0.03,
  col = "lightgrey"
)
points(data_wide$age1, data_wide$mean_ratio, pch=16, col="dimgrey")
lines(new_data$age1, new_data$ratio_gam_fit, col="blue", lwd=2)

######################################################################################################

# Palaeocopida
# Read data (no header)
file_path <- "F:/1-投稿/2022-clades灭绝/Pyrate/12 clades 2025-11/output1myr/Palaeocopida vs. othersdiversity_ratio_2025-11-13.csv"
raw_data <- read.csv(file_path, header = FALSE, stringsAsFactors = FALSE)

# Extract the first column as variable names
var_names <- raw_data[,1]
# Extract the rest as data
datamat <- as.matrix(raw_data[,-1])
# Convert to numeric
storage.mode(datamat) <- "numeric"

# Now, each row is a variable, columns are samples
# Turn it "long" so each variable是一列，每行是一个时间点
data_wide <- as.data.frame(t(datamat))
colnames(data_wide) <- var_names

# Now data_wide should have columns "age1", "mean_ratio", etc.
str(data_wide)
head(data_wide)

# Now do GAM
gam_model <- gam(mean_ratio ~ s(age1), data = data_wide)

# Predict fitted values at the original data points
data_wide$ratio_gam_fit <- predict(gam_model, newdata = data_wide)

# Export
write.csv(data_wide, file = "F:/1-投稿/2022-clades灭绝/Pyrate/12 clades 2025-11/output1myr/Palaeocopida_ratio_with_gamfit_PyRate.csv", row.names = FALSE)

# Sequence for smooth plot
new_age <- seq(min(data_wide$age1, na.rm = TRUE), max(data_wide$age1, na.rm = TRUE), length.out = 200)
new_data <- data.frame(age1 = new_age)
new_data$ratio_gam_fit <- predict(gam_model, newdata = new_data)

# Plot
plot(data_wide$age1, data_wide$mean_ratio, pch=16, col="grey", xlim=c(540,0),, xlab="Age (Ma)", ylab="Proportional diversity", main="Palaeocopida", ylim=c(0,1), type="n")
arrows(
  x0 = data_wide$age1,
  y0 = data_wide$hpd95lower,
  x1 = data_wide$age1,
  y1 = data_wide$hpd95higher,
  angle = 90,
  code = 3,
  length = 0.03,
  col = "lightgrey"
)
points(data_wide$age1, data_wide$mean_ratio, pch=16, col="dimgrey")
lines(new_data$age1, new_data$ratio_gam_fit, col="blue", lwd=2)

######################################################################################################

# Rugosa
# Read data (no header)
file_path <- "F:/1-投稿/2022-clades灭绝/Pyrate/12 clades 2025-11/output1myr/Rugosa vs. Porifera,Bryozoadiversity_ratio_2025-11-13.csv"
raw_data <- read.csv(file_path, header = FALSE, stringsAsFactors = FALSE)

# Extract the first column as variable names
var_names <- raw_data[,1]
# Extract the rest as data
datamat <- as.matrix(raw_data[,-1])
# Convert to numeric
storage.mode(datamat) <- "numeric"

# Now, each row is a variable, columns are samples
# Turn it "long" so each variable是一列，每行是一个时间点
data_wide <- as.data.frame(t(datamat))
colnames(data_wide) <- var_names

# Now data_wide should have columns "age1", "mean_ratio", etc.
str(data_wide)
head(data_wide)

# Now do GAM
gam_model <- gam(mean_ratio ~ s(age1), data = data_wide)

# Predict fitted values at the original data points
data_wide$ratio_gam_fit <- predict(gam_model, newdata = data_wide)

# Export
write.csv(data_wide, file = "F:/1-投稿/2022-clades灭绝/Pyrate/12 clades 2025-11/output1myr/Rugosa_ratio_with_gamfit_PyRate.csv", row.names = FALSE)

# Sequence for smooth plot
new_age <- seq(min(data_wide$age1, na.rm = TRUE), max(data_wide$age1, na.rm = TRUE), length.out = 200)
new_data <- data.frame(age1 = new_age)
new_data$ratio_gam_fit <- predict(gam_model, newdata = new_data)

# Plot
plot(data_wide$age1, data_wide$mean_ratio, pch=16, col="grey", xlim=c(540,0),, xlab="Age (Ma)", ylab="Proportional diversity", main="Rugosa", ylim=c(0,1), type="n")
arrows(
  x0 = data_wide$age1,
  y0 = data_wide$hpd95lower,
  x1 = data_wide$age1,
  y1 = data_wide$hpd95higher,
  angle = 90,
  code = 3,
  length = 0.03,
  col = "lightgrey"
)
points(data_wide$age1, data_wide$mean_ratio, pch=16, col="dimgrey")
lines(new_data$age1, new_data$ratio_gam_fit, col="blue", lwd=2)

######################################################################################################

# Spiriferinida
# Read data (no header)
file_path <- "F:/1-投稿/2022-clades灭绝/Pyrate/12 clades 2025-11/output1myr/Spiriferinida vs. othersdiversity_ratio_2025-11-13.csv"
raw_data <- read.csv(file_path, header = FALSE, stringsAsFactors = FALSE)

# Extract the first column as variable names
var_names <- raw_data[,1]
# Extract the rest as data
datamat <- as.matrix(raw_data[,-1])
# Convert to numeric
storage.mode(datamat) <- "numeric"

# Now, each row is a variable, columns are samples
# Turn it "long" so each variable是一列，每行是一个时间点
data_wide <- as.data.frame(t(datamat))
colnames(data_wide) <- var_names

# Now data_wide should have columns "age1", "mean_ratio", etc.
str(data_wide)
head(data_wide)

# Now do GAM
gam_model <- gam(mean_ratio ~ s(age1), data = data_wide)

# Predict fitted values at the original data points
data_wide$ratio_gam_fit <- predict(gam_model, newdata = data_wide)

# Export
write.csv(data_wide, file = "F:/1-投稿/2022-clades灭绝/Pyrate/12 clades 2025-11/output1myr/Spiriferinida_ratio_with_gamfit_PyRate.csv", row.names = FALSE)

# Sequence for smooth plot
new_age <- seq(min(data_wide$age1, na.rm = TRUE), max(data_wide$age1, na.rm = TRUE), length.out = 200)
new_data <- data.frame(age1 = new_age)
new_data$ratio_gam_fit <- predict(gam_model, newdata = new_data)

# Plot
plot(data_wide$age1, data_wide$mean_ratio, pch=16, col="grey", xlim=c(540,0),, xlab="Age (Ma)", ylab="Proportional diversity", main="Spiriferinida", ylim=c(0,0.6), type="n")
arrows(
  x0 = data_wide$age1,
  y0 = data_wide$hpd95lower,
  x1 = data_wide$age1,
  y1 = data_wide$hpd95higher,
  angle = 90,
  code = 3,
  length = 0.03,
  col = "lightgrey"
)
points(data_wide$age1, data_wide$mean_ratio, pch=16, col="dimgrey")
lines(new_data$age1, new_data$ratio_gam_fit, col="blue", lwd=2)

######################################################################################################

# Tabulata
# Read data (no header)
file_path <- "F:/1-投稿/2022-clades灭绝/Pyrate/12 clades 2025-11/output1myr/Tabulata vs. Porifera,Bryozoadiversity_ratio_2025-11-13.csv"
raw_data <- read.csv(file_path, header = FALSE, stringsAsFactors = FALSE)

# Extract the first column as variable names
var_names <- raw_data[,1]
# Extract the rest as data
datamat <- as.matrix(raw_data[,-1])
# Convert to numeric
storage.mode(datamat) <- "numeric"

# Now, each row is a variable, columns are samples
# Turn it "long" so each variable是一列，每行是一个时间点
data_wide <- as.data.frame(t(datamat))
colnames(data_wide) <- var_names

# Now data_wide should have columns "age1", "mean_ratio", etc.
str(data_wide)
head(data_wide)

# Now do GAM
gam_model <- gam(mean_ratio ~ s(age1), data = data_wide)

# Predict fitted values at the original data points
data_wide$ratio_gam_fit <- predict(gam_model, newdata = data_wide)

# Export
write.csv(data_wide, file = "F:/1-投稿/2022-clades灭绝/Pyrate/12 clades 2025-11/output1myr/Tabulata_ratio_with_gamfit_PyRate.csv", row.names = FALSE)

# Sequence for smooth plot
new_age <- seq(min(data_wide$age1, na.rm = TRUE), max(data_wide$age1, na.rm = TRUE), length.out = 200)
new_data <- data.frame(age1 = new_age)
new_data$ratio_gam_fit <- predict(gam_model, newdata = new_data)

# Plot
plot(data_wide$age1, data_wide$mean_ratio, pch=16, col="grey", xlim=c(540,0),, xlab="Age (Ma)", ylab="Proportional diversity", main="Tabulata", ylim=c(0,0.8), type="n")
arrows(
  x0 = data_wide$age1,
  y0 = data_wide$hpd95lower,
  x1 = data_wide$age1,
  y1 = data_wide$hpd95higher,
  angle = 90,
  code = 3,
  length = 0.03,
  col = "lightgrey"
)
points(data_wide$age1, data_wide$mean_ratio, pch=16, col="dimgrey")
lines(new_data$age1, new_data$ratio_gam_fit, col="blue", lwd=2)

######################################################################################################

# Trilobita
# Read data (no header)
file_path <- "F:/1-投稿/2022-clades灭绝/Pyrate/12 clades 2025-11/output1myr/Trilobita vs. Chelicerata,Pancrustaceadiversity_ratio_2025-11-13.csv"
raw_data <- read.csv(file_path, header = FALSE, stringsAsFactors = FALSE)

# Extract the first column as variable names
var_names <- raw_data[,1]
# Extract the rest as data
datamat <- as.matrix(raw_data[,-1])
# Convert to numeric
storage.mode(datamat) <- "numeric"

# Now, each row is a variable, columns are samples
# Turn it "long" so each variable是一列，每行是一个时间点
data_wide <- as.data.frame(t(datamat))
colnames(data_wide) <- var_names

# Now data_wide should have columns "age1", "mean_ratio", etc.
str(data_wide)
head(data_wide)

# Now do GAM
gam_model <- gam(mean_ratio ~ s(age1), data = data_wide)

# Predict fitted values at the original data points
data_wide$ratio_gam_fit <- predict(gam_model, newdata = data_wide)

# Export
write.csv(data_wide, file = "F:/1-投稿/2022-clades灭绝/Pyrate/12 clades 2025-11/output1myr/Trilobita_ratio_with_gamfit_PyRate.csv", row.names = FALSE)

# Sequence for smooth plot
new_age <- seq(min(data_wide$age1, na.rm = TRUE), max(data_wide$age1, na.rm = TRUE), length.out = 200)
new_data <- data.frame(age1 = new_age)
new_data$ratio_gam_fit <- predict(gam_model, newdata = new_data)

# Plot
plot(data_wide$age1, data_wide$mean_ratio, pch=16, col="grey", xlim=c(540,0),, xlab="Age (Ma)", ylab="Proportional diversity", main="Trilobita", ylim=c(0,1), type="n")
arrows(
  x0 = data_wide$age1,
  y0 = data_wide$hpd95lower,
  x1 = data_wide$age1,
  y1 = data_wide$hpd95higher,
  angle = 90,
  code = 3,
  length = 0.03,
  col = "lightgrey"
)
points(data_wide$age1, data_wide$mean_ratio, pch=16, col="dimgrey")
lines(new_data$age1, new_data$ratio_gam_fit, col="blue", lwd=2)



