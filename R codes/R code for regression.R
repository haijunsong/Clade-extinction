
###############################################################################################
# Ammonoidea
# Read the data
Proportion_data <- read.csv("F:/1-Í¶¸å/2022-cladesÃð¾ø/Codes and data/Data and resuls for regression/Ammonoidea_ratio_summary_decine phase.csv", header = TRUE)

# Extract and order variables
age <- as.vector(Proportion_data$age)
ratio_mean <- as.vector(Proportion_data$ratio_mean)
ratio_ci <- as.vector(Proportion_data$ratio_ci)
ord <- order(-age)
age <- age[ord]
ratio_mean <- ratio_mean[ord]
ratio_ci <- ratio_ci[ord]

# Fit linear model and predict for a dense sequence (for smoother CI plotting)
age_seq <- seq(max(age), min(age), length.out = 500)
lm1 <- lm(ratio_mean ~ age, data = data.frame(age = age, ratio_mean = ratio_mean))
pred <- predict(lm1, newdata = data.frame(age = age_seq), interval = "confidence", level = 0.95)

# Basic plot, no points yet
par(mfrow=c(3,4), mar=c(4,4,2,1)) # 12 clades
plot(
  age, ratio_mean,
  type = "n",
  xlab = "Age",
  ylab = "Ratio Mean",
  xlim = c(530, 0),
  ylim = c(0, 1)
)

# 1. Draw confidence interval as a shaded polygon
polygon(
  c(age_seq, rev(age_seq)),
  c(pred[, 3], rev(pred[, 2])),
  col = rgb(0.7, 0.7, 0.7, 0.5),
  border = NA
)

# 2. Draw regression line
lines(age_seq, pred[, 1], col = "blue", lwd = 2)

# 3. Draw data points (in gray)
points(age, ratio_mean, pch = 19, cex = 0.9, col = "black")

# 4. Draw error bars (confidence intervals) on top
offset <- 3
for (i in seq_along(age)) {
  lower_limit <- ratio_mean[i] - ratio_ci[i]
  upper_limit <- ratio_mean[i] + ratio_ci[i]
  segments(x0 = age[i], y0 = lower_limit, x1 = age[i], y1 = upper_limit, col = "blue", lwd = 2)
  segments(x0 = age[i] - offset, y0 = lower_limit, x1 = age[i] + offset, y1 = lower_limit, col = "blue", lwd = 2)
  segments(x0 = age[i] - offset, y0 = upper_limit, x1 = age[i] + offset, y1 = upper_limit, col = "blue", lwd = 2)
}
###############################################################################################
# Camerata
# Read the data
Proportion_data <- read.csv("F:/1-Í¶¸å/2022-cladesÃð¾ø/Codes and data/Data and resuls for regression/Camerata_ratio_summary_decine phase.csv", header = TRUE)

# Extract and order variables
age <- as.vector(Proportion_data$age)
ratio_mean <- as.vector(Proportion_data$ratio_mean)
ratio_ci <- as.vector(Proportion_data$ratio_ci)
ord <- order(-age)
age <- age[ord]
ratio_mean <- ratio_mean[ord]
ratio_ci <- ratio_ci[ord]

# Fit linear model and predict for a dense sequence (for smoother CI plotting)
age_seq <- seq(max(age), min(age), length.out = 500)
lm1 <- lm(ratio_mean ~ age, data = data.frame(age = age, ratio_mean = ratio_mean))
pred <- predict(lm1, newdata = data.frame(age = age_seq), interval = "confidence", level = 0.95)

# Basic plot, no points yet
plot(
  age, ratio_mean,
  type = "n",
  xlab = "Age",
  ylab = "Ratio Mean",
  xlim = c(530, 0),
  ylim = c(0, 1)
)

# 1. Draw confidence interval as a shaded polygon
polygon(
  c(age_seq, rev(age_seq)),
  c(pred[, 3], rev(pred[, 2])),
  col = rgb(0.7, 0.7, 0.7, 0.5),
  border = NA
)

# 2. Draw regression line
lines(age_seq, pred[, 1], col = "blue", lwd = 2)

# 3. Draw data points (in gray)
points(age, ratio_mean, pch = 19, cex = 0.9, col = "black")

# 4. Draw error bars (confidence intervals) on top
offset <- 3
for (i in seq_along(age)) {
  lower_limit <- ratio_mean[i] - ratio_ci[i]
  upper_limit <- ratio_mean[i] + ratio_ci[i]
  segments(x0 = age[i], y0 = lower_limit, x1 = age[i], y1 = upper_limit, col = "blue", lwd = 2)
  segments(x0 = age[i] - offset, y0 = lower_limit, x1 = age[i] + offset, y1 = lower_limit, col = "blue", lwd = 2)
  segments(x0 = age[i] - offset, y0 = upper_limit, x1 = age[i] + offset, y1 = upper_limit, col = "blue", lwd = 2)
}
###############################################################################################
# Conodonta
# Read the data
Proportion_data <- read.csv("F:/1-Í¶¸å/2022-cladesÃð¾ø/Codes and data/Data and resuls for regression/Conodonta_ratio_summary_decine phase.csv", header = TRUE)

# Extract and order variables
age <- as.vector(Proportion_data$age)
ratio_mean <- as.vector(Proportion_data$ratio_mean)
ratio_ci <- as.vector(Proportion_data$ratio_ci)
ord <- order(-age)
age <- age[ord]
ratio_mean <- ratio_mean[ord]
ratio_ci <- ratio_ci[ord]

# Fit linear model and predict for a dense sequence (for smoother CI plotting)
age_seq <- seq(max(age), min(age), length.out = 500)
lm1 <- lm(ratio_mean ~ age, data = data.frame(age = age, ratio_mean = ratio_mean))
pred <- predict(lm1, newdata = data.frame(age = age_seq), interval = "confidence", level = 0.95)

# Basic plot, no points yet
plot(
  age, ratio_mean,
  type = "n",
  xlab = "Age",
  ylab = "Ratio Mean",
  xlim = c(530, 0),
  ylim = c(0, 1.2)
)

# 1. Draw confidence interval as a shaded polygon
polygon(
  c(age_seq, rev(age_seq)),
  c(pred[, 3], rev(pred[, 2])),
  col = rgb(0.7, 0.7, 0.7, 0.5),
  border = NA
)

# 2. Draw regression line
lines(age_seq, pred[, 1], col = "blue", lwd = 2)

# 3. Draw data points (in gray)
points(age, ratio_mean, pch = 19, cex = 0.9, col = "black")

# 4. Draw error bars (confidence intervals) on top
offset <- 3
for (i in seq_along(age)) {
  lower_limit <- ratio_mean[i] - ratio_ci[i]
  upper_limit <- ratio_mean[i] + ratio_ci[i]
  segments(x0 = age[i], y0 = lower_limit, x1 = age[i], y1 = upper_limit, col = "blue", lwd = 2)
  segments(x0 = age[i] - offset, y0 = lower_limit, x1 = age[i] + offset, y1 = lower_limit, col = "blue", lwd = 2)
  segments(x0 = age[i] - offset, y0 = upper_limit, x1 = age[i] + offset, y1 = upper_limit, col = "blue", lwd = 2)
}
###############################################################################################
# Fenestrida
# Read the data
Proportion_data <- read.csv("F:/1-Í¶¸å/2022-cladesÃð¾ø/Codes and data/Data and resuls for regression/Fenestrida_ratio_summary_decine phase.csv", header = TRUE)

# Extract and order variables
age <- as.vector(Proportion_data$age)
ratio_mean <- as.vector(Proportion_data$ratio_mean)
ratio_ci <- as.vector(Proportion_data$ratio_ci)
ord <- order(-age)
age <- age[ord]
ratio_mean <- ratio_mean[ord]
ratio_ci <- ratio_ci[ord]

# Fit linear model and predict for a dense sequence (for smoother CI plotting)
age_seq <- seq(max(age), min(age), length.out = 500)
lm1 <- lm(ratio_mean ~ age, data = data.frame(age = age, ratio_mean = ratio_mean))
pred <- predict(lm1, newdata = data.frame(age = age_seq), interval = "confidence", level = 0.95)

# Basic plot, no points yet
plot(
  age, ratio_mean,
  type = "n",
  xlab = "Age",
  ylab = "Ratio Mean",
  xlim = c(530, 0),
  ylim = c(0, 1)
)

# 1. Draw confidence interval as a shaded polygon
polygon(
  c(age_seq, rev(age_seq)),
  c(pred[, 3], rev(pred[, 2])),
  col = rgb(0.7, 0.7, 0.7, 0.5),
  border = NA
)

# 2. Draw regression line
lines(age_seq, pred[, 1], col = "blue", lwd = 2)

# 3. Draw data points (in gray)
points(age, ratio_mean, pch = 19, cex = 0.9, col = "black")

# 4. Draw error bars (confidence intervals) on top
offset <- 3
for (i in seq_along(age)) {
  lower_limit <- ratio_mean[i] - ratio_ci[i]
  upper_limit <- ratio_mean[i] + ratio_ci[i]
  segments(x0 = age[i], y0 = lower_limit, x1 = age[i], y1 = upper_limit, col = "blue", lwd = 2)
  segments(x0 = age[i] - offset, y0 = lower_limit, x1 = age[i] + offset, y1 = lower_limit, col = "blue", lwd = 2)
  segments(x0 = age[i] - offset, y0 = upper_limit, x1 = age[i] + offset, y1 = upper_limit, col = "blue", lwd = 2)
}
###############################################################################################
# Fusulinoidea
# Read the data
Proportion_data <- read.csv("F:/1-Í¶¸å/2022-cladesÃð¾ø/Codes and data/Data and resuls for regression/Fusulinoidea_ratio_summary_decine phase.csv", header = TRUE)

# Extract and order variables
age <- as.vector(Proportion_data$age)
ratio_mean <- as.vector(Proportion_data$ratio_mean)
ratio_ci <- as.vector(Proportion_data$ratio_ci)
ord <- order(-age)
age <- age[ord]
ratio_mean <- ratio_mean[ord]
ratio_ci <- ratio_ci[ord]

# Fit linear model and predict for a dense sequence (for smoother CI plotting)
age_seq <- seq(max(age), min(age), length.out = 500)
lm1 <- lm(ratio_mean ~ age, data = data.frame(age = age, ratio_mean = ratio_mean))
pred <- predict(lm1, newdata = data.frame(age = age_seq), interval = "confidence", level = 0.95)

# Basic plot, no points yet
plot(
  age, ratio_mean,
  type = "n",
  xlab = "Age",
  ylab = "Ratio Mean",
  xlim = c(530, 0),
  ylim = c(0, 1)
)

# 1. Draw confidence interval as a shaded polygon
polygon(
  c(age_seq, rev(age_seq)),
  c(pred[, 3], rev(pred[, 2])),
  col = rgb(0.7, 0.7, 0.7, 0.5),
  border = NA
)

# 2. Draw regression line
lines(age_seq, pred[, 1], col = "blue", lwd = 2)

# 3. Draw data points (in gray)
points(age, ratio_mean, pch = 19, cex = 0.9, col = "black")

# 4. Draw error bars (confidence intervals) on top
offset <- 3
for (i in seq_along(age)) {
  lower_limit <- ratio_mean[i] - ratio_ci[i]
  upper_limit <- ratio_mean[i] + ratio_ci[i]
  segments(x0 = age[i], y0 = lower_limit, x1 = age[i], y1 = upper_limit, col = "blue", lwd = 2)
  segments(x0 = age[i] - offset, y0 = lower_limit, x1 = age[i] + offset, y1 = lower_limit, col = "blue", lwd = 2)
  segments(x0 = age[i] - offset, y0 = upper_limit, x1 = age[i] + offset, y1 = upper_limit, col = "blue", lwd = 2)
}
###############################################################################################
# Graptoloidea
# Read the data
Proportion_data <- read.csv("F:/1-Í¶¸å/2022-cladesÃð¾ø/Codes and data/Data and resuls for regression/Graptoloidea_ratio_summary_decine phase.csv", header = TRUE)

# Extract and order variables
age <- as.vector(Proportion_data$age)
ratio_mean <- as.vector(Proportion_data$ratio_mean)
ratio_ci <- as.vector(Proportion_data$ratio_ci)
ord <- order(-age)
age <- age[ord]
ratio_mean <- ratio_mean[ord]
ratio_ci <- ratio_ci[ord]

# Fit linear model and predict for a dense sequence (for smoother CI plotting)
age_seq <- seq(max(age), min(age), length.out = 500)
lm1 <- lm(ratio_mean ~ age, data = data.frame(age = age, ratio_mean = ratio_mean))
pred <- predict(lm1, newdata = data.frame(age = age_seq), interval = "confidence", level = 0.95)

# Basic plot, no points yet
plot(
  age, ratio_mean,
  type = "n",
  xlab = "Age",
  ylab = "Ratio Mean",
  xlim = c(530, 0),
  ylim = c(0, 1)
)

# 1. Draw confidence interval as a shaded polygon
polygon(
  c(age_seq, rev(age_seq)),
  c(pred[, 3], rev(pred[, 2])),
  col = rgb(0.7, 0.7, 0.7, 0.5),
  border = NA
)

# 2. Draw regression line
lines(age_seq, pred[, 1], col = "blue", lwd = 2)

# 3. Draw data points (in gray)
points(age, ratio_mean, pch = 19, cex = 0.9, col = "black")

# 4. Draw error bars (confidence intervals) on top
offset <- 3
for (i in seq_along(age)) {
  lower_limit <- ratio_mean[i] - ratio_ci[i]
  upper_limit <- ratio_mean[i] + ratio_ci[i]
  segments(x0 = age[i], y0 = lower_limit, x1 = age[i], y1 = upper_limit, col = "blue", lwd = 2)
  segments(x0 = age[i] - offset, y0 = lower_limit, x1 = age[i] + offset, y1 = lower_limit, col = "blue", lwd = 2)
  segments(x0 = age[i] - offset, y0 = upper_limit, x1 = age[i] + offset, y1 = upper_limit, col = "blue", lwd = 2)
}
###############################################################################################
# Orthoceratoidea
# Read the data
Proportion_data <- read.csv("F:/1-Í¶¸å/2022-cladesÃð¾ø/Codes and data/Data and resuls for regression/Orthoceratoidea_ratio_summary_decine phase.csv", header = TRUE)

# Extract and order variables
age <- as.vector(Proportion_data$age)
ratio_mean <- as.vector(Proportion_data$ratio_mean)
ratio_ci <- as.vector(Proportion_data$ratio_ci)
ord <- order(-age)
age <- age[ord]
ratio_mean <- ratio_mean[ord]
ratio_ci <- ratio_ci[ord]

# Fit linear model and predict for a dense sequence (for smoother CI plotting)
age_seq <- seq(max(age), min(age), length.out = 500)
lm1 <- lm(ratio_mean ~ age, data = data.frame(age = age, ratio_mean = ratio_mean))
pred <- predict(lm1, newdata = data.frame(age = age_seq), interval = "confidence", level = 0.95)

# Basic plot, no points yet
plot(
  age, ratio_mean,
  type = "n",
  xlab = "Age",
  ylab = "Ratio Mean",
  xlim = c(530, 0),
  ylim = c(0, 1.2)
)

# 1. Draw confidence interval as a shaded polygon
polygon(
  c(age_seq, rev(age_seq)),
  c(pred[, 3], rev(pred[, 2])),
  col = rgb(0.7, 0.7, 0.7, 0.5),
  border = NA
)

# 2. Draw regression line
lines(age_seq, pred[, 1], col = "blue", lwd = 2)

# 3. Draw data points (in gray)
points(age, ratio_mean, pch = 19, cex = 0.9, col = "black")

# 4. Draw error bars (confidence intervals) on top
offset <- 3
for (i in seq_along(age)) {
  lower_limit <- ratio_mean[i] - ratio_ci[i]
  upper_limit <- ratio_mean[i] + ratio_ci[i]
  segments(x0 = age[i], y0 = lower_limit, x1 = age[i], y1 = upper_limit, col = "blue", lwd = 2)
  segments(x0 = age[i] - offset, y0 = lower_limit, x1 = age[i] + offset, y1 = lower_limit, col = "blue", lwd = 2)
  segments(x0 = age[i] - offset, y0 = upper_limit, x1 = age[i] + offset, y1 = upper_limit, col = "blue", lwd = 2)
}
###############################################################################################
# Palaeocopida
# Read the data
Proportion_data <- read.csv("F:/1-Í¶¸å/2022-cladesÃð¾ø/Codes and data/Data and resuls for regression/Palaeocopida_ratio_summary_decine phase.csv", header = TRUE)

# Extract and order variables
age <- as.vector(Proportion_data$age)
ratio_mean <- as.vector(Proportion_data$ratio_mean)
ratio_ci <- as.vector(Proportion_data$ratio_ci)
ord <- order(-age)
age <- age[ord]
ratio_mean <- ratio_mean[ord]
ratio_ci <- ratio_ci[ord]

# Fit linear model and predict for a dense sequence (for smoother CI plotting)
age_seq <- seq(max(age), min(age), length.out = 500)
lm1 <- lm(ratio_mean ~ age, data = data.frame(age = age, ratio_mean = ratio_mean))
pred <- predict(lm1, newdata = data.frame(age = age_seq), interval = "confidence", level = 0.95)

# Basic plot, no points yet
plot(
  age, ratio_mean,
  type = "n",
  xlab = "Age",
  ylab = "Ratio Mean",
  xlim = c(530, 0),
  ylim = c(0, 1.2)
)

# 1. Draw confidence interval as a shaded polygon
polygon(
  c(age_seq, rev(age_seq)),
  c(pred[, 3], rev(pred[, 2])),
  col = rgb(0.7, 0.7, 0.7, 0.5),
  border = NA
)

# 2. Draw regression line
lines(age_seq, pred[, 1], col = "blue", lwd = 2)

# 3. Draw data points (in gray)
points(age, ratio_mean, pch = 19, cex = 0.9, col = "black")

# 4. Draw error bars (confidence intervals) on top
offset <- 3
for (i in seq_along(age)) {
  lower_limit <- ratio_mean[i] - ratio_ci[i]
  upper_limit <- ratio_mean[i] + ratio_ci[i]
  segments(x0 = age[i], y0 = lower_limit, x1 = age[i], y1 = upper_limit, col = "blue", lwd = 2)
  segments(x0 = age[i] - offset, y0 = lower_limit, x1 = age[i] + offset, y1 = lower_limit, col = "blue", lwd = 2)
  segments(x0 = age[i] - offset, y0 = upper_limit, x1 = age[i] + offset, y1 = upper_limit, col = "blue", lwd = 2)
}
###############################################################################################
# Rugosa
# Read the data
Proportion_data <- read.csv("F:/1-Í¶¸å/2022-cladesÃð¾ø/Codes and data/Data and resuls for regression/Rugosa_ratio_summary_decine phase.csv", header = TRUE)

# Extract and order variables
age <- as.vector(Proportion_data$age)
ratio_mean <- as.vector(Proportion_data$ratio_mean)
ratio_ci <- as.vector(Proportion_data$ratio_ci)
ord <- order(-age)
age <- age[ord]
ratio_mean <- ratio_mean[ord]
ratio_ci <- ratio_ci[ord]

# Fit linear model and predict for a dense sequence (for smoother CI plotting)
age_seq <- seq(max(age), min(age), length.out = 500)
lm1 <- lm(ratio_mean ~ age, data = data.frame(age = age, ratio_mean = ratio_mean))
pred <- predict(lm1, newdata = data.frame(age = age_seq), interval = "confidence", level = 0.95)

# Basic plot, no points yet
plot(
  age, ratio_mean,
  type = "n",
  xlab = "Age",
  ylab = "Ratio Mean",
  xlim = c(530, 0),
  ylim = c(0, 1)
)

# 1. Draw confidence interval as a shaded polygon
polygon(
  c(age_seq, rev(age_seq)),
  c(pred[, 3], rev(pred[, 2])),
  col = rgb(0.7, 0.7, 0.7, 0.5),
  border = NA
)

# 2. Draw regression line
lines(age_seq, pred[, 1], col = "blue", lwd = 2)

# 3. Draw data points (in gray)
points(age, ratio_mean, pch = 19, cex = 0.9, col = "black")

# 4. Draw error bars (confidence intervals) on top
offset <- 3
for (i in seq_along(age)) {
  lower_limit <- ratio_mean[i] - ratio_ci[i]
  upper_limit <- ratio_mean[i] + ratio_ci[i]
  segments(x0 = age[i], y0 = lower_limit, x1 = age[i], y1 = upper_limit, col = "blue", lwd = 2)
  segments(x0 = age[i] - offset, y0 = lower_limit, x1 = age[i] + offset, y1 = lower_limit, col = "blue", lwd = 2)
  segments(x0 = age[i] - offset, y0 = upper_limit, x1 = age[i] + offset, y1 = upper_limit, col = "blue", lwd = 2)
}
###############################################################################################
# Spiriferinida
# Read the data
Proportion_data <- read.csv("F:/1-Í¶¸å/2022-cladesÃð¾ø/Codes and data/Data and resuls for regression/Spiriferinida_ratio_summary_decine phase.csv", header = TRUE)

# Extract and order variables
age <- as.vector(Proportion_data$age)
ratio_mean <- as.vector(Proportion_data$ratio_mean)
ratio_ci <- as.vector(Proportion_data$ratio_ci)
ord <- order(-age)
age <- age[ord]
ratio_mean <- ratio_mean[ord]
ratio_ci <- ratio_ci[ord]

# Fit linear model and predict for a dense sequence (for smoother CI plotting)
age_seq <- seq(max(age), min(age), length.out = 500)
lm1 <- lm(ratio_mean ~ age, data = data.frame(age = age, ratio_mean = ratio_mean))
pred <- predict(lm1, newdata = data.frame(age = age_seq), interval = "confidence", level = 0.95)

# Basic plot, no points yet
plot(
  age, ratio_mean,
  type = "n",
  xlab = "Age",
  ylab = "Ratio Mean",
  xlim = c(530, 0),
  ylim = c(0, 1)
)

# 1. Draw confidence interval as a shaded polygon
polygon(
  c(age_seq, rev(age_seq)),
  c(pred[, 3], rev(pred[, 2])),
  col = rgb(0.7, 0.7, 0.7, 0.5),
  border = NA
)

# 2. Draw regression line
lines(age_seq, pred[, 1], col = "blue", lwd = 2)

# 3. Draw data points (in gray)
points(age, ratio_mean, pch = 19, cex = 0.9, col = "black")

# 4. Draw error bars (confidence intervals) on top
offset <- 3
for (i in seq_along(age)) {
  lower_limit <- ratio_mean[i] - ratio_ci[i]
  upper_limit <- ratio_mean[i] + ratio_ci[i]
  segments(x0 = age[i], y0 = lower_limit, x1 = age[i], y1 = upper_limit, col = "blue", lwd = 2)
  segments(x0 = age[i] - offset, y0 = lower_limit, x1 = age[i] + offset, y1 = lower_limit, col = "blue", lwd = 2)
  segments(x0 = age[i] - offset, y0 = upper_limit, x1 = age[i] + offset, y1 = upper_limit, col = "blue", lwd = 2)
}
###############################################################################################
# Tabulata
# Read the data
Proportion_data <- read.csv("F:/1-Í¶¸å/2022-cladesÃð¾ø/Codes and data/Data and resuls for regression/Tabulata_ratio_summary_decine phase.csv", header = TRUE)

# Extract and order variables
age <- as.vector(Proportion_data$age)
ratio_mean <- as.vector(Proportion_data$ratio_mean)
ratio_ci <- as.vector(Proportion_data$ratio_ci)
ord <- order(-age)
age <- age[ord]
ratio_mean <- ratio_mean[ord]
ratio_ci <- ratio_ci[ord]

# Fit linear model and predict for a dense sequence (for smoother CI plotting)
age_seq <- seq(max(age), min(age), length.out = 500)
lm1 <- lm(ratio_mean ~ age, data = data.frame(age = age, ratio_mean = ratio_mean))
pred <- predict(lm1, newdata = data.frame(age = age_seq), interval = "confidence", level = 0.95)

# Basic plot, no points yet
plot(
  age, ratio_mean,
  type = "n",
  xlab = "Age",
  ylab = "Ratio Mean",
  xlim = c(530, 0),
  ylim = c(0, 1)
)

# 1. Draw confidence interval as a shaded polygon
polygon(
  c(age_seq, rev(age_seq)),
  c(pred[, 3], rev(pred[, 2])),
  col = rgb(0.7, 0.7, 0.7, 0.5),
  border = NA
)

# 2. Draw regression line
lines(age_seq, pred[, 1], col = "blue", lwd = 2)

# 3. Draw data points (in gray)
points(age, ratio_mean, pch = 19, cex = 0.9, col = "black")

# 4. Draw error bars (confidence intervals) on top
offset <- 3
for (i in seq_along(age)) {
  lower_limit <- ratio_mean[i] - ratio_ci[i]
  upper_limit <- ratio_mean[i] + ratio_ci[i]
  segments(x0 = age[i], y0 = lower_limit, x1 = age[i], y1 = upper_limit, col = "blue", lwd = 2)
  segments(x0 = age[i] - offset, y0 = lower_limit, x1 = age[i] + offset, y1 = lower_limit, col = "blue", lwd = 2)
  segments(x0 = age[i] - offset, y0 = upper_limit, x1 = age[i] + offset, y1 = upper_limit, col = "blue", lwd = 2)
}
###############################################################################################
# Trilobita
# Read the data
Proportion_data <- read.csv("F:/1-Í¶¸å/2022-cladesÃð¾ø/Codes and data/Data and resuls for regression/Trilobita_ratio_summary_decine phase.csv", header = TRUE)

# Extract and order variables
age <- as.vector(Proportion_data$age)
ratio_mean <- as.vector(Proportion_data$ratio_mean)
ratio_ci <- as.vector(Proportion_data$ratio_ci)
ord <- order(-age)
age <- age[ord]
ratio_mean <- ratio_mean[ord]
ratio_ci <- ratio_ci[ord]

# Fit linear model and predict for a dense sequence (for smoother CI plotting)
age_seq <- seq(max(age), min(age), length.out = 500)
lm1 <- lm(ratio_mean ~ age, data = data.frame(age = age, ratio_mean = ratio_mean))
pred <- predict(lm1, newdata = data.frame(age = age_seq), interval = "confidence", level = 0.95)

# Basic plot, no points yet
plot(
  age, ratio_mean,
  type = "n",
  xlab = "Age",
  ylab = "Ratio Mean",
  xlim = c(530, 0),
  ylim = c(0, 1)
)

# 1. Draw confidence interval as a shaded polygon
polygon(
  c(age_seq, rev(age_seq)),
  c(pred[, 3], rev(pred[, 2])),
  col = rgb(0.7, 0.7, 0.7, 0.5),
  border = NA
)

# 2. Draw regression line
lines(age_seq, pred[, 1], col = "blue", lwd = 2)

# 3. Draw data points (in gray)
points(age, ratio_mean, pch = 19, cex = 0.9, col = "black")

# 4. Draw error bars (confidence intervals) on top
offset <- 3
for (i in seq_along(age)) {
  lower_limit <- ratio_mean[i] - ratio_ci[i]
  upper_limit <- ratio_mean[i] + ratio_ci[i]
  segments(x0 = age[i], y0 = lower_limit, x1 = age[i], y1 = upper_limit, col = "blue", lwd = 2)
  segments(x0 = age[i] - offset, y0 = lower_limit, x1 = age[i] + offset, y1 = lower_limit, col = "blue", lwd = 2)
  segments(x0 = age[i] - offset, y0 = upper_limit, x1 = age[i] + offset, y1 = upper_limit, col = "blue", lwd = 2)
}
##