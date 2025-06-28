library(metafor)

# read data
Proportion_data <- read.csv("F:/1-Í¶¸å/2022-cladesÃð¾ø/PBDB Data/SQS results/Proportion_Trilobita.csv", header = TRUE)

# Sort the data by age in decreasing order
Proportion_data <- Proportion_data[order(Proportion_data$age, decreasing = TRUE), ]

# Convert each column in the data frame to a vector
age <- as.vector(Proportion_data$age)
proportion <- as.vector(Proportion_data$proportion)
error1 <- as.vector(Proportion_data$error1)
error2 <- as.vector(Proportion_data$error2)

# Method
lm1 <- lm(data = Proportion_data, proportion ~ age)

# Calculate R-squared and p-value
summary_lm1 <- summary(lm1)
r_squared <- summary_lm1$r.squared
p_value <- summary_lm1$coefficients[2, 4]

# Print R-squared and p-value
cat("R2=", r_squared, "\n")
cat("p=", p_value, "\n")

# 95% confidence interval
conf_interval1 <- predict(lm1, interval = "confidence", level = 0.95)

# Plot
plot(age, proportion, pch = 19, xlab = "Age (Ma)", ylab = "Proportion", xlim = c(550, 0), ylim = c(0, 1))

# Add regression line
abline(lm1, col = "blue", lwd = 2)

# Add confidence intervals
matlines(age, conf_interval1[, 2:3], col = "blue", lty = 1)

# Add R-squared and p-value to the plot
text(x = 250, y = 0.9, labels = paste("R2=", round(r_squared, 3), "\n", "p=", format.pval(p_value, digits = 3)), pos = 4)