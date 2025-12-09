library(divDyn)

######################################################################################################

# Ammonoidea

library(divDyn)

# 1. Read data:
# Ammonoidea_Nautiloidea_Coleoidea_Fish.csv == "all cephalopod and fish" group
dat_All <- read.csv("F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Ammonoidea,Nautiloidea,Coleoidea,Fish.csv", header = TRUE)
# Ammonoidea.csv (for numerator)
dat_Ammon <- read.csv("F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Ammonoidea.csv", header = TRUE)

# 2. Set parameters
n_iter <- 50
q_level <- 0.7

# 3. Prepare time bins (ensure binno is numeric)
dat_Ammon$binno  <- as.numeric(dat_Ammon$binno)
dat_All$binno    <- as.numeric(dat_All$binno)
bins <- sort(unique(c(dat_Ammon$binno, dat_All$binno)))
colnames_bins <- as.character(bins)

# 4. Create result matrices for each iteration
All_results <- matrix(NA, nrow=n_iter, ncol=length(bins))
colnames(All_results) <- colnames_bins
Ammonoidea_results <- matrix(NA, nrow=n_iter, ncol=length(bins))
colnames(Ammonoidea_results) <- colnames_bins
Ratio_results <- matrix(NA, nrow=n_iter, ncol=length(bins))
colnames(Ratio_results) <- colnames_bins

# 5. Vectors to store mean value in each iteration
divSIB_All_vec <- numeric(n_iter)
divSIB_Ammon_vec <- numeric(n_iter)
ratio_vec <- numeric(n_iter)

# 6. Main loop
for(i in 1:n_iter){
  # Ammonoidea SQS
  res_Ammon <- subsample(dat_Ammon, q=q_level, tax="genus", bin="binno", iter=1, type="sqs")
  # All groups SQS
  res_All <- subsample(dat_All, q=q_level, tax="genus", bin="binno", iter=1, type="sqs")
  
  # Initialize vectors for this iteration
  ammon_div_vec <- rep(NA, length(bins))
  all_div_vec <- rep(NA, length(bins))
  names(ammon_div_vec) <- names(all_div_vec) <- colnames_bins
  
  # Assign SQS results to correct bins
  if (nrow(res_Ammon) > 0 && "binno" %in% names(res_Ammon)) {
    idx <- match(as.character(res_Ammon$binno), colnames_bins)
    ok <- !is.na(idx)
    if (any(ok)) ammon_div_vec[idx[ok]] <- res_Ammon$divSIB[ok]
  }
  if (nrow(res_All) > 0 && "binno" %in% names(res_All)) {
    idx <- match(as.character(res_All$binno), colnames_bins)
    ok <- !is.na(idx)
    if (any(ok)) all_div_vec[idx[ok]] <- res_All$divSIB[ok]
  }
  
  # Store each iteration results
  Ammonoidea_results[i,] <- ammon_div_vec
  All_results[i,] <- all_div_vec
  
  # Calculate ratio for this iteration
  ratio_this <- ammon_div_vec / all_div_vec
  Ratio_results[i, ] <- ratio_this
  
  # Store mean values for this iteration
  divSIB_Ammon_vec[i] <- mean(ammon_div_vec, na.rm=TRUE)
  divSIB_All_vec[i] <- mean(all_div_vec, na.rm=TRUE)
  ratio_vec[i] <- mean(ratio_this, na.rm=TRUE)
}

# 7. Descriptive statistics (mean and 95% CI half-width)
ratio_mean <- mean(ratio_vec, na.rm=TRUE)
ratio_sd   <- sd(ratio_vec, na.rm=TRUE)
n_eff      <- sum(!is.na(ratio_vec))
ratio_se   <- ratio_sd / sqrt(n_eff)
ratio_ci_halfwidth <- 1.96 * ratio_se  # Only output half-width

cat("Mean ratio of divSIB (Ammonoidea / All groups) after", n_iter, "iterations:\n")
cat("Mean:", ratio_mean, "\n")
cat("95% CI half-width:", ratio_ci_halfwidth, "\n")

# 8. Save all iteration results
Ammonoidea_results_with_mean <- rbind(Ammonoidea_results, Mean=apply(Ammonoidea_results, 2, mean, na.rm=TRUE))
write.csv(Ammonoidea_results_with_mean, file="F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Ammonoidea_SQS_50iter.csv")
write.csv(All_results, file="F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/AllCephFish_SQS_50iter.csv")
write.csv(Ratio_results, file="F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Ammonoidea_ratio_50iter.csv")

# 9. Calculate mean and 95% CI half-width for the ratio in each bin
ratio_mean_perbin <- apply(Ratio_results, 2, mean, na.rm=TRUE)
ratio_sd_perbin   <- apply(Ratio_results, 2, sd, na.rm=TRUE)
ratio_n_perbin    <- apply(Ratio_results, 2, function(x) sum(!is.na(x)))
ratio_se_perbin   <- ratio_sd_perbin / sqrt(ratio_n_perbin)
ratio_ci_perbin   <- 1.96 * ratio_se_perbin # Only output this

# Binno与Age对应关系表
bin_age <- data.frame(
  binno = 1:100,
  age = c(0.00585, 0.07035, 0.4515, 1.287, 2.19, 3.09, 4.4665, 6.2895, 9.438, 12.725,
          14.9, 18.21, 21.735, 25.425, 30.86, 35.805, 39.455, 44.5, 51.9, 57.6,
          60.4, 63.8, 69.05, 77.85, 84.95, 88.05, 91.85, 97.2, 106.75, 117.2,
          123.585, 129.185, 136.2, 142.4, 147.1, 152, 158.15, 163.4, 166.75, 169.55,
          172.8, 179.45, 188.55, 196.2, 200.4, 204.9, 217.75, 232, 239.5, 244.6,
          249.2, 251.551, 253.021, 256.825, 261.895, 265.59, 269.955, 278.255, 286.8, 291.81,
          296.21, 301.3, 305.35, 311.1, 319.2, 327.05, 338.8, 352.8, 365.55, 377.45,
          385.2, 390.5, 400.45, 409.2, 415, 421.1, 424.3, 426.5, 428.95, 431.95,
          435.95, 439.65, 442.3, 444.5, 449.1, 455.7, 462.85, 468.65, 473.85, 481.55,
          487.45, 491.75, 495.5, 498.75, 502.5, 506.75, 511.5, 517.5, 525, 533.9)
)

summary_df <- data.frame(
  binno = as.numeric(names(ratio_mean_perbin)), 
  ratio_mean = ratio_mean_perbin, 
  ratio_ci = ratio_ci_perbin
)

# 合并age列
summary_df <- merge(summary_df, bin_age, by = "binno", all.x = TRUE)
# Optional: 按binno排序
summary_df <- summary_df[order(summary_df$binno), ]

write.csv(
  summary_df,
  file="F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Ammonoidea_ratio_summary.csv", 
  row.names=FALSE
)

######################################################################################################
# Camerata

library(divDyn)

# 1. Read data:
# Camerata,Pentacrinoidea.csv == "all camerates and pentacrinoids" group
dat_All <- read.csv("F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Camerata,Pentacrinoidea.csv", header = TRUE)
# Camerata.csv (for numerator)
dat_Camerata <- read.csv("F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Camerata.csv", header = TRUE)

# 2. Set parameters
n_iter <- 50
q_level <- 0.7

# 3. Prepare time bins (ensure binno is numeric)
dat_Camerata$binno  <- as.numeric(dat_Camerata$binno)
dat_All$binno    <- as.numeric(dat_All$binno)
bins <- sort(unique(c(dat_Camerata$binno, dat_All$binno)))
colnames_bins <- as.character(bins)

# 4. Create result matrices for each iteration
All_results <- matrix(NA, nrow=n_iter, ncol=length(bins))
colnames(All_results) <- colnames_bins
Camerata_results <- matrix(NA, nrow=n_iter, ncol=length(bins))
colnames(Camerata_results) <- colnames_bins
Ratio_results <- matrix(NA, nrow=n_iter, ncol=length(bins))
colnames(Ratio_results) <- colnames_bins

# 5. Vectors to store mean value in each iteration
divSIB_All_vec <- numeric(n_iter)
divSIB_Camerata_vec <- numeric(n_iter)
ratio_vec <- numeric(n_iter)

# 6. Main loop
for(i in 1:n_iter){
  # Camerata SQS
  res_Camerata <- subsample(dat_Camerata, q=q_level, tax="genus", bin="binno", iter=1, type="sqs")
  # All groups SQS
  res_All <- subsample(dat_All, q=q_level, tax="genus", bin="binno", iter=1, type="sqs")
  
  # Initialize vectors for this iteration
  camerata_div_vec <- rep(NA, length(bins))
  all_div_vec <- rep(NA, length(bins))
  names(camerata_div_vec) <- names(all_div_vec) <- colnames_bins
  
  # Assign SQS results to correct bins
  if (nrow(res_Camerata) > 0 && "binno" %in% names(res_Camerata)) {
    idx <- match(as.character(res_Camerata$binno), colnames_bins)
    ok <- !is.na(idx)
    if (any(ok)) camerata_div_vec[idx[ok]] <- res_Camerata$divSIB[ok]
  }
  if (nrow(res_All) > 0 && "binno" %in% names(res_All)) {
    idx <- match(as.character(res_All$binno), colnames_bins)
    ok <- !is.na(idx)
    if (any(ok)) all_div_vec[idx[ok]] <- res_All$divSIB[ok]
  }
  
  # Store each iteration results
  Camerata_results[i,] <- camerata_div_vec
  All_results[i,] <- all_div_vec
  
  # Calculate ratio for this iteration
  ratio_this <- camerata_div_vec / all_div_vec
  Ratio_results[i, ] <- ratio_this
  
  # Store mean values for this iteration
  divSIB_Camerata_vec[i] <- mean(camerata_div_vec, na.rm=TRUE)
  divSIB_All_vec[i] <- mean(all_div_vec, na.rm=TRUE)
  ratio_vec[i] <- mean(ratio_this, na.rm=TRUE)
}

# 7. Descriptive statistics (mean and 95% CI half-width)
ratio_mean <- mean(ratio_vec, na.rm=TRUE)
ratio_sd   <- sd(ratio_vec, na.rm=TRUE)
n_eff      <- sum(!is.na(ratio_vec))
ratio_se   <- ratio_sd / sqrt(n_eff)
ratio_ci_halfwidth <- 1.96 * ratio_se  # Only output half-width

cat("Mean ratio of divSIB (Camerata / All groups) after", n_iter, "iterations:\n")
cat("Mean:", ratio_mean, "\n")
cat("95% CI half-width:", ratio_ci_halfwidth, "\n")

# 8. Save all iteration results
Camerata_results_with_mean <- rbind(Camerata_results, Mean=apply(Camerata_results, 2, mean, na.rm=TRUE))
write.csv(Camerata_results_with_mean, file="F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Camerata_SQS_50iter.csv")
write.csv(All_results, file="F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/AllCamerataPentacrinoidea_SQS_50iter.csv")
write.csv(Ratio_results, file="F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Camerata_ratio_50iter.csv")

# 9. Calculate mean and 95% CI half-width for the ratio in each bin
ratio_mean_perbin <- apply(Ratio_results, 2, mean, na.rm=TRUE)
ratio_sd_perbin   <- apply(Ratio_results, 2, sd, na.rm=TRUE)
ratio_n_perbin    <- apply(Ratio_results, 2, function(x) sum(!is.na(x)))
ratio_se_perbin   <- ratio_sd_perbin / sqrt(ratio_n_perbin)
ratio_ci_perbin   <- 1.96 * ratio_se_perbin # Only output this

# Binno与Age对应关系表
bin_age <- data.frame(
  binno = 1:100,
  age = c(0.00585, 0.07035, 0.4515, 1.287, 2.19, 3.09, 4.4665, 6.2895, 9.438, 12.725,
          14.9, 18.21, 21.735, 25.425, 30.86, 35.805, 39.455, 44.5, 51.9, 57.6,
          60.4, 63.8, 69.05, 77.85, 84.95, 88.05, 91.85, 97.2, 106.75, 117.2,
          123.585, 129.185, 136.2, 142.4, 147.1, 152, 158.15, 163.4, 166.75, 169.55,
          172.8, 179.45, 188.55, 196.2, 200.4, 204.9, 217.75, 232, 239.5, 244.6,
          249.2, 251.551, 253.021, 256.825, 261.895, 265.59, 269.955, 278.255, 286.8, 291.81,
          296.21, 301.3, 305.35, 311.1, 319.2, 327.05, 338.8, 352.8, 365.55, 377.45,
          385.2, 390.5, 400.45, 409.2, 415, 421.1, 424.3, 426.5, 428.95, 431.95,
          435.95, 439.65, 442.3, 444.5, 449.1, 455.7, 462.85, 468.65, 473.85, 481.55,
          487.45, 491.75, 495.5, 498.75, 502.5, 506.75, 511.5, 517.5, 525, 533.9)
)

summary_df <- data.frame(
  binno = as.numeric(names(ratio_mean_perbin)), 
  ratio_mean = ratio_mean_perbin, 
  ratio_ci = ratio_ci_perbin
)

# 合并age列
summary_df <- merge(summary_df, bin_age, by = "binno", all.x = TRUE)
# Optional: 按binno排序
summary_df <- summary_df[order(summary_df$binno), ]

write.csv(
  summary_df,
  file="F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Camerata_ratio_summary.csv", 
  row.names=FALSE
)



######################################################################################################

# Conodonta

library(divDyn)

# 1. Read data:
# Conodonta,Fish.csv == "all conodonta and fish" group
dat_All <- read.csv("F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Conodonta,Fish.csv", header = TRUE)
# Conodonta.csv (for numerator)
dat_Conodonta <- read.csv("F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Conodonta.csv", header = TRUE)

# 2. Set parameters
n_iter <- 50
q_level <- 0.7

# 3. Prepare time bins (ensure binno is numeric)
dat_Conodonta$binno  <- as.numeric(dat_Conodonta$binno)
dat_All$binno    <- as.numeric(dat_All$binno)
bins <- sort(unique(c(dat_Conodonta$binno, dat_All$binno)))
colnames_bins <- as.character(bins)

# 4. Create result matrices for each iteration
All_results <- matrix(NA, nrow=n_iter, ncol=length(bins))
colnames(All_results) <- colnames_bins
Conodonta_results <- matrix(NA, nrow=n_iter, ncol=length(bins))
colnames(Conodonta_results) <- colnames_bins
Ratio_results <- matrix(NA, nrow=n_iter, ncol=length(bins))
colnames(Ratio_results) <- colnames_bins

# 5. Vectors to store mean value in each iteration
divSIB_All_vec <- numeric(n_iter)
divSIB_Conodonta_vec <- numeric(n_iter)
ratio_vec <- numeric(n_iter)

# 6. Main loop
for(i in 1:n_iter){
  # Conodonta SQS
  res_Conodonta <- subsample(dat_Conodonta, q=q_level, tax="genus", bin="binno", iter=1, type="sqs")
  # All groups SQS
  res_All <- subsample(dat_All, q=q_level, tax="genus", bin="binno", iter=1, type="sqs")
  
  # Initialize vectors for this iteration
  conodonta_div_vec <- rep(NA, length(bins))
  all_div_vec <- rep(NA, length(bins))
  names(conodonta_div_vec) <- names(all_div_vec) <- colnames_bins
  
  # Assign SQS results to correct bins
  if (nrow(res_Conodonta) > 0 && "binno" %in% names(res_Conodonta)) {
    idx <- match(as.character(res_Conodonta$binno), colnames_bins)
    ok <- !is.na(idx)
    if (any(ok)) conodonta_div_vec[idx[ok]] <- res_Conodonta$divSIB[ok]
  }
  if (nrow(res_All) > 0 && "binno" %in% names(res_All)) {
    idx <- match(as.character(res_All$binno), colnames_bins)
    ok <- !is.na(idx)
    if (any(ok)) all_div_vec[idx[ok]] <- res_All$divSIB[ok]
  }
  
  # Store each iteration results
  Conodonta_results[i,] <- conodonta_div_vec
  All_results[i,] <- all_div_vec
  
  # Calculate ratio for this iteration
  ratio_this <- conodonta_div_vec / all_div_vec
  Ratio_results[i, ] <- ratio_this
  
  # Store mean values for this iteration
  divSIB_Conodonta_vec[i] <- mean(conodonta_div_vec, na.rm=TRUE)
  divSIB_All_vec[i] <- mean(all_div_vec, na.rm=TRUE)
  ratio_vec[i] <- mean(ratio_this, na.rm=TRUE)
}

# 7. Descriptive statistics (mean and 95% CI half-width)
ratio_mean <- mean(ratio_vec, na.rm=TRUE)
ratio_sd   <- sd(ratio_vec, na.rm=TRUE)
n_eff      <- sum(!is.na(ratio_vec))
ratio_se   <- ratio_sd / sqrt(n_eff)
ratio_ci_halfwidth <- 1.96 * ratio_se  # Only output half-width

cat("Mean ratio of divSIB (Conodonta / All groups) after", n_iter, "iterations:\n")
cat("Mean:", ratio_mean, "\n")
cat("95% CI half-width:", ratio_ci_halfwidth, "\n")

# 8. Save all iteration results
Conodonta_results_with_mean <- rbind(Conodonta_results, Mean=apply(Conodonta_results, 2, mean, na.rm=TRUE))
write.csv(Conodonta_results_with_mean, file="F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Conodonta_SQS_50iter.csv")
write.csv(All_results, file="F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/AllConodontaFish_SQS_50iter.csv")
write.csv(Ratio_results, file="F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Conodonta_ratio_50iter.csv")

# 9. Calculate mean and 95% CI half-width for the ratio in each bin
ratio_mean_perbin <- apply(Ratio_results, 2, mean, na.rm=TRUE)
ratio_sd_perbin   <- apply(Ratio_results, 2, sd, na.rm=TRUE)
ratio_n_perbin    <- apply(Ratio_results, 2, function(x) sum(!is.na(x)))
ratio_se_perbin   <- ratio_sd_perbin / sqrt(ratio_n_perbin)
ratio_ci_perbin   <- 1.96 * ratio_se_perbin # Only output this

# Binno与Age对应关系表
bin_age <- data.frame(
  binno = 1:100,
  age = c(0.00585, 0.07035, 0.4515, 1.287, 2.19, 3.09, 4.4665, 6.2895, 9.438, 12.725,
          14.9, 18.21, 21.735, 25.425, 30.86, 35.805, 39.455, 44.5, 51.9, 57.6,
          60.4, 63.8, 69.05, 77.85, 84.95, 88.05, 91.85, 97.2, 106.75, 117.2,
          123.585, 129.185, 136.2, 142.4, 147.1, 152, 158.15, 163.4, 166.75, 169.55,
          172.8, 179.45, 188.55, 196.2, 200.4, 204.9, 217.75, 232, 239.5, 244.6,
          249.2, 251.551, 253.021, 256.825, 261.895, 265.59, 269.955, 278.255, 286.8, 291.81,
          296.21, 301.3, 305.35, 311.1, 319.2, 327.05, 338.8, 352.8, 365.55, 377.45,
          385.2, 390.5, 400.45, 409.2, 415, 421.1, 424.3, 426.5, 428.95, 431.95,
          435.95, 439.65, 442.3, 444.5, 449.1, 455.7, 462.85, 468.65, 473.85, 481.55,
          487.45, 491.75, 495.5, 498.75, 502.5, 506.75, 511.5, 517.5, 525, 533.9)
)

summary_df <- data.frame(
  binno = as.numeric(names(ratio_mean_perbin)), 
  ratio_mean = ratio_mean_perbin, 
  ratio_ci = ratio_ci_perbin
)

# Merge with age
summary_df <- merge(summary_df, bin_age, by = "binno", all.x = TRUE)
summary_df <- summary_df[order(summary_df$binno), ]

write.csv(
  summary_df,
  file="F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Conodonta_ratio_summary.csv", 
  row.names=FALSE
)

######################################################################################################
# Fenestrida

library(divDyn)

# 1. Read data:
# Rugosa,Tabulata,Porifera,Bryozoa.csv == "all rugose corals, tabulate corals, porifera and bryozoa" group
dat_All <- read.csv("F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Rugosa,Tabulata,Porifera,Bryozoa.csv", header = TRUE)
# Fenestrida.csv (for numerator)
dat_Fenestrida <- read.csv("F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Fenestrida.csv", header = TRUE)

# 2. Set parameters
n_iter <- 50
q_level <- 0.7

# 3. Prepare time bins (ensure binno is numeric)
dat_Fenestrida$binno  <- as.numeric(dat_Fenestrida$binno)
dat_All$binno    <- as.numeric(dat_All$binno)
bins <- sort(unique(c(dat_Fenestrida$binno, dat_All$binno)))
colnames_bins <- as.character(bins)

# 4. Create result matrices for each iteration
All_results <- matrix(NA, nrow=n_iter, ncol=length(bins))
colnames(All_results) <- colnames_bins
Fenestrida_results <- matrix(NA, nrow=n_iter, ncol=length(bins))
colnames(Fenestrida_results) <- colnames_bins
Ratio_results <- matrix(NA, nrow=n_iter, ncol=length(bins))
colnames(Ratio_results) <- colnames_bins

# 5. Vectors to store mean value in each iteration
divSIB_All_vec <- numeric(n_iter)
divSIB_Fenestrida_vec <- numeric(n_iter)
ratio_vec <- numeric(n_iter)

# 6. Main loop
for(i in 1:n_iter){
  # Fenestrida SQS
  res_Fenestrida <- subsample(dat_Fenestrida, q=q_level, tax="genus", bin="binno", iter=1, type="sqs")
  # All groups SQS
  res_All <- subsample(dat_All, q=q_level, tax="genus", bin="binno", iter=1, type="sqs")
  
  # Initialize vectors for this iteration
  fenestrida_div_vec <- rep(NA, length(bins))
  all_div_vec <- rep(NA, length(bins))
  names(fenestrida_div_vec) <- names(all_div_vec) <- colnames_bins
  
  # Assign SQS results to correct bins
  if (nrow(res_Fenestrida) > 0 && "binno" %in% names(res_Fenestrida)) {
    idx <- match(as.character(res_Fenestrida$binno), colnames_bins)
    ok <- !is.na(idx)
    if (any(ok)) fenestrida_div_vec[idx[ok]] <- res_Fenestrida$divSIB[ok]
  }
  if (nrow(res_All) > 0 && "binno" %in% names(res_All)) {
    idx <- match(as.character(res_All$binno), colnames_bins)
    ok <- !is.na(idx)
    if (any(ok)) all_div_vec[idx[ok]] <- res_All$divSIB[ok]
  }
  
  # Store each iteration results
  Fenestrida_results[i,] <- fenestrida_div_vec
  All_results[i,] <- all_div_vec
  
  # Calculate ratio for this iteration
  ratio_this <- fenestrida_div_vec / all_div_vec
  Ratio_results[i, ] <- ratio_this
  
  # Store mean values for this iteration
  divSIB_Fenestrida_vec[i] <- mean(fenestrida_div_vec, na.rm=TRUE)
  divSIB_All_vec[i] <- mean(all_div_vec, na.rm=TRUE)
  ratio_vec[i] <- mean(ratio_this, na.rm=TRUE)
}

# 7. Descriptive statistics (mean and 95% CI half-width)
ratio_mean <- mean(ratio_vec, na.rm=TRUE)
ratio_sd   <- sd(ratio_vec, na.rm=TRUE)
n_eff      <- sum(!is.na(ratio_vec))
ratio_se   <- ratio_sd / sqrt(n_eff)
ratio_ci_halfwidth <- 1.96 * ratio_se  # Only output half-width

cat("Mean ratio of divSIB (Fenestrida / All groups) after", n_iter, "iterations:\n")
cat("Mean:", ratio_mean, "\n")
cat("95% CI half-width:", ratio_ci_halfwidth, "\n")

# 8. Save all iteration results
Fenestrida_results_with_mean <- rbind(Fenestrida_results, Mean=apply(Fenestrida_results, 2, mean, na.rm=TRUE))
write.csv(Fenestrida_results_with_mean, file="F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Fenestrida_SQS_50iter.csv")
write.csv(All_results, file="F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/AllRugosaTabulataPoriferaBryozoa_SQS_50iter.csv")
write.csv(Ratio_results, file="F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Fenestrida_ratio_50iter.csv")

# 9. Calculate mean and 95% CI half-width for the ratio in each bin
ratio_mean_perbin <- apply(Ratio_results, 2, mean, na.rm=TRUE)
ratio_sd_perbin   <- apply(Ratio_results, 2, sd, na.rm=TRUE)
ratio_n_perbin    <- apply(Ratio_results, 2, function(x) sum(!is.na(x)))
ratio_se_perbin   <- ratio_sd_perbin / sqrt(ratio_n_perbin)
ratio_ci_perbin   <- 1.96 * ratio_se_perbin # Only output this

# Binno与Age对应关系表
bin_age <- data.frame(
  binno = 1:100,
  age = c(0.00585, 0.07035, 0.4515, 1.287, 2.19, 3.09, 4.4665, 6.2895, 9.438, 12.725,
          14.9, 18.21, 21.735, 25.425, 30.86, 35.805, 39.455, 44.5, 51.9, 57.6,
          60.4, 63.8, 69.05, 77.85, 84.95, 88.05, 91.85, 97.2, 106.75, 117.2,
          123.585, 129.185, 136.2, 142.4, 147.1, 152, 158.15, 163.4, 166.75, 169.55,
          172.8, 179.45, 188.55, 196.2, 200.4, 204.9, 217.75, 232, 239.5, 244.6,
          249.2, 251.551, 253.021, 256.825, 261.895, 265.59, 269.955, 278.255, 286.8, 291.81,
          296.21, 301.3, 305.35, 311.1, 319.2, 327.05, 338.8, 352.8, 365.55, 377.45,
          385.2, 390.5, 400.45, 409.2, 415, 421.1, 424.3, 426.5, 428.95, 431.95,
          435.95, 439.65, 442.3, 444.5, 449.1, 455.7, 462.85, 468.65, 473.85, 481.55,
          487.45, 491.75, 495.5, 498.75, 502.5, 506.75, 511.5, 517.5, 525, 533.9)
)

summary_df <- data.frame(
  binno = as.numeric(names(ratio_mean_perbin)), 
  ratio_mean = ratio_mean_perbin, 
  ratio_ci = ratio_ci_perbin
)

# 合并age列
summary_df <- merge(summary_df, bin_age, by = "binno", all.x = TRUE)
summary_df <- summary_df[order(summary_df$binno), ]

write.csv(
  summary_df,
  file="F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Fenestrida_ratio_summary.csv", 
  row.names=FALSE
)





######################################################################################################

# Fusulinoidea

library(divDyn)

# 1. Read data:
dat_All <- read.csv("F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Foraminifera.csv", header = TRUE)
# Fusulinoidea.csv (for numerator)
dat_Fusu <- read.csv("F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Fusulinoidea.csv", header = TRUE)

# 2. Set parameters
n_iter <- 50
q_level <- 0.7

# 3. Prepare time bins (ensure binno is numeric)
dat_Fusu$binno  <- as.numeric(dat_Fusu$binno)
dat_All$binno    <- as.numeric(dat_All$binno)
bins <- sort(unique(c(dat_Fusu$binno, dat_All$binno)))
colnames_bins <- as.character(bins)

# 4. Create result matrices for each iteration
All_results <- matrix(NA, nrow=n_iter, ncol=length(bins))
colnames(All_results) <- colnames_bins
Fusulinoidea_results <- matrix(NA, nrow=n_iter, ncol=length(bins))
colnames(Fusulinoidea_results) <- colnames_bins
Ratio_results <- matrix(NA, nrow=n_iter, ncol=length(bins))
colnames(Ratio_results) <- colnames_bins

# 5. Vectors to store mean value in each iteration
divSIB_All_vec <- numeric(n_iter)
divSIB_Fusu_vec <- numeric(n_iter)
ratio_vec <- numeric(n_iter)

# 6. Main loop
for(i in 1:n_iter){
  # Fusulinoidea SQS
  res_Fusu <- subsample(dat_Fusu, q=q_level, tax="genus", bin="binno", iter=1, type="sqs")
  # All groups SQS
  res_All <- subsample(dat_All, q=q_level, tax="genus", bin="binno", iter=1, type="sqs")
  
  # Initialize vectors for this iteration
  fusu_div_vec <- rep(NA, length(bins))
  all_div_vec <- rep(NA, length(bins))
  names(fusu_div_vec) <- names(all_div_vec) <- colnames_bins
  
  # Assign SQS results to correct bins
  if (nrow(res_Fusu) > 0 && "binno" %in% names(res_Fusu)) {
    idx <- match(as.character(res_Fusu$binno), colnames_bins)
    ok <- !is.na(idx)
    if (any(ok)) fusu_div_vec[idx[ok]] <- res_Fusu$divSIB[ok]
  }
  if (nrow(res_All) > 0 && "binno" %in% names(res_All)) {
    idx <- match(as.character(res_All$binno), colnames_bins)
    ok <- !is.na(idx)
    if (any(ok)) all_div_vec[idx[ok]] <- res_All$divSIB[ok]
  }
  
  # Store each iteration results
  Fusulinoidea_results[i,] <- fusu_div_vec
  All_results[i,] <- all_div_vec
  
  # Calculate ratio for this iteration
  ratio_this <- fusu_div_vec / all_div_vec
  Ratio_results[i, ] <- ratio_this
  
  # Store mean values for this iteration
  divSIB_Fusu_vec[i] <- mean(fusu_div_vec, na.rm=TRUE)
  divSIB_All_vec[i] <- mean(all_div_vec, na.rm=TRUE)
  ratio_vec[i] <- mean(ratio_this, na.rm=TRUE)
}

# 7. Descriptive statistics (mean and 95% CI half-width)
ratio_mean <- mean(ratio_vec, na.rm=TRUE)
ratio_sd   <- sd(ratio_vec, na.rm=TRUE)
n_eff      <- sum(!is.na(ratio_vec))
ratio_se   <- ratio_sd / sqrt(n_eff)
ratio_ci_halfwidth <- 1.96 * ratio_se  # Only output half-width

cat("Mean ratio of divSIB (Fusulinoidea / All groups) after", n_iter, "iterations:\n")
cat("Mean:", ratio_mean, "\n")
cat("95% CI half-width:", ratio_ci_halfwidth, "\n")

# 8. Save all iteration results
Fusulinoidea_results_with_mean <- rbind(Fusulinoidea_results, Mean=apply(Fusulinoidea_results, 2, mean, na.rm=TRUE))
write.csv(Fusulinoidea_results_with_mean, file="F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Fusulinoidea_SQS_50iter.csv")
write.csv(All_results, file="F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/AllForaminifera_SQS_50iter.csv")
write.csv(Ratio_results, file="F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Fusulinoidea_ratio_50iter.csv")

# 9. Calculate mean and 95% CI half-width for the ratio in each bin
ratio_mean_perbin <- apply(Ratio_results, 2, mean, na.rm=TRUE)
ratio_sd_perbin   <- apply(Ratio_results, 2, sd, na.rm=TRUE)
ratio_n_perbin    <- apply(Ratio_results, 2, function(x) sum(!is.na(x)))
ratio_se_perbin   <- ratio_sd_perbin / sqrt(ratio_n_perbin)
ratio_ci_perbin   <- 1.96 * ratio_se_perbin # Only output this

# Binno与Age对应关系表
bin_age <- data.frame(
  binno = 1:100,
  age = c(0.00585, 0.07035, 0.4515, 1.287, 2.19, 3.09, 4.4665, 6.2895, 9.438, 12.725,
          14.9, 18.21, 21.735, 25.425, 30.86, 35.805, 39.455, 44.5, 51.9, 57.6,
          60.4, 63.8, 69.05, 77.85, 84.95, 88.05, 91.85, 97.2, 106.75, 117.2,
          123.585, 129.185, 136.2, 142.4, 147.1, 152, 158.15, 163.4, 166.75, 169.55,
          172.8, 179.45, 188.55, 196.2, 200.4, 204.9, 217.75, 232, 239.5, 244.6,
          249.2, 251.551, 253.021, 256.825, 261.895, 265.59, 269.955, 278.255, 286.8, 291.81,
          296.21, 301.3, 305.35, 311.1, 319.2, 327.05, 338.8, 352.8, 365.55, 377.45,
          385.2, 390.5, 400.45, 409.2, 415, 421.1, 424.3, 426.5, 428.95, 431.95,
          435.95, 439.65, 442.3, 444.5, 449.1, 455.7, 462.85, 468.65, 473.85, 481.55,
          487.45, 491.75, 495.5, 498.75, 502.5, 506.75, 511.5, 517.5, 525, 533.9)
)

summary_df <- data.frame(
  binno = as.numeric(names(ratio_mean_perbin)), 
  ratio_mean = ratio_mean_perbin, 
  ratio_ci = ratio_ci_perbin
)

# Merge with age
summary_df <- merge(summary_df, bin_age, by = "binno", all.x = TRUE)
summary_df <- summary_df[order(summary_df$binno), ]

write.csv(
  summary_df,
  file="F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Fusulinoidea_ratio_summary.csv", 
  row.names=FALSE
)






######################################################################################################

# Graptoloidea

library(divDyn)

# 1. Read data:
# Graptoloidea,Tentaculita.csv == "all graptoloids and tentaculita" group
dat_All <- read.csv("F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Graptoloidea,Tentaculita.csv", header = TRUE)
# Graptoloidea.csv (for numerator)
dat_Graptoloidea <- read.csv("F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Graptoloidea.csv", header = TRUE)

# 2. Set parameters
n_iter <- 50
q_level <- 0.7

# 3. Prepare time bins (ensure binno is numeric)
dat_Graptoloidea$binno  <- as.numeric(dat_Graptoloidea$binno)
dat_All$binno    <- as.numeric(dat_All$binno)
bins <- sort(unique(c(dat_Graptoloidea$binno, dat_All$binno)))
colnames_bins <- as.character(bins)

# 4. Create result matrices for each iteration
All_results <- matrix(NA, nrow=n_iter, ncol=length(bins))
colnames(All_results) <- colnames_bins
Graptoloidea_results <- matrix(NA, nrow=n_iter, ncol=length(bins))
colnames(Graptoloidea_results) <- colnames_bins
Ratio_results <- matrix(NA, nrow=n_iter, ncol=length(bins))
colnames(Ratio_results) <- colnames_bins

# 5. Vectors to store mean value in each iteration
divSIB_All_vec <- numeric(n_iter)
divSIB_Graptoloidea_vec <- numeric(n_iter)
ratio_vec <- numeric(n_iter)

# 6. Main loop
for(i in 1:n_iter){
  # Graptoloidea SQS
  res_Graptoloidea <- subsample(dat_Graptoloidea, q=q_level, tax="genus", bin="binno", iter=1, type="sqs")
  # All groups SQS
  res_All <- subsample(dat_All, q=q_level, tax="genus", bin="binno", iter=1, type="sqs")
  
  # Initialize vectors for this iteration
  graptoloidea_div_vec <- rep(NA, length(bins))
  all_div_vec <- rep(NA, length(bins))
  names(graptoloidea_div_vec) <- names(all_div_vec) <- colnames_bins
  
  # Assign SQS results to correct bins
  if (nrow(res_Graptoloidea) > 0 && "binno" %in% names(res_Graptoloidea)) {
    idx <- match(as.character(res_Graptoloidea$binno), colnames_bins)
    ok <- !is.na(idx)
    if (any(ok)) graptoloidea_div_vec[idx[ok]] <- res_Graptoloidea$divSIB[ok]
  }
  if (nrow(res_All) > 0 && "binno" %in% names(res_All)) {
    idx <- match(as.character(res_All$binno), colnames_bins)
    ok <- !is.na(idx)
    if (any(ok)) all_div_vec[idx[ok]] <- res_All$divSIB[ok]
  }
  
  # Store each iteration results
  Graptoloidea_results[i,] <- graptoloidea_div_vec
  All_results[i,] <- all_div_vec
  
  # Calculate ratio for this iteration
  ratio_this <- graptoloidea_div_vec / all_div_vec
  Ratio_results[i, ] <- ratio_this
  
  # Store mean values for this iteration
  divSIB_Graptoloidea_vec[i] <- mean(graptoloidea_div_vec, na.rm=TRUE)
  divSIB_All_vec[i] <- mean(all_div_vec, na.rm=TRUE)
  ratio_vec[i] <- mean(ratio_this, na.rm=TRUE)
}

# 7. Descriptive statistics (mean and 95% CI half-width)
ratio_mean <- mean(ratio_vec, na.rm=TRUE)
ratio_sd   <- sd(ratio_vec, na.rm=TRUE)
n_eff      <- sum(!is.na(ratio_vec))
ratio_se   <- ratio_sd / sqrt(n_eff)
ratio_ci_halfwidth <- 1.96 * ratio_se  # Only output half-width

cat("Mean ratio of divSIB (Graptoloidea / All groups) after", n_iter, "iterations:\n")
cat("Mean:", ratio_mean, "\n")
cat("95% CI half-width:", ratio_ci_halfwidth, "\n")

# 8. Save all iteration results
Graptoloidea_results_with_mean <- rbind(Graptoloidea_results, Mean=apply(Graptoloidea_results, 2, mean, na.rm=TRUE))
write.csv(Graptoloidea_results_with_mean, file="F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Graptoloidea_SQS_50iter.csv")
write.csv(All_results, file="F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/AllGraptoloideaTentaculita_SQS_50iter.csv")
write.csv(Ratio_results, file="F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Graptoloidea_ratio_50iter.csv")

# 9. Calculate mean and 95% CI half-width for the ratio in each bin
ratio_mean_perbin <- apply(Ratio_results, 2, mean, na.rm=TRUE)
ratio_sd_perbin   <- apply(Ratio_results, 2, sd, na.rm=TRUE)
ratio_n_perbin    <- apply(Ratio_results, 2, function(x) sum(!is.na(x)))
ratio_se_perbin   <- ratio_sd_perbin / sqrt(ratio_n_perbin)
ratio_ci_perbin   <- 1.96 * ratio_se_perbin # Only output this

# Binno与Age对应关系表
bin_age <- data.frame(
  binno = 1:100,
  age = c(0.00585, 0.07035, 0.4515, 1.287, 2.19, 3.09, 4.4665, 6.2895, 9.438, 12.725,
          14.9, 18.21, 21.735, 25.425, 30.86, 35.805, 39.455, 44.5, 51.9, 57.6,
          60.4, 63.8, 69.05, 77.85, 84.95, 88.05, 91.85, 97.2, 106.75, 117.2,
          123.585, 129.185, 136.2, 142.4, 147.1, 152, 158.15, 163.4, 166.75, 169.55,
          172.8, 179.45, 188.55, 196.2, 200.4, 204.9, 217.75, 232, 239.5, 244.6,
          249.2, 251.551, 253.021, 256.825, 261.895, 265.59, 269.955, 278.255, 286.8, 291.81,
          296.21, 301.3, 305.35, 311.1, 319.2, 327.05, 338.8, 352.8, 365.55, 377.45,
          385.2, 390.5, 400.45, 409.2, 415, 421.1, 424.3, 426.5, 428.95, 431.95,
          435.95, 439.65, 442.3, 444.5, 449.1, 455.7, 462.85, 468.65, 473.85, 481.55,
          487.45, 491.75, 495.5, 498.75, 502.5, 506.75, 511.5, 517.5, 525, 533.9)
)

summary_df <- data.frame(
  binno = as.numeric(names(ratio_mean_perbin)), 
  ratio_mean = ratio_mean_perbin, 
  ratio_ci = ratio_ci_perbin
)

# 合并age列
summary_df <- merge(summary_df, bin_age, by = "binno", all.x = TRUE)
summary_df <- summary_df[order(summary_df$binno), ]

write.csv(
  summary_df,
  file="F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Graptoloidea_ratio_summary.csv", 
  row.names=FALSE
)

######################################################################################################

# Orthoceratoidea

library(divDyn)

# 1. Read data:
# Orthoceratoidea,Nautiloidea.csv == "all orthoceratoidea and nautiloidea" group
dat_All <- read.csv("F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Orthoceratoidea,Nautiloidea.csv", header = TRUE)
# Orthoceratoidea.csv (for numerator)
dat_Orthoceratoidea <- read.csv("F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Orthoceratoidea.csv", header = TRUE)

# 2. Set parameters
n_iter <- 50
q_level <- 0.7

# 3. Prepare time bins (ensure binno is numeric)
dat_Orthoceratoidea$binno  <- as.numeric(dat_Orthoceratoidea$binno)
dat_All$binno    <- as.numeric(dat_All$binno)
bins <- sort(unique(c(dat_Orthoceratoidea$binno, dat_All$binno)))
colnames_bins <- as.character(bins)

# 4. Create result matrices for each iteration
All_results <- matrix(NA, nrow=n_iter, ncol=length(bins))
colnames(All_results) <- colnames_bins
Orthoceratoidea_results <- matrix(NA, nrow=n_iter, ncol=length(bins))
colnames(Orthoceratoidea_results) <- colnames_bins
Ratio_results <- matrix(NA, nrow=n_iter, ncol=length(bins))
colnames(Ratio_results) <- colnames_bins

# 5. Vectors to store mean value in each iteration
divSIB_All_vec <- numeric(n_iter)
divSIB_Orthoceratoidea_vec <- numeric(n_iter)
ratio_vec <- numeric(n_iter)

# 6. Main loop
for(i in 1:n_iter){
  # Orthoceratoidea SQS
  res_Orthoceratoidea <- subsample(dat_Orthoceratoidea, q=q_level, tax="genus", bin="binno", iter=1, type="sqs")
  # All groups SQS
  res_All <- subsample(dat_All, q=q_level, tax="genus", bin="binno", iter=1, type="sqs")
  
  # Initialize vectors for this iteration
  orthoceratoidea_div_vec <- rep(NA, length(bins))
  all_div_vec <- rep(NA, length(bins))
  names(orthoceratoidea_div_vec) <- names(all_div_vec) <- colnames_bins
  
  # Assign SQS results to correct bins
  if (nrow(res_Orthoceratoidea) > 0 && "binno" %in% names(res_Orthoceratoidea)) {
    idx <- match(as.character(res_Orthoceratoidea$binno), colnames_bins)
    ok <- !is.na(idx)
    if (any(ok)) orthoceratoidea_div_vec[idx[ok]] <- res_Orthoceratoidea$divSIB[ok]
  }
  if (nrow(res_All) > 0 && "binno" %in% names(res_All)) {
    idx <- match(as.character(res_All$binno), colnames_bins)
    ok <- !is.na(idx)
    if (any(ok)) all_div_vec[idx[ok]] <- res_All$divSIB[ok]
  }
  
  # Store each iteration results
  Orthoceratoidea_results[i,] <- orthoceratoidea_div_vec
  All_results[i,] <- all_div_vec
  
  # Calculate ratio for this iteration
  ratio_this <- orthoceratoidea_div_vec / all_div_vec
  Ratio_results[i, ] <- ratio_this
  
  # Store mean values for this iteration
  divSIB_Orthoceratoidea_vec[i] <- mean(orthoceratoidea_div_vec, na.rm=TRUE)
  divSIB_All_vec[i] <- mean(all_div_vec, na.rm=TRUE)
  ratio_vec[i] <- mean(ratio_this, na.rm=TRUE)
}

# 7. Descriptive statistics (mean and 95% CI half-width)
ratio_mean <- mean(ratio_vec, na.rm=TRUE)
ratio_sd   <- sd(ratio_vec, na.rm=TRUE)
n_eff      <- sum(!is.na(ratio_vec))
ratio_se   <- ratio_sd / sqrt(n_eff)
ratio_ci_halfwidth <- 1.96 * ratio_se  # Only output half-width

cat("Mean ratio of divSIB (Orthoceratoidea / All groups) after", n_iter, "iterations:\n")
cat("Mean:", ratio_mean, "\n")
cat("95% CI half-width:", ratio_ci_halfwidth, "\n")

# 8. Save all iteration results
Orthoceratoidea_results_with_mean <- rbind(Orthoceratoidea_results, Mean=apply(Orthoceratoidea_results, 2, mean, na.rm=TRUE))
write.csv(Orthoceratoidea_results_with_mean, file="F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Orthoceratoidea_SQS_50iter.csv")
write.csv(All_results, file="F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/AllOrthoceratoideaNautiloidea_SQS_50iter.csv")
write.csv(Ratio_results, file="F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Orthoceratoidea_ratio_50iter.csv")

# 9. Calculate mean and 95% CI half-width for the ratio in each bin
ratio_mean_perbin <- apply(Ratio_results, 2, mean, na.rm=TRUE)
ratio_sd_perbin   <- apply(Ratio_results, 2, sd, na.rm=TRUE)
ratio_n_perbin    <- apply(Ratio_results, 2, function(x) sum(!is.na(x)))
ratio_se_perbin   <- ratio_sd_perbin / sqrt(ratio_n_perbin)
ratio_ci_perbin   <- 1.96 * ratio_se_perbin # Only output this

# Binno与Age对应关系表
bin_age <- data.frame(
  binno = 1:100,
  age = c(0.00585, 0.07035, 0.4515, 1.287, 2.19, 3.09, 4.4665, 6.2895, 9.438, 12.725,
          14.9, 18.21, 21.735, 25.425, 30.86, 35.805, 39.455, 44.5, 51.9, 57.6,
          60.4, 63.8, 69.05, 77.85, 84.95, 88.05, 91.85, 97.2, 106.75, 117.2,
          123.585, 129.185, 136.2, 142.4, 147.1, 152, 158.15, 163.4, 166.75, 169.55,
          172.8, 179.45, 188.55, 196.2, 200.4, 204.9, 217.75, 232, 239.5, 244.6,
          249.2, 251.551, 253.021, 256.825, 261.895, 265.59, 269.955, 278.255, 286.8, 291.81,
          296.21, 301.3, 305.35, 311.1, 319.2, 327.05, 338.8, 352.8, 365.55, 377.45,
          385.2, 390.5, 400.45, 409.2, 415, 421.1, 424.3, 426.5, 428.95, 431.95,
          435.95, 439.65, 442.3, 444.5, 449.1, 455.7, 462.85, 468.65, 473.85, 481.55,
          487.45, 491.75, 495.5, 498.75, 502.5, 506.75, 511.5, 517.5, 525, 533.9)
)

summary_df <- data.frame(
  binno = as.numeric(names(ratio_mean_perbin)), 
  ratio_mean = ratio_mean_perbin, 
  ratio_ci = ratio_ci_perbin
)

# 合并age列
summary_df <- merge(summary_df, bin_age, by = "binno", all.x = TRUE)
# Optional: 按binno排序
summary_df <- summary_df[order(summary_df$binno), ]

write.csv(
  summary_df,
  file="F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Orthoceratoidea_ratio_summary.csv", 
  row.names=FALSE
)


######################################################################################################

# Palaeocopida
library(divDyn)

# 1. Read data:
# Ostracoda.csv == "all ostracoda" group
dat_All <- read.csv("F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Ostracoda.csv", header = TRUE)
# Palaeocopida.csv (for numerator)
dat_Palaeocopida <- read.csv("F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Palaeocopida.csv", header = TRUE)

# 2. Set parameters
n_iter <- 50
q_level <- 0.7

# 3. Prepare time bins (ensure binno is numeric)
dat_Palaeocopida$binno  <- as.numeric(dat_Palaeocopida$binno)
dat_All$binno    <- as.numeric(dat_All$binno)
bins <- sort(unique(c(dat_Palaeocopida$binno, dat_All$binno)))
colnames_bins <- as.character(bins)

# 4. Create result matrices for each iteration
All_results <- matrix(NA, nrow=n_iter, ncol=length(bins))
colnames(All_results) <- colnames_bins
Palaeocopida_results <- matrix(NA, nrow=n_iter, ncol=length(bins))
colnames(Palaeocopida_results) <- colnames_bins
Ratio_results <- matrix(NA, nrow=n_iter, ncol=length(bins))
colnames(Ratio_results) <- colnames_bins

# 5. Vectors to store mean value in each iteration
divSIB_All_vec <- numeric(n_iter)
divSIB_Palaeocopida_vec <- numeric(n_iter)
ratio_vec <- numeric(n_iter)

# 6. Main loop
for(i in 1:n_iter){
  # Palaeocopida SQS
  res_Palaeocopida <- subsample(dat_Palaeocopida, q=q_level, tax="genus", bin="binno", iter=1, type="sqs")
  # All groups SQS
  res_All <- subsample(dat_All, q=q_level, tax="genus", bin="binno", iter=1, type="sqs")
  
  # Initialize vectors for this iteration
  palaeocopida_div_vec <- rep(NA, length(bins))
  all_div_vec <- rep(NA, length(bins))
  names(palaeocopida_div_vec) <- names(all_div_vec) <- colnames_bins
  
  # Assign SQS results to correct bins
  if (nrow(res_Palaeocopida) > 0 && "binno" %in% names(res_Palaeocopida)) {
    idx <- match(as.character(res_Palaeocopida$binno), colnames_bins)
    ok <- !is.na(idx)
    if (any(ok)) palaeocopida_div_vec[idx[ok]] <- res_Palaeocopida$divSIB[ok]
  }
  if (nrow(res_All) > 0 && "binno" %in% names(res_All)) {
    idx <- match(as.character(res_All$binno), colnames_bins)
    ok <- !is.na(idx)
    if (any(ok)) all_div_vec[idx[ok]] <- res_All$divSIB[ok]
  }
  
  # Store each iteration results
  Palaeocopida_results[i,] <- palaeocopida_div_vec
  All_results[i,] <- all_div_vec
  
  # Calculate ratio for this iteration
  ratio_this <- palaeocopida_div_vec / all_div_vec
  Ratio_results[i, ] <- ratio_this
  
  # Store mean values for this iteration
  divSIB_Palaeocopida_vec[i] <- mean(palaeocopida_div_vec, na.rm=TRUE)
  divSIB_All_vec[i] <- mean(all_div_vec, na.rm=TRUE)
  ratio_vec[i] <- mean(ratio_this, na.rm=TRUE)
}

# 7. Descriptive statistics (mean and 95% CI half-width)
ratio_mean <- mean(ratio_vec, na.rm=TRUE)
ratio_sd   <- sd(ratio_vec, na.rm=TRUE)
n_eff      <- sum(!is.na(ratio_vec))
ratio_se   <- ratio_sd / sqrt(n_eff)
ratio_ci_halfwidth <- 1.96 * ratio_se  # Only output half-width

cat("Mean ratio of divSIB (Palaeocopida / All groups) after", n_iter, "iterations:\n")
cat("Mean:", ratio_mean, "\n")
cat("95% CI half-width:", ratio_ci_halfwidth, "\n")

# 8. Save all iteration results
Palaeocopida_results_with_mean <- rbind(Palaeocopida_results, Mean=apply(Palaeocopida_results, 2, mean, na.rm=TRUE))
write.csv(Palaeocopida_results_with_mean, file="F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Palaeocopida_SQS_50iter.csv")
write.csv(All_results, file="F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/AllOstracoda_SQS_50iter.csv")
write.csv(Ratio_results, file="F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Palaeocopida_ratio_50iter.csv")

# 9. Calculate mean and 95% CI half-width for the ratio in each bin
ratio_mean_perbin <- apply(Ratio_results, 2, mean, na.rm=TRUE)
ratio_sd_perbin   <- apply(Ratio_results, 2, sd, na.rm=TRUE)
ratio_n_perbin    <- apply(Ratio_results, 2, function(x) sum(!is.na(x)))
ratio_se_perbin   <- ratio_sd_perbin / sqrt(ratio_n_perbin)
ratio_ci_perbin   <- 1.96 * ratio_se_perbin # Only output this

# Binno与Age对应关系表
bin_age <- data.frame(
  binno = 1:100,
  age = c(0.00585, 0.07035, 0.4515, 1.287, 2.19, 3.09, 4.4665, 6.2895, 9.438, 12.725,
          14.9, 18.21, 21.735, 25.425, 30.86, 35.805, 39.455, 44.5, 51.9, 57.6,
          60.4, 63.8, 69.05, 77.85, 84.95, 88.05, 91.85, 97.2, 106.75, 117.2,
          123.585, 129.185, 136.2, 142.4, 147.1, 152, 158.15, 163.4, 166.75, 169.55,
          172.8, 179.45, 188.55, 196.2, 200.4, 204.9, 217.75, 232, 239.5, 244.6,
          249.2, 251.551, 253.021, 256.825, 261.895, 265.59, 269.955, 278.255, 286.8, 291.81,
          296.21, 301.3, 305.35, 311.1, 319.2, 327.05, 338.8, 352.8, 365.55, 377.45,
          385.2, 390.5, 400.45, 409.2, 415, 421.1, 424.3, 426.5, 428.95, 431.95,
          435.95, 439.65, 442.3, 444.5, 449.1, 455.7, 462.85, 468.65, 473.85, 481.55,
          487.45, 491.75, 495.5, 498.75, 502.5, 506.75, 511.5, 517.5, 525, 533.9)
)

summary_df <- data.frame(
  binno = as.numeric(names(ratio_mean_perbin)), 
  ratio_mean = ratio_mean_perbin, 
  ratio_ci = ratio_ci_perbin
)

# 合并age列
summary_df <- merge(summary_df, bin_age, by = "binno", all.x = TRUE)
# Optional: 按binno排序
summary_df <- summary_df[order(summary_df$binno), ]

write.csv(
  summary_df,
  file="F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Palaeocopida_ratio_summary.csv", 
  row.names=FALSE
)


######################################################################################################

# Rugosa
library(divDyn)

# 1. Read data:
# Rugosa,Porifera,Bryozoa.csv == "all rugosa, porifera and bryozoa" group
dat_All <- read.csv("F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Rugosa,Porifera,Bryozoa.csv", header = TRUE)
# Rugosa.csv (for numerator)
dat_Rugosa <- read.csv("F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Rugosa.csv", header = TRUE)

# 2. Set parameters
n_iter <- 50
q_level <- 0.7

# 3. Prepare time bins (ensure binno is numeric)
dat_Rugosa$binno  <- as.numeric(dat_Rugosa$binno)
dat_All$binno    <- as.numeric(dat_All$binno)
bins <- sort(unique(c(dat_Rugosa$binno, dat_All$binno)))
colnames_bins <- as.character(bins)

# 4. Create result matrices for each iteration
All_results <- matrix(NA, nrow=n_iter, ncol=length(bins))
colnames(All_results) <- colnames_bins
Rugosa_results <- matrix(NA, nrow=n_iter, ncol=length(bins))
colnames(Rugosa_results) <- colnames_bins
Ratio_results <- matrix(NA, nrow=n_iter, ncol=length(bins))
colnames(Ratio_results) <- colnames_bins

# 5. Vectors to store mean value in each iteration
divSIB_All_vec <- numeric(n_iter)
divSIB_Rugosa_vec <- numeric(n_iter)
ratio_vec <- numeric(n_iter)

# 6. Main loop
for(i in 1:n_iter){
  # Rugosa SQS
  res_Rugosa <- subsample(dat_Rugosa, q=q_level, tax="genus", bin="binno", iter=1, type="sqs")
  # All groups SQS
  res_All <- subsample(dat_All, q=q_level, tax="genus", bin="binno", iter=1, type="sqs")
  
  # Initialize vectors for this iteration
  rugosa_div_vec <- rep(NA, length(bins))
  all_div_vec <- rep(NA, length(bins))
  names(rugosa_div_vec) <- names(all_div_vec) <- colnames_bins
  
  # Assign SQS results to correct bins
  if (nrow(res_Rugosa) > 0 && "binno" %in% names(res_Rugosa)) {
    idx <- match(as.character(res_Rugosa$binno), colnames_bins)
    ok <- !is.na(idx)
    if (any(ok)) rugosa_div_vec[idx[ok]] <- res_Rugosa$divSIB[ok]
  }
  if (nrow(res_All) > 0 && "binno" %in% names(res_All)) {
    idx <- match(as.character(res_All$binno), colnames_bins)
    ok <- !is.na(idx)
    if (any(ok)) all_div_vec[idx[ok]] <- res_All$divSIB[ok]
  }
  
  # Store each iteration results
  Rugosa_results[i,] <- rugosa_div_vec
  All_results[i,] <- all_div_vec
  
  # Calculate ratio for this iteration
  ratio_this <- rugosa_div_vec / all_div_vec
  Ratio_results[i, ] <- ratio_this
  
  # Store mean values for this iteration
  divSIB_Rugosa_vec[i] <- mean(rugosa_div_vec, na.rm=TRUE)
  divSIB_All_vec[i] <- mean(all_div_vec, na.rm=TRUE)
  ratio_vec[i] <- mean(ratio_this, na.rm=TRUE)
}

# 7. Descriptive statistics (mean and 95% CI half-width)
ratio_mean <- mean(ratio_vec, na.rm=TRUE)
ratio_sd   <- sd(ratio_vec, na.rm=TRUE)
n_eff      <- sum(!is.na(ratio_vec))
ratio_se   <- ratio_sd / sqrt(n_eff)
ratio_ci_halfwidth <- 1.96 * ratio_se  # Only output half-width

cat("Mean ratio of divSIB (Rugosa / All groups) after", n_iter, "iterations:\n")
cat("Mean:", ratio_mean, "\n")
cat("95% CI half-width:", ratio_ci_halfwidth, "\n")

# 8. Save all iteration results
Rugosa_results_with_mean <- rbind(Rugosa_results, Mean=apply(Rugosa_results, 2, mean, na.rm=TRUE))
write.csv(Rugosa_results_with_mean, file="F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Rugosa_SQS_50iter.csv")
write.csv(All_results, file="F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/AllRugosaPoriferaBryozoa_SQS_50iter.csv")
write.csv(Ratio_results, file="F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Rugosa_ratio_50iter.csv")

# 9. Calculate mean and 95% CI half-width for the ratio in each bin
ratio_mean_perbin <- apply(Ratio_results, 2, mean, na.rm=TRUE)
ratio_sd_perbin   <- apply(Ratio_results, 2, sd, na.rm=TRUE)
ratio_n_perbin    <- apply(Ratio_results, 2, function(x) sum(!is.na(x)))
ratio_se_perbin   <- ratio_sd_perbin / sqrt(ratio_n_perbin)
ratio_ci_perbin   <- 1.96 * ratio_se_perbin # Only output this

# Binno与Age对应关系表
bin_age <- data.frame(
  binno = 1:100,
  age = c(0.00585, 0.07035, 0.4515, 1.287, 2.19, 3.09, 4.4665, 6.2895, 9.438, 12.725,
          14.9, 18.21, 21.735, 25.425, 30.86, 35.805, 39.455, 44.5, 51.9, 57.6,
          60.4, 63.8, 69.05, 77.85, 84.95, 88.05, 91.85, 97.2, 106.75, 117.2,
          123.585, 129.185, 136.2, 142.4, 147.1, 152, 158.15, 163.4, 166.75, 169.55,
          172.8, 179.45, 188.55, 196.2, 200.4, 204.9, 217.75, 232, 239.5, 244.6,
          249.2, 251.551, 253.021, 256.825, 261.895, 265.59, 269.955, 278.255, 286.8, 291.81,
          296.21, 301.3, 305.35, 311.1, 319.2, 327.05, 338.8, 352.8, 365.55, 377.45,
          385.2, 390.5, 400.45, 409.2, 415, 421.1, 424.3, 426.5, 428.95, 431.95,
          435.95, 439.65, 442.3, 444.5, 449.1, 455.7, 462.85, 468.65, 473.85, 481.55,
          487.45, 491.75, 495.5, 498.75, 502.5, 506.75, 511.5, 517.5, 525, 533.9)
)

summary_df <- data.frame(
  binno = as.numeric(names(ratio_mean_perbin)), 
  ratio_mean = ratio_mean_perbin, 
  ratio_ci = ratio_ci_perbin
)

# Merge with age column
summary_df <- merge(summary_df, bin_age, by = "binno", all.x = TRUE)
summary_df <- summary_df[order(summary_df$binno), ]

write.csv(
  summary_df,
  file="F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Rugosa_ratio_summary.csv", 
  row.names=FALSE
)


######################################################################################################

# Spiriferinida

library(divDyn)

# 1. Read data:
# Brachiopoda.csv == "all brachiopods" group
dat_All <- read.csv("F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Brachiopoda.csv", header = TRUE)
# Spiriferinida.csv (for numerator)
dat_Spiriferinida <- read.csv("F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Spiriferinida.csv", header = TRUE)

# 2. Set parameters
n_iter <- 50
q_level <- 0.7

# 3. Prepare time bins (ensure binno is numeric)
dat_Spiriferinida$binno  <- as.numeric(dat_Spiriferinida$binno)
dat_All$binno    <- as.numeric(dat_All$binno)
bins <- sort(unique(c(dat_Spiriferinida$binno, dat_All$binno)))
colnames_bins <- as.character(bins)

# 4. Create result matrices for each iteration
All_results <- matrix(NA, nrow=n_iter, ncol=length(bins))
colnames(All_results) <- colnames_bins
Spiriferinida_results <- matrix(NA, nrow=n_iter, ncol=length(bins))
colnames(Spiriferinida_results) <- colnames_bins
Ratio_results <- matrix(NA, nrow=n_iter, ncol=length(bins))
colnames(Ratio_results) <- colnames_bins

# 5. Vectors to store mean value in each iteration
divSIB_All_vec <- numeric(n_iter)
divSIB_Spiriferinida_vec <- numeric(n_iter)
ratio_vec <- numeric(n_iter)

# 6. Main loop
for(i in 1:n_iter){
  # Spiriferinida SQS
  res_Spiriferinida <- subsample(dat_Spiriferinida, q=q_level, tax="genus", bin="binno", iter=1, type="sqs")
  # All groups SQS
  res_All <- subsample(dat_All, q=q_level, tax="genus", bin="binno", iter=1, type="sqs")
  
  # Initialize vectors for this iteration
  spiriferinida_div_vec <- rep(NA, length(bins))
  all_div_vec <- rep(NA, length(bins))
  names(spiriferinida_div_vec) <- names(all_div_vec) <- colnames_bins
  
  # Assign SQS results to correct bins
  if (nrow(res_Spiriferinida) > 0 && "binno" %in% names(res_Spiriferinida)) {
    idx <- match(as.character(res_Spiriferinida$binno), colnames_bins)
    ok <- !is.na(idx)
    if (any(ok)) spiriferinida_div_vec[idx[ok]] <- res_Spiriferinida$divSIB[ok]
  }
  if (nrow(res_All) > 0 && "binno" %in% names(res_All)) {
    idx <- match(as.character(res_All$binno), colnames_bins)
    ok <- !is.na(idx)
    if (any(ok)) all_div_vec[idx[ok]] <- res_All$divSIB[ok]
  }
  
  # Store each iteration results
  Spiriferinida_results[i,] <- spiriferinida_div_vec
  All_results[i,] <- all_div_vec
  
  # Calculate ratio for this iteration
  ratio_this <- spiriferinida_div_vec / all_div_vec
  Ratio_results[i, ] <- ratio_this
  
  # Store mean values for this iteration
  divSIB_Spiriferinida_vec[i] <- mean(spiriferinida_div_vec, na.rm=TRUE)
  divSIB_All_vec[i] <- mean(all_div_vec, na.rm=TRUE)
  ratio_vec[i] <- mean(ratio_this, na.rm=TRUE)
}

# 7. Descriptive statistics (mean and 95% CI half-width)
ratio_mean <- mean(ratio_vec, na.rm=TRUE)
ratio_sd   <- sd(ratio_vec, na.rm=TRUE)
n_eff      <- sum(!is.na(ratio_vec))
ratio_se   <- ratio_sd / sqrt(n_eff)
ratio_ci_halfwidth <- 1.96 * ratio_se  # Only output half-width

cat("Mean ratio of divSIB (Spiriferinida / All groups) after", n_iter, "iterations:\n")
cat("Mean:", ratio_mean, "\n")
cat("95% CI half-width:", ratio_ci_halfwidth, "\n")

# 8. Save all iteration results
Spiriferinida_results_with_mean <- rbind(Spiriferinida_results, Mean=apply(Spiriferinida_results, 2, mean, na.rm=TRUE))
write.csv(Spiriferinida_results_with_mean, file="F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Spiriferinida_SQS_50iter.csv")
write.csv(All_results, file="F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/AllBrachiopoda_SQS_50iter.csv")
write.csv(Ratio_results, file="F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Spiriferinida_ratio_50iter.csv")

# 9. Calculate mean and 95% CI half-width for the ratio in each bin
ratio_mean_perbin <- apply(Ratio_results, 2, mean, na.rm=TRUE)
ratio_sd_perbin   <- apply(Ratio_results, 2, sd, na.rm=TRUE)
ratio_n_perbin    <- apply(Ratio_results, 2, function(x) sum(!is.na(x)))
ratio_se_perbin   <- ratio_sd_perbin / sqrt(ratio_n_perbin)
ratio_ci_perbin   <- 1.96 * ratio_se_perbin # Only output this

# Binno与Age对应关系表
bin_age <- data.frame(
  binno = 1:100,
  age = c(0.00585, 0.07035, 0.4515, 1.287, 2.19, 3.09, 4.4665, 6.2895, 9.438, 12.725,
          14.9, 18.21, 21.735, 25.425, 30.86, 35.805, 39.455, 44.5, 51.9, 57.6,
          60.4, 63.8, 69.05, 77.85, 84.95, 88.05, 91.85, 97.2, 106.75, 117.2,
          123.585, 129.185, 136.2, 142.4, 147.1, 152, 158.15, 163.4, 166.75, 169.55,
          172.8, 179.45, 188.55, 196.2, 200.4, 204.9, 217.75, 232, 239.5, 244.6,
          249.2, 251.551, 253.021, 256.825, 261.895, 265.59, 269.955, 278.255, 286.8, 291.81,
          296.21, 301.3, 305.35, 311.1, 319.2, 327.05, 338.8, 352.8, 365.55, 377.45,
          385.2, 390.5, 400.45, 409.2, 415, 421.1, 424.3, 426.5, 428.95, 431.95,
          435.95, 439.65, 442.3, 444.5, 449.1, 455.7, 462.85, 468.65, 473.85, 481.55,
          487.45, 491.75, 495.5, 498.75, 502.5, 506.75, 511.5, 517.5, 525, 533.9)
)

summary_df <- data.frame(
  binno = as.numeric(names(ratio_mean_perbin)), 
  ratio_mean = ratio_mean_perbin, 
  ratio_ci = ratio_ci_perbin
)

# 合并age列
summary_df <- merge(summary_df, bin_age, by = "binno", all.x = TRUE)
# Optional: 按binno排序
summary_df <- summary_df[order(summary_df$binno), ]

write.csv(
  summary_df,
  file="F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Spiriferinida_ratio_summary.csv", 
  row.names=FALSE
)

######################################################################################################

# Tabulata
library(divDyn)

# 1. Read data:
# Tabulata,Porifera,Bryozoa.csv == "all tabulata, porifera, bryozoa" group
dat_All <- read.csv("F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Tabulata,Porifera,Bryozoa.csv", header = TRUE)
# Tabulata.csv (for numerator)
dat_Tabulata <- read.csv("F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Tabulata.csv", header = TRUE)

# 2. Set parameters
n_iter <- 50
q_level <- 0.7

# 3. Prepare time bins (ensure binno is numeric)
dat_Tabulata$binno  <- as.numeric(dat_Tabulata$binno)
dat_All$binno    <- as.numeric(dat_All$binno)
bins <- sort(unique(c(dat_Tabulata$binno, dat_All$binno)))
colnames_bins <- as.character(bins)

# 4. Create result matrices for each iteration
All_results <- matrix(NA, nrow=n_iter, ncol=length(bins))
colnames(All_results) <- colnames_bins
Tabulata_results <- matrix(NA, nrow=n_iter, ncol=length(bins))
colnames(Tabulata_results) <- colnames_bins
Ratio_results <- matrix(NA, nrow=n_iter, ncol=length(bins))
colnames(Ratio_results) <- colnames_bins

# 5. Vectors to store mean value in each iteration
divSIB_All_vec <- numeric(n_iter)
divSIB_Tabulata_vec <- numeric(n_iter)
ratio_vec <- numeric(n_iter)

# 6. Main loop
for(i in 1:n_iter){
  # Tabulata SQS
  res_Tabulata <- subsample(dat_Tabulata, q=q_level, tax="genus", bin="binno", iter=1, type="sqs")
  # All groups SQS
  res_All <- subsample(dat_All, q=q_level, tax="genus", bin="binno", iter=1, type="sqs")
  
  # Initialize vectors for this iteration
  tabulata_div_vec <- rep(NA, length(bins))
  all_div_vec <- rep(NA, length(bins))
  names(tabulata_div_vec) <- names(all_div_vec) <- colnames_bins
  
  # Assign SQS results to correct bins
  if (nrow(res_Tabulata) > 0 && "binno" %in% names(res_Tabulata)) {
    idx <- match(as.character(res_Tabulata$binno), colnames_bins)
    ok <- !is.na(idx)
    if (any(ok)) tabulata_div_vec[idx[ok]] <- res_Tabulata$divSIB[ok]
  }
  if (nrow(res_All) > 0 && "binno" %in% names(res_All)) {
    idx <- match(as.character(res_All$binno), colnames_bins)
    ok <- !is.na(idx)
    if (any(ok)) all_div_vec[idx[ok]] <- res_All$divSIB[ok]
  }
  
  # Store each iteration results
  Tabulata_results[i,] <- tabulata_div_vec
  All_results[i,] <- all_div_vec
  
  # Calculate ratio for this iteration
  ratio_this <- tabulata_div_vec / all_div_vec
  Ratio_results[i, ] <- ratio_this
  
  # Store mean values for this iteration
  divSIB_Tabulata_vec[i] <- mean(tabulata_div_vec, na.rm=TRUE)
  divSIB_All_vec[i] <- mean(all_div_vec, na.rm=TRUE)
  ratio_vec[i] <- mean(ratio_this, na.rm=TRUE)
}

# 7. Descriptive statistics (mean and 95% CI half-width)
ratio_mean <- mean(ratio_vec, na.rm=TRUE)
ratio_sd   <- sd(ratio_vec, na.rm=TRUE)
n_eff      <- sum(!is.na(ratio_vec))
ratio_se   <- ratio_sd / sqrt(n_eff)
ratio_ci_halfwidth <- 1.96 * ratio_se  # Only output half-width

cat("Mean ratio of divSIB (Tabulata / All groups) after", n_iter, "iterations:\n")
cat("Mean:", ratio_mean, "\n")
cat("95% CI half-width:", ratio_ci_halfwidth, "\n")

# 8. Save all iteration results
Tabulata_results_with_mean <- rbind(Tabulata_results, Mean=apply(Tabulata_results, 2, mean, na.rm=TRUE))
write.csv(Tabulata_results_with_mean, file="F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Tabulata_SQS_50iter.csv")
write.csv(All_results, file="F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/AllTabuPorBryo_SQS_50iter.csv")
write.csv(Ratio_results, file="F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Tabulata_ratio_50iter.csv")

# 9. Calculate mean and 95% CI half-width for the ratio in each bin
ratio_mean_perbin <- apply(Ratio_results, 2, mean, na.rm=TRUE)
ratio_sd_perbin   <- apply(Ratio_results, 2, sd, na.rm=TRUE)
ratio_n_perbin    <- apply(Ratio_results, 2, function(x) sum(!is.na(x)))
ratio_se_perbin   <- ratio_sd_perbin / sqrt(ratio_n_perbin)
ratio_ci_perbin   <- 1.96 * ratio_se_perbin # Only output this

# Binno与Age对应关系表
bin_age <- data.frame(
  binno = 1:100,
  age = c(0.00585, 0.07035, 0.4515, 1.287, 2.19, 3.09, 4.4665, 6.2895, 9.438, 12.725,
          14.9, 18.21, 21.735, 25.425, 30.86, 35.805, 39.455, 44.5, 51.9, 57.6,
          60.4, 63.8, 69.05, 77.85, 84.95, 88.05, 91.85, 97.2, 106.75, 117.2,
          123.585, 129.185, 136.2, 142.4, 147.1, 152, 158.15, 163.4, 166.75, 169.55,
          172.8, 179.45, 188.55, 196.2, 200.4, 204.9, 217.75, 232, 239.5, 244.6,
          249.2, 251.551, 253.021, 256.825, 261.895, 265.59, 269.955, 278.255, 286.8, 291.81,
          296.21, 301.3, 305.35, 311.1, 319.2, 327.05, 338.8, 352.8, 365.55, 377.45,
          385.2, 390.5, 400.45, 409.2, 415, 421.1, 424.3, 426.5, 428.95, 431.95,
          435.95, 439.65, 442.3, 444.5, 449.1, 455.7, 462.85, 468.65, 473.85, 481.55,
          487.45, 491.75, 495.5, 498.75, 502.5, 506.75, 511.5, 517.5, 525, 533.9)
)

summary_df <- data.frame(
  binno = as.numeric(names(ratio_mean_perbin)), 
  ratio_mean = ratio_mean_perbin, 
  ratio_ci = ratio_ci_perbin
)

# 合并age列
summary_df <- merge(summary_df, bin_age, by = "binno", all.x = TRUE)
summary_df <- summary_df[order(summary_df$binno), ]

write.csv(
  summary_df,
  file="F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Tabulata_ratio_summary.csv", 
  row.names=FALSE
)


######################################################################################################

# Trilobita

library(divDyn)

# 1. Read data:
# Trilobita,Chelicerata,Pancrustacea.csv == "all trilobite and other arthropods" group
dat_All <- read.csv("F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Trilobita,Chelicerata,Pancrustacea.csv", header = TRUE)
# Trilobita.csv (for numerator)
dat_Trilobita <- read.csv("F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Trilobita.csv", header = TRUE)

# 2. Set parameters
n_iter <- 50
q_level <- 0.7

# 3. Prepare time bins (ensure binno is numeric)
dat_Trilobita$binno  <- as.numeric(dat_Trilobita$binno)
dat_All$binno    <- as.numeric(dat_All$binno)
bins <- sort(unique(c(dat_Trilobita$binno, dat_All$binno)))
colnames_bins <- as.character(bins)

# 4. Create result matrices for each iteration
All_results <- matrix(NA, nrow=n_iter, ncol=length(bins))
colnames(All_results) <- colnames_bins
Trilobita_results <- matrix(NA, nrow=n_iter, ncol=length(bins))
colnames(Trilobita_results) <- colnames_bins
Ratio_results <- matrix(NA, nrow=n_iter, ncol=length(bins))
colnames(Ratio_results) <- colnames_bins

# 5. Vectors to store mean value in each iteration
divSIB_All_vec <- numeric(n_iter)
divSIB_Trilobita_vec <- numeric(n_iter)
ratio_vec <- numeric(n_iter)

# 6. Main loop
for(i in 1:n_iter){
  # Trilobita SQS
  res_Trilobita <- subsample(dat_Trilobita, q=q_level, tax="genus", bin="binno", iter=1, type="sqs")
  # All groups SQS
  res_All <- subsample(dat_All, q=q_level, tax="genus", bin="binno", iter=1, type="sqs")
  
  # Initialize vectors for this iteration
  trilobita_div_vec <- rep(NA, length(bins))
  all_div_vec <- rep(NA, length(bins))
  names(trilobita_div_vec) <- names(all_div_vec) <- colnames_bins
  
  # Assign SQS results to correct bins
  if (nrow(res_Trilobita) > 0 && "binno" %in% names(res_Trilobita)) {
    idx <- match(as.character(res_Trilobita$binno), colnames_bins)
    ok <- !is.na(idx)
    if (any(ok)) trilobita_div_vec[idx[ok]] <- res_Trilobita$divSIB[ok]
  }
  if (nrow(res_All) > 0 && "binno" %in% names(res_All)) {
    idx <- match(as.character(res_All$binno), colnames_bins)
    ok <- !is.na(idx)
    if (any(ok)) all_div_vec[idx[ok]] <- res_All$divSIB[ok]
  }
  
  # Store each iteration results
  Trilobita_results[i,] <- trilobita_div_vec
  All_results[i,] <- all_div_vec
  
  # Calculate ratio for this iteration
  ratio_this <- trilobita_div_vec / all_div_vec
  Ratio_results[i, ] <- ratio_this
  
  # Store mean values for this iteration
  divSIB_Trilobita_vec[i] <- mean(trilobita_div_vec, na.rm=TRUE)
  divSIB_All_vec[i] <- mean(all_div_vec, na.rm=TRUE)
  ratio_vec[i] <- mean(ratio_this, na.rm=TRUE)
}

# 7. Descriptive statistics (mean and 95% CI half-width)
ratio_mean <- mean(ratio_vec, na.rm=TRUE)
ratio_sd   <- sd(ratio_vec, na.rm=TRUE)
n_eff      <- sum(!is.na(ratio_vec))
ratio_se   <- ratio_sd / sqrt(n_eff)
ratio_ci_halfwidth <- 1.96 * ratio_se  # Only output half-width

cat("Mean ratio of divSIB (Trilobita / All groups) after", n_iter, "iterations:\n")
cat("Mean:", ratio_mean, "\n")
cat("95% CI half-width:", ratio_ci_halfwidth, "\n")

# 8. Save all iteration results
Trilobita_results_with_mean <- rbind(Trilobita_results, Mean=apply(Trilobita_results, 2, mean, na.rm=TRUE))
write.csv(Trilobita_results_with_mean, file="F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Trilobita_SQS_50iter.csv")
write.csv(All_results, file="F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/AllArthropod_SQS_50iter.csv")
write.csv(Ratio_results, file="F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Trilobita_ratio_50iter.csv")

# 9. Calculate mean and 95% CI half-width for the ratio in each bin
ratio_mean_perbin <- apply(Ratio_results, 2, mean, na.rm=TRUE)
ratio_sd_perbin   <- apply(Ratio_results, 2, sd, na.rm=TRUE)
ratio_n_perbin    <- apply(Ratio_results, 2, function(x) sum(!is.na(x)))
ratio_se_perbin   <- ratio_sd_perbin / sqrt(ratio_n_perbin)
ratio_ci_perbin   <- 1.96 * ratio_se_perbin # Only output this

# Binno与Age对应关系表
bin_age <- data.frame(
  binno = 1:100,
  age = c(0.00585, 0.07035, 0.4515, 1.287, 2.19, 3.09, 4.4665, 6.2895, 9.438, 12.725,
          14.9, 18.21, 21.735, 25.425, 30.86, 35.805, 39.455, 44.5, 51.9, 57.6,
          60.4, 63.8, 69.05, 77.85, 84.95, 88.05, 91.85, 97.2, 106.75, 117.2,
          123.585, 129.185, 136.2, 142.4, 147.1, 152, 158.15, 163.4, 166.75, 169.55,
          172.8, 179.45, 188.55, 196.2, 200.4, 204.9, 217.75, 232, 239.5, 244.6,
          249.2, 251.551, 253.021, 256.825, 261.895, 265.59, 269.955, 278.255, 286.8, 291.81,
          296.21, 301.3, 305.35, 311.1, 319.2, 327.05, 338.8, 352.8, 365.55, 377.45,
          385.2, 390.5, 400.45, 409.2, 415, 421.1, 424.3, 426.5, 428.95, 431.95,
          435.95, 439.65, 442.3, 444.5, 449.1, 455.7, 462.85, 468.65, 473.85, 481.55,
          487.45, 491.75, 495.5, 498.75, 502.5, 506.75, 511.5, 517.5, 525, 533.9)
)

summary_df <- data.frame(
  binno = as.numeric(names(ratio_mean_perbin)), 
  ratio_mean = ratio_mean_perbin, 
  ratio_ci = ratio_ci_perbin
)

# 合并age列
summary_df <- merge(summary_df, bin_age, by = "binno", all.x = TRUE)
summary_df <- summary_df[order(summary_df$binno), ]

write.csv(
  summary_df,
  file="F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Trilobita_ratio_summary.csv", 
  row.names=FALSE
)
