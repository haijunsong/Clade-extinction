##If you encounter an error: Function 'Rcpp' is not available, you need to update Bioconductor
install.packages("Rcpp")
BiocManager::install("org.Hs.eg.db")

######################################################################################################

dat_Hb = read.csv("F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Foraminifera.csv",header = TRUE) 

library(divDyn) 

SQS_result_Hb = subsample(dat_Hb,q = 0.7,tax = "genus", bin = "binno",iter = 50, type = "sqs")

write.csv(SQS_result_Hb, file = "F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Foraminifera_result-0.7.csv") 

######################################################################################################

dat_Hb = read.csv("F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Fusulinoidea.csv",header = TRUE) 

library(divDyn) 

SQS_result_Hb = subsample(dat_Hb,q = 0.7,tax = "genus", bin = "binno",iter = 50, type = "sqs")

write.csv(SQS_result_Hb, file = "F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Fusulinoidea_result-0.7.csv") 


######################################################################################################

dat_Hb = read.csv("F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Brachiopoda.csv",header = TRUE) 

library(divDyn) 

SQS_result_Hb = subsample(dat_Hb,q = 0.7,tax = "genus", bin = "binno",iter = 50, type = "sqs")

write.csv(SQS_result_Hb, file = "F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Brachiopoda_result-0.7.csv") 


######################################################################################################

dat_Hb = read.csv("F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Spiriferinida.csv",header = TRUE) 

library(divDyn) 

SQS_result_Hb = subsample(dat_Hb,q = 0.7,tax = "genus", bin = "binno",iter = 50, type = "sqs")

write.csv(SQS_result_Hb, file = "F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Spiriferinida_result-0.7.csv") 

######################################################################################################

dat_Hb = read.csv("F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Ostracoda.csv",header = TRUE) 

library(divDyn) 

SQS_result_Hb = subsample(dat_Hb,q = 0.7,tax = "genus", bin = "binno",iter = 50, type = "sqs")

write.csv(SQS_result_Hb, file = "F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Ostracoda_result-0.7.csv") 


######################################################################################################

dat_Hb = read.csv("F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Palaeocopida.csv",header = TRUE) 

library(divDyn) 

SQS_result_Hb = subsample(dat_Hb,q = 0.7,tax = "genus", bin = "binno",iter = 50, type = "sqs")

write.csv(SQS_result_Hb, file = "F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Palaeocopida_result-0.7.csv") 


######################################################################################################

dat_Hb = read.csv("F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Fenestrida.csv",header = TRUE) 

library(divDyn) 

SQS_result_Hb = subsample(dat_Hb,q = 0.7,tax = "genus", bin = "binno",iter = 50, type = "sqs")

write.csv(SQS_result_Hb, file = "F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Fenestrida_result-0.7.csv") 


######################################################################################################

dat_Hb = read.csv("F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Rugosa,Tabulata,Porifera,Bryozoa.csv",header = TRUE) 

library(divDyn) 

SQS_result_Hb = subsample(dat_Hb,q = 0.7,tax = "genus", bin = "binno",iter = 50, type = "sqs")

write.csv(SQS_result_Hb, file = "F:/1-投稿/2022-clades灭绝/PBDB Data/SQS 原数据/stage level/Rugosa,Tabulata,Porifera,Bryozoa_result-0.7.csv") 
