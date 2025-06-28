# Created by Xiaokang Liu on 03/29/2025 => xklliu@cug.edu.cn

library(stats4)
library(scales)
library(HDInterval)

### PLOT EMPIRICAL
add_geochrono <- function(Y1,Y2){	
  polygon(-c(541.0,541.0,521.0,521.0), c(Y1,Y2,Y2,Y1), col = "#8CB06C", lwd = 0.5) # Terreneuvian
  polygon(-c(521.0,521.0,509.0,509.0), c(Y1,Y2,Y2,Y1), col = "#99C078", lwd = 0.5) # Series 2
  polygon(-c(509.0,509.0,497.0,497.0), c(Y1,Y2,Y2,Y1), col = "#A6CF86", lwd = 0.5) # Miaolingian
  polygon(-c(497.0,497.0,485.4,485.4), c(Y1,Y2,Y2,Y1), col = "#B3E095", lwd = 0.5) # Furongian
  
  polygon(-c(485.4,485.4,470.0,470.0), c(Y1,Y2,Y2,Y1), col = "#1A9D6F", lwd = 0.5) # Lower Ordovician
  polygon(-c(470.0,470.0,458.4,458.4), c(Y1,Y2,Y2,Y1), col = "#4DB47E", lwd = 0.5) # Middle Ordovician
  polygon(-c(458.4,458.4,443.8,443.8), c(Y1,Y2,Y2,Y1), col = "#7FCA93", lwd = 0.5) # Upper Ordovician
  
  polygon(-c(443.8,443.8,433.4,433.4), c(Y1,Y2,Y2,Y1), col = "#99D7B3", lwd = 0.5) # Llandovery
  polygon(-c(433.4,433.4,427.4,427.4), c(Y1,Y2,Y2,Y1), col = "#B3E1C2", lwd = 0.5) # Wenlock
  polygon(-c(427.4,427.4,423.0,423.0), c(Y1,Y2,Y2,Y1), col = "#BFE6CF", lwd = 0.5) # Ludlow
  polygon(-c(423.0,423.0,419.2,419.2), c(Y1,Y2,Y2,Y1), col = "#E6F5E1", lwd = 0.5) # Pridoli
  
  polygon(-c(419.2,419.2,393.3,393.3), c(Y1,Y2,Y2,Y1), col = "#E5AC4D", lwd = 0.5) # Lower Devonian
  polygon(-c(393.3,393.3,382.7,382.7), c(Y1,Y2,Y2,Y1), col = "#F1C868", lwd = 0.5) # Middle Devonian
  polygon(-c(382.7,382.7,358.9,358.9), c(Y1,Y2,Y2,Y1), col = "#F1E19D", lwd = 0.5) # Upper Devonian
  
  polygon(-c(358.9,358.9,323.2,323.2), c(Y1,Y2,Y2,Y1), col = "#678F66", lwd = 0.5) # Mississippian
  polygon(-c(323.2,323.2,298.9,298.9), c(Y1,Y2,Y2,Y1), col = "#7EBCC6", lwd = 0.5) # Pennsylvanian
  
  polygon(-c(298.9,298.9,273,273),     c(Y1,Y2,Y2,Y1), col = "#EF5845", lwd = 0.5) # Cisuralian
  polygon(-c(273,273,259.1,259.1),     c(Y1,Y2,Y2,Y1), col = "#FB745C", lwd = 0.5) # Guadalupian
  polygon(-c(259.1,259.1,251.9,251.9), c(Y1,Y2,Y2,Y1), col = "#f9b4a3", lwd = 0.5) # Lopingian
  
  polygon(-c(251.9,251.9,247.2,247.2), c(Y1,Y2,Y2,Y1), col = "#a05da5", lwd = 0.5) # Lower Triassic
  polygon(-c(247.2,247.2,237,237),     c(Y1,Y2,Y2,Y1), col = "#b282ba", lwd = 0.5) # Middle Triassic
  polygon(-c(237,237,201.3,201.3),     c(Y1,Y2,Y2,Y1), col = "#bc9dca", lwd = 0.5) # Upper Triassic
  
  polygon(-c(201.3,201.3,174.1,174.1), c(Y1,Y2,Y2,Y1), col = "#00b4eb", lwd = 0.5) # Lower Jurassic
  polygon(-c(174.1,174.1,163.5,163.5), c(Y1,Y2,Y2,Y1), col = "#71cfeb", lwd = 0.5) # Middle Jurassic
  polygon(-c(163.5,163.5,145,145),     c(Y1,Y2,Y2,Y1), col = "#abe1fa", lwd = 0.5) # Upper Jurassic
  
  polygon(-c(145,145,100.5,100.5),     c(Y1,Y2,Y2,Y1), col = "#A0C96D", lwd = 0.5) # Lower Cretaceous
  polygon(-c(100.5,100.5,66,66),       c(Y1,Y2,Y2,Y1), col = "#BAD25F", lwd = 0.5) # Upper Cretaceous
  
  polygon(-c(66,66,56,56),             c(Y1,Y2,Y2,Y1), col = "#F8B77D", lwd = 0.5) # Paleocene
  polygon(-c(56,56,33.9,33.9),         c(Y1,Y2,Y2,Y1), col = "#FAC18A", lwd = 0.5) # Eocene
  polygon(-c(33.9,33.9,23.03,23.03),   c(Y1,Y2,Y2,Y1), col = "#FBCC98", lwd = 0.5) # Oligocene
  polygon(-c(23.03,23.03,5.33,5.33),   c(Y1,Y2,Y2,Y1), col = "#FFED00", lwd = 0.5) # Miocene
  polygon(-c(5.33,5.33,2.58,2.58),     c(Y1,Y2,Y2,Y1), col = "#FFF7B2", lwd = 0.5) # Pliocene
  polygon(-c(2.58,2.58,0.0117,0.0117), c(Y1,Y2,Y2,Y1), col = "#FFF1C4", lwd = 0.5) # Pleistocene
  polygon(-c(0.0117,0.0117,0,0),       c(Y1,Y2,Y2,Y1), col = "#FEEBD2", lwd = 0.5) # Holocene
}

plot_polygon <- function(x,t1,t2,color){
  for (lev in seq(0.95,0.10, length.out =10)){
    hpd = hdi(x,lev)
    polygon(x=-c(t1,t1,t2,t2), y = c(as.numeric(hpd[1:2]),rev(as.numeric(hpd[1:2]))),border=F,col=alpha(color,0.075))	
  }
}

extract_data <- function(log_file){
  tbl_post = read.table(log_file, h=T)
  tbl_post_burnin = tbl_post[10:dim(tbl_post)[1],]
  tbl = t(as.matrix(tbl_post_burnin[7:dim(tbl_post_burnin)[2]]))
  # PARSE AGES
  ages = as.numeric(gsub("t_","",row.names(tbl)))
  # get relative diversity
  tbl_frac = t(t(tbl[,-1]))
  #cat(P_predicted_diversity_mean)
  P_predicted_diversity_mean = apply(tbl_frac, FUN=mean,1)
  ages = -ages
  return(list(tbl_frac,ages,P_predicted_diversity_mean))
}

plot_diversity <- function(log_file1,log_file2,col_alpha=0.2,skyline_plot=0, color="#3dab3c", title=title){
  
  tbl_frac1 <- t(read.csv(source_data[i,1],header=TRUE,row.names = NULL, sep = ",", check.names = FALSE))
  ages1 <- -as.numeric(rownames(tbl_frac1))
  # data1 <- extract_data (log_file1)
  # tbl_frac1 <- data1[[1]]
  # ages1 <- data1[[2]]
  P_predicted_diversity_mean1 <- rowMeans(tbl_frac1)
  
  Ylab= "Diversity"		
  hpd501 = hdi(t(tbl_frac1),0.50)
  hpd751 = hdi(t(tbl_frac1),0.75)
  hpd951 = hdi(t(tbl_frac1),0.95)
  # cat(length(ages1),'\n')
  # cat(ages1,'\n')
  
  tbl_frac2 <- t(read.csv(source_data[i,2],header=TRUE,row.names = NULL, sep = ",", check.names = FALSE))
  ages2 <- -as.numeric(rownames(tbl_frac2))
  P_predicted_diversity_mean2 = rowMeans(tbl_frac2)
  col_alpha=0.2
  hpd502 = hdi(t(tbl_frac2),0.50)
  hpd752 = hdi(t(tbl_frac2),0.75)
  hpd952 = hdi(t(tbl_frac2),0.95)
  
  minY = min(hpd951,hpd952)
  maxY = max(hpd951,hpd952)
  pdf(paste('./output1/',row.names(source_data)[1], "_diversity_",Sys.Date(), ".pdf"), width = 6, height = 4)
  plot(ages1,P_predicted_diversity_mean1,type="n",main=row.names(source_data)[1],xlab="Time (Ma)",ylab = Ylab, ylim=c(minY,maxY),xlim = c(-541,0))#min(ages),max(ages) #-541,0
  lines(ages2,P_predicted_diversity_mean2,col='Orange',lwd=1)# #F08519
  if (0==1){
    x1 = -ages
    age_m = x1+c(diff(x1)/2,mean(diff(x1)/2))
    ages_m = c(age_m[1:length(age_m)],max(x1))
    for (i in 2:length(ages)){
      plot_polygon(t(tbl_frac)[,i],ages_m[i],ages_m[i-1],color )
      add_geochrono(0, -0.05*(maxY-minY))
    }
  }else{
    hpd_list1 = list(hpd951,hpd751,hpd501)
    colors1 = c("#7fc97f","#7fc97f","#7fc97f")
    for (i in 1:length(hpd_list1)){
      hpd_temp = hpd_list1[[i]]
      polygon(c(ages1, rev(ages1)), c(hpd_temp[1,], rev(hpd_temp[2,])), col = alpha(colors1[i],col_alpha), border = NA)
    }
    lines(ages1,P_predicted_diversity_mean1,col="#7fc97f",lwd=1)
    
    hpd_list2 = list(hpd952,hpd752,hpd502)
    colors2 = c("#F08519","#F08519","#F08519")
    for (i in 1:length(hpd_list2)){
      hpd_temp2 = hpd_list2[[i]]
      polygon(c(ages2, rev(ages2)), c(hpd_temp2[1,], rev(hpd_temp2[2,])), col = alpha(colors2[i],col_alpha), border = NA)
    }
    #lines(ages2,P_predicted_diversity_mean2,col="#F08519",lwd=1)
    add_geochrono(0, -0.05*(maxY-minY))
  }
  dev.off()
  filename1 <- paste('./output1/',title, "_diversity_",Sys.Date(), ".csv", sep = "")
  write.table(rbind(ages1,P_predicted_diversity_mean1,hpd501,hpd751,hpd951), file = filename1, sep = ",", col.names = FALSE, 
              row.names = c('age1','diversity_mean','hpd50lower','hpd50higher','hpd75lower','hpd75higher','hpd95lower','hpd95higher'))#, append = TRUE
  write.table(rbind(ages2,P_predicted_diversity_mean2,hpd502,hpd752,hpd952), file = filename1, sep = ",", col.names = FALSE, 
              row.names = c('age2','diversity_mean','hpd50lower','hpd50higher','hpd75lower','hpd75higher','hpd95lower','hpd95higher'), append = TRUE)#, append = TRUE
  return(list(-ages1,tbl_frac1,-ages2,tbl_frac2))
}

plot_diversity_ratio <- function(age1,age2,div1,div2,title){
  col_alpha=0.2
  num_resampling <- 10000
  ratio <- array(NA,dim=c(length(age1),num_resampling))
  com_age <- rep(0,length(age1))
  #target <- age1[1]
  #index <- which.min(abs(age2 - age1[n]))
  for (n in 1:length(age1)){#length(index1)
    choice1 <- sample(div1[n,],size=num_resampling,replace=TRUE)
    nindex <- which.min(abs(age2 - age1[n]))
    if (abs(age2[nindex]-age1[n]) <= 1){
      choice2 <- sample(div2[nindex,],size=num_resampling,replace=TRUE)
      com_age[n] <- age2[nindex]
    }else{
      choice2 <- rep(0,num_resampling)
    }
    per_ratio <- choice1 / (choice2 + choice1)
    ratio[n,] <- per_ratio
  }
  ages = -age1
  P_predicted_diversity_mean = apply(ratio, FUN=mean,1)
  Ylab= "ratio Diversity"
  hpd50 = hdi(t(ratio),0.50)#dim = [2,time_bins]
  hpd75 = hdi(t(ratio),0.75)
  hpd95 = hdi(t(ratio),0.95)
  minY = min(hpd95)
  maxY = max(hpd95)#minY,maxY
  cat(dim(hpd50))
  pdf(paste('./output1/',title, "diversity_ratio_", Sys.Date(), ".pdf"), width = 6, height = 4)
  plot(ages,P_predicted_diversity_mean,type="n",main=title,xlab="Time (Ma)",ylab = Ylab, ylim=c(minY,maxY),xlim = c(min(ages),max(ages)))#min(ages),max(ages)  #-541,0
  hpd_list = list(hpd95,hpd75,hpd50)
  colors = c("#4169E1","#4169E1","#4169E1")#colors = c("#cccccc","#969696","#525252")
  for (i in 1:length(hpd_list)){
    hpd_temp = hpd_list[[i]]
    polygon(c(ages, rev(ages)), c(hpd_temp[1,], rev(hpd_temp[2,])), col = alpha(colors[i],col_alpha), border = NA)
    #print(c(hpd_temp[1,], rev(hpd_temp[2,])))
  }
  lines(ages,P_predicted_diversity_mean,col="#4169E1",lwd=1)
  add_geochrono(0, -0.1*(maxY-minY))
  dev.off()
  #return
  filename1 <- paste('./output1/',title, "diversity_ratio_", Sys.Date(), ".csv", sep = "")
  write.table(rbind(age1,com_age,P_predicted_diversity_mean,hpd50,hpd75,hpd95), file = filename1, sep = ",", col.names = FALSE, 
              row.names = c('age1','com_age','mean_ratio','hpd50lower','hpd50higher','hpd75lower','hpd75higher','hpd95lower','hpd95higher'))#, append = TRUE
}



source_data = read.csv("D:/PyRate-master/Song/path.csv",header=T,row.names = 1)

for (i in 1:1){ # 9
  per_title = row.names(source_data)[i]
  
  total <- plot_diversity(source_data[i,1],source_data[i,2],title=row.names(source_data)[i])
  clade1_age <- total[[1]]
  clade1_tbl_frac <- total[[2]]
  clade2_age <- total[[3]]
  clade2_tbl_frac <- total[[4]]
  
  ############plot_diversity_ratio############
  raito_ages <- plot_diversity_ratio(clade1_age,clade2_age,clade1_tbl_frac,clade2_tbl_frac,title=row.names(source_data)[i])
  
}


tbl_frac1 <- t(read.csv(source_data[i,1],header=TRUE,row.names = NULL, sep = ",", check.names = FALSE))
ages1 <- -as.numeric(rownames(tbl_frac1))
# data1 <- extract_data (log_file1)
# tbl_frac1 <- data1[[1]]
# ages1 <- data1[[2]]
P_predicted_diversity_mean1 <- rowMeans(tbl_frac1)

Ylab= "Diversity"		
hpd501 = hdi(t(tbl_frac1),0.50)
hpd751 = hdi(t(tbl_frac1),0.75)
hpd951 = hdi(t(tbl_frac1),0.95)
# cat(length(ages1),'\n')
# cat(ages1,'\n')

tbl_frac2 <- t(read.csv(source_data[i,2],header=TRUE,row.names = NULL, sep = ",", check.names = FALSE))
ages2 <- -as.numeric(rownames(tbl_frac2))
P_predicted_diversity_mean2 = rowMeans(tbl_frac2)
col_alpha=0.2
hpd502 = hdi(t(tbl_frac2),0.50)
hpd752 = hdi(t(tbl_frac2),0.75)
hpd952 = hdi(t(tbl_frac2),0.95)

minY = min(hpd951,hpd952)
maxY = max(hpd951,hpd952)
#pdf(paste('./output1/',row.names(source_data)[1], "_diversity_",Sys.Date(), ".pdf"), width = 6, height = 4)
plot(ages1,P_predicted_diversity_mean1,type="n",main=row.names(source_data)[1],xlab="Time (Ma)",ylab = Ylab, ylim=c(minY,maxY),xlim = c(-541,0))#min(ages),max(ages) #-541,0
lines(ages2,P_predicted_diversity_mean2,col='Orange',lwd=1)# #F08519
if (0==1){
  x1 = -ages
  age_m = x1+c(diff(x1)/2,mean(diff(x1)/2))
  ages_m = c(age_m[1:length(age_m)],max(x1))
  for (i in 2:length(ages)){
    plot_polygon(t(tbl_frac)[,i],ages_m[i],ages_m[i-1],color )
    add_geochrono(0, -0.05*(maxY-minY))
  }
}else{
  hpd_list1 = list(hpd951,hpd751,hpd501)
  colors1 = c("#7fc97f","#7fc97f","#7fc97f")
  for (i in 1:length(hpd_list1)){
    hpd_temp = hpd_list1[[i]]
    polygon(c(ages1, rev(ages1)), c(hpd_temp[1,], rev(hpd_temp[2,])), col = alpha(colors1[i],col_alpha), border = NA)
  }
  lines(ages1,P_predicted_diversity_mean1,col="#7fc97f",lwd=1)
  
  hpd_list2 = list(hpd952,hpd752,hpd502)
  colors2 = c("#F08519","#F08519","#F08519")
  for (i in 1:length(hpd_list2)){
    hpd_temp2 = hpd_list2[[i]]
    polygon(c(ages2, rev(ages2)), c(hpd_temp2[1,], rev(hpd_temp2[2,])), col = alpha(colors2[i],col_alpha), border = NA)
  }
  #lines(ages2,P_predicted_diversity_mean2,col="#F08519",lwd=1)
  add_geochrono(0, -0.05*(maxY-minY))
}

