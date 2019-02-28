rowData <- read.csv(file="~/Documents/5615/Assignments/Assignment_1/row_sum.csv", header=T, sep=",");
colData <- read.csv(file="~/Documents/5615/Assignments/Assignment_1/col_sum.csv", header=T, sep=",");
reduce <- read.csv(file="~/Documents/5615/Assignments/Assignment_1/reduce.csv", header=T, sep=",")

plotSU<-function(DF,logscale=1){
  if(!logscale)
    DF$Speedup = 10^(DF$Speedup)
    
  plot(log2(DF$Block.size[1:9]), log10(DF$Speedup[1:9]),
       pch=17,cex = 1, col="red", xlab="Block-size", ylab="Speedup",
       xaxt="n", yaxt="n", ylim=c(floor(min(log10(DF$Speedup))),logscale+ceiling(max(log10(DF$Speedup)))))
  abline(h=0,lwd=2,lty=2,col="grey")
  axis(side=1, at=log2(DF$Block.size[1:9]), labels=DF$Block.size[1:9])
  
  if(logscale){
    tmp = seq(from=floor(min(log10(DF$Speedup))),to=logscale+ceiling(max(log10(DF$Speedup))),by=1)
    #tmp2 = c(paste("(1E",tmp,")x"))
    #tmp2[tmp2=="(1E 0 )x"] <- "0 x"
    tmp2 = c(paste(10^(tmp),"x"))
    axis(side=2, at=c(tmp),labels=tmp2,las=2)
  } else {
    tmp = seq(from=floor(min(log10(DF$Speedup))),to=logscale+ceiling(max(log10(DF$Speedup))),length.out=5)
    tmp2 = c(paste(tmp,"x"))
    
    axis(side=2, at=c(tmp),labels=tmp2,las=2)
  }
  
  points(log2(DF$Block.size[10:18]), log10(DF$Speedup[10:18]), pch = 17, col='green')
  points(log2(DF$Block.size[19:27]), log10(DF$Speedup[19:27]), pch = 17, col='blue')
  points(log2(DF$Block.size[28:36]), log10(DF$Speedup[28:36]), pch = 17, col='purple')
  
  lines(log2(DF$Block.size[1:9]), log10(DF$Speedup[1:9]), lwd = 2, col='red')
  lines(log2(DF$Block.size[10:18]), log10(DF$Speedup[10:18]), lwd = 2, col='green')
  lines(log2(DF$Block.size[19:27]), log10(DF$Speedup[19:27]), lwd = 2, col='blue')
  lines(log2(DF$Block.size[28:36]), log10(DF$Speedup[28:36]), lwd = 2, col='purple')
  
  legend("topright", legend=c("NxM: 1000x1000","NxM: 5000x5000","NxM: 10000x10000","NxM: 30000x30000"), 
         col=c("red","green","blue","purple"), lty=1,lwd=2,pch=17,cex=0.85)
}

plotACC<-function(DF){
  DF$SSE.Err = DF$SSE.Err/100
  plot(log2(DF$Block.size[1:9]), log10(DF$SSE.Err[1:9]),
       pch=17,cex = 1, col='red', xlab="Block-size", ylab="Error",
       xaxt="n", yaxt="n", ylim=c(floor(min(log10(DF$SSE.Err))),1+ceiling(max(log10(DF$SSE.Err)))))
  abline(h=0,lwd=2,lty=2,col="grey")
  axis(side=1, at=log2(DF$Block.size[1:9]), labels=DF$Block.size[1:9])
  tmp = seq(from=floor(min(log10(DF$SSE.Err))),to=1+ceiling(max(log10(DF$Speedup))),by=1)
  tmp2 = c(paste(10^(tmp),"x"))
  #tmp2 = c(paste("1E",tmp))
  #tmp2[tmp2=="1E 0"] <- "0"
  axis(side=2, at=c(tmp),labels=tmp2,las=2)
  
  points(log2(DF$Block.size[10:18]), log10(DF$SSE.Err[10:18]), pch = 17, col='green')
  points(log2(DF$Block.size[19:27]), log10(DF$SSE.Err[19:27]), pch = 17, col='blue')
  points(log2(DF$Block.size[28:36]), log10(DF$SSE.Err[28:36]), pch = 17, col='purple')
  
  lines(log2(DF$Block.size[1:9]), log10(DF$SSE.Err[1:9]), lwd = 2, col='red')
  lines(log2(DF$Block.size[10:18]), log10(DF$SSE.Err[10:18]), lwd = 2, col='green')
  lines(log2(DF$Block.size[19:27]), log10(DF$SSE.Err[19:27]), lwd = 2, col='blue')
  lines(log2(DF$Block.size[28:36]), log10(DF$SSE.Err[28:36]), lwd = 2, col='purple')
  
  legend("topright", legend=c("NxM: 1000x1000","NxM: 5000x5000","NxM: 10000x10000","NxM: 30000x30000"), 
         col=c("red","green","blue","purple"), lty=1,lwd=2,pch=17,cex=0.85)
}


plotTau<-function(DF){
  
  plot(log2(DF$Block.size[1:9]), DF$GPU.time[1:9],
       pch=17,cex = 1, col="red", xlab="Block-size", ylab="GPU-time (s)",
       xaxt="n", ylim=c(min(DF$GPU.time),max(DF$GPU.time)),las=1)
  abline(h=0,lwd=2,lty=2,col="grey")
  axis(side=1, at=log2(DF$Block.size[1:9]), labels=DF$Block.size[1:9])
  
  points(log2(DF$Block.size[10:18]), DF$GPU.time[10:18], pch = 17, col='green')
  points(log2(DF$Block.size[19:27]), DF$GPU.time[19:27], pch = 17, col='blue')
  points(log2(DF$Block.size[28:36]), DF$GPU.time[28:36], pch = 17, col='purple')
  
  lines(log2(DF$Block.size[1:9]), DF$GPU.time[1:9], lwd = 2, col='red')
  lines(log2(DF$Block.size[10:18]), DF$GPU.time[10:18], lwd = 2, col='green')
  lines(log2(DF$Block.size[19:27]), DF$GPU.time[19:27], lwd = 2, col='blue')
  lines(log2(DF$Block.size[28:36]), DF$GPU.time[28:36], lwd = 2, col='purple')
  
  legend("topright", legend=c("NxM: 1000x1000","NxM: 5000x5000","NxM: 10000x10000","NxM: 30000x30000"), 
         col=c("red","green","blue","purple"), lty=1,lwd=2,pch=17,cex=0.85)
}

plotCPUvsGPU<-function(DF){
  yvals = c(log10(DF$GPU.time),log10(DF$CPU.time))

  plot(log2(DF$Block.size[1:9]), log10(DF$GPU.time[1:9]),
       pch=17,cex = 1, col="red", xlab="Block-size", ylab="GPU-time vs CPU-time",
       xaxt="n", yaxt="n", ylim=c(floor(min(yvals)),ceiling(max(yvals))))
  abline(h=0,lwd=2,lty=2,col="grey")
  axis(side=1, at=log2(DF$Block.size[1:9]), labels=DF$Block.size[1:9])
  
  tmp = seq(from=floor(min(yvals)),to=ceiling(max(yvals)),by=1)
  tmp2 = c(paste(10^(tmp),"s"))
  #tmp2[tmp2=="(1E 0 )s"] <- "0 x"
  axis(side=2, at=c(tmp),labels=tmp2,las=2)
  
  points(log2(DF$Block.size[10:18]), log10(DF$GPU.time[10:18]), pch = 17, col='green')
  points(log2(DF$Block.size[19:27]), log10(DF$GPU.time[19:27]), pch = 17, col='blue')
  points(log2(DF$Block.size[28:36]), log10(DF$GPU.time[28:36]), pch = 17, col='purple')
  
  lines(log2(DF$Block.size[1:9]), log10(DF$GPU.time[1:9]), lwd = 2, col='red')
  lines(log2(DF$Block.size[10:18]), log10(DF$GPU.time[10:18]), lwd = 2, col='green')
  lines(log2(DF$Block.size[19:27]), log10(DF$GPU.time[19:27]), lwd = 2, col='blue')
  lines(log2(DF$Block.size[28:36]), log10(DF$GPU.time[28:36]), lwd = 2, col='purple')
  
  points(log2(DF$Block.size[1:9]), log10(DF$CPU.time[1:9]), pch = 17, col='red')
  points(log2(DF$Block.size[10:18]), log10(DF$CPU.time[10:18]), pch = 17, col='green')
  points(log2(DF$Block.size[19:27]), log10(DF$CPU.time[19:27]), pch = 17, col='blue')
  points(log2(DF$Block.size[28:36]), log10(DF$CPU.time[28:36]), pch = 17, col='purple')
  
  lines(log2(DF$Block.size[1:9]), log10(DF$CPU.time[1:9]), lwd = 2, col='red')
  lines(log2(DF$Block.size[10:18]), log10(DF$CPU.time[10:18]), lwd = 2, col='green')
  lines(log2(DF$Block.size[19:27]), log10(DF$CPU.time[19:27]), lwd = 2, col='blue')
  lines(log2(DF$Block.size[28:36]), log10(DF$CPU.time[28:36]), lwd = 2, col='purple')
  
  #legend("topleft", legend=c("NxM: 1000x1000","NxM: 5000x5000","NxM: 10000x10000","NxM: 30000x30000"), 
         #col=c("red","green","blue","purple"), lty=1,lwd=2,pch=17,cex=0.85)
}

plotSU(rowData)
plotSU(colData)
plotSU(reduce)
plotSU(reduce,0)

plotACC(reduce)

plotTau(rowData)
plotTau(colData)
plotTau(reduce)

plotCPUvsGPU(rowData)
