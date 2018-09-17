#------------------------------------------------------------#
# Laboratorio 3
# Grupo 7
#------------------------------------------------------------#

#----- Instalación e importación de librerias necesarias para correr el programa ----#
for (libreria in c("rela","psych","FactoMineR","cluster","mclust","fpc","plyr","ggplot2","tidyverse","factoextra")) {
  if (!require(libreria, character.only=T)) {
    install.packages(libreria)
    library(libreria, character.only=T)
  }
}
#----------------- Descarga de datos y extraccion de los mismos ---------------------#
#Ya que el dataset es de Kaggle y se necesita un sign-in para descargarlo, ya se incluye en el folder

#Se descomprime el archivo y se coloca en un folder llamado datos
unzip("all.zip", exdir = "./datos")

#----------------------------- Lectura de datos ------------------------------------#
#Se lee el archivo descomprimido para realizar un análisis preliminar
train_raw <- read.csv("./datos/train.csv", stringsAsFactors = F)
summary(train_raw)
