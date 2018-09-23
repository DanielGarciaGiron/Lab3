#------------------------------------------------------------#
# Laboratorio 3
# Grupo 7
#------------------------------------------------------------#

#----- Instalación e importación de librerias necesarias para correr el programa ----#
for (libreria in c("readr","h2o","neuralnet","readr","ggplot2","caret","Matrix","xgboost")) {
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
train <- read.csv("./datos/train.csv", stringsAsFactors = F)
test <- read.csv("./datos/test.csv", stringsAsFactors = F)
summary(train)

#--------------------------- Graficación de Digitos --------------------------------#
# Se crea una matriz de 28*28 con los valores de los colores de los pixeles
m = matrix(unlist(train[10,-1]),nrow = 28,byrow = T)
# Luego se grafica la matriz, para verificar si el número aparece correctamente
image(m,col=grey.colors(255))

#Como la imagen aparece rotada, debemos de utilizar una función para que aparezca correctamente cada número
rotar <- function(x) t(apply(x, 2, rev)) #rotación de la matriz

#Se grafican varias imagenes y se rotan con la función anterior, para verificar si aparecen correctamente
par(mfrow=c(2,3))
lapply(1:10, 
       function(x) image(
         rotar(matrix(unlist(train[x,-1]),nrow = 28,byrow = T)),
         col=grey.colors(255),
         xlab=train[x,1]
       )
)
#Luego, se regresan las opciones de ploteo a default
par(mfrow=c(1,1))

barplot(table(train[,1]), col=topo.colors(15), main="Cantidad de numeros en Train.csv")


#------------------------------- Modelo Redes Neuronales --------------------------------#
#Se crean sub dataframes con menos datos puesto que las 784 variables tardan bastante tiempo en calcularse
ind <- sample((1:28000), 1000)
sampleTrain <- train[ind,]

#se toman todos los nombres de las columnas y se ingresan como funcion en la redneuronal
nombres<-colnames(sampleTrain[2:785])
nombresEnFormula<-paste(nombres,collapse = "+")

#se calcula la red neuronal con 5 neuronas
form=as.formula(paste("label~",nombresEnFormula,collapse="+"))
RedNeuronal <- neuralnet(formula =  form,
                         data = sampleTrain, hidden = 5)

#se parte el dataset de test de la misma forma que el train, asi se puede comparar resultados obtenidos
sampleTest <- test[ind,]

#se realiza la prediccion
RedResultados <- compute(RedNeuronal,sampleTest)

#se crea una tabla para visualizar los resultados.
TablaResultados <- data.frame(actual = sampleTrain$label, prediccion = round(RedResultados$net.result))

#------------------------------- Modelo Deep Learning --------------------------------#
## start a local h2o cluster
H2O = h2o.init(max_mem_size = '6g', nthreads = -1)
#Usar 6GB de RAM y usar todas los núcleos posibles de la CPU

#Conventir train y test a formato h2o y se prepara el modelo
train[,1] = as.factor(train[,1])
#Se convierten a factores para clasificación
train_h2o = as.h2o(train)
test_h2o = as.h2o(test)

#Se toma el tiempo para determinar efectividad
s <- proc.time()

#Se entrena el modelo, ya que tiene muchos argumentos, se separan en líneas
model = 
  h2o.deeplearning(x = 2:785,  #Número de columnas para predecir 
                   y = 1,   #Número de columna para el label
                   training_frame = train_h2o, #Los datos en formato h2o
                   activation = "RectifierWithDropout", #Algoritmo a utilizar
                   input_dropout_ratio = 0.2,
                   hidden_dropout_ratios = c(0.5,0.5),
                   balance_classes = TRUE, 
                   hidden = c(100,100), #Dos layers de 100 nodos
                   momentum_stable = 0.99,
                   nesterov_accelerated_gradient = T, #Se utiliza para agilizar el algoritmo
                   epochs = 15)

#Se imprime la matriz de confusión
h2o.confusionMatrix(model)

#Se imprime el tiempo pasado
s - proc.time()

#Se predicen los datos y se guarda el resultado
#Clasificar el modelo de test
h2o_y_test <- h2o.predict(model, test_h2o)

#Para leer los resultados, convertir formato H2O format a data frame y guardar como csv
df_y_test = as.data.frame(h2o_y_test)
df_y_test = data.frame(ImageId = seq(1,length(df_y_test$predict)), Label = df_y_test$predict)
write.csv(df_y_test, file = "DeepLearning.csv", row.names=F)

#Se apaga el cluster virtural de h2o para continuar trabajando
h2o.shutdown(prompt = F)


#------------------------------- Modelo Otro Algoritmo (EXtreme Gradient Boosting Training) --------------------------------#
train2 <- train
test2 <- test

#Se remueve label del dataset
Label <- train2$label
train2$label <- NULL

# Quita todos los vectores que sean combinaciones lineales de otros vectores
CombinacionLineal <- findLinearCombos(train2)
train2 <- train2[, -CombinacionLineal$remove]
test2 <- test2[, -CombinacionLineal$remove]

# Quita vectores con varianza cercana a 0
Varianza <- nearZeroVar(train2, saveMetrics = TRUE)
train2 <- train2[, -which(Varianza[1:nrow(Varianza),]$nzv == TRUE)]
test2 <- test2[, -which(Varianza[1:nrow(Varianza),]$nzv == TRUE)]

train2$Label <- Label


#Se definen los parametros de entrenamiento de xgb, todos son los que trae por default.
PARAM <- list(
  booster            = "gbtree",          
  silent             = 0,                 
  eta                = 0.05,              
  gamma              = 0,                 
  max_depth          = 5,                 
  min_child_weight   = 1,                 
  subsample          = 0.70,              
  colsample_bytree   = 0.95,              
  num_parallel_tree  = 1,                 
  lambda             = 0,                 
  lambda_bias        = 0,                 
  alpha              = 0,                 
  objective          = "multi:softmax",   
  num_class          = 10,                
  base_score         = 0.5,               
  eval_metric        = "merror"           
)


#Se convierte train2 en una matriz de diseño. 
train2_SMM <- sparse.model.matrix(Label ~ ., data = train2)
train2_XGB <- xgb.DMatrix(data = train2_SMM, label = Label)

set.seed(1)

#Se entrena el modelo
Modelo <- xgb.train(params      = PARAM, 
                   data        = train2_XGB, 
                   nrounds     = 50, 
                   verbose     = 2, #si es menor a 0 no imprime nada en consola
                   watchlist   = list(train2_SMM = train2_XGB)
)

test2$Label <- 0

#Predice las variables
test2_SMM <- sparse.model.matrix(Label ~ ., data = test2)
PRED <- predict(Modelo, test2_SMM)

#imprime los resultados
print(PRED)