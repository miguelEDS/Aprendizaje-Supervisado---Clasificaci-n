#----------------------------------------------------------------------------------------------#
#                            Universidad del Valle - Escuela de Estadística                    #
#                                         Programa Académico de:                               #
#                                        Pregrado en Estadística                               #
#                Asignatura : Técnicas de Minería de Datos y Aprendizaje Automático            #           
#                                  Profesor - Jaime Mosquera Restrepo                          #
#----------------------------------------------------------------------------------------------#
# Estudiantes: Yeimy Tatiana Marín código:1524344-3752 -                                       #
#              Miguel Enriquez     código:2023796-334                                          #
#----------------------------------------------------------------------------------------------#
#                         0. Configuración inicial-Librerías requeridas                     ####
#----------------------------------------------------------------------------------------------#
wd="G:\\Mi unidad\\Economía\\7mo semestre\\Mineria de datos\\Tercer taller"       # Ruta al Directorio de trabajo
setwd(wd)                                # Establecer el directorio de trabajo 

#install.packages("easypackages")        # Libreria especial para hacer carga automática de librerias
library("easypackages")

# Listado de librerias requeridas por el script
lib_req<-c("MASS","visdat","corrplot","plotrix","doBy","FactoMineR","factoextra","caret","e1071","pROC","class","rpart","rpart.plot","randomForest")

# Verificación, instalación y carga de librerias.
easypackages::packages(lib_req)         

#----------------------------------------------------------------------------------------------#
#              Aprendizaje Supervisado - Clasificación                                      ####
#              Caso - Data Clientes Cooperativa                                                #
#----------------------------------------------------------------------------------------------#

#----------------------------------------------------------------------------------------------#
## 1. Lectura de Datos, transformación y ajuste de estructura                               ####
#----------------------------------------------------------------------------------------------#

# Lectura de Datos en R
dataclientes=data.frame(read.table("Data Clientes Cooperativa.txt", header = TRUE))

# Visualización de los datos
View(dataclientes); str(dataclientes)

# Consistencia de los datos (Se elimina columna de ID y registro con GENERO = NA)
clientes <- subset(dataclientes[,-1], GENERO != " ")
View(clientes); str(clientes)

# Transformación de las variables categóricas a factor
clientes$GENERO<-as.factor(clientes$GENERO)
clientes$ESTADO_CIVIL<-as.factor(clientes$ESTADO_CIVIL)
clientes$MODALIDAD_PAGO<-as.factor(clientes$MODALIDAD_PAGO)
clientes$HIPOTECA<-as.factor(clientes$HIPOTECA)
clientes$RIESGO<-as.factor(clientes$RIESGO)

# Visualización de datos faltantes
windows(height=10,width=15)
visdat::vis_miss(clientes)   

nobs.comp=sum(complete.cases(clientes))       # Cuenta los registros completos
nobs.miss = sum(!complete.cases(clientes))    # Cuenta los registros con datos faltantes.

#----------------------------------------------------------------------------------------------#
## 2. Análisis exploratorio de los datos                                                    ####
#----------------------------------------------------------------------------------------------#

# Descripción de los datos
summary(clientes)

# Representación Gráfica
color=c("Gray","Blue")             # Define colores para los gráficos, Gris (Cumplimiento) y Azul (Impago)

# Boxplots individuales de todas las variables (Cuantitativas) vs Impago
windows(height=10,width=15);par(mfrow=c(2,3))                 
attach(clientes)
tabla<-prop.table(table(RIESGO))
coord<-barplot(tabla, col=color,ylim=c(0,1), main="Impago (V)")
text(coord,tabla,labels=round(tabla,2), pos=3)
lapply(names(clientes[,-c(3,4,7,8,10)]),function(y){
  boxplot(clientes[,y]~clientes[,"RIESGO"],ylab= y, xlab="Impago",boxwex = 0.5,col=NULL)
  stripchart(clientes[,y] ~ clientes[,"RIESGO"], vertical = T,
             method = "jitter", pch = 19,
             col = color, add = T)
})
detach(clientes)

# Barplot individuales de todas las variables (Cualitativas) vs Impago
windows(height=10,width=15)   
crear_barplots <- function(data, variable, color) {
  par(mfrow = c(2, 2))
  
  lapply(variable, function(var) {
    coord <- barplot(prop.table(table(data$RIESGO, data[[var]])), col = color, beside = TRUE, legend = TRUE,
                     main = var, ylab = "Frecuencia", ylim = c(0, 0.6))
    text(coord, prop.table(table(data$RIESGO, data[[var]])), labels = round(prop.table(table(data$RIESGO, data[[var]])),2), pos = 3)
  })
}

# Se llama la función para generar los gráficos de barras
variables <- c("GENERO", "ESTADO_CIVIL", "MODALIDAD_PAGO", "HIPOTECA")
x11()
crear_barplots(clientes, variables, color)

# Indicadores Resumen de todas las variables cuantitativas a través de la media y la desviación por grupo de Impago
coef.var=function(x){sd(x)/mean(x)}

doBy::summaryBy(.~RIESGO,data=clientes,FUN=mean)
doBy::summaryBy(.~RIESGO,data=clientes,FUN=sd)
doBy::summaryBy(.~RIESGO,data=clientes,FUN=coef.var)

# Visualización bivariada, gráficos de dispersión X´s versus Impago
data<-clientes[,-c(3,4,7,8,10)]

windows(height=10,width=15)
par(mfrow=c(3,4),oma=c(1,1,1,1), mar=c(4,4,1,1))
attach(data)
for (i in 1:4){
  x=names(data)[i]
  rango.x=range(data[,x])
  for(j in (i+1):5){
    y=names(data)[j]
    rango.y=range(data[,y])
    plot(data[RIESGO=="F",x],data[RIESGO=="F",y],xlab=x,ylab=y,xlim=rango.x,ylim=rango.y,col=color[1],pch=20)
    points(data[RIESGO=="V",x],data[RIESGO=="V",y],col=color[2],pch=20)
  }
}

# Visualización Multivariada apoyada en Componentes principales.
Modelo_PCA <- FactoMineR::PCA(data)

windows(height=10,width=15)
factoextra::fviz_pca_ind(Modelo_PCA ,habillage=clientes$RIESGO,col.ind =color,addEllipses = T, ellipse.level = 0.95) 
windows(height=10,width=15)
factoextra::fviz_pca_biplot(Modelo_PCA ,habillage=clientes$RIESGO,col.ind =color) 

#----------------------------------------------------------------------------------------------#
## 3. Entrenamiento y comparación de modelos de clasificación                               ####
#----------------------------------------------------------------------------------------------#

#----------------------------------------------------------------------------------------------#
# 3.1. División de datos entrenamiento/test                                                 ####
#----------------------------------------------------------------------------------------------#

#Selección del porcentaje para dividir los datos.
p.tr=0.8;N=nrow(clientes)
set.seed(222); index.tr=sample(1:N,ceiling(p.tr*N),replace=FALSE)
clientes.tr=clientes[index.tr,]         # Datos de entrenamiento del modelo
clientes.te=clientes[-index.tr,]        # Datos de prueba para evaluar el performance del modelo

# En este punto ajustaremos diferentes tipos de clasificadores a los datos
# disponibles, almacenando sus medidas de desempeño.
Sum.Performance.tr=list()

#----------------------------------------------------------------------------------------------#
### 3.2 Modelo Estadístico. Regresión Logística (RL)                                        ####
#----------------------------------------------------------------------------------------------#
## Ajuste de un Modelo saturado, es decir, modelo con todas las variables.
Model_RLMsat = glm(formula= RIESGO~., data=clientes.tr,family = "binomial")  

# Verifica la significancia individual
summary(Model_RLMsat)                                               
anova(Model_RLMsat) 

# Score generados para convertir a clasificaciones
Score_RLMsat = predict(Model_RLMsat)       # Evalúa los Score estimados para el modelo logístico
                                              
Prob_RLMsat = predict(Model_RLMsat,type = "response") # Transformación de los score a probabilidades

# Creación de las clasificaciones
pc=0.5                              # Seleccionamos un punto de corte pc=0.5
Predict_RLMsat = as.factor(ifelse(Prob_RLMsat>pc,"V","F")) 

# Indicadores de correcta clasificación.
ICC_RLMsat=caret::confusionMatrix(Predict_RLMsat,clientes.tr$RIESGO,dnn=c("Impago - RLSat", "Impago - Obs"),positive = "V",mode="everything")
ICC_RLMsat$byClass                     

# Estrategia Backward para la selección de variables
Model_RLMred=step(Model_RLMsat,direction = "backward")
summary(Model_RLMred)   
anova(Model_RLMred,Model_RLMsat,test="Chisq")
anova(Model_RLMred,test="Chisq")
Model_RLNull= glm(formula= RIESGO~1, data=clientes.tr,family = "binomial")
anova(Model_RLNull,Model_RLMred,test="Chisq")

# Interpretación de los parametros
Int.parRL=function(Modelo){
  Tabla_coef=cbind(Coef=coef(Modelo),IC=confint(Modelo,level=0.95))  # Coeficientes del Modelo e IC 
  Tabla_odds = exp(Tabla_coef)
  colnames(Tabla_odds)[1]="e-beta"
  list(Coeficientes= Tabla_coef,OR=Tabla_odds)
}

Int.parRL(Model_RLMred)

# Predicciones y clasificación con el modelo reducido
Prob_RLMred = predict(Model_RLMred,type = "response")
pc=0.5                              # Seleccionamos un punto de corte pc=0.5
Predict_RLMred = as.factor(ifelse(Prob_RLMred>pc,"V","F")) 

# Indicadores de correcta clasificación.
ICC_RLMred=caret::confusionMatrix(Predict_RLMred,clientes.tr$RIESGO,dnn=c("IMPAGO - RLred", "IMPAGO - Obs"),positive = "V",mode="everything")
ICC_RLMred

# Criterios para evaluar la bondad de ajuste del modelo clasificador

# Visualización de la situación
windows(height=10,width=15) 
with(clientes.tr,{
  dD=density(Prob_RLMred[RIESGO=="V"])
  dS=density(Prob_RLMred[RIESGO=="F"])
  plot(NULL, xlim=range(Prob_RLMred), ylim=range(c(dD$y,dS$y)), 
       type="n",ylab="Densidad",xlab="Probabilidad Predicha")
  lines(dD,col=color[2]);lines(dS,col=color[1])
  text( 0.08,3, "Cumplimiento"); text(0.82,1, "Impago")
  PC.option=c(0.5,0.25,0.75)
  abline(v=PC.option,col="red",lty=2)
  text(PC.option,3,paste("PC = ",PC.option),cex=0.7,adj=0)
})

# Curva ROC: Criterio para explorar un mejor punto de corte para el Modelo RL (Tuning)
roc <- pROC::roc(clientes.tr$RIESGO,Prob_RLMred, auc = TRUE, ci = TRUE)
print(roc)
windows(height=10,width=15)
pROC::plot.roc(roc, legacy.axes = TRUE, print.thres = "best", print.auc = TRUE,
               auc.polygon = FALSE, max.auc.polygon = FALSE, auc.polygon.col = "gainsboro",
               col = 2, grid = TRUE) 

# Probando el nuevo punto de corte (pc)
pc=0.181
Predict_RLMred = as.factor(ifelse(Prob_RLMred>pc,"V","F")) 

# Indicadores de correcta clasificación.
caret::confusionMatrix(Predict_RLMred,clientes.tr$RIESGO,positive = "V")
ICC_RLMred=caret::confusionMatrix(Predict_RLMred,clientes.tr$RIESGO,positive = "V")
ICC_RLMred$byClass                      # Se generan las predicciones 

# Actualizamos el cuadro de comparación de desempeño
Sum.Performance.tr$RL=ICC_RLMred$byClass
rm(Model_RLMsat,Model_RLNull)
#----------------------------------------------------------------------------------------------#
### 3.3 Algoritmo de Aprendizaje Automático. SVM                                            ####                       ####
#----------------------------------------------------------------------------------------------#
Model_svm= e1071::svm(formula = RIESGO ~ ., data = clientes.tr, scale=TRUE,
                      type = 'C-classification',kernel = 'linear',cost=1,epsilon=0.1) 

coef(Model_svm)
summary(Model_svm)

Predict_svm=predict(Model_svm)

# Indicadores de correcta clasificación.
ICC_svm=caret::confusionMatrix(Predict_svm ,clientes.tr$RIESGO,positive = "V")
ICC_svm$byClass

# Se puede mejorar el desempeño de svm?, tuning los hiperparamétros c (penalidad) y epsilon (intensidad)
set.seed(1)        # usaremos la función tune de la libreria e1071
tune_svm = e1071::tune(svm, RIESGO~., data=clientes.tr, nrepeat=10,ranges = list(cost = 2^(0:5)),kernel="linear")
summary(tune_svm)

windows(height=10,width=15)
plot(tune_svm$performances$error,ylim = c(0,1))
plot(tune_svm)

Model_svm = tune_svm$best.model
Predict_svm=predict(Model_svm)

# Indicadores de correcta clasificación.
ICC_svm=caret::confusionMatrix(Predict_svm ,clientes.tr$RIESGO,positive = "V")
ICC_svm$byClass
Sum.Performance.tr$svm=ICC_svm$byClass

#----------------------------------------------------------------------------------------------#
### 3.4 Modelo de Ensamble. Random Forest - Bosque Aleatorio                                ####
#----------------------------------------------------------------------------------------------#
m = ceiling(sqrt(ncol(clientes.tr[,-9])))     # Número de variables predictoras para el proceso de aleatorización.
n.T= 500                                  # Número de Arboles en el bosque.

set.seed(10)
Model_RF = randomForest::randomForest(RIESGO~. ,data=clientes.tr,ntree=n.T,mtry=m,importance=T)

# Visualización de la importancia de las variables.
windows(height=10,width=15)
importancia=importance(Model_RF)
par(mfrow=c(1,2))
barplot(sort(importancia[,4],decreasing=F),col="blue",horiz=T,main=colnames(importancia)[4])
barplot(sort(importancia[,3],decreasing=F),col="blue",horiz=T,main=colnames(importancia)[3])

# Indicadores de correcta clasificación
Predict_RF = predict(Model_RF)
ICC_RF=caret::confusionMatrix(Predict_RF, clientes.tr$RIESGO, positive = "F")
ICC_RF$byClass

# Sintonizando mtry en el bosque aleatorio por validación cruzada.
caret::modelLookup('rf')

set.seed(10)
ctrl<- trainControl(method = "repeatedcv", number=10,repeats = 10 ) # K-fold repetido 10 veces.
valor.mtry=expand.grid(mtry=2:7)     # malla de exploración para mtry =1:7
Model_RF<-train(RIESGO~. ,data=clientes.tr,method="rf",
                trControl=ctrl,tuneGrid=valor.mtry)
Model_RF
windows(height=10,width=15)
plot(Model_RF, ylim= c(0,1),metric="Accuracy")
windows(height=10,width=15)
plot(Model_RF, ylim= c(0,1),metric="Kappa")

# Indicadores de correcta clasificación - RF tuneado
set.seed(10)
Model_RF = randomForest::randomForest(RIESGO~. ,data=clientes.tr,ntree=n.T,mtry=2)
Predict_RF = predict(Model_RF)
ICC_RF=caret::confusionMatrix(Predict_RF,clientes.tr$RIESGO,positive = "V")
ICC_RF$byClass
Sum.Performance.tr$RF=ICC_RF$byClass
#----------------------------------------------------------------------------------------------#
## 4. Comparación del desempeño                                                             ####
#----------------------------------------------------------------------------------------------#

#----------------------------------------------------------------------------------------------#
### 4.1  Comparación con datos de entrenamiento - bondad de ajuste                          ####
#----------------------------------------------------------------------------------------------#
Sum.Performance.tr = do.call(cbind,Sum.Performance.tr)
View(Sum.Performance.tr[,34:36])
#----------------------------------------------------------------------------------------------#
### 4.2  Comparación con datos de test - Capacidad predictiva                               ####
#----------------------------------------------------------------------------------------------#
Sum.Performance.te=list()

# Model_RLMred(pc=0.181)
pc.RL=0.181
Predict_RL.te = predict(Model_RLMred,newdata=clientes.te,type="response")
Class.RL.te= as.factor(ifelse(Predict_RL.te<pc.RL,"F","V"))
ICC_RL.te=caret::confusionMatrix(Class.RL.te,clientes.te$RIESGO,positive = "V")
Sum.Performance.te$RL=ICC_RL.te$byClass

# Model_svm
Predict_svm.te = predict(Model_svm,newdata=clientes.te,type="class")
ICC_svm.te=caret::confusionMatrix(Predict_svm.te,clientes.te$RIESGO,positive = "V")
Sum.Performance.te$svm=ICC_svm.te$byClass

# Model_Model_RF
Predict_RF.te = predict(Model_RF,newdata=clientes.te)
ICC_RF.te=caret::confusionMatrix(Predict_RF.te,clientes.te$RIESGO,positive = "V")
Sum.Performance.te$RF=ICC_RF.te$byClass

Sum.Performance.te = do.call(cbind,Sum.Performance.te)
View(Sum.Performance.te)

