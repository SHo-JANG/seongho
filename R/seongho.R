#' Creates histogram, boxplot
#' @export
#' @param x numeric variable
#'#####        12.철강사 Data -기출문제      ######
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

# Question No. 1
# 주어진 데이터 파일은 특정 철강사의 제품코드, 불량코드, 그리고 공정 과정에서 발생한 데이터를 담고 있다.
# 해당 데이터를 이용하여 다음 문제의 답변을 작성하시오.
#
# 제공 데이터 파일: E15Q1_data_raw.csv
#
# 1-24번 컬럼: Analog Data
# 25번 컬럼: 제품코드 (Binary)
# 26번 컬럼: 불량코드 (Integer with range 1 to 7)
# 1) EDA를 실시하여 결과값을 제시하고, 상관분석을 시행하여 변수 선택 및 파생 변수 생성과정을 풀이하시오.
# 2) 전체 데이터를 Train, Validation, Test 용도로 분할하고 시각화 하기
# 3) 불량코드 1에 대하여 Logistic Regression 을 활용하여 이항분류 모델을 생성하라.
#    생성한 모형에 대한 최적의 Cut-off value를 선정 후, confusionMatrix를 제시하라.
#    (반드시 시각화와 통계량을 포함시킬 것)
# 4) Logisitc Regression 을 제외하고 SVM을 포함하여 3가지 다항분류모형을 만들어
#    Precision 과 Sensitivity(TPR)를 제시하시오. 또한 모델향상과정과 최적화과정을
#    통해 ConfusionMatrix 를 도출하시오.
# 5) 상기 3) , 4) 번 4가지 모형 중 1가지를 선택하여 최적의 클러스터링개수(단일집단 ~ 5개)
#    를 제시하시오. 모형성능 향상 과정을 수행하여 Clustering 전후의 F1 Score 와 모형평가
#    결과를 제시하시오.
# library(tidyverse)
# library(data.table)
# data_raw <- fread("E15Q1_data_raw.csv")
#
# View(data_raw)
# str(data_raw)
#
# sum(is.na(data_raw))
#
# data <-
#   data_raw %>%
#   mutate(SteelType = as.factor(SteelType),
#          Fault = as.factor(Fault))
#
# summary(data)
# str(data)
#
# # pivot_longer 에 필요한 패키지
# library(tidyr)
#
# library(dplyr)
# library(ggplot2)
# library(tibble)
#
# # select_if 에 필요한 패키지
# # 히스토그램 변수별 한번에 그리기
# ggplot(data = select_if(data, is.numeric) %>%
#          pivot_longer(cols=everything())) +
#   geom_histogram(aes(x=value), bins=50) +
#   facet_wrap(~name, scales = "free_x")
#
# # 상관계수 한번에 임계치 이상 나오게 하기
#
# data_cor = select_if(data, is.numeric)
# cor_matx  <-  cor(data_cor)
#
# # 상관행렬의 대각선 아랫방향을 0으로 만든다.
# cor_matx[lower.tri(cor_matx, diag=TRUE)] = 0
#
# data.frame(cor_matx) %>%
#   rownames_to_column %>%
#   pivot_longer(cols=(-rowname)) %>%
#   filter(value > 0.7)
#
# library(corrplot)
# corrplot(cor_matx)
# # chart.Correlation(cor_matx)
#
# # Fault 분포
# plot(data$Fault, col = 'red')
#
# # SteelType 별 Fault histogram 그리기
# library(dplyr)
# library(ggplot2)
# library(tibble)
# library(tidyr)
#
# data %>%
#   ggplot(stat = 'count', aes(x=SteelType, fill=Fault)) +
#   geom_bar() +
#   theme_classic()
#
# # 변수 선택 및 파생변수 선택
# # 상관계수가 0.7 이상인 변수들만 추려내서 그것들만 PCA 를 수행한다.
#
# high_cor_col <-
#   data.frame(cor_matx) %>%
#   rownames_to_column %>%
#   pivot_longer(cols=(-rowname)) %>%
#   filter(value > 0.7) %>%
#   select(rowname,name) %>%
#   pivot_longer(cols=c(rowname, name)) %>%
#   pull(value) %>%
#   unique
#
# data_pca  <-  select(data, all_of(high_cor_col))
# data.pca  <-  prcomp(data_pca, center=T, scale=T)
#
# summary(data.pca)
# plot(data.pca, type='l')
# biplot(data.pca)
#
#
# data.pca2  <-  prcomp(data_pca[-392,], center = T, scale=T)
#
# summary(data.pca2)
# plot(data.pca2, type='l')
# biplot(data.pca2)
# # 주성분 변수 1~6번까지 파생변수로 선택
# data_new <-
#   data %>%
#   select(-all_of(high_cor_col)) %>%
#   cbind(data.frame(data.pca$x[,1:6]))
#
# str(data_new)
#
# # 2) 전체 데이터를 Train(50%), Validation(30%), Test(20%) 용도로 분할하고 시각화 하기
#
# library(caret)
# set.seed(93)
#
# folds <- createFolds(data_new$Fault, k=10)
#
# idx_tr <- folds[1:5] %>% unlist %>% unname
# idx_val <- folds[6:8] %>% unlist %>% unname
# idx_te <- folds[9:10] %>% unlist %>% unname
#
# data_tr <- data_new[idx_tr,]
# data_val <- data_new[idx_val,]
# data_te <- data_new[idx_te,]
#
# # 테스트 분리한 것 시각화 (각 분리 데이터 별 Fault 빈도 막대그래프로 그리기)
#
# # 원본 데이터에 분리한 데이터셋 태그 붙이기
# data_new <-
#   data_new %>%
#   mutate(idx = seq(NROW(.)),gubun = ifelse(idx %in% idx_tr, 'tr',
#                                            ifelse(idx %in% idx_val, 'va',
#                                                   'te'))) %>% select(-idx)
#
#
# data_new %>%
#   ggplot(stat = 'count', aes(x = gubun, fill = Fault)) +
#   geom_bar()
#
# # 3) 불량코드 1에 대하여 Logistic Regression 을 활용하여 이항분류 모델을 생성하라.
# # 생성한 모형에 대한 최적의 Cut-off value를 선정 후, confusionMatrix를 제시하라.
# # (반드시 시각화와 통계량을 포함시킬 것)
#
# data_tr_bin <- data_tr %>%
#   mutate(Fault = as.factor(ifelse(Fault == 1, 1, 0)))
# target_vd <- as.factor(ifelse(data_val$Fault == 1, 1, 0))
# target_te <- as.factor(ifelse(data_te$Fault == 1, 1, 0))
#
# data.glm = glm(Fault~., data = data_tr_bin, family = 'binomial')
# data.glm.step <- step(object = data.glm, scope = list(lower = ~1, upper = ~.), direction = 'both')
#
# summary(data.glm.step)
#
# #312.09
#
# ### 최적의 cut-off 찾기
#
# library(Metrics)
#
#
# cut_off <- seq(0,1,0.01)
#
# pred <- predict(data.glm.step, newdata = data_val, type = 'response')
#
# f1 = numeric(0)
# acc = numeric(0)
#
# for(i in seq_along(cut_off)) {
#   pred.class <- ifelse(pred >= cut_off[i], 1, 0)
#   f1[i] = fbeta_score(actual = as.numeric(as.character(target_vd)), predicted = as.numeric(pred.class)) # beta =1 디폴트 f1
#   acc[i] = Metrics::accuracy(actual = as.numeric(as.character(target_vd)), predicted = as.numeric(pred.class)) # beta =1 디폴트 f1
# }
#
# f1_all <- as.data.frame(cbind(cut_off, f1, acc))
# library(dplyr)
#
# f1_all %>% arrange(-f1)
# f1_all %>% arrange(-acc)
# # 0.58426966
#
# # f1, accuracy 모두 최적의 cut-off 는 0.38, f1으로 판단해야 함.
#
# plot(x=f1_all$cut_off, y=f1_all$f1, type = 'l', col = 'red', xlab = "cut_off" , ylab = "score", main = 'cut_off plot')
# par(new=T)
# plot(x=f1_all$cut_off, y=f1_all$acc, type = 'l', col = 'blue', xlab = "" , ylab = "")
# # h, v
# abline(v=0.38, lty = 'dashed')
# legend("topright", legend = c("f1", "accuracy"), pch = c(20,20), col=c("red", "blue"))
#
# # test 셋으로 confusion matrix 작성
#
# pred_te <- predict(data.glm.step, newdata = data_te , type = 'response')
# pred.class.te <- ifelse(pred_te >= 0.38, 1, 0)
#
# # positive = '1' 중요
# conf <- confusionMatrix(as.factor(pred.class.te), as.factor(target_te), positive = '1')
#
# # ROC 커브 작성
#
# # 숫자형으로 넣어야 한다.
# library(ROCR)
# pred.roc <- prediction(as.numeric(pred_te), as.numeric(as.character(target_te)))
#
# plot(performance(pred.roc, "tpr", "fpr"))
#
# abline(a=0, b=1, lty=2, col="black")
#
# # performance의 전체값이 안 나옴 확인 필요
# auc <- performance(pred.roc, "auc")@y.values
#
# # 0.9362711
#
# #4) Logisitc Regression 을 제외하고 SVM을 포함하여 3가지 다항분류모형을 만들어
# # Precision 과 Sensitivity(TPR)를 제시하시오. 또한 모델향상과정과 최적화과정을
# # 통해 ConfusionMatrix 를 도출하시오.
#
# library(caret)
#
# names(getModelInfo())
#
# #SVM
# fit_svm <-
#   train(
#     form = paste0("Y", Fault) ~.,
#     #    form = Fault ~.,
#     data = data_tr_bin,
#     trControl = trainControl(method = "none", classProbs = TRUE),
#     method = "svmLinear",
#     preprocess = c("center", "scale")
#   )
#
# # Random Forest
# fit_rf <-
#   train(
#     form = Fault ~.,
#     data = data_tr_bin,
#     trControl = trainControl(method = "none"),
#     method = "rf",
#     preprocess = c("center", "scale")
#   )
#
# # KNN
# fit_knn <-
#   train(
#     form = Fault ~.,
#     data = data_tr_bin,
#     trControl = trainControl(method = "none"),
#     method = "knn"
#     #    preprocess = c("center", "scale")
#   )
#
# # glm
# fit_glm <-
#   train(
#     form = Fault ~.,
#     data = data_tr_bin,
#     trControl = trainControl(method = "none"),
#     method = "glm",
#     #method = "glmStepAIC",
#     family = "binomial"
#   )
#
#
# # NN
# fit_nnet <-
#   train(
#     form = Fault ~.,
#     data = data_tr_bin,
#     trControl = trainControl(method = "none"),
#     method = "nnet",
#     preprocess = c("center", "scale")
#   )
#
# pred_svm = predict(fit_svm, data_te, type = "prob")[,2]
# pred_rf  = predict(fit_rf,  data_te, type = "prob")[,2]
# pred_knn = predict(fit_knn, data_te, type = "prob")[,2]
#
# # cut-off 0.38 대입
# pred_svm_bin = as.factor(as.numeric(pred_svm >= 0.38))
# pred_rf_bin  = as.factor(as.numeric(pred_rf  >= 0.38))
# pred_knn_bin = as.factor(as.numeric(pred_knn >= 0.38))
#
# confusionMatrix(pred_svm_bin, target_te, positive = '1')
# confusionMatrix(pred_rf_bin, target_te,  positive = '1')
# confusionMatrix(pred_knn_bin, target_te, positive = '1')
#
# # 5) 상기 3) , 4) 번 4가지 모형 중 1가지를 선택하여 최적의 클러스터링개수(단일집단 ~ 5개)
# # 를 제시하시오. 모형성능 향상 과정을 수행하여 Clustering 전후의 F1 Score 와 모형평가
# # 결과를 제시하시오.
#
# # 최적의 군집갯수를 찾은 다음 cluster 를 범주형 변수로 추가해서 모형에 넣고 돌린 후, 성능 차이 분석
# # 원 데이터에서 Fault(종속변수) 제거, 범주형 변수는 numeric 을 변경, 모두 표준화 수행
#
# str(data)
#
# data_scale <-
#   data %>%
#   select(-Fault) %>%
#   mutate(SteelType = as.numeric(SteelType)) %>%
#   mutate_all(~scale(.))
#
# library(NbClust)
# #set.seed(93)
#
# nc <- NbClust(data_scale, min.nc = 2, max.nc=5, method = 'kmeans')
# table(nc$Best.n[1,])
#
# # 최적의 군집갯수는 2
#
# kmeans_data <-kmeans(data_scale,2)
# colnames(data_cluster)
#
# # 원래 데이터에 cluster 를 붙인다.
#
# data_cluster <- cbind(data, kmeans_data$cluster) %>%
#   mutate(Fault = as.factor(ifelse(Fault == 1, '1', '0')))
#
# colnames(data_cluster)[28] <- 'cluster'
#
# data_cluster$cluster <- as.factor(data_cluster$cluster)
#
# #str(data_cluster)
#
# # 해당 데이터 테스트 셋 분리
# data_cluster_tr <- data_cluster[idx_tr,]
# data_cluster_val <- data_cluster[idx_val,]
# data_cluster_te <- data_cluster[idx_te,]
# target_val_c <- data_cluster_val$Fault
# target_te_c <- data_cluster_te$Fault
#
# # SVM 으로 모델 생성 및 비교 수행
#
# fit_svm_cluster <-
#   train(
#     form = paste0("Y", Fault) ~.,
#     #    form = Fault ~.,
#     data = data_cluster_tr,
#     trControl = trainControl(method = "none", classProbs = TRUE),
#     method = "svmLinear",
#     preprocess = c("center", "scale")
#   )
#
#
# # 최적의 f1 찾기
# f1_c = numeric(0)
# acc_c = numeric(0)
# pred_svm_cluster_val = predict(fit_svm_cluster, data_cluster_val, type = "prob")[,2]
#
#
# for(i in seq_along(cut_off)) {
#   pred.class_c <- ifelse(pred_svm_cluster_val >= cut_off[i], 1, 0)
#   f1_c[i] = fbeta_score(actual = as.numeric(as.character(target_val_c)), predicted = as.numeric(pred.class_c)) # beta =1 디폴트 f1
#   acc_c[i] = accuracy(actual = as.numeric(as.character(target_val_c)), predicted = as.numeric(pred.class_c)) # beta =1 디폴트 f1
# }
#
# f1_all_c <- as.data.frame(cbind(cut_off, f1_c, acc_c))
#
#
# f1_all_c %>% arrange(-f1_c)
#
# #0.53435115
#
# plot(x=f1_all_c$cut_off, y=f1_all_c$f1_c, type = 'l', col = 'red', xlab = "cut_off" , ylab = "score", main = 'cut_off plot')
# par(new=T)
# plot(x=f1_all_c$cut_off, y=f1_all_c$acc, type = 'l', col = 'blue', xlab = "" , ylab = "")
# # h, v
# abline(v=0.17, lty = 'dashed')
# legend("topright", legend = c("f1", "accuracy"), pch = c(20,20), col=c("red", "blue"))
#
#
# # 0.16일때 최고값(할 때마다 값이 계속 변한다)
#
# # 테스트 데이터로 성능 측정
# # Cluster 데이터 결과
# pred_svm_cluster = predict(fit_svm_cluster, data_cluster_te, type = "prob")[,2]
# pred_svm_bin_cluster = as.factor(as.numeric(pred_svm_cluster >= 0.46))
# caret::confusionMatrix(pred_svm_bin_cluster, target_te_c, positive = '1')
#
# pred_svm_c.roc <- prediction(as.numeric(pred_svm_cluster), as.numeric(target_te_c))
#
# plot(performance(pred_svm_c.roc, "tpr", "fpr"))
#
# abline(a=0, b=1, lty=2, col="black")
#
# # performance의 전체값이 안 나옴 확인 필요
# performance(pred_svm_c.roc, "auc")@y.values
#
# # 0.16일때 0.9243329
#
# # Confusion Matrix and Statistics
# #
# # Reference
# # Prediction   0   1
# # 0 325  10
# # 1  31  22
# #
# # Accuracy : 0.8943
# # 95% CI : (0.8594, 0.9231)
# # No Information Rate : 0.9175
# # P-Value [Acc > NIR] : 0.956119
#
# # Cluster 전의 orginal 결과
# caret::confusionMatrix(pred_svm_bin, target_te, positive = '1')
#
# pred_svm.roc <- prediction(as.numeric(pred_svm), as.numeric(target_te))
#
# plot(performance(pred_svm.roc, "tpr", "fpr"))
#
# abline(a=0, b=1, lty=2, col="black")
#
# # performance의 전체값이 안 나옴 확인 필요
# performance(pred_svm.roc, "auc")@y.values
#
# #0.9308287
#
# # Confusion Matrix and Statistics
# #
# # Reference
# # Prediction   0   1
# # 0 350  22
# # 1   6  10
# #
# # Accuracy : 0.9278
# # 95% CI : (0.8974, 0.9515)
# # No Information Rate : 0.9175
# # P-Value [Acc > NIR] : 0.264091
#
# ## --> 군집분석을 수행 후, SVM 을 수행하였는데 오히려 성능이 떨어진다.





ds<- function(x){
  # 1. ROW AND 2 COLMNS
  par(mfrow=c(1,2))

  # Histogram
  hist(x,col= rainbow(30))

  #Box plot
  boxplot(x,col="green")
  par(mfrow=c(1,1))
  #Numeric summary
  data.frame(min=min(x),
             median=median(x),
             mean=mean(x),
             max=max(x))
}
