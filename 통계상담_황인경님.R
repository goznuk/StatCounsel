# install.packages('foreign')
library(foreign)

df <- read.spss("통계상담_황인경의뢰자/1. Survival_Analysis_consult.sav", to.data.frame = T, reencode = 'utf-8')
View(df)


# 필요한 패키지 로드
library(survival)
library(dplyr)

# 생존 변수 변환 (0 = 생존, 1 = 실패)
df$survival_status <- ifelse(df$survival == "survive", 0, 1)

# factor 변환
df <- df %>%
  mutate_at(vars(Sex, Systemic_disease, bone_augmentation_procedure, implant_site, jaw), as.factor)
str(df)

# Age 변수 변환
df <- df %>% mutate(Age_group = factor(case_when(
  Age < 40 ~ "Under 40",
  Age >= 40 & Age < 60 ~ "40-59", Age >= 60 ~ "60 and over"
), levels = c("Under 40", "40-59", "60 and over")))

# 결측값 제거 후 분석 (보철물 종류 결측값 제거)
df_filtered <- df %>% filter(!is.na(prosthesis_type))

# Univariate Cox Regression (단변량 Cox 회귀)
variables <- c("Sex", "Age_group", "type_of_disability_Group2", "compliance_with_SPT",
               "Systemic_disease", "bone_augmentation_procedure", "tooth_loss_reason",
               "implant_diameter_group", "implant_length_group", "implant_site",
               "jaw", "prosthesis_type", "periodontal_diagnosis_group")

univariate_results <- lapply(variables, function(var) {
  formula <- as.formula(paste("Surv(fu_total_yr, survival_status) ~", var))
  cox_model <- coxph(formula, data = df, ties = "breslow")
  summary(cox_model)
})
?coxph
# 결과 출력
univariate_results

# 다변량 Cox 회귀 분석 (shared frailty 고려X, AIC 기준 )
multivariate_model <- coxph(Surv(fu_total_yr, survival_status) ~
                              Age_group + type_of_disability_Group2 + tooth_loss_reason + 
                              implant_site + prosthesis_type + periodontal_diagnosis_group, data = df)

summary(multivariate_model)

# 생존곡선 플롯
plot(survfit(multivariate_model), main = "Multivariate Cox Model Survival Curve", xlab = "Time (years)", ylab = "Survival Probability", ylim = c(0.7, 1))



# 다변량 Cox 회귀 분석에 대해 비례위험가정 검정
NPH_CHECK <- cox.zph(multivariate_model) #Schoenfeld 잔차 검정
NPH_CHECK


par(mfrow=c(2,3))
plot(NPH_CHECK)




#log-log plot 시각화 테스트

# Kaplan-Meier 모델 적합 (예: Sex 변수)
km_fit <- survfit(Surv(fu_total_yr, survival_status) ~ Sex, data = df)

# 로그-로그 생존곡선 플롯
plot(km_fit, fun = "cloglog", col = c("blue", "red"), lty = 1:2,
     xlab = "Time", ylab = "Log(-Log(Survival))",
     main = "Log-Log Survival Curves by Sex")

# 범례 추가
legend("topright", legend = levels(factor(df$Sex)), col = c("blue", "red"), lty = 1:2)


library(survival)
library(dplyr)

################## Time dependent 변수 추가 start ################
# Step 1: Dummy 변수 생성
df <- df %>%
  mutate(
    # implant_site (p = Posterior, a = Anterior)
    implant_site_p = ifelse(implant_site == "p", 1, 0),  
    implant_site_a = ifelse(implant_site == "a", 1, 0),  
    
    # prosthesis_type (bridge, single, overdenture)
    prosthesis_bridge = ifelse(prosthesis_type == "bridge", 1, 0),
    prosthesis_single = ifelse(prosthesis_type == "single", 1, 0),
    prosthesis_overdenture = ifelse(prosthesis_type == "overdenture", 1, 0)
  )

# 변환된 데이터 확인
str(df)
# Cox 모델 적합 (time-dependent interaction 포함)
cox_model_fixed <- coxph(Surv(fu_total_yr, survival_status) ~ 
                           type_of_disability_Group2 + tooth_loss_reason + jaw + 
                           implant_site_p + 
                           prosthesis_bridge + prosthesis_single + tt(implant_site_p) + tt(prosthesis_bridge) + tt(prosthesis_single), 
                         data = df, tt=function(x,t,...) x*t)

summary(cox_model_fixed)
################## Time dependent 변수 추가 end ################
