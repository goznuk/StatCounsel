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

# 다변량 Cox 회귀 분석 (shared frailty 고려X, AIC 기준, 변수 선택 (4.2.1 참고) )
multivariate_model <- coxph(Surv(fu_total_yr, survival_status) ~
                              Age_group + type_of_disability_Group2 + tooth_loss_reason + 
                              implant_site + prosthesis_type + periodontal_diagnosis_group, data = df)

summary(multivariate_model)

# 다변량 Cox 회귀 분석에 대해 비례위험가정 검정
NPH_CHECK <- cox.zph(multivariate_model) #Schoenfeld 잔차 검정
NPH_CHECK

par(mfrow=c(2,3))
plot(NPH_CHECK)

################## Time dependent 변수 추가 ################
# Step 1: 변수 더미화 (Dummy Variable Transformation)
df <- df %>%
  mutate(
    implant_site_p = ifelse(implant_site == "p", 1, 0),  
    prosthesis_bridge = ifelse(prosthesis_type == "bridge", 1, 0),
    prosthesis_single = ifelse(prosthesis_type == "single", 1, 0),
    age_under_40 = ifelse(Age_group == "Under 40", 1, 0),
    age_40_59 = ifelse(Age_group == "40-59", 1, 0)
  )

# Step 2: Cox 모델 적합 (시간의존적 효과 포함)
cox_model_fixed <- coxph(Surv(fu_total_yr, survival_status) ~ 
                           type_of_disability_Group2 + tooth_loss_reason + jaw + 
                           implant_site_p + 
                           prosthesis_bridge + prosthesis_single + 
                           age_under_40 + age_40_59 +
                           tt(implant_site_p) + tt(prosthesis_bridge) + tt(prosthesis_single) +
                           tt(age_under_40) + tt(age_40_59), 
                         data = df, tt=function(x,t,...) x*t)

summary(cox_model_fixed)







############## Time dependent 변수 추가 영향 비교 #############

# Time dependent 변수 추가 전
multivariate_model <- coxph(Surv(fu_total_yr, survival_status) ~
                              Age_group + type_of_disability_Group2 + tooth_loss_reason + 
                              implant_site + prosthesis_type + periodontal_diagnosis_group, data = df)

summary(multivariate_model)

# Time dependent 변수 추가 후
cox_model_fixed <- coxph(Surv(fu_total_yr, survival_status) ~ 
                           type_of_disability_Group2 + tooth_loss_reason + jaw + 
                           implant_site_p + 
                           prosthesis_bridge + prosthesis_single + 
                           age_under_40 + age_40_59 +
                           tt(implant_site_p) + tt(prosthesis_bridge) + tt(prosthesis_single) +
                           tt(age_under_40) + tt(age_40_59), 
                         data = df, tt=function(x,t,...) x*t)

summary(cox_model_fixed)