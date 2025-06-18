library(gtsummary)
library(dplyr)

df <- read.csv("../output/clustering/final_clusters.csv")

print("Factor conversions...")
# Factor Conversions
df$sex <- as.factor(df$sex)
df$sex <- recode(df$sex, "1"="Male", "0"="Female")

binary_cols <- c(
  "montreal_upper_gi",
  "montreal_perianal",
  "has_active_symptoms",
  "sampling_steroids",
  "sampling_abx",
  "sampling_asa",
  "sampling_aza",
  "sampling_mp",
  "sampling_ifx",
  "sampling_ada",
  "sampling_vedo",
  "sampling_uste",
  "sampling_tofa",
  "sampling_mtx",
  "sampling_ciclosporin",
  "sampling_filgo",
  "sampling_upa",
  "sampling_risa"
)

df <- df %>%
  mutate(across(binary_cols, as.factor))

df <- df %>%
  mutate(
    study_group = case_when(
      study_group_name_CD == 1   ~ "CD",
      study_group_name_UC == 1   ~ "UC",
      study_group_name_IBDU == 1 ~ "IBDU",
      TRUE                       ~ NA_character_
    ),
    study_group = factor(study_group, levels = c("CD", "UC", "IBDU"))
  ) %>%
  select(-study_group_name_CD, -study_group_name_UC, -study_group_name_IBDU)

df <- df %>%
  mutate(
    montreal_cd_location = case_when(
      montreal_cd_location_L1.Ileal == 1   ~ "L1",
      montreal_cd_location_L2.Colonic == 1   ~ "L2",
      montreal_cd_location_L3.Ileocolonic == 1 ~ "L3",
    ),
    montreal_cd_location = factor(montreal_cd_location, levels = c("L1", "L2", "L3"))
  ) %>%
  select(-montreal_cd_location_L1.Ileal, -montreal_cd_location_L2.Colonic, -montreal_cd_location_L3.Ileocolonic)

df <- df %>%
  mutate(
    montreal_cd_behaviour = case_when(
      montreal_cd_behaviour_B1.Non.stricturing..non.penetrating == 1   ~ "B1",
      montreal_cd_behaviour_B2.Stricturing == 1   ~ "B2",
      montreal_cd_behaviour_B3.Penetrating == 1 ~ "B3",
    ),
    montreal_cd_behaviour = factor(montreal_cd_behaviour, levels = c("B1", "B2", "B3"))
  ) %>%
  select(-montreal_cd_behaviour_B1.Non.stricturing..non.penetrating, -montreal_cd_behaviour_B2.Stricturing, -montreal_cd_behaviour_B3.Penetrating)

df <- df %>%
  mutate(
    montreal_uc_extent = case_when(
      montreal_uc_extent_E1.Proctitis == 1   ~ "E1",
      montreal_uc_extent_E2.Left.sided == 1   ~ "E2",
      montreal_uc_extent_E3.Extensive == 1 ~ "E3",
    ),
    montreal_uc_extent = factor(montreal_uc_extent, levels = c("E1", "E2", "E3"))
  ) %>%
  select(-montreal_uc_extent_E1.Proctitis, -montreal_uc_extent_E2.Left.sided, -montreal_uc_extent_E3.Extensive)

df <- df %>%
  mutate(
    montreal_uc_severity = case_when(
      montreal_uc_severity_S0.Remission == 1   ~ "S0",
      montreal_uc_severity_S1.Mild == 1   ~ "S1",
      montreal_uc_severity_S2.Moderate == 1 ~ "S2",
      montreal_uc_severity_S3.Severe == 1 ~ "S3"
    ),
    montreal_uc_severity = factor(montreal_uc_severity, levels = c("S0", "S1", "S2", "S3"))
  ) %>%
  select(-montreal_uc_severity_S0.Remission, -montreal_uc_severity_S1.Mild, -montreal_uc_severity_S2.Moderate, -montreal_uc_severity_S3.Severe)


df <- df %>%
  mutate(
    smoking_status = case_when(
      is_smoker_Ex.smoker == 1   ~ "Ex-smoker",
      is_smoker_Non.smoker == 1   ~ "Non-smoker",
      is_smoker_Smoker == 1 ~ "Smoker",
    ),
    smoking_status = factor(smoking_status, levels = c("Non-smoker", "Ex-smoker", "Smoker"))
  ) %>%
  select(-is_smoker_Ex.smoker, -is_smoker_Non.smoker, -is_smoker_Smoker)


df <- df %>%
  mutate(
    season = case_when(
      season_spring == 1   ~ "Spring",
      season_summer == 1   ~ "Summer",
      season_autumn == 1   ~ "Autumn",
      season_winter == 1   ~ "Winter"
    ),
    season = factor(season, levels = c("Spring", "Summer", "Autumn", "Winter"))
  ) %>%
  select(-season_spring, -season_summer, -season_autumn, -season_winter)


str(df)
theme_gtsummary_journal(journal = "jama")
print("Creating simplified summary table...")
table <- df %>% 
  select(-cluster) %>%
  tbl_summary(
  by = simplified_cluster
) %>% 
  add_p(test = list(study_group ~ "chisq.test")) %>%
  as_gt() %>%
  gt::gtsave("output/clustering/simplified_cluster_comparisons_table.docx")

print("Creating full summary table...")
table <- df %>% 
  select(-simplified_cluster) %>%
  tbl_summary(
  by = cluster
) %>% 
  add_p(test = list(study_group ~ "chisq.test")) %>%
  as_gt() %>%
  gt::gtsave("output/clustering/full_cluster_comparisons_table.docx")

print("Creating simplified 1 vs 2 table...")
table <- df %>% 
  select(-cluster) %>%
  subset(simplified_cluster %in% c(1, 2)) %>%
  tbl_summary(
    by = simplified_cluster
  ) %>% 
  add_p(test = list(study_group ~ "chisq.test")) %>%
  as_gt() %>%
  gt::gtsave("output/clustering/cluster_1_vs_2.docx")
print("Complete.")
