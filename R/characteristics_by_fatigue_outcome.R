# Depends on output from 10_validation_combined.ipynb

library(gtsummary)
library(dplyr)

df <- read.csv("../output/demographics/all_cohorts_demographics.csv")

print("Factor conversions...")
# Factor Conversions
df$sex <- as.factor(df$sex)

binary_cols <- c(
  
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
  "sampling_risa",
  "baseline_aza",
  "baseline_mp",
  "baseline_mtx",
  "baseline_asa",
  "baseline_ifx",
  "baseline_ada",
  "baseline_goli",
  "baseline_vedo",
  "baseline_uste",
  "baseline_risa",
  "baseline_tofa",
  "baseline_filgo",
  "baseline_upa",
  "study_group_name",
  "montreal_cd_location",
  "montreal_cd_behaviour",
  "montreal_upper_gi",
  "montreal_perianal",
  "montreal_uc_extent",
  "montreal_uc_severity",
  "is_smoker",
  "fatigue_outcome",
  "participant_location",
  "self_reported_disease_activity"
  
  
)

df <- df %>%
  mutate(across(binary_cols, as.factor)) 

str(df)

theme_gtsummary_journal(journal = "jama")
print("Creating summary table...")
table <- df %>% 
  filter(study %in% c("GIDAMPs", "MUSIC")) %>%
  tbl_summary(
    by = fatigue_outcome,
    missing = "no",
  ) %>% 
  add_p(test = list(study_group_name ~ "chisq.test")) %>%
  as_gt() %>%
  gt::gtsave("output/all_cohorts/characteristics_by_fatigue_outcome.docx")

print(table)


print("Complete.")
