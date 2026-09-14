import logging
import logging
import os
import time
import glob
import shutil
import cProfile
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window


logging.basicConfig(filename='program_log_pre.txt', level=logging.INFO)

ICD_RANGES = [
    ('A00-B99', 'Certain infectious and parasitic diseases'),
    ('C00-D49', 'Neoplasms'),
    ('D50-D89', 'Diseases of the blood and blood-forming organs and certain disorders involving the immune mechanism'),
    ('E00-E89', 'Endocrine, nutritional and metabolic diseases'),
    ('F01-F99', 'Mental, Behavioral and Neurodevelopmental disorders'),
    ('G00-G99', 'Diseases of the nervous system'),
    ('H00-H59', 'Diseases of the eye and adnexa'),
    ('H60-H95', 'Diseases of the ear and mastoid process'),
    ('I00-I99', 'Diseases of the circulatory system'),
    ('J00-J99', 'Diseases of the respiratory system'),
    ('K00-K95', 'Diseases of the digestive system'),
    ('L00-L99', 'Diseases of the skin and subcutaneous tissue'),
    ('M00-M99', 'Diseases of the musculoskeletal system and connective tissue'),
    ('N00-N99', 'Diseases of the genitourinary system'),
    ('O00-O9A', 'Pregnancy, childbirth and the puerperium'),
    ('P00-P96', 'Certain conditions originating in the perinatal period'),
    ('Q00-Q99', 'Congenital malformations, deformations and chromosomal abnormalities'),
    ('R00-R99', 'Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified'),
    ('S00-T88', 'Injury, poisoning and certain other consequences of external causes'),
    ('U00-U85', 'Codes for special purposes'),
    ('V00-Y99', 'External causes of morbidity'),
    ('Z00-Z99', 'Factors influencing health status and contact with health services')
]

def log_time_and_step(step_name):
    logging.info(f"{step_name}: {time.ctime()}")

def load_data(spark):
    log_time_and_step("Loading Data")
    csv_file_path = "../OASIS/OASIS_BASE_FILE_031822.csv"
    df = spark.read.csv(csv_file_path, header=True, inferSchema=True)#.sample(0.1,seed=42)
    return df

def prepare_data(df):
    log_time_and_step("Preparing Data")
    rename_map = {
        "BENE_ID": "Beneficiary_ID",
        "ASMT_ID": "Assessment_ID",
        "ASMTFFDT": "Assessment_Effective_Date",
        "CLCHPSCD": "Calculated_HIPPS_Code",
        "FACINTID": "Facility_Internal_ID",
        "HHSMTNTD": "HHA_Assessment_Internal_ID",
        "STATE_ID": "State_ID",
        "SBMSSNDT": "Submission_Date",
        "SBMHPSCD": "Submitted_HIPPS_Code",
        "M0010": "Agency_Medicare_Number",
        "M0014": "Branch_State",
        "M0016": "Branch_Identifier_Number",
        "M0030": "Start_of_Care_Date",
        "M0032RNA": "Resumption_of_Care_Date_NA",
        "M0032RDT": "Resumption_of_Care_Date",
        "M0050": "Patient_State",
        "M0060": "Patient_ZIP_Code",
        "M0066": "Patient_Birth_Date",
        "M0069": "Gender",
        "M0080": "Discipline_of_Person_Completing_Assessment",
        "M0140AIN": "American_Indian_or_Alaska_Native",
        "M0140ASN": "Asian",
        "M0140BLK": "Black_or_African_American",
        "M0140HSP": "Hispanic_or_Latino",
        "M0140HPI": "Native_Hawiian_or_Pacific_Islander",
        "M0140UNK": "Unknown_Race_Ethnicity",
        "M0140WHT": "White",
        "M0150C": "Medicare_Fee_For_Service",
        "M0150D": "Medicare_HMO_Managed_Care",
        "M0903": "Date_of_Last_Home_Visit",
        "M0906": "Discharge_Transfer_Death_Date",
        "M0110ETM": "Medicare_HHA_pymnt_episode_asmt_case_mix_grp_early_later_episode",
        "M1000_DC_IPPS_14_DA": "Discharged_Past_14_Days_From_IPPS",
        "M1000_DC_IRF_14_DA": "Discharged_Past_14_Days_From_IRF",
        "M1000_DC_LTC_14_DA": "Discharged_Past_14_Days_From_LTC",
        "M1000_DC_LTCH_14_DA": "Discharged_Past_14_Days_From_LTCH",
        "M1000_DC_OTH_14_DA": "Discharged_Past_14_Days_From_Other",
        "M1000_DC_PSYCH_14_DA": "Discharged_Past_14_Days_From_Psychiatric_Hospital_Or_Unit",
        "M1000_DC_SNF_14_DA": "Discharged_Past_14_Days_From_SNF_TCU",
        "M1000_DC_NON_14_DA": "Discharged_Past_14_Days_NA",
        "M1005_DSCHG_UK": "Most_Recent_Inpat_Discharge_Date_UK",
        "M1005_INP_DSCHG_DT": "Most_Recent_Inpatient_Discharge_Date",
        "M1030_THH_ENT_NUTR": "Therapies_In_Home_Enteral_Nutrition",
        "M1030_THH_IV_INFUS": "Therapies_In_Home_IV_Infusion",
        "M1030_THH_NONE_ABV": "Therapies_In_Home_None_Above",
        "M1030_THH_PAR_NUTR": "Therapies_In_Home_Parenteral_Nutrition",
        "M1034_PTNT_OVRAL_STUS": "Patient_Overall_Status",
        "M1036_RSK_ALCOHOL": "High_Risk_Factor_Alcohol_Dependency",
        "M1036_RSK_DRUGS": "High_Risk_Factor_Drug_Dependency",
        "M1036_RSK_NONE": "High_Risk_Factor_None_Of_The_Above",
        "M1036_RSK_OBESITY": "High_Risk_Factor_Obesity",
        "M1036_RSK_SMOKING": "High_Risk_Factor_Smoking",
        "M1036_RSK_UK": "High_Risk_Factor_UK",
        "M1100_PTNT_LVG_STUTN": "Patient_Living_Situation",
        "M1200_VISION": "Vision",
        "M1210_HEARG_ABLTY": "Ability_To_Hear",
        "M1220_UNDRSTG_VERBAL_CNTNT": "Understanding_Of_Verbal_Content",
        "M1230_SPEECH": "Speech_And_Oral_Expression",
        "M1240_FRML_PAIN_ASMT": "Formal_Pain_Assessment",
        "M1242_PAIN_FREQ_ACTVTY_MVMT": "Frequency_Of_Pain_Interfering_With_Activity",
        "M1300_PRSR_ULCR_RISK_ASMT": "Pressure_Ulcer_Assessment",
        "M1302_RISK_OF_PRSR_ULCR": "Risk_Of_Developing_Pressure_Ulcers",
        "M1306_UNHLD_STG2_PRSR_ULCR": "Unhealed_Pressure_Ulcer_at_Least_Stage_II",
        "M1307_OLDST_STG2_ONST_DT": "Oldest_Stage_II_Onset_Date",
        "M1307_OLDST_STG2_AT_DSCHRG": "Status_Oldst_Stg_2_Pressure_Ulcer_At_Discharge",
        "M1320_STUS_PRBLM_PRSR_ULCR": "Status_Of_Most_Problematic_Pressure_Ulcer",
        "M1322_NBR_PRU_STG1": "Current_Number_Of_Stage_I_Pressure_Ulcers",
        "M1324_STG_PRBL_PRU": "Stage_Of_Most_Problematic_Pressure_Ulcer",
        "M1330_STAS_ULCR_PRSNT": "Stasis_Ulcer_Present",
        "M1332_NUM_STAS_ULCR": "Current_Number_Of_(Observable)_Stasis_Ulcers",
        "M1334_STUS_PRBLM_STAS_ULCR": "Status_Of_Most_Problematic_Stasis_Ulcer",
        "M1340_SRGCL_WND_PRSNT": "Does_This_Patient_Have_A_Surgical_Wound",
        "M1342_STUS_PRBLM_SRGCL_WND": "Status_Of_Most_Problematic_Surgical_Wound",
        "M1350_LESION_OPEN_WND": "Skin_Lesion_Or_Open_Wound",
        "M1400_WHEN_DYSPNIC": "When_Is_Patient_Dyspneic",
        "M1410_RESPTX_AIRPR": "Resprtry_Treat_At_Home_Airway_Press",
        "M1410_RESPTX_NONE": "Resprtry_Treat_At_Home_None",
        "M1410_RESPTX_OXYGN": "Resprtry_Treat_At_Home_Oxygen",
        "M1410_RESPTX_VENT": "Resprtry_Treat_At_Home_Ventilator",
        "M1600_UTI": "Patient_Treated_For_UTI_Last_14_Days",
        "M1610_UR_INCONT": "Urinary_Incontinence_Or_Catheter_Presence",
        "M1615_INCNTNT_TIMING": "When_Does_Urinary_Incontinence_Occur",
        "M1620_BWL_INCONT": "Bowel_Incontinence_Frequency",
        "M1630_OSTOMY": "Ostomy_For_Bowel_Elimination",
        "M1700_COG_FUNCTION": "Cognitive_Functioning",
        "M1710_WHEN_CONFUSD": "When_Confused",
        "M1720_WHEN_ANXIOUS": "When_Anxious",
        "M1730_STDZ_DPRSN_SCRNG": "Depression_Screening",
        "M1730_PHQ2_DPRSN": "Feeling_Down_Depressed_Or_Hopeless",
        "M1730_PHQ2_LACK_INTRST": "Little_Interest_Or_Pleasure_In_Doing_Things",
        "M1740_BD_DELUSIONS": "Cog_Behavr_Psych_Symp_Delusional",
        "M1740_BD_IMP_DCSN": "Cog_Behavr_Psych_Symp_Impaired_Decision",
        "M1740_BD_MEM_DFICT": "Cog_Behavr_Psych_Symp_Memory_Deficit",
        "M1740_BD_NONE": "Cog_Behavr_Psych_Symp_None_Of_The_Above",
        "M1740_BD_PHYSICAL": "Cog_Behavr_Psych_Symp_Physical_Aggression",
        "M1740_BD_SOC_INAPP": "Cog_Behavr_Psych_Symp_Socially_Inapp",
        "M1740_BD_VERBAL": "Cog_Behavr_Psych_Symp_Verbal_Disruption",
        "M1745_BEH_PROB_FRQ": "Frequency_Of_Disruptive_Behavior_Symptoms",
        "M1750_REC_PSYCH": "Receives_Psych_Nursing_Services",
        "M1800_CU_GROOMING": "Current_Grooming",
        "M1810_CU_DRESS_UPR": "Current_Dress_Upper",
        "M1820_CU_DRESS_LOW": "Current_Dress_Lower",
        "M1830_CRNT_BATHG": "Current_Bathing",
        "M1840_CUR_TOILTG": "Toilet_Transferring",
        "M1845_CUR_TOILTG_HYGN": "Current_Toileting_Hygiene",
        "M1850_CUR_TRNSFRNG": "Transferring",
        "M1860_CRNT_AMBLTN": "Ambulation_Locomotion",
        "M1870_CU_FEEDING": "Current_Feeding",
        "M1880_CU_PREP_MEAL": "Current_Preparing_Light_Meals",
        "M1890_CU_PHONE_USE": "Current_Phone_Use",
        "M1900_PRIOR_ADLIADL_AMBLTN": "Prior_Functioning_ADL_IADL_Ambulation",
        "M1900_PRIOR_ADLIADL_HSEHOLD": "Prior_Functioning_ADL_IADL_Household_Tasks",
        "M1900_PRIOR_ADLIADL_SELF": "Prior_Functioning_ADL_IADL_Self_Care",
        "M1900_PRIOR_ADLIADL_TRNSFR": "Prior_Functioning_ADL_IADL_Transfer",
        "M1910_MLT_FCTR_FALL_RISK_ASMT": "Multi_Factor_Fall_Risk_Assessment",
        "M2010_HIGH_RISK_DRUG_EDCTN": "Patient_Caregiver_High_Risk_Drug_Educ",
        "M2020_CRNT_MGMT_ORAL_MDCTN": "Current_Management_Of_Oral_Medications",
        "M2030_CRNT_MGMT_INJCTN_MDCTN": "Current_Management_Of_Injectable_Meds",
        "M2040_PRIOR_MGMT_INJCTN_MDCTN": "Prior_Medication_Management_Injectable_Meds",
        "M2040_PRIOR_MGMT_ORAL_MDCTN": "Prior_Medication_Management_Oral_Meds",
        "M2110_ADL_IADL_ASTNC_FREQ": "Frequency_Of_ADL_Or_IADL_Assistance_From_Caregiver",
        "M2200_THRPY_NEED_NA_NUM": "Therapy_Need_NA",
        "M2200_THRPY_NEED_NUM": "Therapy_Need_Number_Of_Visits",
        "M2250_PLAN_SMRY_FALL_PRVNT": "Plan_Of_Care_Synopsis_At_Risk_For_Falls",
        "M2250_PLAN_SMRY_DPRSN_INTRVTN": "Plan_Of_Care_Synopsis_Depression",
        "M2250_PLAN_SMRY_DBTS_FT_CARE": "Plan_Of_Care_Synopsis_Diabetic_Foot_Care",
        "M2250_PLAN_SMRY_PAIN_INTRVTN": "Plan_Of_Care_Synopsis_Pain_Intervention",
        "M2250_PLAN_SMRY_PTNT_SPECF": "Plan_Of_Care_Synopsis_Patient_Specific",
        "M2250_PLAN_SMRY_PRSULC_TRTMT": "Plan_Of_Care_Synopsis_Pressure_Ulcer_Moist_Treatment",
        "M2250_PLAN_SMRY_PRSULC_PRVNT": "Plan_Of_Care_Synopsis_Pressure_Ulcer_Prevention",
        "M2310_ECR_MENTL_BHVRL_PRBLM": "Emergent_Care_Reason_Acute_Mental_Behavioral",
        "M2310_ECR_CRDC_DSRTHM": "Emergent_Care_Reason_Cardiac_Dysrhythmia",
        "M2310_ECR_DHYDRTN_MALNTR": "Emergent_Care_Reason_Dehydration_Malnutrition",
        "M2310_ECR_DVT_PULMNRY": "Emergent_Care_Reason_DVT_Pulmonary_Embolus",
        "M2310_ECR_GI_PRBLM": "Emergent_Care_Reason_GI_Issues",
        "M2310_ECR_HRT_FAILR": "Emergent_Care_Reason_Heart_Failure",
        "M2310_ECR_HYPOGLYC": "Emergent_Care_Hypo_Hyperglycemia",
        "M2310_ECR_MEDICAT": "Emergent_Care_Improper_Medication_Administration",
        "M2310_ECR_INJRY_BY_FALL": "Emergent_Care_Reason_Injury_Caused_By_Fall",
        "M2310_ECR_CTHTR_CMPLCTN": "Emergent_Care_Reason_IV_Catheter_Infection",
        "M2310_ECR_MI_CHST_PAIN": "Emergent_Care_Reason_Myocardial_Infarction",
        "M2310_ECR_OTHR_HRT_DEASE": "Emergent_Care_Reason_Other_Heart_Disease",
        "M2310_ECR_RSPRTRY_OTHR": "Emergent_Care_Reason_Other_Respiratory_Problem",
        "M2310_ECR_OTHER": "Emergent_Care_Reason_Other_Than_Above",
        "M2310_ECR_UK": "Emergent_Care_Reason_Unknown",
        "M2310_ECR_RSPRTRY_INFCTN": "Emergent_Care_Reason_Respiratory_Infection",
        "M2310_ECR_STROKE_TIA": "Emergent_Care_Reason_Stroke_CVA_Or_TIA",
        "M2310_ECR_UNCNTLD_PAIN": "Emergent_Care_Reason_Uncontrolled_Pain",
        "M2310_ECR_UTI": "Emergent_Care_Reason_Urinary_Tract_Infection",
        "M2310_ECR_WND_INFCTN_DTRORTN": "Emergent_Care_Reason_Wound_Infection_Or_Deter",
        "M2410_INPAT_FAC": "Inpatient_Facility_Admitted",
        "M2420_DSCHRG_DISP": "Discharge_Disposition",
        "M2430_HOSP_MENTL_BHVRL_PRBLM": "Hospital_Reason_Acute_Mental_Behavioral",
        "M2430_HOSP_CRDC_DSRTHM": "Hospital_Reason_Cardiac_Dysrhythmia",
        "M2430_HOSP_DHYDRTN_MALNTR": "Hospital_Reason_Dehydration_Malnutrition",
        "M2430_HOSP_VN_PULM": "Hospital_Reason_DVT_Pulmonary_Embolus",
        "M2430_HOSP_GI_PRBLM": "Hospital_Reason_GI_Issues",
        "M2430_HOSP_HRT_FAILR": "Hospital_Reason_Heart_Failure",
        "M2430_HOS_HYPOGLYC": "Hospital_Reason_Hypo_Hyperglycemic",
        "M2430_HOSP_MED": "Hospital_Reason_Improper_Medication_Administration",
        "M2430_HOSP_INJRY_BY_FALL": "Hospital_Reason_Injury_Caused_By_Fall",
        "M2430_HOSP_CTHTR_CMPLCTN": "Hospital_Reason_IV_Catheter_Infection_Complication",
        "M2430_HOSP_MI_CHST_PAIN": "Hospital_Reason_Myocardial_Infarction",
        "M2430_HOSP_OTHR_HRT_DEASE": "Hospital_Reason_Other_Heart_Disease",
        "M2430_HOSP_RSPRTRY_OTHR": "Hospital_Reason_Other_Respiratory_Problem",
        "M2430_HOSP_OTHER": "Hospital_Reason_Other_Than_Above",
        "M2430_HOSP_UK": "Hospital_Reason_Reason_Unknown",
        "M2430_HOSP_RSPRTRY_INFCTN": "Hospital_Reason_Respiratory_Infection",
        "M2430_HOSP_SCHLD_TRTMT": "Hospital_Reason_Scheduled_Treatment_Or_Procedure",
        "M2430_HOSP_STROKE_TIA": "Hospital_Reason_Stroke_CVA_Or_TIA",
        "M2430_HOSP_PAIN": "Hospital_Reason_Uncontrolled_Pain",
        "M2430_HOSP_UR_TRCT": "Hospital_Reason_Urinary_Tract_Infect",
        "M2430_HOSP_WND_INFCTN": "Hospital_Reason_Wound_Infection_Deterioration",
        "M1011_14D_INP1_ICD": "Inpatient_Diagnosis_1_ICD_10_C_M",
        "M1011_14D_INP2_ICD": "Inpatient_Diagnosis_2_ICD_10_C_M",
        "M1011_14_DAY_INP3_ICD": "Inpatient_Diagnosis_3_ICD_10_C_M",
        "M1011_14_DAY_INP4_ICD": "Inpatient_Diagnosis_4_ICD_10_C_M",
        "M1011_14_DAY_INP5_ICD": "Inpatient_Diagnosis_5_ICD_10_C_M",
        "M1011_14_DAY_INP6_ICD": "Inpatient_Diagnosis_6_ICD_10_C_M",
        "M1011_14_DAY_ICD_NA": "Inpatient_Diagnosis_ICD_10_C_M_Not_Applicable",
        "M1017_CHGREG_ICD1": "Regimen_Change_Diagnosis_1_ICD_10_C_M",
        "M1017_CHGREG_ICD2": "Regimen_Change_Diagnosis_2_ICD_10_C_M",
        "M1017_CHGREG_ICD3": "Regimen_Change_Diagnosis_3_ICD_10_C_M",
        "M1017_CHGREG_ICD4": "Regimen_Change_Diagnosis_4_ICD_10_C_M",
        "M1017_CHGREG_ICD5": "Regimen_Change_Diagnosis_5_ICD_10_C_M",
        "M1017_CHGREG_ICD6": "Regimen_Change_Diagnosis_6_ICD_10_C_M",
        "M1017_CHGREG_ICD_NA": "Regimen_Change_Not_Applicable_ICD_10_C_M_Code",
        "M1021_PRI_DGN_ICD": "Primary_Diagnosis_ICD_10_C_M_Code",
        "M1021_PRI_DGN_SEV": "Primary_Diagnosis_Severity_Rating_ICD_10_C_M_Code",
        "M1023_OTH_DGN1_ICD": "Other_Diagnosis_Code_1_ICD_10_C_M",
        "M1023_OTH_DGN1_SEV": "Other_Diagnosis_Code_1_Severity_ICD_10_C_M",
        "M1023_OTH_DGN2_ICD": "Other_Diagnosis_Code_2_ICD_10_C_M",
        "M1023_OTH_DGN2_SEV": "Other_Diagnosis_Code_2_Severity_ICD_10_C_M",
        "M1023_OTH_DGN3_ICD": "Other_Diagnosis_Code_3_ICD_10_C_M",
        "M1023_OTH_DGN3_SEV": "Other_Diagnosis_Code_3_Severity_ICD_10_C_M",
        "M1023_OTH_DGN4_ICD": "Other_Diagnosis_Code_4_ICD_10_C_M",
        "M1023_OTH_DGN4_SEV": "Other_Diagnosis_Code_4_Severity_ICD_10_C_M",
        "M1023_OTH_DGN5_ICD_I10": "Other_Diagnosis_Code_5_ICD_10_C_M",
        "M1023_OTH_DGN5_SEV_I10": "Other_Diagnosis_Code_5_Severity_ICD_10_C_M",
        "M1025_PMT_DGNS_ICD_A3_CD": "Primary_Optional_Diagnosis_ICD_10_C_M_Code",
        "M1025_PMT_DGNS_ICD_A4_CD": "Primary_Optional_Diagnosis_Multiple_Codes_ICD_10_C_M",
        "M1025_PMT_DGNS_ICD_B3_CD": "Optional_Diagnosis_Code_1_ICD_10_C_M",
        "M1025_PMT_DGNS_ICD_B4_CD": "Optional_Diagnosis_Multiple_Code_1_ICD_10_C_M",
        "M1025_PMT_DGNS_ICD_C3_CD": "Optional_Diagnosis_Code_2_ICD_10_C_M",
        "M1025_PMT_DGNS_ICD_C4_CD": "Optional_Diagnosis_Multiple_Code_2_ICD_10_C_M",
        "M1025_PMT_DGNS_ICD_D3_CD": "Optional_Diagnosis_Code_3_ICD_10_C_M",
        "M1025_PMT_DGNS_ICD_D4_CD": "Optional_Diagnosis_Multiple_Code_3_ICD_10_C_M",
        "M1025_PMT_DGNS_ICD_E3_CD": "Optional_Diagnosis_Code_4_ICD_10_C_M",
        "M1025_PMT_DGNS_ICD_E4_CD": "Optional_Diagnosis_Multiple_Code_4_ICD_10_C_M",
        "M1025_PMT_DGNS_ICD_F3_CD": "Optional_Diagnosis_Code_5_ICD_10_C_M",
        "M1025_PMT_DGNS_ICD_F4_CD": "Optional_Diagnosis_Multiple_Code_5_ICD_10_C_M",
        "M1033_HOSP_RISK_HSTRY_FALLS": "Risk_for_Hospitalization_Fall_History",
        "M1033_HOSP_RISK_WGHT_LOSS": "Risk_For_Hospitalization_Weight_Loss",
        "M1033_HOSP_RISK_MLTPL_HOSPZTN": "Risk_For_Hospitalization_Multiple_Hospitalizations",
        "M1033_HOSP_RISK_PLTPL_ER_VSTS": "Risk_For_Hospitalization_Multiple_ER_Visits",
        "M1033_HOSP_RISK_RCNT_DCLN": "Risk_For_Hospitalization_Mental_Emotional_Behavioral",
        "M1033_HOSP_RISK_CMPLY_MED_INSTR": "Risk_For_Hospitalization_Difficulty_with_Medical_Instructions",
        "M1033_HOSP_RISK_5PLUS_MDCTN": "Risk_For_Hospitalization_Taking_Five_or_More_Meds",
        "M1033_HOSP_RISK_EXHAUSTION": "Risk_For_Hospitalization_Exhaustion",
        "M1033_HOSP_RISK_OTHR": "Risk_For_Hospitalization_Other",
        "M1033_HOSP_RISK_NONE_ABOVE": "Risk_For_Hospitalization_None_of_the_Above",
        "M2102_CARE_TYPE_SRC_ADL": "Care_Assistance_ADL_Assistance",
        "M2102_CARE_TYPE_SRC_IADL": "Care_Assistance_IADL_Assistance",
        "M2102_CARE_TYPE_SRC_MDCTN": "Care_Assistance_Medication_Administration",
        "M2102_CARE_TYPE_SRC_PRCDR": "Care_Assistance_Medical_Procedures_Treatments",
        "M2102_CARE_ASTNC_EQUIP_CD": "Care_Assistance_Management_of_Equipment",
        "M2102_CARE_TYPE_SRC_SPRVSN": "Care_Assistance_Supervision_And_Safety",
        "M2102_CARE_TYPE_SRC_ADVCY": "Care_Assistance_Advocacy_Or_Facilitation",
        "M1511_HRT_FAILR_PHYSN_TRTMT": "Heart_failure_follow_up_phys_ordered_treatmnt",
        "GG0170C_MBLTY_DSCHRG_GOAL_CD": "Mobility_Discharge_Goal_Lying_to_Sitting",
        "GG0170C_MBLTY_PRFMNC_CD": "Mobility_SOC_ROC_Performance_Lying_to_Sitting",
        "HHA_AGNCY_ID": "HHA_Agency_ID",
        "M1028_ACTV_DGNS_DML_IND": "Active_Diagnoses_Diabetes_Mellitus",
        "M1028_ACTV_DGNS_IND": "Active_Diagnoses_PVD_or_PAD",
        "M1060_HEIGHT": "Height_in_inches",
        "M1060_WEIGHT": "Weight_in_pounds",
        "M1311_NBR_STG2": "Number_of_Stage_2_Pressure_Ulcers",
        "M1311_NBR_STG2_AT_SOC_ROC": "Number_of_Stage_2_pressure_ulcers_at_SOC_ROC",
        "M1311_NBR_STG3": "Number_of_Stage_3_Pressure_Ulcers",
        "M1311_NBR_STG3_AT_SOC_ROC": "Number_of_Stage_3_pressure_ulcers_at_SOC_ROC",
        "M1311_NBR_STG4": "Number_of_Stage_4_Pressure_Ulcers",
        "M1311_NBR_STG4_AT_SOC_ROC": "Number_of_Stage_4_pressure_ulcers_at_SOC_ROC",
        "M1311_NSTG_CVRG": "Unstageable_coverage_by_slough_or_eschar",
        "M1311_NSTG_CVRG_SOC_ROC": "Unstageable_coverage_by_slough_or_eschar_SOC_ROC",
        "M1311_NSTG_DEEP_TISSUE": "Unstageable_suspect_deep_tissue_injury",
        "M1311_NSTG_DEEP_TISSUE_SOC_ROC": "Unstageable_suspect_deep_tissue_injury_SOC_ROC",
        "M1311_NSTG_DRSG": "Num_unstage_pressure_ulcer_non_remov_dress",
        "M1311_NSTG_DRSG_SOC_ROC": "Num_unstage_pressure_ulcer_non_remov_dress_SOC_ROC",
        "M1313_NBR_STG2_NOT_PRESENT": "New_worsening_Stage_2",
        "M1313_NBR_STG3_NOT_PRESENT": "New_worsening_Stage_3",
        "M1313_NBR_STG4_NOT_PRESENT": "New_worsening_Stage_4",
        "M1313_NSTG_DRSG_NOT_PRESENT": "New_worsening_unstageable_non_removable_drs_dv",
        "M1313_NSTG_CVRG_NOT_PRESENT": "New_worsening_unstageable_coverage_slough_eschar",
        "M1313_NSTG_DEEP_TISSUE_NOT_PRSNT": "New_worsening_unstageable_deep_tissue_injury",
        "M1501_SYMTM_HRT_FAILR_PTNTS": "Symptoms_in_heart_failure_patients",
        "M1511_HRT_FAILR_CARE_PLAN_CHG": "Heart_failure_follow_up_change_in_care_plan",
        "M1511_HRT_FAILR_CLNCL_INTRVTN": "Heart_failure_follow_up_pt_educ_other_clinical",
        "M1511_HRT_FAILR_ER_TRTMT": "Heart_failure_follow_up_ER_treatment_advised",
        "M1511_HRT_FAILR_NO_ACTN": "Heart_failure_follow_up_no_action",
        "M1511_HRT_FAILR_PHYSN_CNTCT": "Heart_failure_follow_up_physician_contacted",
        "M2001_DRUG_RGMN_RVW": "Drug_regimen_review",
        "M2003_MDCTN_FLWP": "Medication_follow_up",
        "M2005_MDCTN_INTRVTN": "Medication_intervention",
        "M2016_DRUG_EDCTN_INTRVTN": "Patient_caregiver_drug_education_intervention",
        "M2301_EMER_USE_AFTR_LAST_ASMT": "Emergent_care_use_since_most_recent_SOC_ROC",
        "M2401_INTRVTN_SMRY_DBTS_FT": "Intervention_synopsis_diabetic_foot_care",
        "M2401_INTRVTN_SMRY_DPRSN": "Intervention_synopsis_depression_intervention",
        "M2401_INTRVTN_SMRY_FALL_PRVNT": "Intervention_synopsis_falls_prevention",
        "M2401_INTRVTN_SMRY_PAIN_MNTR": "Intervention_synopsis_monitor_and_mitigate_pain",
        "M2401_INTRVTN_SMRY_PRSULC_PRVN": "Intervention_synopsis_prevent_pressure_ulcers",
        "M2401_INTRVTN_SMRY_PRSULC_WET": "Intervention_synopsis_PU_moist_wound_treatment"
    }
    # --- Rename Columns ---
    for old, new in rename_map.items():
        if old in df.columns:
            df = df.withColumnRenamed(old, new)

    # --- Date Processing ---
    df = (
        df.withColumn("Assessment_Effective_Date", F.to_date(F.col("Assessment_Effective_Date"), "ddMMMyyyy"))
          .withColumn("AssessmentYear", F.year(F.col("Assessment_Effective_Date")))
          .withColumn("Start_of_Care_Date", F.to_date(F.col("Start_of_Care_Date"), "ddMMMyyyy"))
          .withColumn("Patient_Birth_Date", F.to_date(F.col("Patient_Birth_Date"), "ddMMMyyyy"))
    )

    # --- Visit History ---
    window_spec = Window.partitionBy("Beneficiary_ID", "Facility_Internal_ID").orderBy("Assessment_Effective_Date")

    df = (
        df.withColumn("NumVisits", F.count("Assessment_ID").over(window_spec))
          .withColumn("PrevVisitDate", F.lag("Assessment_Effective_Date").over(window_spec))
          .withColumn("DaysBetweenVisits", F.datediff(F.col("Assessment_Effective_Date"), F.col("PrevVisitDate")))
    )

    # --- Last Admission Date & Care Duration ---
    last_admission = (
        df.groupBy("Beneficiary_ID")
          .agg(F.max("Assessment_Effective_Date").alias("Last_Assessment_Date"))
    )

    df = (
        df.join(last_admission, on="Beneficiary_ID", how="left")
          .withColumn("Days_Cared_For", F.datediff(F.col("Last_Assessment_Date"), F.col("Start_of_Care_Date")))
    )

    # --- Patient Info ---
    df = (
        df.withColumn("Age", (F.datediff(F.col("Assessment_Effective_Date"), F.col("Patient_Birth_Date")) / 365).cast("int"))
          .withColumn("Gender",
                      F.when(F.col("Gender") == 1, "Male")
                       .when(F.col("Gender") == 2, "Female"))
    )

    # --- Discipline Mapping ---
    df = (
        df.withColumn(
            "ByDiscipline",
            F.when(F.col("Discipline_of_Person_Completing_Assessment") == 1, "RN")
             .when(F.col("Discipline_of_Person_Completing_Assessment") == 2, "PT")
             .when(F.col("Discipline_of_Person_Completing_Assessment") == 3, "SLP/ST")
             .when(F.col("Discipline_of_Person_Completing_Assessment") == 4, "OT")
        )
        .drop("Discipline_of_Person_Completing_Assessment")
    )

    return df

def df_to_csv(df, temp_path, final_csv):
    df.coalesce(1).write.mode("overwrite").option("header", "true").csv(temp_path)

    part_files = glob.glob(os.path.join(temp_path, "part-*.csv"))
    if not part_files:
        raise FileNotFoundError(f"No CSV part files found in {temp_path}")

    os.makedirs(os.path.dirname(final_csv), exist_ok=True)
    if os.path.exists(final_csv):
        os.remove(final_csv)

    shutil.move(part_files[0], final_csv)
    shutil.rmtree(temp_path, ignore_errors=True)
    log_time_and_step(f"CSV file saved to {final_csv}")

def safe_contains(icd_value, target):
    # Handle NaN or None
    if pd.isna(icd_value):
        return 0
    # Convert tuple or list to comma-separated string
    if isinstance(icd_value, (tuple, list)):
        icd_value = ', '.join(map(str, icd_value))
    # Ensure string comparison
    if isinstance(icd_value, str):
        return 1 if target in icd_value else 0
    return 0


def get_icd_section_and_range(code):
    if not isinstance(code, str) or len(code) < 3:
        return None, None  # Return None for both section and range if invalid code
    prefix = code[:3]  # Extract the first three characters of the ICD code
    for range_code, section in ICD_RANGES:
        start, end = range_code.split('-')
        if start <= prefix <= end:
            return section, range_code  # Return both section and range
    return None, None


def merging_icd(df_prepped, icd_ranges_spark):
    return df_prepped.join(icd_ranges_spark, on='Beneficiary_ID', how='left')


def find_icds(df, spark):
    columns = ["Beneficiary_ID", "Inpatient_Diagnosis_1_ICD_10_C_M","Inpatient_Diagnosis_2_ICD_10_C_M","Inpatient_Diagnosis_3_ICD_10_C_M","Inpatient_Diagnosis_4_ICD_10_C_M",
           "Inpatient_Diagnosis_5_ICD_10_C_M","Inpatient_Diagnosis_6_ICD_10_C_M","Inpatient_Diagnosis_ICD_10_C_M_Not_Applicable",
           "Primary_Diagnosis_ICD_10_C_M_Code","Primary_Diagnosis_Severity_Rating_ICD_10_C_M_Code",
           "Primary_Optional_Diagnosis_ICD_10_C_M_Code","Primary_Optional_Diagnosis_Multiple_Codes_ICD_10_C_M"]
    icds = df.select(columns).drop_duplicates().toPandas()
    columns = icds.columns

    output_file = "icd_unique_counts.txt"
    with open(output_file, "w") as f:
        for c in columns:
            line = f"{c}: {icds[c].nunique()}\n"
            f.write(line)

    log_time_and_step(f"Unique ICD counts saved to {output_file}")
    # source : https://www.aapc.com/codes/icd-10-codes-range/?srsltid=AfmBOoog5_u_SYP-InM8WmaLmuTMld2ifGbeOfkFiaALHsFRC8vKcko5
    icd_ranges = [
        ('A00-B99', 'Certain infectious and parasitic diseases'),
        ('C00-D49', 'Neoplasms'),
        ('D50-D89', 'Diseases of the blood and blood-forming organs and certain disorders involving the immune mechanism'),
        ('E00-E89', 'Endocrine, nutritional and metabolic diseases'),
        ('F01-F99', 'Mental, Behavioral and Neurodevelopmental disorders'),
        ('G00-G99', 'Diseases of the nervous system'),
        ('H00-H59', 'Diseases of the eye and adnexa'),
        ('H60-H95', 'Diseases of the ear and mastoid process'),
        ('I00-I99', 'Diseases of the circulatory system'),
        ('J00-J99', 'Diseases of the respiratory system'),
        ('K00-K95', 'Diseases of the digestive system'),
        ('L00-L99', 'Diseases of the skin and subcutaneous tissue'),
        ('M00-M99', 'Diseases of the musculoskeletal system and connective tissue'),
        ('N00-N99', 'Diseases of the genitourinary system'),
        ('O00-O9A', 'Pregnancy, childbirth and the puerperium'),
        ('P00-P96', 'Certain conditions originating in the perinatal period'),
        ('Q00-Q99', 'Congenital malformations, deformations and chromosomal abnormalities'),
        ('R00-R99', 'Symptoms, signs and abnormal clinical and laboratory findings, not elsewhere classified'),
        ('S00-T88', 'Injury, poisoning and certain other consequences of external causes'),
        ('U00-U85', 'Codes for special purposes'),
        ('V00-Y99', 'External causes of morbidity'),
        ('Z00-Z99', 'Factors influencing health status and contact with health services')
    ]

    icd_df = pd.DataFrame(icd_ranges, columns=['CODE_RANGE', 'ICD_SECTION'])

    # Apply the function across all ICD columns
    icds[['ICD_Section', 'ICD_Range']] = icds[columns].apply(
        lambda row: pd.Series({
            'ICD_Section': ', '.join({get_icd_section_and_range(code)[0] for code in row if get_icd_section_and_range(code)[0]}),
            'ICD_Range': ', '.join({get_icd_section_and_range(code)[1] for code in row if get_icd_section_and_range(code)[1]})
        }), axis=1)

    # Replace empty values with 'Unknown' (if no section or range is found)
    icds['ICD_Section'] = icds['ICD_Section'].apply(lambda x: x if x else 'Unknown')
    icds['ICD_Range'] = icds['ICD_Range'].apply(lambda x: x if x else 'Unknown')

    icds_sections = icds[['Beneficiary_ID', 'ICD_Range' ,'ICD_Section']].drop_duplicates()
    icds_sections.to_csv('data/icd_sections.csv', index=False)
    log_time_and_step(f"ICD sections saved to csv")

    ranges = [ 'A00-B99','C00-D49','D50-D89','E00-E89','F01-F99', 'G00-G99','H00-H59', 'H60-H95', 'I00-I99', 'J00-J99', 'K00-K95', \
               'L00-L99', 'M00-M99', 'N00-N99','O00-O9A', 'P00-P96', 'Q00-Q99','R00-R99', 'S00-T88', 'U00-U85', 'V00-Y99', 'Z00-Z99']
    icds_sections = icds_sections.assign(**{r: '' for r in ranges})

    # Apply safely for each ICD range
    for icd_range in ranges:
        icds_sections[icd_range] = icds_sections['ICD_Range'].apply(lambda x: safe_contains(x, icd_range))
    ICD_ranges = icds_sections.drop(['ICD_Range', 'ICD_Section'], axis=1).sort_values(by='Beneficiary_ID')
    # Drop non-numeric columns (if any besides Beneficiary_ID)
    numeric_cols = ICD_ranges.drop('Beneficiary_ID', axis=1).columns

    # Group by Beneficiary_ID and sum counts
    ICD_ranges_per_patient = ICD_ranges.groupby('Beneficiary_ID')[numeric_cols].sum().reset_index()
    ICD_ranges_per_patient.to_csv('data/icd_ranges_per_patient.csv', index=False)
    log_time_and_step(f"ICD ranges per patients saved to csv")
    ICD_ranges_spark = spark.createDataFrame(ICD_ranges_per_patient)
    return ICD_ranges_spark




def main():
    start_time = time.time()
    log_time_and_step("Start Program")
    spark = SparkSession.builder \
        .appName("Split") \
        .config("spark.driver.memory", "8g") \
        .config("spark.executor.memory", "8g") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    spark.conf.set("spark.sql.shuffle.partitions", "400")
    spark.conf.set("spark.sql.debug.maxToStringFields", 1000)

    output_dir = "data"
    os.makedirs(output_dir, exist_ok=True)

    df = load_data(spark)
    df_prepped = prepare_data(df)

    temp_path = os.path.join(output_dir, "OASIS_single")
    final_csv = os.path.join(output_dir, "OASIS_v1.csv")
    df_to_csv(df_prepped, temp_path, final_csv)

    icd_ranges = find_icds(df_prepped, spark)
    final_df = merging_icd(df_prepped, icd_ranges)

    final_csv2 = os.path.join(output_dir, "OASIS_v2.csv")
    df_to_csv(final_df, temp_path, final_csv2)

    spark.stop()  # Stop the Spark session when done
    log_time_and_step("End Program")
    logging.info(f"Total Time: {time.time() - start_time:.2f} seconds")

    # Profile the code

if __name__ == "__main__":
    cProfile.run('main()')
