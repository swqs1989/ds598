import importlib
import func
import numpy as np
import pandas as pd

df_101191 = pd.read_excel("data/191_BWSC101 Release Log Form.xlsx")
df_101592 = pd.read_excel("data/592_BWSC101 Release Log Form.xlsx")
df_101607 = pd.read_excel("data/607_BWSC101 Release Log Form.xlsx")

df_101191["RTN"] = df_101191.apply(func.completeRTN, axis=1)
df_101592["RTN"] = df_101592.apply(func.completeRTN, axis=1)
df_101607["RTN"] = df_101607.apply(func.completeRTN, axis=1)

func.preprocess(df_101607, "101607proc.xlsx", "101607")
print(df_101607.shape)
func.preprocess(df_101592, "101592proc.xlsx", "101592")
print(df_101592.shape)
func.preprocess(df_101191, "101191proc.xlsx", "101191")
print(df_101191.shape)

df_101 = df_101191.append(df_101592)
df_101 = df_101.append(df_101607)

df_101 = df_101[(df_101["A3A"] == 1) | (df_101["A2A"] == 1)]

print(df_101.shape)

# TCLass
tclass = pd.read_excel('data/TClass Phase Action Dates All RTNs mgf A 04-10-2018.xlsm', sheetname="All")
df_tclass = tclass

df_tclass = df_tclass.set_index("RTN")

exclude_status = ['ADQREG', 'DEPMOU', 'DEPNDS', 'DEPNFA', 'DPS', 'DPSTRM', 'INVSUB', 'STMRET', 'URAM', 'UNCLSS']

df_tclass = df_tclass[(df_tclass["Notification"] >= "2006-06-01") & (df_tclass["Notification"] <= "2016-03-03")]
df_tclass = df_tclass[~df_tclass["Status"].isin(exclude_status)]

drop_index = df_tclass[(df_tclass["Status"].isin(["REMOPS", "ROSTRM", "TCLASS", "TIERI", "TIERII"])) & df_tclass["Phase1Cs"].isnull() & df_tclass["RaoNr"].isnull()].index

df_tclass = df_tclass.drop(drop_index, axis=0)

df_tclass["length"] = df_tclass.apply(func.daylength, axis=1)

df_tclass = df_tclass[df_tclass["length"] >= 0]

df_tclass["Tier1D"] = df_tclass.apply(func.isTier1D, axis=1)

df_tier = df_101.set_index("RTN").join(df_tclass.loc[:, ["Tier1D"]], how='inner')

print(df_tier.shape)

# GIS
CENSUS_2010_B = pd.read_excel("data/Intersect_Release_CENSUS_2010_Blocks.xls")
T_group_quarters_pop                = pd.read_excel("data/GIS/TRACT/group_quarters_pop.xlsx")
T_households_by_age_family_children = pd.read_excel("data/GIS/TRACT/households_by_age_family_children.xlsx")
T_households_size_by_family = pd.read_excel("data/GIS/TRACT/households_size_by_family.xlsx")
T_housing_owner_rental_demographics = pd.read_excel("data/GIS/TRACT/housing_owner_rental_demographics.xlsx")
T_housing_residency_characteristics = pd.read_excel("data/GIS/TRACT/housing_residency_characteristics.xlsx")
T_pop_by_age_gender = pd.read_excel("data/GIS/TRACT/pop_by_age_gender.xlsx")
T_pop_by_race = pd.read_excel("data/GIS/TRACT/pop_by_race.xlsx")

CENSUS_2010_B = CENSUS_2010_B.loc[:, ["rtn", "GEOID10"]]
gis_block = T_group_quarters_pop.join(T_households_by_age_family_children.set_index("LOGRECNO"), how='inner', on='LOGRECNO', rsuffix='_')
gis_block = gis_block.join(T_households_size_by_family.set_index("LOGRECNO"), how='inner', on='LOGRECNO', rsuffix='_')
gis_block = gis_block.join(T_housing_owner_rental_demographics.set_index("LOGRECNO"), how='inner', on='LOGRECNO', rsuffix='_')
gis_block = gis_block.join(T_housing_residency_characteristics.set_index("LOGRECNO"), how='inner', on='LOGRECNO', rsuffix='_')
gis_block = gis_block.join(T_pop_by_age_gender.set_index("LOGRECNO"), how='inner', on='LOGRECNO', rsuffix='_')
gis_block = gis_block.join(T_pop_by_race.set_index("LOGRECNO"), how='inner', on='LOGRECNO', rsuffix='_')

CENSUS_2010_B['GEOID10'] = CENSUS_2010_B['GEOID10'].apply(lambda x : str(x)[:11])
gis_block["GEOID10"] = gis_block["GEOID10"].astype(str)
gis_block = CENSUS_2010_B.join(gis_block.set_index("GEOID10"), how='inner', on="GEOID10", rsuffix='_')
gis_block = gis_block.set_index("rtn")
features = gis_block.columns.tolist()
features = list(filter(lambda a: a != 'GEOID10_', features))
features = list(filter(lambda a: a != 'GEOID10', features))
features = list(filter(lambda a: a != 'LOGRECNO', features))
gis_block = gis_block.loc[:, features]

df_tier = df_tier.join(gis_block, how='inner')

print(df_tier.shape)
# Chemicals
df_chemicals = pd.read_excel("data/Chemical_Class_Features1.xlsx")
df_chemicals = df_chemicals.set_index("RTN")

list_chemicals = df_chemicals.columns.tolist()

dict_chemicals = {}
for che in list_chemicals:
    dict_chemicals[che] = 0.0
df_tierc = df_tier.join(df_chemicals, how='left')
df_tierc = df_tierc.fillna(dict_chemicals)

print(df_tierc.shape)

df_tierc.to_excel("data/df_tierc700.xlsx")
