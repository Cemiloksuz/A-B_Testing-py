########################## A-B TESTING  ###########################
#The customer stopped using the maximum bidding method and started using a new one.
#We were interested in seeing how well the new method worked at turning people into customers.
# This is done with A-B testing.
# There are two sets of data that contain purchasing counts.


########################## Importing Library and Settings  ###########################
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu
import os
import statsmodels.stats.api as sms

path = "/Users/cemiloksuz/PycharmProjects/EuroTechMiullDataScience/week_6_A-B Testing.py"
os.chdir(path)

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 10)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


########################## Loading  The Date  ###########################
def load_data(dataframe, sheet = 0):
    return pd.read_excel(dataframe, sheet_name = sheet)


control = load_data("ab_testing.xlsx")
df_control = control.copy()
df_control.head()

test = load_data("ab_testing.xlsx", sheet = 1)
df_test = test.copy()
df_test.head()


########################## Data Analysis  ###########################
def analysis_function(dataframe, target):
    print("####################  Summary of Data   ##############\n")
    print(dataframe.describe().T)
    print("####################  Mean of Target  ############\n")
    print(dataframe[target].mean())
    print("####################  Confidence Interval of Target  #######\n")
    print(sms.DescrStatsW(dataframe[target]).tconfint_mean())
    print("####################  Correlation of The Target with Other Variables  #########\n")
    others = dataframe.drop(target, axis = 1).columns
    for col in others:
        print(f"{col} correlation with {target}")
        print(dataframe[target].corr(dataframe[col]))


analysis_function(df_control, "Purchase")
analysis_function(df_test, "Purchase")

cols = ['Impression', 'Click', 'Purchase', 'Earning']
for col in cols:
    fig, ax = plt.subplots(1, 2)
    fig.set_figheight(5)
    fig.set_figwidth(12)
    sns.distplot(df_control[col], hist = False, ax = ax[0])
    sns.distplot(df_test[col], hist = False, ax = ax[1])
    ax[0].set_title('Control')
    ax[1].set_title('Test')
    plt.show()


########################## Feature Engineering  ###########################
df_control["Conversion_Rate"] = (df_control["Purchase"] / df_control["Click"]) * 100
df_test["Conversion_Rate"] = (df_test["Purchase"] / df_test["Click"]) * 100

df_control["Earning_Per_Purchase"] = (df_control["Earning"] / df_control["Purchase"]) * 100
df_test["Earning_Per_Purchase"] = (df_test["Earning"] / df_test["Purchase"]) * 100

df_control["Group"] = "Control"
df_test["Group"] = "Test"

df = pd.concat([df_control, df_test], ignore_index = True)
df.head()

df.groupby("Group").agg({"Purchase": "mean",
                         "Conversion_Rate": "mean",
                         "Earning_Per_Purchase": "mean"})


########################## Outlier Control  ###########################
def outlier_detect(dataframe):
    cols = [col for col in dataframe.columns if dataframe[col].dtype != "object"]

    for feature in cols:
        Q1 = dataframe[feature].quantile(0.05)
        Q3 = dataframe[feature].quantile(0.95)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        if dataframe[(dataframe[feature] > upper)].any(axis = None):
            print(feature, "-- it has OUTLIER")
        else:
            print(feature, "-- NO, outlier")


outlier_detect(df)


########################## A-B Testing  ###########################
# H0: M1 = M2
# There is no statistically significant difference between control group purchase mean and test group's purchase mean.
# H1: M1 != M2
# There is statistically significant difference.

def ab_testing(dataframe, target):
    normal = []
    print("#####  normality test   #####")
    # H0 : Distribution is Normal
    # H1 : Distribution is not Normal
    for value in list(dataframe["Group"].unique()):
        pvalue = shapiro(dataframe.loc[dataframe["Group"] == value, target])[1]
        normal.append(pvalue)
        print(value, 'group p-value: %.4f' % pvalue)
    print("#################################")
    if normal[0] > 0.05 and normal[1] > 0.05:
        test_stat, pvalue_var = levene(dataframe.loc[dataframe["Group"] == "Control", target],
                                       dataframe.loc[dataframe["Group"] == "Test", target], )
        print("#####  variance test   #####")
        print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue_var))
        print("###############################################")
        if pvalue_var > 0.05:
            print("#####  homogenous parametric test   #####")
            test_stat, pvalue = ttest_ind(dataframe.loc[dataframe["Group"] == "Control", target],
                                          dataframe.loc[dataframe["Group"] == "Test", target],
                                          equal_var = True)

            print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))
        else:
            print("#####  non-homogenous parametric test   #####")
            test_stat, pvalue = ttest_ind(dataframe.loc[dataframe["Group"] == "Control", target],
                                          dataframe.loc[dataframe["Group"] == "Test", target],
                                          equal_var = False)

            print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))

    else:
        print("#####  non-parametric test   #####")
        test_stat, pvalue = mannwhitneyu(dataframe.loc[dataframe["Group"] == "Control", target],
                                         dataframe.loc[dataframe["Group"] == "Test", target])

        print('Test Stat = %.4f, p-value = %.4f' % (test_stat, pvalue))


ab_testing(df, "Purchase")
ab_testing(df, "Conversion_Rate")
ab_testing(df, "Earning_Per_Purchase")

# 1--
# Shapiro and Levene test assumptions.
# These tests provide a "Purchase" variable p value larger than 0.05.
# Thus, hypothesis testing uses two independent samples t-test.
# This test shows H0 hypothesis is not rejected because p > 0.05.
# So, statistically, two groups are similar.
# Customers can use maximum bidding.
# 2--
# "Conversion_Rate" and "Earning_Per_Purchase" employ the same assumption tests.
# These tests provide p values below 0.05 for these factors.
# Thus, hypothesis testing uses nonparametric mann-whitney u test.
# This test rejects H0 because p is less than 0.05.
# In statistical terms, two groups differ significantly.
# Test group had higher "Conversion_Rate" and "Earning_Per_Purchase" means.
# We can tell customers average bidding is more profitable.
