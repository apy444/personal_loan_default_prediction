{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Technical notebook"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Lending Club Data Set"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import scipy.stats as stats\n",
    "\n",
    "from sklearn.preprocessing import scale\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.preprocessing import MinMaxScaler\n",
    "from sklearn.impute import SimpleImputer\n",
    "from sklearn.preprocessing import RobustScaler\n",
    "\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.model_selection import cross_validate\n",
    "from sklearn.model_selection import StratifiedKFold\n",
    "from sklearn.model_selection import cross_val_predict\n",
    "\n",
    "from sklearn.metrics import confusion_matrix\n",
    "from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score\n",
    "from sklearn.metrics import classification_report\n",
    "from sklearn.metrics import auc\n",
    "from sklearn.metrics import precision_recall_curve\n",
    "\n",
    "import statsmodels.api as sm\n",
    "from statsmodels.stats.outliers_influence import variance_inflation_factor\n",
    "\n",
    "# from loan_helper import data_cleaning\n",
    "from loan_helper import data_converting\n",
    "from loan_helper import column_description\n",
    "\n",
    "#SMOTE\n",
    "from imblearn.over_sampling import SMOTE\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "!ls -lath LendingClub"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- LendingClub data source:\n",
    "\n",
    "https://www.lendingclub.com/info/download-data.action"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#reading excel\n",
    "description = pd.read_excel('LendingClub/LCDataDictionary.xlsx')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#reading Lending Club loan data from 2014\n",
    "data_lc = pd.read_csv('LendingClub/LoanStats3c_securev1.csv', low_memory=False, header=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "data_lc.loan_amnt.isna().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#removed two rows with full NAN values\n",
    "data_lc = data_lc.loc[data_lc.loan_amnt.notnull()]\n",
    "data_lc.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Feature selection and feature engineering"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Understanding the columns"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "In order to understand the columns we created a dataframe with column names, two examples, datatype, number of missing values, and the long description. The dataframe was exported to excel to make decision on columns. The result is stored in col_selection.xlsx."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "desc = column_description(data_lc, description)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "pd.set_option('display.max_colwidth', -1) #this allows us to see the very long description, if exceeds 50 char\n",
    "desc.head(5)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "desc.to_excel('col_desc_2014.xlsx')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### First round feauture selection"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "Originally the dataset contained 150 columns. When we were reducing the features in order to avoid the overfitting the model we selected features according to the following:\n",
    "\n",
    "- Discarded columns that contained payment or collection information (47 columns)\n",
    "- Discarded columns that contained information that were not available at the time of credit application\n",
    "- Discarded features that require too much data processing (typically free input i.e. emp_title)\n",
    "- Discarded redundant features (subgrade - grade, title - purpose)\n",
    "- Discarded feauters that contain too much NAN values (mnths_since_last_delinq, mths_since_recent_bc_dlq, mths_since_recent_revol_delinq)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "col_selection = pd.read_excel('col_selection_2014.xlsx')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "col_selection.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "selected_col = col_selection.loc[col_selection.Decision == 'keep', 'col_name'].to_list()\n",
    "len(selected_col)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "dataset = data_lc.loc[:, selected_col]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "dataset.shape"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Converting data types\n",
    "\n",
    "- emp_length column to convert numeric and missing values replaced with average\n",
    "- earliest credit line: convert date to numeric (years)\n",
    "- revol_util (revolving utilization) convert to numeric\n",
    "- creating regions from state\n",
    "- reduce categories of loan purpose"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "dataset = data_converting(dataset)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "regions = {'W':  ['CA', 'OR', 'UT','WA', 'CO', 'NV', 'AK', 'MT', 'HI', 'WY', 'ID'], \n",
    "           'SW': ['AZ', 'TX', 'NM', 'OK'],\n",
    "           'SE': ['GA', 'NC', 'VA', 'FL', 'KY', 'SC', 'LA', 'AL', 'WV', 'DC', 'AR', 'DE', 'MS', 'TN' ],\n",
    "           'MW': ['IL', 'MO', 'MN', 'OH', 'WI', 'KS', 'MI', 'SD', 'IA', 'NE', 'IN', 'ND'],\n",
    "           'NE': ['CT', 'NY', 'PA', 'NJ', 'RI','MA', 'MD', 'VT', 'NH', 'ME']}"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "for region, states in regions.items():\n",
    "    for state in states:\n",
    "        dataset.loc[dataset.addr_state == state, 'region'] = region\n",
    "dataset.drop(columns = ['addr_state'], inplace=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Determing the target feature"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "dataset.loan_status.value_counts()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Meaning of the categories\n",
    "<b>Fully paid:</b> Loan has been fully repaid, either at the expiration of the 3- or 5-year year term or as a result of a prepayment.\n",
    "\n",
    "<b>Current:</b> Loan is up to date on all outstanding payments. \n",
    "\n",
    "<b>In Grace Period:</b> Loan is past due but within the 15-day grace period. \n",
    "\n",
    "<b>Late (16-30):</b> Loan has not been current for 16 to 30 days. Learn more about the tools LendingClub has to deal with delinquent borrowers.\n",
    "\n",
    "<b>Late (31-120):</b> Loan has not been current for 31 to 120 days. Learn more about the tools LendingClub has to deal with delinquent borrowers.\n",
    "\n",
    "<b>Default:</b> Loan has not been current for an extended period of time. Learn more about the difference between “default” and “charge off”.\n",
    "\n",
    "<b>Charged Off:</b> Loan for which there is no longer a reasonable expectation of further payments. Upon Charge Off, the remaining principal balance of the Note is deducted from the account balance. Learn more about the difference between “default” and “charge off”.\n",
    "\n",
    "Sosurce: https://help.lendingclub.com/hc/en-us/articles/215488038-What-do-the-different-Note-statuses-mean-"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "pd.crosstab(columns=dataset['loan_status'], index=dataset['term'],)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "We decided to ignore the 'gray' categories, where there might chance to the recovery of the loan. The 'Current' category contains the 60 months term loans, removing them would panalize the long term loans by increasing the default rate within this category. "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#Select default categories:\n",
    "dataset.loc[dataset.loan_status == 'Fully Paid', 'default'] = 0\n",
    "dataset.loc[dataset.loan_status == 'Charged Off', 'default'] = 1\n",
    "dataset.loc[dataset.loan_status == 'Current', 'default'] = 0\n",
    "\n",
    "dataset = dataset.loc[dataset.default.notnull()]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#remove loan_status, default replace it\n",
    "dataset = dataset.drop(columns='loan_status')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "dataset = dataset.reset_index()\n",
    "dataset = dataset.drop(columns='index')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.figure(figsize=(6,5))\n",
    "plt.bar(x=['non default', 'default'], height=dataset.default.value_counts()/len(dataset), width=0.6,)\n",
    "plt.title('The distribution of defaulted and non defaulted loans\\n')\n",
    "ax=plt.gca();"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# dataset.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "x_feats = dataset.columns.to_list()\n",
    "x_feats.remove('default')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Multicollinearity examination"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "x_feats_cont =  x_feats\n",
    "x_feats_cont.remove('term')\n",
    "x_feats_cont.remove('home_ownership')\n",
    "x_feats_cont.remove('verification_status')\n",
    "x_feats_cont.remove('purpose')\n",
    "x_feats_cont.remove('region')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from statsmodels.stats.outliers_influence import variance_inflation_factor"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "X = dataset[x_feats_cont]\n",
    "imp_vif = SimpleImputer(strategy='median', copy=True, fill_value=None)\n",
    "imp_vif.fit(X)  \n",
    "X = imp_vif.transform(X)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "vif = [variance_inflation_factor(X, i) for i in range(X.shape[1])]\n",
    "list(zip(x_feats_cont, vif))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "x_feats = ['revol_util','revol_bal','fico_range_low','grade','installment','loan_amnt','emp_length','annual_inc','delinq_2yrs','dti','delinq_2yrs','inq_last_6mths',\n",
    "           'pub_rec','collections_12_mths_ex_med', 'tot_coll_amt', 'total_rev_hi_lim','acc_open_past_24mths',\n",
    "           'avg_cur_bal','chargeoff_within_12_mths','delinq_amnt', 'mo_sin_rcnt_rev_tl_op', 'mo_sin_rcnt_tl',\n",
    "          'mths_since_recent_bc', 'mths_since_recent_inq','num_accts_ever_120_pd','num_tl_120dpd_2m',\n",
    "          'num_tl_30dpd','num_tl_90g_dpd_24m', 'num_tl_op_past_12m','percent_bc_gt_75', 'pub_rec_bankruptcies',\n",
    "          'tax_liens']\n",
    "x_feats += ['term','home_ownership','verification_status','purpose','region']"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "x_feats"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "x_feats.remove('level_0')"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Preparing dataset for modeling"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "X = pd.get_dummies(dataset[x_feats], drop_first=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "y = dataset.default"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "y.value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "X.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "X.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "X.columns"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Train-Test Split"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#'Stratify=y' provide us the same ratio in the target variable then it was in the original dataset \n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=10, stratify=y) #25%"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "y_train.mean()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "y_test.mean()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Scaling"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train_norm = scale(X_train, axis = 0) \n",
    "X_test_norm = scale(X_test, axis = 0)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The following scaling method assures that the variables of X_train are within a 0-1 range"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "scaler = MinMaxScaler()\n",
    "scaler.fit(X_train)\n",
    "X_train_scaled = scaler.transform(X_train)\n",
    "X_test_scaled = scaler.transform(X_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Scaling using StandarScaler"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- For SMOTE"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "scaler = StandardScaler()\n",
    "scaler.fit(X_train_imp)\n",
    "X_train_scaled = scaler.transform(X_train_imp)\n",
    "X_test_scaled = scaler.transform(X_test_imp)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "The following scaling method robost to outliers"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "scaler = RobustScaler()\n",
    "scaler.fit(X_train_imp)\n",
    "X_train_scaled = scaler.transform(X_train_imp)\n",
    "X_test_scaled = scaler.transform(X_test_imp)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Objective: Increasing predictibility of loan defaults from actual default "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Maximize the F1 score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_metric(y_train, y_probability):\n",
    "    '''\n",
    "    inputs: y_train values and from the trained model the y probabilities for default\n",
    "    output: maximized F1 score, cut-off and the corresponding y_hat\n",
    "    '''\n",
    "    y = 0\n",
    "    F1_score = 0\n",
    "    cut_off = 0\n",
    "    for cutoff in np.linspace(0,1,101):\n",
    "        y_hat = (y_probability > cutoff) * 1\n",
    "        f1 = f1_score(y_train, y_hat)\n",
    "        if f1> F1_score:\n",
    "            F1_score = f1\n",
    "            cut_off = cutoff\n",
    "            y = y_hat\n",
    "    \n",
    "    print('Recall:', recall_score(y_train, y))\n",
    "    print('Precision:', precision_score(y_train, y))\n",
    "    print('F1_score:', F1_score)\n",
    "    print('Cut_off:', cut_off)\n",
    "    \n",
    "    conf_matrix = pd.DataFrame(confusion_matrix(y_train, y),\n",
    "                                    index=['actual 0', 'actual 1'],\n",
    "                                    columns=['predicted 0', 'predicted 1'])\n",
    "    return conf_matrix"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    " ## Model Selection - Logistic regression"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### (I) Baseline: vanilla logistic regression w/o imbalance strategy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Vanilla regression\n",
    "logreg_vanilla = LogisticRegression(C=1e9, solver='liblinear', max_iter=200)\n",
    "\n",
    "model_vanilla = logreg_vanilla.fit(X_train_scaled, y_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "y_probability = model_vanilla.predict_proba(X_train_scaled)[:,1]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {
    "scrolled": true
   },
   "outputs": [],
   "source": [
    "get_metric(y_train, y_probability)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "p, r, t = precision_recall_curve(y_train, model_vanilla.decision_function(X_train_scaled))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from inspect import signature\n",
    "step_kwargs = ({'step': 'post'} if 'step' in signature(plt.fill_between).parameters else {})\n",
    "plt.step(r, p, color='b', alpha=0.4, where='post')\n",
    "plt.fill_between(r, p, color='b', alpha=0.4, **step_kwargs)\n",
    "plt.xlabel('precision')\n",
    "plt.xlabel('Recall')\n",
    "plt.ylabel('Precision')\n",
    "plt.ylim([0.0, 1.05])\n",
    "plt.xlim([0.0, 1])\n",
    "plt.title('Precision-Recall curve');"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### (II) Lasso regression with different C values (w/o imbalance strategy)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "C_values = [0.01, 0.1, 1, 10, 100, 1000, 10000]  # low value means high l1 penalty on coefficients\n",
    "\n",
    "for C in C_values:\n",
    "    logreg_l1 = LogisticRegression(C=C, penalty='l1',\n",
    "                                   solver='liblinear',\n",
    "                                   max_iter=200)\n",
    "    print('-'*40,f'\\nLasso regression with C = {C}')\n",
    "    model_l1 = logreg_l1.fit(X_train_scaled, y_train)\n",
    "    y_probability = model_l1.predict_proba(X_train_scaled)[:,1]\n",
    "    get_metric(y_train, y_probability)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### (III) Ridge regression with different C values (w/o imbalance strategy)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "C_values = [0.01, 0.1, 1, 10, 100, 1000, 10000]  # low value means high l1 penalty on coefficients\n",
    "\n",
    "for C in C_values:\n",
    "    logreg_l2 = LogisticRegression(C=C, penalty='l2',\n",
    "                                   solver='newton-cg',\n",
    "                                   max_iter=200)\n",
    "    \n",
    "    print('-'*40,f'\\nRidge regression with C = {C}')\n",
    "    model_l2 = logreg_l2.fit(X_train_scaled, y_train)\n",
    "    y_probability = model_l2.predict_proba(X_train_scaled)[:,1]\n",
    "    get_metric(y_train, y_probability)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### (IV) Cross-Validation (w/o imbalance strategy)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "cv = StratifiedKFold(n_splits= 5, random_state=1000, shuffle=True)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Vanilla"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "lr_vanilla = LogisticRegression(C=1e9,\n",
    "                                solver='newton-cg',\n",
    "                                max_iter=200)\n",
    "\n",
    "\n",
    "cv_vanilla = cross_validate(estimator=lr_vanilla,\n",
    "                            X=X_train_scaled, y=y_train,\n",
    "                            cv=cv,\n",
    "                            n_jobs=-1,\n",
    "                            return_train_score=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "y_probability = cross_val_predict(lr_vanilla, X_train_scaled, y_train, cv=cv, method='predict_proba')[:,1]\n",
    "get_metric(y_train, y_probability)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Ridge"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "l2_reg = LogisticRegression(C=1,\n",
    "                            solver='newton-cg',\n",
    "                            penalty='l2',\n",
    "                            max_iter=200)\n",
    "\n",
    "cv_l2 = cross_validate(estimator=l2_reg, X=X_train_scaled, y=y_train,\n",
    "                       cv=cv,\n",
    "                       n_jobs=-1,\n",
    "                       return_estimator=True,\n",
    "                       return_train_score=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "y_probability = cross_val_predict(l2_reg, X_train_scaled, y_train, cv=cv, method='predict_proba')[:,1]\n",
    "get_metric(y_train, y_probability)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Lasso"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "l1_reg = LogisticRegression(C=1,\n",
    "                            solver='saga',\n",
    "                            penalty='l1',\n",
    "                            max_iter=200)\n",
    "cv_l1 = cross_validate(estimator=l1_reg, X=X_train_scaled, y=y_train,\n",
    "                       cv=cv,\n",
    "                       n_jobs=-1,\n",
    "                       return_estimator=True,\n",
    "                       return_train_score=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "y_probability = cross_val_predict(l1_reg, X_train_scaled, y_train, cv=cv, method='predict_proba')[:,1]\n",
    "get_metric(y_train, y_probability)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## (V) Imbalance Strategy: Random Oversampling"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## (VI) IMBALANCE STRATEGY: SMOTE"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(y.value_counts()) #Previous original class distribution\n",
    "smote = SMOTE()\n",
    "X_train_smote, y_train_smote = smote.fit_sample(X_train_scaled, y_train) \n",
    "print(pd.Series(y_train_smote).value_counts()) #Preview synthetic sample class distributi"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### A) Baseline Vanilla"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Vanilla regression\n",
    "logreg_vanilla = LogisticRegression(C=1e9, solver='liblinear', max_iter=200)\n",
    "\n",
    "model_vanilla = logreg_vanilla.fit(X_train_smote, y_train_smote)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "y_probability = model_vanilla.predict_proba(X_train_scaled)[:,1]\n",
    "\n",
    "get_metric(y_train, y_probability)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "y_probability = model_vanilla_balance.predict_proba(X_test_scaled)[:,1]\n",
    "\n",
    "get_metric(y_test, y_probability)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- Visualization"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "p_balance, r_balance, t_balance = precision_recall_curve(y_train, model_vanilla_balance.decision_function(X_train_scaled))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "fig, axs = plt.subplots(1,2, figsize=(13,6))\n",
    "\n",
    "step_kwargs = ({'step': 'post'} if 'step' in signature(plt.fill_between).parameters else {})\n",
    "\n",
    "\n",
    "axs[0].fill_between(r, p, color='#8c8c8c', alpha=0.4, **step_kwargs)\n",
    "axs[0].set(title='Imbalance Precision-Recall Curve', xlabel='Recall', ylabel='Precision', xlim=(0.0, 1), ylim=(0.0, 1.05))\n",
    "\n",
    "axs[1].fill_between(r_balance, p_balance, color='r', alpha=0.4, **step_kwargs)\n",
    "axs[1].set(title='Balanced Precision-Recall Curve', xlabel='Recall', ylabel='Precision', xlim=(0.0, 1), ylim=(0.0, 1.05))\n",
    "# fig.savefig('Precision-recall curve')\n",
    "\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "####  B) Lasso regression with different C values"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "C_values = [0.01]  # low value means high l1 penalty on coefficients\n",
    "\n",
    "for C in C_values:\n",
    "    logreg_l1 = LogisticRegression(C=C, penalty='l1',\n",
    "                                   solver='liblinear',\n",
    "                                   max_iter=200)\n",
    "    print('-'*40,f'\\nLasso regression with C = {C}')\n",
    "    model_l1 = logreg_l1.fit(X_train_smote, y_train_smote)\n",
    "    y_probability = model_l1.predict_proba(X_train_scaled)[:,1]\n",
    "    get_metric(y_train, y_probability)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### C) Ridge regression with different C values "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "C_values = [0.01]  # low value means high l1 penalty on coefficients\n",
    "\n",
    "for C in C_values:\n",
    "    logreg_l2 = LogisticRegression(C=C, penalty='l2',\n",
    "                                   solver='newton-cg',\n",
    "                                   max_iter=200)\n",
    "    \n",
    "    print('-'*40,f'\\nRidge regression with C = {C}')\n",
    "    model_l2 = logreg_l2.fit(X_train_smote, y_train_smote)\n",
    "    y_probability = model_l2.predict_proba(X_train_scaled)[:,1]\n",
    "    get_metric(y_train, y_probability)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### D) Cross-Validation"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "cv = StratifiedKFold(n_splits= 5, random_state=1000, shuffle=True)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Vanilla"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "lr_vanilla = LogisticRegression(C=1e9,\n",
    "                                solver='newton-cg',\n",
    "                                max_iter=200)\n",
    "\n",
    "\n",
    "cv_vanilla = cross_validate(estimator=lr_vanilla,\n",
    "                            X=X_train_smote, y=y_train_smote,\n",
    "                            cv=cv,\n",
    "                            n_jobs=-1,\n",
    "                            return_train_score=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "y_probability = cross_val_predict(lr_vanilla, X_train_scaled, y_train, cv=cv, method='predict_proba')[:,1]\n",
    "get_metric(y_train, y_probability)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Ridge"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "l2_reg = LogisticRegression(C=1,\n",
    "                            solver='newton-cg',\n",
    "                            penalty='l2',\n",
    "                            max_iter=200)\n",
    "\n",
    "cv_l2 = cross_validate(estimator=l2_reg, X=X_train_smote, y=y_train_smote,\n",
    "                       cv=cv,\n",
    "                       n_jobs=-1,\n",
    "                       return_estimator=True,\n",
    "                       return_train_score=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "y_probability = cross_val_predict(l2_reg, X_train_scaled, y_train, cv=cv, method='predict_proba')[:,1]\n",
    "get_metric(y_train, y_probability)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Lasso"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "l1_reg = LogisticRegression(C=1,\n",
    "                            solver='saga',\n",
    "                            penalty='l1',\n",
    "                            max_iter=200)\n",
    "cv_l1 = cross_validate(estimator=l1_reg, X=X_train_smote, y=y_train_smote,\n",
    "                       cv=cv,\n",
    "                       n_jobs=-1,\n",
    "                       return_estimator=True,\n",
    "                       return_train_score=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "y_probability = cross_val_predict(l1_reg, X_train_scaled, y_train, cv=cv, method='predict_proba')[:,1]\n",
    "get_metric(y_train, y_probability)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.3"
  },
  "varInspector": {
   "cols": {
    "lenName": 16,
    "lenType": 16,
    "lenVar": 40
   },
   "kernels_config": {
    "python": {
     "delete_cmd_postfix": "",
     "delete_cmd_prefix": "del ",
     "library": "var_list.py",
     "varRefreshCmd": "print(var_dic_list())"
    },
    "r": {
     "delete_cmd_postfix": ") ",
     "delete_cmd_prefix": "rm(",
     "library": "var_list.r",
     "varRefreshCmd": "cat(var_dic_list()) "
    }
   },
   "types_to_exclude": [
    "module",
    "function",
    "builtin_function_or_method",
    "instance",
    "_Feature"
   ],
   "window_display": false
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
