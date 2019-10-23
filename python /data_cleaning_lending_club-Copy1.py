{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Data Cleaning"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## LendingClub data source:\n",
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
    "description = pd.read_excel('LendingClub/LCDataDictionary.xlsx')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# from zipfile import ZipFile\n",
    "# zip_file = ZipFile('LendingClub/LoanStats3d_securev1.csv.zip')\n",
    "# data_lc = pd.read_csv(zip_file.open('LoanStats3d_securev1.csv'), low_memory=False, header=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#read LendingClub loan data from 2014\n",
    "data_lc = pd.read_csv('LendingClub/LoanStats3c_securev1.csv', low_memory=False, header=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "#* * * IMPORTANT * * *\n",
    "#removed two rows with full NAN values\n",
    "data_lc = data_lc.loc[data_lc.loan_amnt.notnull()]\n",
    "data_lc.shape"
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
    "### Further cleaning separately for train and test"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "\n",
    "\n",
    "- annual_inc: in case of annual income we had to handle extreme values (there are many strategies, we were choosing truncating the extreme values to the value of the 99 quantile)\n",
    "- imputing mean values in place of missing values\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "imp = SimpleImputer(strategy='median', copy=True, fill_value=None)\n",
    "imp.fit(X_train)  \n",
    "\n",
    "X_train_imp = imp.transform(X_train)      \n",
    "X_test_imp = imp.transform(X_test)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "income_trunc = X_train.annual_inc.quantile(q=0.99)\n",
    "income_trunc"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "b_train = X_train_imp > income_trunc\n",
    "b_test = X_test_imp > income_trunc"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train_imp[b_train] = income_trunc\n",
    "X_test_imp[b_test] = income_trunc"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "plt.hist(X_train_imp[:,4])\n",
    "plt.title('Annual income');"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "### Further datacleaning separately for train and test\n",
    "\n",
    "\n",
    "\n",
    "- annual_inc: in case of annual income we had to handle extreme values (there are many strategies, we were choosing truncating the extreme values to the value of the 99 quantile)\n",
    "- imputing mean values in place of missing values\n",
    "\n",
    "\n",
    "imp = SimpleImputer(strategy='median', copy=True, fill_value=None)\n",
    "imp.fit(X_train)  \n",
    "\n",
    "X_train_imp = imp.transform(X_train)      \n",
    "X_test_imp = imp.transform(X_test)\n",
    "\n",
    "income_trunc = X_train.annual_inc.quantile(q=0.99)\n",
    "income_trunc\n",
    "\n",
    "b_train = X_train_imp > income_trunc\n",
    "b_test = X_test_imp > income_trunc\n",
    "\n",
    "X_train_imp[b_train] = income_trunc\n",
    "X_test_imp[b_test] = income_trunc\n",
    "\n",
    "plt.hist(X_train_imp[:,4])\n",
    "plt.title('Annual income');"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train = data_cleaning(X_train)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_test = data_cleaning(X_test)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- Setting Null values as mean for emp_lenth for both test and train"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_test.loc[X_test.emp_length.isna(), 'emp_length'] = round(X_test.emp_length.mean())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train.loc[X_train.emp_length.isna(), 'emp_length'] = round(X_train.emp_length.mean())"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- Setting Null values as median for revol_util & dti for both test and train "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_test.loc[X_test.revol_util.isna(), 'revol_util'] = X_test.revol_util.median(skipna=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train.loc[X_train.revol_util.isna(), 'revol_util'] = X_train.revol_util.median(skipna=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train.loc[X_train.dti.isna(), 'dti'] = X_train.dti.median(skipna=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "X_test.loc[X_test.dti.isna(), 'dti'] = X_test.dti.median(skipna=True)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- Truncking income salary for both test and train"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "inc_trunc = X_test.annual_inc.quantile(q=0.995)\n",
    "X_test.loc[X_test.annual_inc>inc_trunc, 'annual_inc'] = X_test.annual_inc.quantile(q=0.995)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "inc_trunc = X_train.annual_inc.quantile(q=0.995)\n",
    "X_train.loc[X_train.annual_inc>inc_trunc, 'annual_inc'] = X_train.annual_inc.quantile(q=0.995)"
   ]
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
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
