# Python Statistics Cheatsheet

> Code on the left, how to read the result on the right

Most cheatsheets give you the code. That's why I wanted to create something that has both: the code on the left and how to read the output on the right.

It covers eight tests and measures I use regularly in real business analysis.

- Descriptive statistics to understand your data before you touch it.
- Z-scores for outlier detection.
- Normality checks before you run tests that assume it.
- IQR for skewed distributions where standard deviation will mislead you.
- T-tests and chi-square for comparing groups.
- Correlation and regression to understand relationships, and how much your model actually explains.

The interpretation column is the part that matters most when a stakeholder is looking at you and waiting for an answer.

---

## 1. Descriptive Statistics

**Central tendency and spread**

| Python | How to read it |
|--------|----------------|
| `df['col'].mean()` | Mean ≠ median → data is skewed, use median |
| `df['col'].median()` | High std → values vary widely from the mean |
| `df['col'].std()` | Low std → values cluster tightly around the mean |
| `df['col'].describe()` | |

---

## 2. Z-score (Outlier Detection)

| Python | How to read it |
|--------|----------------|
| `from scipy import stats` | z = 0 → value equals the mean |
| `z = stats.zscore(df['col'])` | z = 2 → unusually high, worth checking |
| `df[abs(z) > 3]` | \|z\| > 3 → likely outlier, investigate before keeping |

---

## 3. Distributions

### Check Normality

| Python | How to read it |
|--------|----------------|
| `from scipy import stats` | p > 0.05 → data is likely normal |
| `_, p = stats.shapiro(df['col'])` | p < 0.05 → data is not normal, check assumptions |
| `df['col'].skew()` | Skew near 0 → symmetric distribution |
| `df['col'].kurt()` | High skew → long tail on one side |

### Percentiles and IQR

| Python | How to read it |
|--------|----------------|
| `Q1 = df['col'].quantile(0.25)` | IQR = spread of middle 50% of data |
| `Q3 = df['col'].quantile(0.75)` | Values below lower or above upper → outliers |
| `IQR = Q3 - Q1` | More robust than std when data is skewed |
| `lower = Q1 - 1.5 * IQR` | |
| `upper = Q3 + 1.5 * IQR` | |

---

## 4. Hypothesis Testing

### T-test (Compare Two Groups)

| Python | How to read it |
|--------|----------------|
| `from scipy import stats` | p < 0.05 → difference is statistically significant |
| `t, p = stats.ttest_ind(` | p ≥ 0.05 → difference could be random |
| `    group_a, group_b` | Significant ≠ important. Always check effect size too |
| `)` | |

### Chi-square (Categorical Variables)

| Python | How to read it |
|--------|----------------|
| `from scipy.stats import chi2_contingency` | p < 0.05 → relationship between variables is significant |
| `ct = pd.crosstab(df['group'], df['outcome'])` | p ≥ 0.05 → no significant relationship detected |
| `chi2, p, dof, _ = chi2_contingency(ct)` | Use when both variables are categorical |

---

## 5. Correlation and Regression

### Pearson Correlation

| Python | How to read it |
|--------|----------------|
| `df[['x','y','z']].corr()` | r > 0.7 → strong positive relationship |
| `from scipy import stats` | r < -0.7 → strong negative relationship |
| `r, p = stats.pearsonr(df['x'], df['y'])` | r near 0 → no linear relationship |
| | Correlation does not imply causation |

### Linear Regression

| Python | How to read it |
|--------|----------------|
| `from sklearn.linear_model import LinearRegression` | R² = 0.85 → model explains 85% of variation |
| `model = LinearRegression()` | R² near 0 → model explains very little |
| `model.fit(X, y)` | coef = effect of each variable |
| `model.score(X, y)` | High R² does not mean the model is correct |
| `model.coef_` | |

---

> **Save this for your next analysis**
