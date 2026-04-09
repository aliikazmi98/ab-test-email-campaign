# A/B Test Analysis: Marketing Campaign Optimization

End-to-end statistical A/B test analysis on a real marketing 
dataset of 588,000+ users, determining which ad variant drives 
higher conversion rates.

## Results
- **Winning Variant:** Treatment (Ad exposure) 
- **p-value:** 1.71e-13 (astronomically significant)
- **Conversion Lift:** +43.09% (1.785% → 2.555%)
- **Confidence Interval:** [0.60%, 0.94%]
- **Statistical Power:** 100%
- **Confidence Level:** 95%

## Key Findings
- Treatment group converted at 2.555% vs 1.785% for Control
- Both Chi-square and Z-test confirmed identical conclusions
- Result is unambiguous — confidence interval doesn't come 
  close to zero
- Running ads instead of PSAs is clearly the better strategy

## Statistical Tests Used
- **Chi-square test** — for conversion rate comparison
- **Z-test for proportions** — for significance testing
- **Confidence interval analysis** — for effect size estimation
- **Statistical power analysis** — to confirm test reliability

## Visualizations
![Conversion Rate by Group](visualizations/01_conversion_rate_by_group.png)
![Confidence Intervals](visualizations/03_confidence_intervals.png)
![Funnel Chart](visualizations/04_funnel_chart.png)
![Conversion by Day](visualizations/05_conversion_by_day.png)

## Project Structure
- `ab_test_analysis.py` — Full statistical analysis pipeline
- `ab_test_results.txt` — Detailed statistical report
- `ab_summary.csv` — Clean summary table
- `marketing_AB.csv` — Raw dataset (588K+ users)
- `visualizations/` — All 6 chart outputs

## Tech Stack
Python, Pandas, SciPy, Matplotlib

## How to Run
pip install pandas scipy matplotlib

python3 ab_test_analysis.py

## Business Recommendation
The Treatment variant (actual ads) should be deployed 
to all users. The 43% conversion lift represents a 
significant and statistically proven revenue opportunity.
With 588,000+ users, even a 0.77 percentage point improvement 
translates to thousands of additional conversions.

## Dataset
Real marketing A/B test data via Kaggle 
(faviovaz/marketing-ab-testing)
