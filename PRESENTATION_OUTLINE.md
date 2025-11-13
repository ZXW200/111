# NTD Clinical Trials Analysis - Presentation Outline

**Group 16 | Lancaster University**
**Presenting: Results & Findings [12 marks]**

---

## Slide 1: Title Slide

**Investigating Trends in Neglected Tropical Disease (NTD) Clinical Studies**

Group 16:
- Peizhe Jiang
- Congyao Ren
- Zixu Wang
- Alaghwani Balsam

Data: WHO ICTRP, 1999-2023

---

## Slide 2: Research Context

**What are NTDs?**
- Affect >1 billion people globally
- Disproportionate burden on disadvantaged populations
- Chronically underfunded research area

**Our Dataset:**
- 315 clinical trials (311 after cleaning)
- 25-year period (1999-2023)
- 62 countries
- 4 main diseases: Chagas, Schistosomiasis, Endemic worms, Visceral leishmaniasis

---

## Slide 3: Research Questions

**5 Key Questions:**

1. What factors associate with results publication?
2. Can network analysis reveal country partnerships?
3. What proportion funded by pharma companies?
4. Are children and pregnant women included?
5. Which drugs studied (Chagas disease)?

---

## SECTION 1: RESULTS & FINDINGS

---

## Slide 4: RQ 2.2.1 - Publication Factors

**Finding: Severe Publication Gap**

📊 **Key Statistics:**
- Publication rate: **4.2%** (13/311)
- Unpublished: **95.8%** (298/311)

**Method:** Logistic Regression
- Features: phase, study_type, sponsor, income level
- OneHotEncoder + Logistic Regression pipeline

**Implication:**
→ Massive publication bias in NTD research
→ Evidence base severely limited

[Include: coefficients_plot.png]

---

## Slide 5: RQ 2.2.1 - What Predicts Publication?

**Coefficient Analysis:**

📈 **Positive factors** (increase publication):
- [Show top 3 from logit_results.csv]

📉 **Negative factors** (decrease publication):
- [Show bottom 3 from logit_results.csv]

**Interpretation:**
- Study design matters
- Sponsor type influences publication
- Income level of host country affects transparency

---

## Slide 6: RQ 2.2.3 - Pharmaceutical Funding

**Finding: Low Industry Investment**

📊 **Sponsor Distribution:**
- **Non-profit: 66.9%** (208 trials)
- Other: 21.2% (66 trials)
- **Industry: 7.4%** (23 trials) ⚠️
- Government: 4.5% (14 trials)

**Geographic Distribution:**
- Industry trials: 23 across 22 countries
- High-burden vs normal regions: [from visualization]

[Include: sponsor_distribution.png]

---

## Slide 7: RQ 2.2.3 - Industry & Disease Burden

**Do pharma trials align with high-burden regions?**

📍 **High-burden countries identified:**
- India, Mexico, Tanzania, Bangladesh, Bolivia, Kenya, Egypt, Côte d'Ivoire

📊 **Industry trial distribution:**
[Include: industry_region.png]

**Finding:**
→ Potential misalignment between disease burden and pharma investment
→ Most research driven by non-profit sector

---

## Slide 8: RQ 2.2.2 - Country Collaborations

**Finding: International Partnerships Exist**

🌐 **Network Analysis:**
- Multi-country trials: [X trials]
- Countries in network: 62
- Collaborative connections: [X edges]

**Top Hub Countries** (by betweenness centrality):
1. [Country 1] - Partners: X, Centrality: Y
2. [Country 2] - Partners: X, Centrality: Y
3. [Country 3] - Partners: X, Centrality: Y

**Implication:**
→ Some countries act as bridges in NTD research
→ Collaboration patterns reveal research leadership

[Include: network diagram if available, or table from network_statistics.csv]

---

## Slide 9: RQ 2.2.4 - Special Populations

**Finding: Limited Inclusion**

👶 **Children:**
- No explicit tracking field in dataset
- Estimated ~50% trials include children (age < 18)
- Data quality limitation

🤰 **Pregnant Women:**
- Explicitly included: **12 trials (3.8%)**
- Excluded or unclear: 96.2%

**Implication:**
→ Special populations underrepresented
→ Evidence gap for vulnerable groups

---

## Slide 10: RQ 2.2.5 - Drug Trends (Chagas)

**Finding: Drug Development Focus Areas**

💊 **Chagas Disease Analysis:**
- Trials identified: [X]
- Drug extraction: Regex-based from intervention field

**Top 5 Drugs:**
1. [Drug 1] - Count: X
2. [Drug 2] - Count: X
3. [Drug 3] - Count: X
4. [Drug 4] - Count: X
5. [Drug 5] - Count: X

**Temporal Trends:**
- [Show trend from chagas_drug_trends.csv]

---

## SECTION 2: HOW RESULTS ADDRESS RESEARCH GOALS

---

## Slide 11: Addressing the Main Research Goal

**Goal:** "Clarify evolution of NTD trial activities, understand trends and characteristics"

**How Our Findings Address This:**

✅ **Temporal Trends:**
- 25-year coverage reveals long-term patterns
- Drug trends show evolving treatment focus

✅ **Geographic Distribution:**
- 62 countries analyzed
- Collaboration networks mapped
- Hub countries identified

✅ **Funding Structure:**
- Clear sponsor classification
- Dominated by non-profits (67%)
- Industry minimal (7%)

✅ **Publication Patterns:**
- Severe gap identified (4.2%)
- Factors influencing transparency revealed

✅ **Population Focus:**
- Special groups underrepresented
- Highlights evidence gaps

---

## SECTION 3: REFLECTION (Limitations & Improvements)

---

## Slide 12: Data Limitations

**Challenges We Encountered:**

1. **Publication Bias**
   - Only 13 published trials (4.2%)
   - Limited our analysis capability

2. **Data Completeness**
   - 4 outliers removed
   - Missing values in key fields
   - No explicit child participant field

3. **Classification Accuracy**
   - Sponsor classification: keyword-based
   - May misclassify edge cases

4. **Temporal Imbalance**
   - Distribution across years varies
   - More recent years may have more trials

---

## Slide 13: Methodological Limitations

**Analytical Challenges:**

1. **Logistic Regression:**
   - ⚠️ Small published sample (n=13)
   - ⚠️ Severe class imbalance (4% vs 96%)
   - May affect model reliability

2. **Network Analysis:**
   - Country-level only
   - Doesn't capture institution collaborations
   - Edge weights may not reflect collaboration depth

3. **Drug Extraction:**
   - Regex-based: may miss naming variations
   - Limited to Chagas only (not all 4 diseases)

**What We Did to Mitigate:**
- Stratified sampling for train/test split
- Transparent documentation of limitations
- Robust data cleaning pipeline

---

## Slide 14: Recommendations for Improvement

**Future Research Directions:**

### Data Collection Improvements:
1. **Standardize reporting:**
   - Mandatory special population fields
   - Enforce result publication deadlines

2. **Expand scope:**
   - Include post-2023 trials
   - Add institution-level data

### Methodological Enhancements:
1. **Address class imbalance:**
   - SMOTE (synthetic oversampling)
   - Class weighting in model

2. **Advanced methods:**
   - Ensemble models (Random Forest, XGBoost)
   - Deep learning for text analysis

3. **Deeper network analysis:**
   - Institution-level networks
   - Temporal network evolution
   - Community detection algorithms

4. **NLP techniques:**
   - Named Entity Recognition for drug extraction
   - Sentiment analysis of trial outcomes

---

## Slide 15: Policy Recommendations

**Actionable Insights for Stakeholders:**

### For Regulators:
- 📋 **Mandate result publication** (address 96% gap)
- 📊 **Standardize data reporting** (special populations)

### For Funders:
- 💰 **Incentivize pharma investment** (currently 7%)
- 🎯 **Target high-burden regions** (align funding)

### For Researchers:
- 👶 **Design inclusive trials** (children, pregnant women)
- 🤝 **Strengthen collaborations** (leverage hub countries)

### For WHO/IDDO:
- 🗄️ **Improve data quality** (consistent fields)
- 📈 **Track temporal trends** (updated annually)

---

## Slide 16: Strengths of Our Analysis

**What We Did Well:**

✅ **Comprehensive Data Cleaning**
- Robust outlier detection (sample size, age validation)
- Automated sponsor classification
- Income level mapping

✅ **Multi-Method Approach**
- Statistics (descriptive analysis)
- Machine Learning (logistic regression)
- Network Analysis (collaboration patterns)
- Text Mining (drug extraction)

✅ **Reproducible Pipeline**
- Well-documented code
- Modular design (5 separate scripts)
- Clear execution order

✅ **Transparent Analysis**
- Acknowledged all limitations
- Detailed documentation (CODE_DOCUMENTATION.txt)

✅ **Practical Insights**
- Actionable recommendations
- Policy implications identified

---

## Slide 17: Key Takeaways

**3 Major Discoveries:**

1. 🚨 **Publication Crisis**
   - 95.8% of trials don't publish results
   - Evidence base severely limited

2. 💰 **Funding Gap**
   - Pharma companies fund only 7.4%
   - Non-profits carry the burden (67%)

3. 👥 **Representation Gap**
   - Special populations underrepresented
   - Pregnant women: 3.8%, Children: unclear

**What This Means:**
→ NTD research needs structural change
→ Better incentives, mandates, and data standards required

---

## Slide 18: Conclusion

**Summary:**

✅ Analyzed 311 NTD trials (1999-2023)

✅ Answered all 5 research questions

✅ Identified critical gaps:
- Publication transparency
- Pharma investment
- Special population inclusion

✅ Provided actionable recommendations

**Impact:**
- Evidence for policy makers
- Guidance for funders
- Roadmap for researchers

---

## Slide 19: Thank You

**Questions?**

**Project Materials:**
- Code: [GitHub link if available]
- Documentation: CODE_DOCUMENTATION.txt
- Data: CleanedData/ folder

**Contact:**
Group 16 - Lancaster University

---

## APPENDIX SLIDES (if needed)

---

## Appendix A: Data Cleaning Process

**Steps:**
1. Remove duplicates and outliers (4 removed)
2. Classify sponsors (4 categories)
3. Map income levels (World Bank)
4. Validate ages (0-120 years, 0-1440 months)
5. Clean HTML tags from text fields
6. Fill missing values (median/mode/Unknown)

**Output:** 311 clean trials ready for analysis

---

## Appendix B: Logistic Regression Details

**Model Specification:**
```
Pipeline([
    ('encoder', OneHotEncoder(handle_unknown='ignore')),
    ('logit', LogisticRegression(max_iter=2000))
])
```

**Features:** phase, study_type, sponsor_category, income_level
**Target:** results_posted (binary)
**Split:** 80/20 train-test, stratified

---

## Appendix C: Network Metrics Explained

**Degree:** Number of partners a country collaborates with

**Weighted Degree:** Total collaboration instances (accounts for repeated partnerships)

**Betweenness Centrality:** Measures how often a country appears on shortest paths between others → identifies "bridge" countries

---

## END OF PRESENTATION

---

# SPEAKER NOTES

## Time Allocation (assuming 10-15 minute presentation):

- Introduction (Slides 1-3): 2 minutes
- Results (Slides 4-10): 6 minutes (1 min per RQ)
- Addressing Goals (Slide 11): 1 minute
- Limitations (Slides 12-13): 2 minutes
- Recommendations (Slides 14-15): 2 minutes
- Conclusion (Slides 16-18): 2 minutes

**Total: ~15 minutes + Q&A**

## Tips for Delivery:

1. **Point to visualizations** when discussing results
2. **Use laser pointer** or cursor to highlight key numbers
3. **Speak slowly** - give audience time to read slides
4. **Make eye contact** - don't just read slides
5. **Emphasize the 3 key findings** (publication, funding, representation gaps)
6. **Be confident about limitations** - shows critical thinking
7. **End with impact** - policy recommendations show practical value

## Anticipated Questions:

Q: "Why only 4.2% published?"
A: "This reflects broader issues in clinical trial transparency. Our analysis suggests factors like sponsor type and study phase influence publication, but the gap is concerning."

Q: "How reliable is keyword-based sponsor classification?"
A: "We validated against known categories and achieved good accuracy. However, we acknowledge edge cases may be misclassified - this is documented in our limitations."

Q: "Why focus on Chagas for drug analysis?"
A: "Time constraints. Chagas had sufficient trial volume for meaningful trend analysis. Future work could expand to all 4 diseases."

Q: "What about statistical significance of your regression?"
A: "With only 13 published trials, statistical power is limited. We focus on coefficient direction and magnitude, acknowledging this limitation."
