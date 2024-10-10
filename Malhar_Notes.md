# Malhar's Notes

## Questions i Want to answer

	what features do we have with us?
	what do they mean?
	what our “outliers” are?
	How are these features are analysed by a doctor?
	How are these features analysed by current DL models?


## Notes

1. 3 available datasets (AddNeuroMed1 (ANM1), ANM2, and AD Neuroimaging Initiative (ADNI))
    - https://www.ncbi.nlm.nih.gov/geo/download/?acc=GDS4758
    - ADNI and ANM datasets are publicly available (ADNI, http://adni.loni.usc.edu/; ANM, https://www.ncbi.nlm.nih.gov/geo/).
    - I couldn't find ANM dataset

2. Very good Paper - https://www.nature.com/articles/s41598-020-74399-w
    - good quality review of multimodal alzheimers disease prediction, reviews the dataset well

3. ADNI dataset facts
    - 503 MRI scans (cross sectional -consisting of 9108 voxels per patient distributed over 18 slices, with each slice having 22 × 23 voxels)
    - 2004 clinical or EHR (electronic health records) datapoints - (e.g., memory tests, balance tests, and cognitive tests)
    - 808 WGS (whole genome sequences) - (at the time of sequencing, 128 with AD, 415 with MCI, and 267 controls) by illumina

    -  588 patients have SNP and EHR, 283 patients have imaging and EHR, the remaining patients have only EHR data.


## Conclusions

1. I want to work with genomic data

Reasons:
    - I think its the future - to be able to see things just through your genome and epigenome. You won't need any other data to be able to assess your biophysiology. This will become more feasible as sequencing the DNA becomes cheaper and more accessible, which is happening due to new technologies emerging at a rapid rate.
    - Analysing a genome is a very complex problem - Genome is a very broad data set (5 mil features), combined with epigenome and other omics, it creates a tough yet interesting problem to analyse.

2. There is Limited papers on working with genomes for Alzheimers - I found many many more papers working with MRIs and EHRs for the same analysis. While MRIs and EHRs help diagnosing AD, if youve already done your MRI/EHR you might as well show it to the doctor and get it diagnosed

What I think we should do - Apply for all possible omics data (genomic/epigenomic/anything else) and also download from other sources, and build a sexy asf deep network to predict alzheimers well.
- a paper i found doing deep network on genomic for alzheimers [here](https://www.researchgate.net/publication/368486521_Deep_Belief_Network-based_Approach_for_Detecting_Alzheimer's_disease_using_the_Multi-omics_Data/link/640612a50cf1030a5679f607/download?_tp=eyJjb250ZXh0Ijp7ImZpcnN0UGFnZSI6InB1YmxpY2F0aW9uIiwicGFnZSI6InB1YmxpY2F0aW9uIn19)
- another paper [here](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10741889/)
- another one [here](https://www.nature.com/articles/s41598-020-60595-1#data-availability)


## Todo for this goal

- [x] Apply for ADNI relevant data [here](http://adni.loni.usc.edu/)

- [ ] Find the other data on nih geo [here](https://www.ncbi.nlm.nih.gov/geo/download/?acc=GDS4758)

- [ ] Compile all data

- [ ] build large repository (with good references to check back) of research papers to figure out state of the art for:
    - feature selction
    - data preprocessing
    - model architecture
    - results
    - etc

- [ ] Build a Timeline and plan to execute project.
    Include:
    - What basic shallow models well use for benchmarks (SVM KNN etc)
    - How and where well start to execute state of art
    - How we plan to improve on it

- [ ] MISC
    - [ ] Everyone Learn GIT and well!(grading depends on it)
    - [ ] Maintain knowledge base and keep it well organized (papers, etc)
