# PrincipalComponentAnalysis

# Introduction
This program is an application of Principal Component Analysis (PCA) into scientific area using large data set. This program also demonstrates how to apply PCA using Python. Follows are general steps applying PCA:
0. The very first step is to include these codes at the top of your program: 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
1. Import your dataset 
2. Vectorization of dataset as X and y
3. Remove all the rows contains NaN values taking the advange of numpy and pandas.
4. standardization(center&scale the data: 0 means and unit variance for each y):
   X_std = StandardScaler().fit_transform(X)
   X_std = preprocessing.scale(X) also works
5. Create a PCA object
   pca = PCA() 
   pca = PCA(n_components=X.shape[1]) also works
6. Apply
   pca.fit(X_std)
7. Customize your plot

For more information, please see comments in the .py file.

# Study
Harmful algae blooms (HABs) have been implicated in fish kills, wildlife poisonings, and human health impacts related to consumption and recreational use. HAB reports are increasing both globally and within the Great Lakes basin, including the Lake Erie. The frequency and intensity of HABs in Lake Erie have increased over the decades, which impacts surface water quality and public health. The objective of this study is to identify potential indicators of HABs in Lake Erie using Principal Component Analysis (PCA) method. The data set in this study is selected from available weather/monitoring stations located along and, in the Lake. This study shows that the PCA method is an appropriate tool for this purpose. The results also illustrate inflow and suspended sediments are relatively critical indicators of algal bloom in Lake Erie. The study also tests the relevance of this tool for Lake Erie, which could be applied for water resource management.

# PCA
Principal component analysis (PCA) is a multivariate statistical technique that extracts important components from a data set in which the variables are inter-correlated. The goal of PCA is to extract the important information from the data set and to express this information as a set of new orthogonal variables called principal components. Extraction of principal components can be represented geometrically by rotating the original axes. Mathematically, PCA generally includes the following five major steps: (1) standardize the variables x1, x2, … , xp to unit norm and zero means so that they have equivalence weights in the analysis; (2) calculate the correlation or covariance matrix C; (3) table the eigenvalues λ1, λ2, …, λp  and the corresponding eigenvectors a1, a2, …, ap; (4) extract the components that have large possible variance (i.e., inertia, therefore the extracted components will explain large part of the inertia in the data set); and (5) calculate loadings (i.e. coefficients of correlation between variables and components), and plot loading versus indicators. Details for mathematically analyzing PCA are published elsewhere. 

# Analyzing Process and Results
Three basins are analyzed together to obtain which indicator contributes more to the algal bloom in the scope of the entire lake. In a PCA, the number of principal components is equal to the number of variables, however, a component is composed of all the variables used in the study rather than comprised of a single variable. For instance, 6 indicators are analyzed in this study, so 6 principal components are produced. The results show that the first component accounts for approximately 33.4% and the second component account for approximately 24.6% of the total variance of the data set. Because the percentages each component contributes to the total variance are not small enough to be neglected, all the principal components should be taken into consideration in this study. 

Although the relative importance of an indicator within a component is reflected, it does not provide any information on the importance of the indicator itself. Thus, a calculation like the weighted average is performed on the result in Figure 10 using the equation below
Weighted loading=∑_(n=1)^6▒〖(loading ×percentage of explained variance)     (2),〗

Results from this analysis could prove that algal bloom in Lake Erie is a result of the combination of many factors, and the mechanism that the factors act together influencing the intensity and fluency of algal bloom is complicated. Considering the three basins together, inflow, lake water surface temperature, precipitation, suspended sediments, and wind speed have positive relationships with an algal bloom, while cloud has a negative relationship with an algal bloom. Inflow and suspended sediments are primary to algae growth because they provide the nutrient the algae needed to grow, especially phosphorous and nitrogen. Within the appropriate range of temperature, algae growth is accelerated when the lake water surface temperature is warmer. When precipitation increases, the surface runoff will also increase, thereby increasing nutrient input into the lake. Stronger wind event with higher wind speed may promote surface water mixing and brings more nutrient the algae needs. More sunlight will be transmitted through the clouds when cloud coverage is low, thereby promoting photosynthesis of algae and accelerating algae growth. 
