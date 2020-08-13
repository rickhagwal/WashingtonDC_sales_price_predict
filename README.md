# Washington DC Housing- Sales Price prediction

### INTRODUCTION and Research Question-
Here, we will take a closer look at Sale Prices, available for Washington D.C, provided from around 1990 to July, 2018.
After looking through the data, I arrived at a research question for one topic of interest i.e., Predicting Sale Prices for Washington D.C city. 

### Data Manipulation and Cleaning-
There are total 46 features present, divided into- Categorical, Continuous and Target column (PRICE) -

#### Categorical features-
'HEAT', 'AC', 'SALEDATE', 'QUALIFIED', 'STYLE', 'STRUCT', 'GRADE',
       'CNDTN', 'EXTWALL', 'ROOF', 'INTWALL', 'GIS_LAST_MOD_DTTM', 'SOURCE',
       'FULLADDRESS', 'CITY', 'STATE', 'NATIONALGRID', 'ASSESSMENT_NBHD',
       'ASSESSMENT_SUBNBHD', 'CENSUS_BLOCK', 'WARD', 'QUADRANT

#### Continuous features-
'BATHRM', 'HF_BATHRM', 'NUM_UNITS', 'ROOMS', 'BEDRM', 'AYB', 'YR_RMDL', 'EYB', 'STORIES', 'SALE_NUM', 'GBA','BLDG_NUM' 'KITCHENS', 'FIREPLACES', 'USECODE', 'LANDAREA', 'ZIPCODE',  'LATITUDE', 'LONGITUDE', 'CENSUS_TRACT', 'SQUARE', 'X', 'Y'

There are no missing values present in the dataset. 
Dropped few features, which is repeating either same information, as other columns (such as- X,Y are similar to Latitude and Longitude) or is not giving any useful information, to predict ‘PRICE’ such as- GIS_LAST_MOD_DTTM','SOURCE', 'FULLADDRESS', 'CITY' , 'STATE', 'ASSESSMENT_SUBNBHD' , 'CENSUS_BLOCK' , 'NATIONALGRID', 'BLDG_NUM'
(As these features are not an important characteristic to determine Price. For example- FullAddress is not required, as the place is already located in ‘ASSESSMENT_NBHD’ column.)
Manipulate few features such as- SALEDATE column is converted to just Year column. 

#### Convert Categorical columns to continuous features by-
 e.g., Converting features such as- 'AC', 'CNDTN', 'QUALIFIED', via Label Encoding.
And the remaining categorical features via One Hot Encoding.

#### Outliers in Continuous features-
From the graphs below, it can be seen that all continuous features have outliers. Outliers can be a problem for linear model performance. But, since they do not effect tree based models, thus, I haven’t did any preprocessing on them.

![alt text](https://github.com/rickhagwal/WashingtonDC_sales_price_predict/blob/master/images/1_Boxplot_cont.PNG)

#### Skewness of Continuous features-
Few of the features are skewed, as can be seen from the below plots-

![alt text](https://github.com/rickhagwal/WashingtonDC_sales_price_predict/blob/master/images/2_Skewness_continuous_features.PNG)

![alt text](https://github.com/rickhagwal/WashingtonDC_sales_price_predict/blob/master/images/3_skewness_cont.PNG)

![alt text](https://github.com/rickhagwal/WashingtonDC_sales_price_predict/blob/master/images/4_skew.PNG)

To reduce some of the skewness of these features, I did Log transformation on these features.

#### Target Variable('PRICE') skewness-

As can be seen from the below plots and description, that ‘PRICE’ feature also has right skewness in data.-

![alt text](https://github.com/rickhagwal/WashingtonDC_sales_price_predict/blob/master/images/5_price_skew.PNG)

![alt text](https://github.com/rickhagwal/WashingtonDC_sales_price_predict/blob/master/images/6_price.PNG)

As max PRICE is quite far away from the other quartiles of data.
In order to reduce skewness, used InterQuantileRange (IQR), to calculate upper and lower boundary, and used upper boundary (1560000.0), to reduce skewness of data, which is somewhat reduced too, as can be seen in below plot-

![alt text](https://github.com/rickhagwal/WashingtonDC_sales_price_predict/blob/master/images/7_price_unskewed.PNG)

### Data Exploration-

#### Spearman Correlation's Heatmap-

![alt text](https://github.com/rickhagwal/WashingtonDC_sales_price_predict/blob/master/images/heatmap_spearman.png)


Fig.5. Spearman Correlation of Continuous Features via Heatmap(to detect non-linear trend in data)

#### Pairplot-

![alt text](https://github.com/rickhagwal/WashingtonDC_sales_price_predict/blob/master/images/9_pairplot.PNG)

It can be seen from the above plot, that Rooms and Bathrooms, Landarea and GBA  both of them have some kind of linear relationship.

#### Feature Selection-

There came around 163 features after doing data manipulation (i.e, encoding on categorical columns) on dataset. Performed Feature Selection, using ExtraTreesRegressor of RandomForest. Took threshold of around 0.01, and got just 27 features, to select columns contributing to the target (‘PRICE’) column.

![alt text](https://github.com/rickhagwal/WashingtonDC_sales_price_predict/blob/master/images/10_Feature_Selection.png)



Fig.7. Visualizing important features, via Feature Selection

As can be seen from the above plot, following features contributes most (in decreasing order) to the target (‘PRICE’)-

![alt text](https://github.com/rickhagwal/WashingtonDC_sales_price_predict/blob/master/images/11_features_dist.PNG)



#### Data Methodology-

After selecting features, Split dataset into train and test, and performed Standardization on continuous features, to scale all the selected features in the same range. 
In order to determine ‘PRICE’, applied supervised learning techniques (linear and tree-based ML algorithms), such as-
i.	Linear Regression,
ii.	Lasso Regression,
iii.	Ridge Regression,
iv.	Decision Tree
v.	Random Forest
Metrics used for calculating Regression-
Performed MAE (Mean Absolute Error) and RMSE (Root Mean Squared Error).

#### Data Analysis-

![alt text](https://github.com/rickhagwal/WashingtonDC_sales_price_predict/blob/master/images/12_metric.png)



Fig.8. MAE and RMSE of different Machine Learning models on train and test dataset


As can be seen from the above plot, Decision Tree is performing best on train dataset, for both MAE and RMSE, which may lead to overfitting, as for test dataset it is performing around 156124 RMSE. Whereas, for other models i.e., Linear Regression, Lasso and Ridge Regression, MAE Train is around 107000, and RMSE Train is around 143600, and MAE Test is around 109500 and RMSE Test is around 146200. Just Random Forest model performed better than other models, on both training as well as test dataset, with RMSE of 107383 and MAE of 70959 o on test dataset.

#### Conclusion-

As can be seen above, Random Forest is performing best than all the other models, with RMSE on test datset as- 107383 and MAE of 70959. And if will perform hyperparameter optimization on Random Forest model, it may give even more better results.


