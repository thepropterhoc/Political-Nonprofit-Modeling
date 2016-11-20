#AIzaSyADuv1109TAE_eDK-g0abWMkYrIYjJahmw
#AIzaSyCL9rEHZn_GZvIWwxDAnPajDKS1Uz1JWH8

import psycopg2 as dbapi
from datetime import datetime
import sys
import csv
import math
#from SVMCurve import SVMCurve
from random import random
import numpy as np
import sklearn.linear_model
from sklearn.svm import SVR, SVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.learning_curve import learning_curve
from sklearn import cross_validation, metrics
import matplotlib as mpl 
#import pygraphviz as pz

mpl.use('TkAgg')

from matplotlib import pyplot as plt
from mpl_toolkits.basemap import Basemap

def featureFilter(x, selectedFeatures):
	if type(x) == type(tuple()):
		x = list(x)
	selectedColumnsIndices = [idx for idx in range(len(selectedFeatures)) if selectedFeatures[idx] == 1]
	if not len(x) > 1:
		x = list(x)
	zipped = list(zip(*x))
	selectedColumns = [zipped[idx] for idx in selectedColumnsIndices]
	x = list(zip(*selectedColumns))
	return x

def oneHotEncode(columns=None, oneHotColumnIndices=[]):
	numCols = len(columns)
	returnedX = []
	prevColumnIndex = 0
	for columnIndex in oneHotColumnIndices:
		if prevColumnIndex != columnIndex:
			returnedX += list(map(lambda x: list(map(lambda y: float(y), x)), columns[prevColumnIndex : columnIndex]))
		colVals = columns[columnIndex]
		ints = list(map(lambda x: [int(x)], LabelEncoder().fit_transform(colVals)))
		encoder = OneHotEncoder(categorical_features = 'all', sparse=False, dtype='float', n_values='auto', handle_unknown='error')
		fitted = encoder.fit(ints)
		oneHots = encoder.fit_transform(ints).tolist()
		returnedX += list(zip(*oneHots))
		prevColumnIndex = columnIndex + 1
	if prevColumnIndex != numCols:
		returnedX += list(map(lambda x: list(map(lambda y: float(y), x)), columns[prevColumnIndex:]))
	return list(zip(*returnedX))


MAX_ITER = 1000000

con = dbapi.connect(host="localhost", port="5432", database="postgres", user="shelbyvanhooser")

cur = con.cursor()

cur.execute("""
SELECT * 
FROM full_link
WHERE totalBurrage < 100000.0 AND 
	totalIstook < 100000.0 AND  
	totalCornett < 100000.0 AND 
	totalEdmondson < 100000.0 AND 
	totalFallin1 < 100000.0 AND 
	totalPriest < 100000.0
;
""")
rows = cur.fetchall()
print("Total number of samples : ", len(rows))



# Mean income = HC01_EST_VC15
# Median income = HC01_EST_VC13

"""

DROP TABLE IF EXISTS with_home_value;
CREATE TABLE IF NOT EXISTS with_home_value (
	first_name varchar(80), 
	last_name varchar(80),
	zip varchar(10),
	lat double precision,
	lon double precision,
	mean_household_income double precision,
	median_household_income double precision,
	totalBurrage double precision, 
	totalIstook double precision, 
	totalCornett double precision, 
	totalEdmondson double precision, 
	totalFallin1 double precision, 
	totalPriest double precision,
	two_year_total double precision,
	one_year_total double precision,
	ninety_day_total double precision,
	thirty_day_total double precision,
	home_value double precision,
	zillow_id varchar(80));
INSERT INTO with_home_value
SELECT full_link.*, NULL, NULL FROM full_link;


DROP TABLE IF EXISTS full_link;
CREATE TABLE IF NOT EXISTS full_link (
	first_name varchar(80), 
	last_name varchar(80),
	zip varchar(10),
	lat double precision,
	lon double precision,
	mean_household_income double precision,
	median_household_income double precision,
	totalBurrage double precision, 
	totalIstook double precision, 
	totalCornett double precision, 
	totalEdmondson double precision, 
	totalFallin1 double precision, 
	totalPriest double precision,
	two_year_total double precision,
	one_year_total double precision,
	ninety_day_total double precision,
	thirty_day_total double precision);
INSERT INTO full_link
SELECT 
	ddonors.first_name, 
	ddonors.last_name, 
	ddonors.zip, 
	donors_valid.lat, 
	donors_valid.lon, 
	census_data.HC01_EST_VC15 AS mean_household_income,
	census_data.HC01_EST_VC13 AS median_household_income,
	SUM(CASE WHEN tracked_contributions.ethics_num = '110053' THEN tracked_contributions.total_contributions ELSE 0.0 END) AS totalBurrage, 
	SUM(CASE WHEN tracked_contributions.ethics_num = '106127' THEN tracked_contributions.total_contributions ELSE 0.0 END) AS totalIstook, 
	SUM(CASE WHEN tracked_contributions.ethics_num = '714002' THEN tracked_contributions.total_contributions ELSE 0.0 END) AS totalCornett, 
	SUM(CASE WHEN tracked_contributions.ethics_num = '110013' THEN tracked_contributions.total_contributions ELSE 0.0 END) AS totalEdmondson, 
	SUM(CASE WHEN tracked_contributions.ethics_num = '110051' THEN tracked_contributions.total_contributions ELSE 0.0 END) AS totalFallin1, 
	SUM(CASE WHEN tracked_contributions.ethics_num = '110150' THEN tracked_contributions.total_contributions ELSE 0.0 END) AS totalPriest, 
	donor_windows.two_year_total,
	donor_windows.one_year_total,
	donor_windows.ninety_day_total,
	donor_windows.thirty_day_total
FROM (SELECT DISTINCT first_name, last_name, zip, address, city, state FROM donors) ddonors
	INNER JOIN donors_valid 
		ON UPPER(ddonors.address) = UPPER(donors_valid.address) AND 
		UPPER(ddonors.city) = UPPER(donors_valid.city) AND 
		UPPER(ddonors.state) = UPPER(donors_valid.state) AND 
		UPPER(ddonors.zip) = UPPER(donors_valid.zip)
	INNER JOIN tracked_contributions
		USING(lat, lon)
	INNER JOIN donor_windows
		ON UPPER(donor_windows.first_name) = UPPER(ddonors.first_name) AND
		UPPER(donor_windows.last_name) = UPPER(ddonors.last_name)
	INNER JOIN census_data
		ON ddonors.zip = census_data.geo_id2
GROUP BY 
	ddonors.first_name, 
	ddonors.last_name, 
	ddonors.zip, 
	donors_valid.lat, 
	donors_valid.lon,
	mean_household_income,
	median_household_income,
	two_year_total,
	one_year_total,
	ninety_day_total,
	thirty_day_total
;
"""

# Unknown
unknownsQuery = """
select 
points_map.zip,
points_map.lat, 
points_map.lon, 
census_data.HC01_EST_VC15 AS mean_household_income,
census_data.HC01_EST_VC13 AS median_household_income,
SUM(CASE WHEN total_grouped_contributions.ethics_num = '110053' THEN total_grouped_contributions.totalContributions ELSE 0.0 END) AS totalBurrage, 
SUM(CASE WHEN total_grouped_contributions.ethics_num = '106127' THEN total_grouped_contributions.totalContributions ELSE 0.0 END) AS totalIstook, 
SUM(CASE WHEN total_grouped_contributions.ethics_num = '714002' THEN total_grouped_contributions.totalContributions ELSE 0.0 END) AS totalCornett, 
SUM(CASE WHEN total_grouped_contributions.ethics_num = '110013' THEN total_grouped_contributions.totalContributions ELSE 0.0 END) AS totalEdmondson, 
SUM(CASE WHEN total_grouped_contributions.ethics_num = '110051' THEN total_grouped_contributions.totalContributions ELSE 0.0 END) AS totalFallin1, 
SUM(CASE WHEN total_grouped_contributions.ethics_num = '110150' THEN total_grouped_contributions.totalContributions ELSE 0.0 END) AS totalPriest
	from total_grouped_contributions 
	inner join contributor using(contributor_id) 
	inner join points_map on 
		upper(points_map.address) = upper(contributor.street) 
		and upper(points_map.city) = upper(contributor.city) 
		and upper(points_map.state) = upper(contributor.state) 
	inner join census_data on points_map.zip = census_data.geo_id2
group by points_map.zip, points_map.lat, points_map.lon, mean_household_income, median_household_income; 
"""

cur.execute(unknownsQuery)
unknowns =cur.fetchall()



zipped = list(zip(*rows))

x = list(zipped[:-3])
twoYear, oneYear, ninetyDay, thirtyDay = map(lambda x: list(x), zipped[-4:])

categorized = list(map(lambda x: int(math.ceil(math.log(x))) if x > 0.0 else 0, twoYear))

#n, bins, patches = plt.hist(categorized, len(set(categorized)), normed=1, facecolor='green', alpha=0.75)

#print(categorized)

#plt.show()


xPoints = list(zip(*x))


twoYearPoints = list(map(lambda x: float(x), twoYear))#list(zip(twoYear))
oneYearPoints = list(oneYear)
ninetyDayPoints = list(ninetyDay)
thirtyDay = list(thirtyDay)

linear = 		SVC(**{'C' : 1.0, 'kernel' : 'linear', 							'max_iter' : MAX_ITER})
quadratic = SVC(**{'C' : 1.0, 'kernel' : 'poly', 'degree' : 2, 	'max_iter' : MAX_ITER})
cubic = 		SVC(**{'kernel' : 'poly', 'degree' : 3, 	'max_iter' : MAX_ITER})
treeRegressor = DecisionTreeClassifier(random_state=int(random() * 100.0))
forest10 = RandomForestClassifier(n_estimators=10, n_jobs=4)
forest15 = RandomForestClassifier(n_estimators=15, n_jobs=4)
forest20 = RandomForestClassifier(n_estimators=20, n_jobs=4)
net   = 		sklearn.linear_model.SGDClassifier(loss='squared_loss', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, n_iter=20000, shuffle=True, verbose=0, epsilon=0.1, random_state=None, learning_rate='invscaling', eta0=0.01, power_t=0.25, warm_start=False, average=False)

# linear, quadratic, cubic, treeRegressor, forest10, forest15,
# treeRegressor, forest10, forest15, 
modelList = [forest20]
Y = categorized

selectedFeatures = [1] * (len(xPoints[0]) -1)
selectedFeatures[0] = 0
selectedFeatures[1] = 0

oneHots = []

train_sizes=np.linspace(.1, 1.0, 10)

for model in modelList:
	print('##### Analyzing {0} #####\n'.format(model))
	xPointFiltered = featureFilter(xPoints, selectedFeatures)
	xPointFiltered = oneHotEncode(columns=list(zip(*xPointFiltered)), oneHotColumnIndices=oneHots)

	print(xPointFiltered[0])

	#print(xPointFiltered)
	"""
	cv = cross_validation.ShuffleSplit(len(xPointFiltered), n_iter=10,
                                   test_size=0.2, random_state=int(random() * 100.0))
	predictions = np.array(cross_validation.cross_val_predict(model, xPointFiltered, Y, cv=2))
	scores = cross_validation.cross_val_score(model, xPointFiltered, Y, cv=cv, n_jobs=4)
	"""

	model.fit(xPointFiltered, Y)

	#importances = model.feature_importances_
	#std = np.std([model.feature_importances_ for tree in model.estimators_], axis=0)

	#indices = np.argsort(importances)[::-1]

	# Print the feature ranking
	#print("Feature ranking:")

	#for f in range(len(xPointFiltered[0])):
	   # print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

	# Plot the feature importances of the forest
	predictions = model.predict(unknowns) * -1

	l_p = list(predictions)

	best_index = l_p.index(min(l_p))
	print(unknowns[best_index], predictions[best_index])

	
	xlat, xlon = list(zip(*unknowns))[1:3]
	txlat, txlon = list(zip(*xPointFiltered))[1:3]


	x = np.array(xlat + txlat)
	y = np.array(xlon + txlon)


	gridsize=2000

	m  = Basemap(lon_0=-98.0, projection='robin')

	xpt, ypt = m(y, x)

	print(type(predictions))
	print(type(Y))

	totalLabels = Y + list(predictions)
	
	m.hexbin(xpt, ypt, C=totalLabels, gridsize = gridsize)
	#m.scatter(xpt, ypt, c=predictions)
	m.readshapefile('./tl_2013_40_prisecroads', 'Roads')
	
	m.drawcoastlines()
	m.drawparallels(np.arange(0,81,20))
	m.drawmeridians(np.arange(-180,181,60))
	m.colorbar() # draw colorbar
	plt.title('Donors')
	
	#m.hexbin(x1, y1, C=predictions, gridsize=gridsize, cmap=plt.cm.jet)
	#m.hexbin(x, y, gridsize=20, cmap=plt.cm.jet)
	#plt.axis([min(x), max(x), min(y), max(y)])
	#m.colorbar()

	plt.show() 

	
	
	"""
	#print(np.mean(scores))
	underasks = 0
	overgives = 0
	exacts = 0
	trueErrors = 0
	for prediction, actual in zip(predictions, Y):
		if int(math.fabs(prediction - actual)) < 3:
			if prediction == actual:
				exacts += 1
			if prediction > actual:
				underasks += 1
			else:
				overgives += 1
		else:
			trueErrors += 1

	print("Under : {0}  Exact : {1} Overgives : {2}  ERROR : {3}".format(underasks, exacts, overgives, trueErrors))

	close = np.vectorize(lambda x: 1 if x < 3 else 0)
	err = np.abs(np.subtract(predictions, Y))
	numCorrect = float(np.sum(close(err)))
	print(numCorrect / float(len(predictions)))
	"""
