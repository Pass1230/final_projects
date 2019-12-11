# Group members: Yu Zhu, Hua Li, , Chenming Xu
# Hypothesis 1: Chenming Xu
# Hypothesis 2: Yu Zhu
# Hypothesis 3: Hua Li

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def condition(x,y,z):
    """
    :param x: the date frame that used
    :param y: the column choosed
    :param z: the condition applied
    :return: choosed dataframe
    >>> DF1=pd.DataFrame.from_items([('A', [1, 2, 3]), ('B', [4, 5, 6])],orient='index', columns=['one', 'two', 'three'])
    >>> condition(DF1,'one',1)
       one  two  three
    A    1    2      3
    """
    return x[x[y]==z]


def compare(a):
    """ Compare the ratio of male suicide to female suicide
     if ratio > 1, then male suicide rate is higher
     if ratio = 1, then the two rates are equal
     if ratio < 1, then female suicide rate is higher.

     :param a:list of gender suicide ratio
     :return: 1 or 0
    >>> data = {'city': ['Beijing', 'Shanghai', 'Guangzhou', 'Shenzhen', 'Hangzhou', 'Chongqing'], 'year': [2016,2016,2015,2017,2016, 2016], 'population_rate': [2.1, 2.3, 1.1, 0.7, 0.5, 0.5]}
    >>> frame = pd.DataFrame(data, columns = ['year', 'city', 'population_rate'])
    >>> frame['test'] = frame.apply(lambda x: compare(x.population_rate), axis = 1)
    >>> frame['test']
    0    1
    1    1
    2    1
    3    0
    4    0
    5    0
    Name: test, dtype: int64
     """
    if a>1:
        return 1
    else:
        return 0


def merge_default(a):
    """Merge two different dataset."""
    return pd.merge(country_rate,a,left_on='Entity',right_on='Country Name')


def specify_religion(data, data_name, religion_name, variable_name):
    """Extract the data of a specific column from the whole dataset."""
    subset = data[data[data_name] == religion_name]
    variable = subset[variable_name]
    return variable


def select_time(data, name):
    """Select the time interval."""
    data = data[data[name] >= 2000]
    data = data[data[name] <= 2016]
    return data


def groupby_data(data):
    """Group the data by Entity and Year."""
    output = data.groupby(['Entity','Year'])['higher_male'].value_counts()
    return output.unstack()


def compute_mean(data):
    """Compute the mean of the data."""
    output = data['Suicide rate (deaths per 100,000)']
    mean = output.mean()
    return mean


ratio = pd.read_csv('Male-Female-Ratio-of-Suicide-Rates.csv')

# Hypothesis 1:
frame = pd.DataFrame(ratio,columns=['Entity','Code','Year','Male_female_suicide_ratio'])

frame['higher_male'] = frame.apply(lambda x: compare(x.Male_female_suicide_ratio), axis=1)
result = groupby_data(frame)

uk = frame[frame['Entity'] == 'United Kingdom']
result_uk = groupby_data(uk)
result_uk.plot.bar()
plt.show()

china = frame[frame['Entity'] == 'China']
result_china = groupby_data(china)
result_china.plot.bar()
plt.show()

morocco = frame[frame['Entity'] == 'Morocco']
result_morocco = groupby_data(morocco)
result_morocco.plot.bar()
plt.show()



# Hypothesis 2:
suicide = pd.read_csv('suicide-rate-1990-2017.csv', sep = ',')
suicide = select_time(suicide, 'Year')

GDP = pd.read_excel('GDP.xls', sep=',')[['country', 'year', ' gdp_for_year ($) ']]
GDP = select_time(GDP, 'year')
GDP[' gdp_for_year ($) '] = GDP[' gdp_for_year ($) ']/10000000000

years = {'years': range(2000, 2017)}
years = pd.DataFrame(years)

# the trend of the suicide rate.
suicide_year = suicide.groupby('Year').sum()

plt.plot(years['years'], suicide_year['Suicide rate (deaths per 100,000)'])
plt.show()

# Merge the suicide and GDP dataset.
suicide_entity_year = suicide.groupby(['Entity', 'Year'])['Suicide rate (deaths per 100,000)'].sum()
gdp_entity_year = GDP.groupby(['country', 'year'])[' gdp_for_year ($) '].sum()

combine = pd.merge(suicide_entity_year, gdp_entity_year, left_on=['Entity', 'Year'],
                   right_on=['country', 'year'], right_index=True, how='inner')

country_suicide = combine.groupby(['Entity', 'Year'])[['Suicide rate (deaths per 100,000)']].agg(np.sum)
country_GDP = combine.groupby(['Entity'])[[' gdp_for_year ($) ']].agg(np.mean)

# Divide countries into high GDP countries and low GDP countries.
mean = np.mean(country_GDP[' gdp_for_year ($) '])
country_GDP['classify_gdp'] = country_GDP[' gdp_for_year ($) '] > mean

newcombine = pd.merge(country_suicide, country_GDP['classify_gdp'], left_on='Entity', right_on='Entity',
                      right_index=True, how='inner')

# Draw the scatter plot of suicide rate in low and high gdp countries.
plt.plot(newcombine['classify_gdp'], newcombine['Suicide rate (deaths per 100,000)'], 'ro')
plt.show()

newgroup = newcombine.groupby(['classify_gdp', 'Year'], as_index=False)['Suicide rate (deaths per 100,000)'].agg(np.mean)

# Draw the suicide rate in low and high gdp countries respectively.
plt.plot(years['years'], newgroup[newgroup['classify_gdp'] == False]['Suicide rate (deaths per 100,000)'], label='Low GDP')
plt.plot(years['years'], newgroup[newgroup['classify_gdp'] == True]['Suicide rate (deaths per 100,000)'], label='High GDP')
plt.legend()
plt.show()

# Hypothesis 3:
pd.set_option('display.max_columns', 1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.max_rows', 500)

#get the mean suicide rate of each country from 2000-2015
df=pd.read_csv('suicide-rate-1990-2017.csv',sep=',', index_col=0)
special_year=['2000','2001','2002','2003','2004','2005','2006','2007','2008','2009','2010','2011','2012','2013','2014','2015']
df=df[df['Year'].isin(special_year)]
country_rate=df.groupby('Entity')['Suicide rate (deaths per 100,000)'].mean()
country_rate=country_rate.to_frame()


#find each country's predominant religion
religion=pd.read_excel('Religions1.xlsx',engine="openpyxl",header=1)
pre_religion=religion.iloc[range(0,4211,18)]

# Group countries with their predominant religion
Mus= condition(pre_religion,'Religion 1', 'Muslims')
Muslims=Mus['Country Name']
Chris=condition(pre_religion,'Religion 1', 'Christians')
Christians=Chris['Country Name']
Budd=condition(pre_religion,'Religion 1', 'Buddhists')
Buddhists=Budd['Country Name']
Hin=condition(pre_religion,'Religion 1', 'Hindus')
Hindus=Hin['Country Name']
Agn=condition(pre_religion,'Religion 1', 'Agnostics')
Agnostics=Agn['Country Name']
Chi=condition(pre_religion,'Religion 1', 'Chinese folk-religionists')
Chinese_folk_religionists=Chi['Country Name']
J=condition(pre_religion,'Religion 1', 'Jews')
Jews=J['Country Name']

Muslims=merge_default(Muslims)
Christians=merge_default(Christians)
Buddhists=merge_default(Buddhists)
Hindus=merge_default(Hindus)
Agnostics=merge_default(Agnostics)
Chinese_folk_religionists=pd.merge(country_rate,Chinese_folk_religionists,left_on='Entity',right_on='Country Name')
Jews=merge_default(Jews)

#count mean suicide rate of each religion groups


print('Muslims suicide rate=', compute_mean(Muslims))
print('Christians suicide rate=', compute_mean(Christians))
print('Buddhists suicide rate=', compute_mean(Buddhists))
print('Hindus suicide rate=', compute_mean(Hindus))
print('Agnostics suicide rate=', compute_mean(Agnostics))
print('Chinese_folk_religionists suicide rate=', compute_mean(Chinese_folk_religionists))
print('Jews suicide rate=', compute_mean(Jews))

#get mean suicide rate of each religion of each year

# plt.plot(years['years'], groups[groups['E']])

a = pd.merge(pre_religion, df, left_on='Country Name', right_on='Entity', right_index=True, how='inner')
groups=a.groupby(['Year','Religion 1'],as_index=False)['Suicide rate (deaths per 100,000)'].mean()

years = {'years': range(2000, 2016)}

religions = ['Agnostics', 'Christians', 'Muslims', 'Chinese folk-religionists',
             'Jews', 'Buddhists', 'Hindus']

for religion in religions:
    print(religion, specify_religion(groups, 'Religion 1', religion,
                'Suicide rate (deaths per 100,000)'))
    plt.plot(years['years'], specify_religion(groups, 'Religion 1', religion,
                                              'Suicide rate (deaths per 100,000)'), label=religion)
plt.legend()
plt.show()
