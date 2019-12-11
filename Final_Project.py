# Group members: Yu Zhu, Hua Li, , Chenming Xu
# Hypothesis 1: Chenming Xu
# Hypothesis 2 & 3: Yu Zhu
# Hypothesis 4: Hua Li

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ratio = pd.read_csv('Male-Female-Ratio-of-Suicide-Rates.csv')

# Hypothesis 1:
frame = pd.DataFrame(ratio,columns=['Entity','Code','Year','Male_female_suicide_ratio'])

frame['higher_male'] = frame.Male_female_suicide_ratio.apply(lambda x: 1 if x>1 else 0)
result = frame.groupby(['Entity','Year'])['higher_male'].value_counts().unstack()

uk = frame[frame['Entity'] == 'United Kingdom']
result_uk = uk.groupby(['Entity','Year'])['higher_male'].value_counts().unstack()
result_uk.plot.bar()
plt.show()

china = frame[frame['Entity'] == 'China']
result_china = china.groupby(['Entity','Year'])['higher_male'].value_counts().unstack()
result_china.plot.bar()
plt.show()

morocco = frame[frame['Entity'] == 'Morocco']
result_morocco = morocco.groupby(['Entity','Year'])['higher_male'].value_counts().unstack()
result_morocco.plot.bar()
plt.show()

# Hypothesis 2:
suicide = pd.read_csv('suicide-rate-1990-2017.csv', sep = ',')
suicide = suicide[suicide['Year'] >= 2000]
suicide = suicide[suicide['Year'] <= 2016]

GDP = pd.read_excel('GDP.xls', sep=',')[['country', 'year', ' gdp_for_year ($) ']]
GDP = GDP[GDP['year'] >= 2000]
GDP = GDP[GDP['year'] <= 2016]
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

combine = pd.merge(suicide_entity_year, gdp_entity_year, left_on=['Entity', 'Year'],right_on=['country', 'year'],
                   right_index=True, how='inner')

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

#group countries with their predominant religion
Mus=pre_religion[pre_religion['Religion 1']=='Muslims']
Muslims=Mus['Country Name']
Chris=pre_religion[pre_religion['Religion 1']=='Christians']
Christians=Chris['Country Name']
Budd=pre_religion[pre_religion['Religion 1']=='Buddhists']
Buddhists=Budd['Country Name']
Hin=pre_religion[pre_religion['Religion 1']=='Hindus']
Hindus=Hin['Country Name']
Agn=pre_religion[pre_religion['Religion 1']=='Agnostics']
Agnostics=Agn['Country Name']
Chi=pre_religion[pre_religion['Religion 1']=='Chinese folk-religionists']
Chinese_folk_religionists=Chi['Country Name']
J=pre_religion[pre_religion['Religion 1']=='Jews']
Jews=J['Country Name']

#count mean suicide rate of each religion groups
Muslims=pd.merge(country_rate,Muslims,left_on='Entity',right_on='Country Name')
print('Muslims suicide rate=', Muslims['Suicide rate (deaths per 100,000)'].mean())

Christians=pd.merge(country_rate,Christians,left_on='Entity',right_on='Country Name')
print('Christians suicide rate=', Christians['Suicide rate (deaths per 100,000)'].mean())

Buddhists=pd.merge(country_rate,Buddhists,left_on='Entity',right_on='Country Name')
print('Buddhists suicide rate=', Buddhists['Suicide rate (deaths per 100,000)'].mean())

Hindus=pd.merge(country_rate,Hindus,left_on='Entity',right_on='Country Name')
print('Hindus suicide rate=', Hindus['Suicide rate (deaths per 100,000)'].mean())

Agnostics=pd.merge(country_rate,Agnostics,left_on='Entity',right_on='Country Name')
print('Agnostics suicide rate=', Agnostics['Suicide rate (deaths per 100,000)'].mean())

Chinese_folk_religionists=pd.merge(country_rate,Chinese_folk_religionists,left_on='Entity',right_on='Country Name')
print('Chinese_folk_religionists suicide rate=', Chinese_folk_religionists['Suicide rate (deaths per 100,000)'].mean())

Jews=pd.merge(country_rate,Jews,left_on='Entity',right_on='Country Name')
print('Jews suicide rate=', Jews['Suicide rate (deaths per 100,000)'].mean())

#get mean suicide rate of each religion of each year

# plt.plot(years['years'], groups[groups['E']])

a = pd.merge(pre_religion, df, left_on='Country Name', right_on='Entity', right_index=True, how='inner')
groups=a.groupby(['Year','Religion 1'],as_index=False)['Suicide rate (deaths per 100,000)'].mean()

years = {'years': range(2000, 2016)}

plt.plot(years['years'], groups[groups['Religion 1']=='Agnostics']['Suicide rate (deaths per 100,000)'],label='Agnostics')
plt.plot(years['years'], groups[groups['Religion 1']=='Christians']['Suicide rate (deaths per 100,000)'],label='Christians')
plt.plot(years['years'], groups[groups['Religion 1']=='Muslims']['Suicide rate (deaths per 100,000)'],label='Muslims')
plt.plot(years['years'], groups[groups['Religion 1']=='Chinese folk-religionists']['Suicide rate (deaths per 100,000)'],label='Chinese folk-religionists')
plt.plot(years['years'], groups[groups['Religion 1']=='Jews']['Suicide rate (deaths per 100,000)'],label='Jews')
plt.plot(years['years'], groups[groups['Religion 1']=='Buddhists']['Suicide rate (deaths per 100,000)'],label='Buddhists')
plt.plot(years['years'], groups[groups['Religion 1']=='Hindus']['Suicide rate (deaths per 100,000)'],label='Hindus')
plt.legend()
plt.show()
