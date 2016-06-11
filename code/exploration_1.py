# -*- coding: utf-8 -*-
"""
Created on Sun May 29 17:19:52 2016

@author: Giammi
"""

# SETUP ===========================================================================================
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pylab
import warnings
warnings.filterwarnings('ignore')
from pandas.tools.plotting import table

workspace = 'C:\\Users\\Giammi\\OneDrive\\Università\\1 anno\\2 semestre\\'
workspace += 'Data Mining & Text Mining\\kaggle competitions\\san francisco crime'
os.chdir(workspace)

train = './train.csv'
crimeData = pd.read_csv(train, parse_dates=['Dates'], index_col='Dates', delimiter=',')

workspace = "C:\\Users\\Giammi\\OneDrive\\Università\\Machine Learning\\project"
os.chdir(workspace)

head = crimeData.head(5)

ax = plt.subplot(411, frame_on=False) # no visible frame
ax.xaxis.set_visible(False)  # hide the x axis
ax.yaxis.set_visible(False)  # hide the y axis

table(ax, head)  # where head is your data frame

plt.savefig('mytable.png')


# INSPECTION ======================================================================================
pylab.rcParams['figure.figsize'] = (14.5, 6.0)

crimes_rating = crimeData['Category'].value_counts()
print ('San Francisco Crimes\n')
print ('Category\t\tNumber of occurences') 
print (crimes_rating)

top = 18
y_pos = np.arange(len(crimes_rating[0:top].keys()))

plt.barh(y_pos, crimes_rating[0:top].get_values(), align='center', alpha=0.4, color = 'blue')
plt.yticks(y_pos, [x.title() for x in crimes_rating[0:top].keys()], fontsize = 11)
plt.xlabel('Number of occurences', fontsize = 14)
plt.title('San Francisco Crime Classification', fontsize = 26)
plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
plt.savefig("crimes_occurences.png")

"""
Add new features to the dataset:
    Weekday (Monday, Tuesday, ...)
    Hour of day
    Month
    Year
    Day of month
"""
crimeData['DayOfWeek'] = crimeData.index.dayofweek
crimeData['Hour'] = crimeData.index.hour
crimeData['Month'] = crimeData.index.month
crimeData['Year'] = crimeData.index.year
crimeData['DayOfMonth'] = crimeData.index.day

# Crimes per hour =================================================================================
# Collect the most frequent crimes (see analysis above)
larceny = crimeData[crimeData['Category'] == "LARCENY/THEFT"]
assault = crimeData[crimeData['Category'] == "ASSAULT"]
drug = crimeData[crimeData['Category'] == "DRUG/NARCOTIC"]
vehicle = crimeData[crimeData['Category'] == "VEHICLE THEFT"]
vandalism = crimeData[crimeData['Category'] == "VANDALISM"]
burglary = crimeData[crimeData['Category'] == "BURGLARY"]


pylab.rcParams['figure.figsize'] = (16.0, 10.0)
pylab.rcParams['figure.dpi'] = 80

with plt.style.context('fivethirtyeight'):
    ax1 = plt.subplot2grid((3,3), (0,0), colspan=3)
    ax1.plot(crimeData.groupby('Hour').size(), 'ro-')
    ax1.set_title ('All crimes', fontsize=16)
    start, end = ax1.get_xlim()
    ax1.xaxis.set_ticks(np.arange(start, end, 1))
    
    ax2 = plt.subplot2grid((3,3), (1, 0))
    ax2.plot(larceny.groupby('Hour').size(), 'o-')
    ax2.set_title ('Larceny/Theft', fontsize=12)
    
    ax3 = plt.subplot2grid((3,3), (1, 1))
    ax3.plot(assault.groupby('Hour').size(), 'o-')
    ax3.set_title ('Assault')
    
    ax4 = plt.subplot2grid((3,3), (1, 2))
    ax4.plot(drug.groupby('Hour').size(), 'o-')
    ax4.set_title ('Drug/Narcotic', fontsize=12)
    
    ax5 = plt.subplot2grid((3,3), (2, 0))
    ax5.plot(vehicle.groupby('Hour').size(), 'o-')
    ax5.set_title ('Vehicle', fontsize=12)
    
    ax6 = plt.subplot2grid((3,3), (2, 1))
    ax6.plot(vandalism.groupby('Hour').size(), 'o-')
    ax6.set_title ('Vandalism', fontsize=12)
    
    ax7 = plt.subplot2grid((3,3), (2, 2))
    ax7.plot(burglary.groupby('Hour').size(), 'o-')
    ax7.set_title ('Burglary', fontsize=12)

    pylab.gcf().text(0.5, 0.99, 'San Francisco Crimes: Occurences per Hour',
                    horizontalalignment='center', verticalalignment='top', fontsize = 24)
    
plt.tight_layout(3)
plt.savefig("crimes_per_hour.png")
plt.show()



# Crimes per day of week =========================================================================
pylab.rcParams['figure.figsize'] = (16.0, 10.0)
plt.style.use('ggplot')

daysOfWeekIdx = crimeData.groupby('DayOfWeek').size().keys()
occursByWeek  = crimeData.groupby('DayOfWeek').size().get_values()
daysOfWeekLit = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

# Linear plot for all crimes
ax1 = plt.subplot2grid((2,3), (0,0), colspan=3)
ax1.plot(daysOfWeekIdx, occursByWeek, 'ro-', linewidth=2)
ax1.set_xticklabels(daysOfWeekLit)
ax1.set_title ('All Crimes', fontsize=18)
# ensure that ticks are only at the bottom and left parts of the plot
ax1.get_xaxis().tick_bottom()
ax1.get_yaxis().tick_left()

# Bar plot
y = np.empty([6,7]) # top6 crimes, 7 days in a week
h = [None]*6
width = 0.1

ax2 = plt.subplot2grid((2,3), (1,0), colspan=3)

y[0] = larceny.groupby('DayOfWeek').size().get_values()
y[1] = assault.groupby('DayOfWeek').size().get_values()
y[2] = drug.groupby('DayOfWeek').size().get_values()
y[3] = vehicle.groupby('DayOfWeek').size().get_values()
y[4] = vandalism.groupby('DayOfWeek').size().get_values()
y[5] = burglary.groupby('DayOfWeek').size().get_values()

color_sequence = ['#1f77b4', '#ff7f0e', '#2ca02c','#d62728', '#9467bd', '#8c564b']

for i in range(6):
    h[i] = ax2.bar(daysOfWeekIdx + i*width, y[i], width, color=color_sequence[i], alpha = 0.7)

ax2.set_xticks(daysOfWeekIdx + 3*width)
ax2.set_xticklabels(daysOfWeekLit)
# ensure that ticks are only at the bottom and left parts of the plot
ax2.get_xaxis().tick_bottom()
ax2.get_yaxis().tick_left()

ax2.legend((item[0] for item in h), 
           ('Larceny', 'Assault', 'Drug', 'Vehicle', 'Vandalism', 'Burglary'), 
           bbox_to_anchor=(0.88, 1), loc=2, borderaxespad=0., frameon=False)

pylab.gcf().text(0.5, 0.98, 
            'San Francisco Crimes: Occurences per Day Of Week',
            horizontalalignment='center',
            verticalalignment='top', 
             fontsize = 26)

plt.savefig("crimes_per_weekday.png")
plt.show()



# Crimes per month ===============================================================================
pylab.rcParams['figure.figsize'] = (16.0, 10.0)
plt.style.use('ggplot')

monthsIdx     = crimeData.groupby('Month').size().keys() - 1
occursByMonth = crimeData.groupby('Month').size().get_values()
monthsLit = ['January', 'February', 'March', 'April', 'May', 'June',
             'July','August', 'September', 'October', 'Novemeber', 'December']

# Linear plot for all crimes
ax1 = plt.subplot2grid((3,3), (0,0), colspan=3)
ax1.plot(monthsIdx, occursByMonth, 'ro-', linewidth=2)

ax1.set_title ('All Crimes', fontsize=16)

start, end = ax1.get_xlim()
ax1.xaxis.set_ticks(np.arange(start, end, 1))
ax1.set_xticklabels(monthsLit)
# ensure that ticks are only at the bottom and left parts of the plot
ax1.get_xaxis().tick_bottom()
ax1.get_yaxis().tick_left()

# Linear normalized plot for 6 top crimes
ax2 = plt.subplot2grid((3,3), (1,0), colspan=3, rowspan=2)

y = np.empty([6,12])
y[0] = larceny.groupby('Month').size().get_values()
y[1] = assault.groupby('Month').size().get_values()
y[2] = drug.groupby('Month').size().get_values()
y[3] = vehicle.groupby('Month').size().get_values()
y[4] = vandalism.groupby('Month').size().get_values()
y[5] = burglary.groupby('Month').size().get_values()

crimes = ['Larceny/theft', 'Assault', 'Drug', 'Vehicle', 'Vandalism', 'Burglary']
color_sequence = ['#1f77b4', '#ff7f0e', '#2ca02c','#d62728', '#9467bd', '#8c564b']

for i in range(0,6):
    y[i]= (y[i]-min(y[i]))/(max(y[i])-min(y[i]))  # normalization
    h[i] = ax2.plot(monthsIdx, y[i],'o-', color=color_sequence[i], lw=2)

ax2.set_ylabel("Crime occurences by month, normalized")

ax2.xaxis.set_ticks(np.arange(start, end+2, 1))
ax2.set_xticklabels(monthsLit)

ax2.legend((item[0] for item in h), 
           crimes, 
           bbox_to_anchor=(0.87, 1), loc=2, borderaxespad=0., frameon=False)

pylab.gcf().text(0.5, 0.98, 
            'San Francisco Crimes: Occurences per Month',
            horizontalalignment='center',
            verticalalignment='top', 
             fontsize = 26)
             
plt.savefig("crimes_per_month.png")
plt.show()



# Crimes per year 2003 - 2015 ====================================================================
pylab.rcParams['figure.figsize'] = (16.0, 10.0)
plt.style.use('ggplot')

years        = crimeData.groupby('Year').size().keys()
occursByYear = crimeData.groupby('Year').size().get_values()

# Linear plot for all crimes
ax1 = plt.subplot2grid((3,3), (0,0), colspan=3)
ax1.plot(years, occursByYear, 'ro-', linewidth=2)

ax1.set_title ('All Crimes', fontsize=18)

start, end = ax1.get_xlim()
ax1.xaxis.set_ticks(np.arange(start, end, 1))
# ensure that ticks are only at the bottom and left parts of the plot
ax1.get_xaxis().tick_bottom()
ax1.get_yaxis().tick_left()

# Linear normalized plot for 6 top crimes
ax2 = plt.subplot2grid((3,3), (1,0), colspan=3, rowspan=2)

y = np.empty([6,13])
y[0] = larceny.groupby('Year').size().get_values()
y[1] = assault.groupby('Year').size().get_values()
y[2] = drug.groupby('Year').size().get_values()
y[3] = vehicle.groupby('Year').size().get_values()
y[4] = vandalism.groupby('Year').size().get_values()
y[5] = burglary.groupby('Year').size().get_values()

for i in range(0,6):
    h[i] = ax2.plot(years, y[i],'o-', color=color_sequence[i], lw=2)

ax2.set_ylabel("Crime occurences by year")

start, end = ax2.get_xlim()  
ax2.xaxis.set_ticks(np.arange(start, end+2, 1))

ax2.legend((item[0] for item in h), 
           crimes, 
           bbox_to_anchor=(0.87, 1), loc=2, borderaxespad=0., frameon=False)

pylab.gcf().text(0.5, 0.98, 
            'San Francisco Crimes: Occurences per Year',
            horizontalalignment='center',
            verticalalignment='top', 
             fontsize = 26)

plt.savefig("crimes_per_year.png")
plt.show()



# Crimes per year & month ========================================================================
pylab.rcParams['figure.figsize'] = (16.0, 6.0)
yearMonth = crimeData.groupby(['Year','Month']).size()
ax = yearMonth.plot(lw=2)
plt.title('San Francisco Crimes: Trend per Month&Year', fontsize=24)
plt.savefig("crimes_per_year&month.png")
plt.show()



# Crimes per PdDistrict ==========================================================================
pylab.rcParams['figure.figsize'] = (16.0, 10.0)
plt.style.use('ggplot')

districts = crimeData.groupby('PdDistrict').size().keys()
occursByWeek  = crimeData.groupby('PdDistrict').size().get_values()
districtsIdx = np.linspace(0, len(districts)-1, len(districts))

# Bar plots
y = np.empty([6, len(districts)]) # top6 crimes, 10 districts
h = [None]*6
width = 0.1

ax2 = plt.subplot2grid((1,3), (0,0), colspan=3)

y[0] = larceny.groupby('PdDistrict').size().get_values()
y[1] = assault.groupby('PdDistrict').size().get_values()
y[2] = drug.groupby('PdDistrict').size().get_values()
y[3] = vehicle.groupby('PdDistrict').size().get_values()
y[4] = vandalism.groupby('PdDistrict').size().get_values()
y[5] = burglary.groupby('PdDistrict').size().get_values()

color_sequence = ['#1f77b4', '#ff7f0e', '#2ca02c','#d62728', '#9467bd', '#8c564b']

for i in range(6):
    h[i] = ax2.bar(districtsIdx + i*width, y[i], width, color=color_sequence[i], alpha = 0.7)

ax2.set_xticks(districtsIdx + 3*width)
ax2.set_xticklabels(districts)
# ensure that ticks are only at the bottom and left parts of the plot
ax2.get_xaxis().tick_bottom()
ax2.get_yaxis().tick_left()

ax2.legend((item[0] for item in h), 
           ('Larceny', 'Assault', 'Drug', 'Vehicle', 'Vandalism', 'Burglary'), 
           bbox_to_anchor=(0.88, 1), loc=2, borderaxespad=0., frameon=False)

pylab.gcf().text(0.5, 0.96, 
            'San Francisco Crimes: Occurences per Police Department District',
            horizontalalignment='center',
            verticalalignment='top', 
             fontsize = 26)

plt.savefig("crimes_per_pddistrict.png")
plt.show()



# Work on the address ============================================================================
crimeData['Block'] = crimeData.Address.str.contains('.?Block.?')
pylab.rcParams['figure.figsize'] = (16.0, 10.0)
plt.style.use('ggplot')

block = crimeData.groupby('Block').size().keys()
occursByBlock  = crimeData.groupby('Block').size().get_values()
blocksIdx = np.linspace(0, len(block)-1, len(block))

# Bar plots
y = np.empty([6, len(block)]) # top6 crimes, 10 districts
h = [None]*6
width = 0.1

ax2 = plt.subplot2grid((1,3), (0,0), colspan=3)

y[0] = larceny.groupby('Block').size().get_values()
y[1] = assault.groupby('Block').size().get_values()
y[2] = drug.groupby('Block').size().get_values()
y[3] = vehicle.groupby('Block').size().get_values()
y[4] = vandalism.groupby('Block').size().get_values()
y[5] = burglary.groupby('Block').size().get_values()

color_sequence = ['#1f77b4', '#ff7f0e', '#2ca02c','#d62728', '#9467bd', '#8c564b']

for i in range(6):
    h[i] = ax2.bar(blocksIdx + i*width, y[i], width, color=color_sequence[i], alpha = 0.7)

ax2.set_xticks(blocksIdx + 3*width)
ax2.set_xticklabels(block)
# ensure that ticks are only at the bottom and left parts of the plot
ax2.get_xaxis().tick_bottom()
ax2.get_yaxis().tick_left()

ax2.legend((item[0] for item in h), 
           ('Larceny', 'Assault', 'Drug', 'Vehicle', 'Vandalism', 'Burglary'), 
           bbox_to_anchor=(0.88, 1), loc=2, borderaxespad=0., frameon=False)

pylab.gcf().text(0.5, 0.96, 
            'San Francicso Crimes: Occurences per Block or not',
            horizontalalignment='center',
            verticalalignment='top', 
             fontsize = 26)


plt.savefig("crimes_per_block.png")
plt.show()
