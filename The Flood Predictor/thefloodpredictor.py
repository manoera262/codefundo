# -*- coding: utf-8 -*-
"""
Created on Wed Oct 24 20:37:29 2018

@author: Brogrammers
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 12:12:33 2018

@author: Brogrammers
"""


#IMPORTING LIBRARIES
import pandas as pd
from pandas import DataFrame
from sklearn import linear_model
import tkinter as tk 
import statsmodels.api as sm



#Dictionary used for numbering states alphabetically
states={'andhrapradesh': 1,
 'arunachalpradesh': 2,
 'assam': 3,
 'bihar': 4, 'chhattisgarh': 5,
 'goa': 6, 'gujarat': 7, 'haryana': 8, 
 'himachalpradesh': 9, 'jammuandkashmir': 10,
 'jharkhand': 11, 'karnataka': 12, 'kerala': 13,
 'madhyapradesh': 14, 'maharashtra': 15, 
 'manipur': 16, 'meghalaya': 17, 'mizoram': 18, 
 'nagaland': 19, 'odisha': 20, 'punjab': 21, 'rajasthan': 22, 
 'sikkim': 23, 'tamilnadu': 24, 'telangana': 25, 'tripura': 26,
 'uttarpradesh': 27, 'uttarakhand': 28, 'westbengal': 29,
 'newdelhi': 0}

#DTASET CONTAINS AVERAGE YEARLY DATA FOR YEAR 2015,2016,2017 FOR EACH STATE NUMBERED 1 TO 29 ALPHABETICALY
dataset={'state': [10.0, 10.0, 10.0, 9.0, 9.0, 9.0, 28.0, 28.0, 28.0, 21.0, 21.0, 21.0, 8.0, 8.0, 8.0, 22.0, 22.0, 22.0, 27.0, 27.0, 27.0, 4.0, 4.0, 4.0, 11.0, 11.0, 11.0, 29.0, 29.0, 29.0, 20.0, 20.0, 20.0, 14.0, 14.0, 14.0, 5.0, 5.0, 5.0, 7.0, 7.0, 7.0, 15.0, 15.0, 15.0, 6.0, 6.0, 6.0, 25.0, 25.0, 25.0, 1.0, 1.0, 1.0, 24.0, 24.0, 24.0, 12.0, 12.0, 12.0, 13.0, 13.0, 13.0, 23.0, 23.0, 23.0, 3.0, 3.0, 3.0, 17.0, 17.0, 17.0, 2.0, 2.0, 2.0, 19.0, 19.0, 19.0, 16.0, 16.0, 16.0, 18.0, 18.0, 18.0, 26.0, 26.0, 26.0, 0.0, 0.0], 
 'temperature': [13.44, 14.6, 13.78, 23.59, 24.8, 23.2, 13.3, 12.0, 11.5, 24.95, 25.6, 25.0, 24.02, 25.3, 24.7, 26.72, 27.0, 25.3, 25.85, 24.08, 25.3, 24.31, 23.3, 24.07, 25.85, 25.2, 25.0, 26.92, 27.0, 26.4, 27.44, 26.8, 26.1, 25.35, 24.7, 26.9, 25.01, 24.2, 25.0, 27.42, 26.8, 25.1, 27.5, 25.9, 26.5, 27.3, 26.0, 26.6, 28.97, 27.7, 28.0, 28.97, 28.6, 27.0, 29.07, 31.0, 29.5, 24.1, 25.5, 24.4, 27.4, 28.6, 28.0, 22.38, 23.0, 24.3, 23.08, 23.0, 22.3, 23.05, 24.4, 22.7, 23.05, 22.8, 22.76, 21.91, 21.8, 20.3, 22.75, 22.6, 21.6, 23.13, 24.06, 22.4, 23.0, 22.5, 23.7, 25.0, 30.0],
 'relativehumidity': [70.41, 68.0, 69.2, 73.0, 70.0, 72.7, 71.0, 72.0, 73.4, 64.0, 61.0, 63.2, 58.0, 56.4, 57.6, 43.66, 42.0, 45.3, 62.4, 64.2, 63.0, 67.6, 70.0, 69.3, 61.2, 63.6, 62.0, 76.58, 75.2, 77.0, 70.25, 71.2, 72.5, 50.8, 53.0, 49.7, 53.0, 54.8, 52.7, 54.91, 55.6, 57.5, 74.91, 77.0, 75.6, 75.91, 78.6, 76.2, 56.33, 58.0, 56.9, 59.0, 58.2, 57.0, 69.7, 52.0, 63.2, 65.16, 57.2, 65.0, 78.33, 71.0, 77.9, 73.0, 73.9, 74.0, 76.0, 75.4, 77.8, 79.0, 77.6, 81.0, 81.99, 82.4, 82.3, 80.0, 81.0, 84.0, 80.0, 81.0, 83.8, 83.0, 82.3, 85.0, 81.0, 80.5, 81.6, 62.0, 60.0], 
 'windspeed': [3.0, 2.8, 3.1, 5.0, 5.3, 4.7, 5.0, 5.3, 5.6, 4.7, 5.0, 5.1, 4.3, 4.7, 5.2, 6.0, 6.2, 5.8, 5.0, 4.6, 5.2, 5.4, 5.3, 4.8, 4.7, 5.3, 4.9, 4.0, 4.6, 5.0, 7.0, 6.8, 7.3, 4.0, 4.4, 5.0, 4.3, 4.0, 5.3, 5.0, 4.7, 5.7, 6.0, 5.8, 6.2, 7.0, 7.22, 6.4, 4.0, 4.3, 4.8, 6.0, 4.92, 5.7, 6.6, 6.4, 6.12, 6.3, 5.7, 5.9, 9.0, 9.6, 8.98, 4.7, 5.0, 5.2, 5.0, 5.2, 4.8, 5.0, 5.2, 4.87, 5.0, 4.8, 5.12, 4.0, 4.7, 4.1, 4.0, 4.3, 4.05, 4.5, 4.7, 4.3, 4.7, 5.0, 4.5, 4.3, 4.1],
 'pressure': [1.023, 1.022, 1.024, 1.02, 1.021, 1.019, 1.019, 1.022, 1.017, 1.016, 1.018, 1.019, 1.018, 1.016, 1.015, 1.016, 1.018, 1.019, 1.015, 1.018, 1.017, 1.014, 1.012, 1.011, 1.014, 1.016, 1.015, 1.014, 1.013, 1.015, 1.015, 1.017, 1.02, 1.013, 1.015, 1.012, 1.018, 1.022, 1.019, 1.015, 1.017, 1.015, 1.013, 1.014, 1.01, 1.013, 1.011, 1.016, 1.018, 1.02, 1.016, 1.019, 1.016, 1.017, 1.017, 1.02, 1.017, 1.019, 1.016, 1.015, 1.012, 1.009, 1.01, 1.02, 1.023, 1.021, 1.018, 1.019, 1.017, 1.019, 1.022, 1.017, 1.023, 1.02, 1.025, 1.018, 1.016, 1.021, 1.017, 1.019, 1.016, 1.019, 1.021, 1.018, 1.016, 1.019, 1.018, 1.016, 1.011],
 'rainfall ': [1572.6, 902.8, 1278.9, 1223.2, 921.5, 1182.2, 1247.6, 1308.6, 1476.0, 612.6, 444.0, 497.5, 437.8, 398.7, 421.6, 458.6, 347.2, 600.3, 592.3, 871.7, 695.0, 874.0, 1158.4, 1112.0, 1085.6, 1264.0, 1165.8, 1507.7, 1427.0, 1568.0, 1210.1, 1253.0, 1344.5, 1045.4, 1340.0, 840.0, 1136.0, 1315.8, 1124.0, 677.6, 764.9, 1024.0, 629.5, 1163.0, 994.0, 2191.5, 3663.9, 3443.0, 747.9, 1043.0, 815.7, 987.9, 908.3, 892.0, 1204.5, 535.0, 973.0, 1083.8, 687.5, 1061.0, 2602.9, 1870.9, 2664.0, 2501.4, 2624.0, 2684.9, 2308.3, 2266.4, 2711.0, 2624.9, 2356.0, 2803.0, 2593.2, 2706.8, 2745.0, 1947.4, 1956.2, 2805.0, 1840.0, 1902.0, 2769.0, 2013.0, 1992.0, 2842.0, 2200.0, 1905.0, 2394.0, 612.6, 700.1]
 }


df = DataFrame(dataset,columns=['state','temperature','relativehumidity','windspeed','pressure','rainfall ']) 

X = df[['state','temperature','relativehumidity','windspeed','pressure']] # here we have 2 input variables for multiple regression. If you just want to use one variable for simple linear regression, then use X = df['Interest_Rate'] for example.Alternatively, you may add additional variables within the brackets
Y = df['rainfall '] # output variable (what we are trying to predict)

# with sklearn FITTING DATA
regr = linear_model.LinearRegression()
regr.fit(X, Y)

print('Intercept of the graph: \n\n', regr.intercept_)
print('Coefficients : \n\n', regr.coef_)

# Adding a constant FOR BETTER RESULT WITH STATESMODELS

X = sm.add_constant(X)
 
model = sm.OLS(Y, X).fit()
predictions = model.predict(X) 
 



root= tk.Tk() 
root.title("THE FLOOD PREDICTOR")
root.resizable(False, False)
canvas1 = tk.Canvas(root, width = 1200, height = 450,bg="blue")
canvas1.pack()


#PRINTING INTERCEPTS
#Intercept_result = ('Intercept: ', regr.intercept_)
#label_Intercept = tk.Label(root, text=Intercept_result, justify = 'center')
#canvas1.create_window(270, 260, window=label_Intercept)


#PRINTING COEFFICIENTS
#Coefficients_result  = ('Coefficients: ', regr.coef_)
#label_Coefficients = tk.Label(root, text=Coefficients_result, justify = 'center')
#canvas1.create_window(270, 285, window=label_Coefficients)

#With statsmodels
print_model = model.summary()
label_model = tk.Label(root, text=print_model, justify = 'center', relief = 'solid', bg='Blue',fg="yellow")
canvas1.create_window(900, 250, window=label_model)


#adding title
label0 = tk.Label(root, text='THE FLOOD PREDICTOR',justify='center',bg="blue",fg="yellow",font=('times', 30, 'bold'))
canvas1.create_window(260, 60, window=label0)


# inputting new state
label1 = tk.Label(root, text='ENTER STATE: ',justify='left',bg="blue",fg="yellow",font=('times', 10, 'bold'))
canvas1.create_window(100, 100, window=label1)

entry1 = tk.Entry (root) 
canvas1.create_window(270, 100, window=entry1)

# inputting new temperature
label2 = tk.Label(root, text='TEMPERATURE IN CELSIUS: ',justify='left',bg="blue",fg="yellow",font=('times', 10, 'bold'))
canvas1.create_window(100,120 , window=label2)

entry2 = tk.Entry (root) # create 2nd entry box
canvas1.create_window(270, 120, window=entry2)




#inputting new humidity
label3 = tk.Label(root, text='RELATIVE HUMIDITY (In %): ',justify='left',bg="blue",fg="yellow",font=('times', 10, 'bold'))
canvas1.create_window(100, 140, window=label3)

entry3 = tk.Entry (root) # create 2nd entry box
canvas1.create_window(270, 140, window=entry3)

# inputting new windspeed
label4 = tk.Label(root, text='WINDSPEED IN Km/hr: ',justify='left',bg="blue",fg="yellow",font=('times', 10, 'bold'))
canvas1.create_window(100, 160, window=label4)

entry4 = tk.Entry (root) # create 2nd entry box
canvas1.create_window(270, 160, window=entry4)

# inputting new pressure
label5 = tk.Label(root, text='PRESSURE (In Bar): ',justify='left',bg="blue",fg="yellow",font=('times', 10, 'bold'))
canvas1.create_window(100, 180, window=label5)

entry5 = tk.Entry (root) # create 2nd entry box
canvas1.create_window(270, 180, window=entry5)


def values(): 
    
    
    global New_state1 #our 1st input variable
    New_state1 = entry1.get()
    New_state1=New_state1.replace(' ','')
    New_state1=New_state1.lower()
    New_state=states.get(New_state1,11)
    
    
    global New_temperature #our 2nd input variable
    New_temperature = float(entry2.get()) 
    
    global New_humidity #our 3rd input variable
    New_humidity = float(entry3.get()) 
    
    global New_windspeed #our 4th input variable
    New_windspeed = float(entry4.get()) 
    
    global New_pressure #our 5th input variable
    New_pressure = float(entry5.get()) 
    
    
    w = regr.predict([[New_state,New_temperature ,New_humidity,New_windspeed,New_pressure]])
    if w<0:
        w=0
    
      
    Prediction_result  = ('Predicted rainfall in mm   :::   ',str(w))
    label_Prediction = tk.Label(root, text= Prediction_result, bg='blue',fg="yellow",font=('times', 15, 'bold'))
    canvas1.create_window(270, 320, window=label_Prediction)
    
    
    
    #possibility of FLOOD taking rainfall 00mm-> 0.00%flood and 3000mm->100%flood  scale equation y=x*1/30
    if w<3000:
        flood=w*1/30
    if w>3000:
          flood=100.00
    
        
    
    if flood>100:
        flood=100
    
    Prediction_result2  = ('Predicted flood percentage   :::    ',str(flood))
    label_Prediction1 = tk.Label(root, text= Prediction_result2, bg='blue',fg="yellow",font=('times', 15, 'bold'))
    canvas1.create_window(270, 345, window=label_Prediction1)
    print(flood)      
    
button1 = tk.Button (root, text='Predict',command=values, bg='red',font=('times', 15, 'bold')) # button to call the 'values' command above 
canvas1.create_window(270, 210, window=button1)
 

root.mainloop()
