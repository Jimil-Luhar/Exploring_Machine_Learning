import numpy as np
import matplotlib.pyplot as plt

advertising_costs = np.array([100, 200, 300, 400, 500]) #Data Set
sales = np.array([300, 500, 600, 700, 800])

mean_adv = np.mean(advertising_costs) #Mean Computation
mean_sales = np.mean(sales)

m = np.sum((advertising_costs - mean_adv) * (sales - mean_sales)) / np.sum((advertising_costs - mean_adv)**2)
c = mean_sales - m * mean_adv #Coefficients Calculation

print(f"Model: Sales = {c:.2f} + {m:.2f}*Advertising_Costs") #Finally; Model: Sales = 220.00 + 1.20*Advertising_Costs
#'f' is used to format the evaluated expression inside "", as a string
#'{c:.2f}' this embeds the c variable into floating string

plt.scatter(advertising_costs, sales, color = 'blue')
plt.plot(advertising_costs, c + m * advertising_costs, color = 'red')
plt.xlabel('Advertising Costs')
plt.ylabel('Sales')
plt.title('Advertising Costs vs Sales')
plt.show()
