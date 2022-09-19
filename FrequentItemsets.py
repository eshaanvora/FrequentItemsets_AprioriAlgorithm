#Eshaan Vora
#EshaanVora@gmail.com
#Frequent Itemsets Retail Data, Implementing Apriori Algorithm

import pandas as pd
from mlxtend.frequent_patterns import apriori

#Import Dataset
address = "RetailData.csv"

data = pd.read_csv(address)

#Convert to Pandas data frame
df = pd.DataFrame(data)

#Drop Unnecessary Variables
df = df.drop('StockCode', axis = 1)
df = df.drop('InvoiceDate', axis = 1)
df = df.drop('UnitPrice', axis = 1)
df = df.drop('CustomerID', axis = 1)

#Transform dataframe to have transaction by invoice number
#Filter by country to save memory; entire data frame is too large to transform
def byCountry(country):
    transactions = (df[df['Country'] == country]
              .groupby(['InvoiceNo', 'Description'])['Quantity']
              .sum().unstack().reset_index()
              .set_index('InvoiceNo'))
    return transactions

#Create dataframe for each country
countryList = data['Country'].unique()
transactions = {}
for i in countryList:
    if i != 'United Kingdom':
        transactions[i] = byCountry(i)

#Join dataframes from each country to create final clean dataset
final_df = pd.concat(transactions)

#Hot encode data; Convert values to binary values
def convert_to_binary(x):
    if x >= 1:
        return 1
    else:
        return 0

final_df = final_df.applymap(convert_to_binary)

#Calculate frequent itemsets with min support level 2%
#At higher levels of support, frequent itemset is blank
frequent_itemsets = apriori(final_df, min_support = 0.02, use_colnames = True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
frq_itemsets1 = frequent_itemsets[frequent_itemsets['length'] == 3]
print(frq_itemsets1)

#File output
text_file = open("frequent_itemsets.txt", "w")
text_file.write("Min Support Level: 0.02")
text_file.write(str(frq_itemsets1))
text_file.write("\n")
