import pandas as pd
csv_path = '/Users/guffrey/local-chatbot-service/data/csv/data.csv'
df = pd.read_csv(csv_path)
tax_year_105 = df[(df['yyymm'] >= 10501) & (df['yyymm'] <= 10512) & (df['item_of_tax'] == '地價稅')]
total_amount_105_land_tax = sum(tax_year_105['actual_collection_c_thousandtwd'])
result = total_amount_105_land_tax
print(result)